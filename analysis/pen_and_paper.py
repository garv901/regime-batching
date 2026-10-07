"""Pen-and-paper gain search: analytical step-cost model + vLLM-like continuous-batching simulator.

Goal: rank (model x workload x load) scenarios by the gain a regime-aware attention dispatcher could
deliver over vLLM 0.30 defaults (FLASH_ATTN/FA3, FULL_AND_PIECEWISE, max_num_batched_tokens=8192,
max_cudagraph_capture_size=512), before touching a GPU.

Baseline FA3 split behaviour encoded from source (vllm v0.30.0 flash_attn.py:805-815 and
vllm-flash-attention@506341a hopper/flash_api.cpp:get_num_splits, flash_prepare_scheduler.cu):
  * num_actual_tokens <= 512 (full-CG range): max_num_splits=32 -> per-sequence dynamic splits
    proportional to KV length -> load balanced.
  * num_actual_tokens  > 512: num_splits=0 -> heuristic with batch treated as 1:
    total_mblocks = H_kv * ceil(max_q * g / kBlockM). If >= 0.8*132 -> 1 split for EVERY sequence
    -> each long-KV decode is streamed by one CTA per kv-head (straggler).
Proposed: phase/lane-split attention -> decode rows always get balanced (dynamically split) kernels.

All constants are deliberately simple; uncertain ones (single-CTA streaming BW) are swept and will be
calibrated by the S2 microbenchmarks.
"""
from __future__ import annotations

import itertools
import json
import math
import sys
from dataclasses import dataclass, asdict
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, __file__.rsplit("/analysis/", 1)[0])
from regime_batching.features import fa3_static_splits, fa3_tile_time  # noqa: E402

NSM = 132
BW_W = 2.29e12     # weight streaming, in-engine fit (xstress_16k B0: decode floor 6.6 ms for 15.1 GB)
BW_KV = 2.85e12    # balanced paged-KV decode attention, in-engine fit (0.35 ms/GB)
F_GEMM = 480e12    # effective dense step throughput incl. elementwise ops (in-engine: T=512 prefill 15 ms)
F_ATT = 450e12     # achievable prefill attention
OVH_FULLCG = 0.10e-3   # per-step overhead, pure decode in full CUDA graph
OVH_PIECEWISE = 0.40e-3  # per-step overhead, mixed step (piecewise graphs, eager attention)
OVH_SPLIT_PER_LAYER = 4e-6  # proposed: extra launch for split decode attention per layer
KBLOCK_M = 128
# Calibrated from results/microbench/attn_sweep.jsonl (S2, H100, FA3 506341a):
BW_CTA_MEAS = 58e9   # FA3 1-split single sequence: 461 GB/s over 8 kv-head CTAs
BETA_X = 0.8         # unsplit mixed step: t_att ~= t_strag + BETA_X * t_balanced (median of nosplit cases)
STATIC_SPLIT_PENALTY = 1.08  # forcing max_num_splits=32 on non-straggler mixed steps (mean 1.08, max 1.18)

MODELS = {
    # name: layers, q heads, kv heads, head_dim, per-step weight bytes, non-embedding params
    "Qwen3-1.7B": dict(L=28, hq=16, hkv=8, hd=128, wbytes=3.4e9, params=1.4e9, kvcap=550e3),
    "Qwen3-4B": dict(L=36, hq=32, hkv=8, hd=128, wbytes=8.0e9, params=3.6e9, kvcap=400e3),
    "Qwen3-8B": dict(L=36, hq=32, hkv=8, hd=128, wbytes=15.1e9, params=6.9e9, kvcap=340e3),
    "Llama-3.1-8B": dict(L=32, hq=32, hkv=8, hd=128, wbytes=15.0e9, params=7.0e9, kvcap=385e3),
}


@dataclass
class Scenario:
    model: str
    prompt: int          # prompt tokens (median if prompt_sigma > 0)
    out_short: int
    out_long: int        # fixed long output, or 0 -> sample from reasoning_lengths.csv (clipped 1k..32k)
    frac_long: float
    rate: float          # req/s
    bw_cta: float = BW_CTA_MEAS  # single-CTA streaming bandwidth (B/s)
    prompt_sigma: float = 0.0    # lognormal sigma of prompt length (Azure code ~1.0)
    sim_seconds: float = 240.0
    seed: int = 0


def step_cost(m, dec_L: np.ndarray, pre_q: list[int], pre_ctx: list[int], policy: str, bw_cta: float,
              attn_model: str = "additive"):
    """Return (step_time_s, regime, info). dec_L: KV lengths of decode reqs (after appending this token).

    attn_model: "additive" (original straggler model) or "tile" (FA3 tile-scheduler model, features.fa3_tile_time).
    """
    L, hq, hkv, hd = m["L"], m["hq"], m["hkv"], m["hd"]
    g = hq // hkv
    n_dec = len(dec_L)
    T = n_dec + sum(pre_q)
    # GEMMs (whole batch, shared weights)
    t_gemm = max(m["wbytes"] / BW_W, 2 * m["params"] * T / F_GEMM)
    # attention per layer
    tok_bytes_layer = hkv * hd * 2 * 2
    sumL = float(dec_L.sum()) if n_dec else 0.0
    maxL = float(dec_L.max()) if n_dec else 0.0
    t_dec_bal = sumL * tok_bytes_layer / BW_KV
    t_pre = sum(4.0 * q * (c + q / 2) * hd * hq for q, c in zip(pre_q, pre_ctx)) / F_ATT
    max_q = max(pre_q) if pre_q else 1
    maxk = max([maxL] + [c + q for q, c in zip(pre_q, pre_ctx)])
    if T > 512:  # non-full-CG path: FA3 heuristic, batch treated as 1
        n_split, _ = fa3_static_splits(max_q, int(maxk), T, g, hkv, hd, full_cg=False)
    else:
        n_split = 32
    straggler_cfg = T > 512 and n_split < 32
    t_strag = maxL * hd * 2 * 2 / (bw_cta * max(n_split, 1))  # longest seq: <= n_split CTAs per kv-head
    t_bal = t_dec_bal + t_pre
    x_time = max(t_bal, t_strag + BETA_X * t_bal) if (straggler_cfg and n_dec) else t_bal
    is_x = bool(pre_q) and x_time > 1.2 * t_bal
    if policy == "baseline":
        att = x_time
    elif policy == "static_split":  # forced max_num_splits=32 on every non-CG step: fixes X, taxes the rest
        att = t_bal * (STATIC_SPLIT_PENALTY if (T > 512 and pre_q and not is_x) else 1.0)
    else:  # proposed: per-step choice (heuristic unless straggler-bound -> phase/lane split)
        att = t_bal + (OVH_SPLIT_PER_LAYER if is_x else 0.0) if x_time > t_bal else x_time
    excess = att - t_bal
    split_ovh = 0.0
    if attn_model == "tile" and T:  # same policies, priced by the tile model; X = default split >= 1.2x the 32-split bound
        qv = np.array([1] * n_dec + list(pre_q), dtype=np.int64)
        kvv = np.array(list(dec_L.astype(np.int64)) + [c + q for q, c in zip(pre_q, pre_ctx)], dtype=np.int64)
        t_def = fa3_tile_time(qv, kvv, n_split, g, hkv, hd)[0]
        t_32 = fa3_tile_time(qv, kvv, 32, g, hkv, hd)[0] if n_split < 32 else t_def
        is_x = bool(pre_q) and n_dec > 0 and t_def > 1.2 * t_32
        att = {"baseline": t_def, "static_split": t_32}.get(policy, min(t_def, t_32 + OVH_SPLIT_PER_LAYER))
        t_bal, excess = min(t_def, t_32), att - min(t_def, t_32)
        t_dec_bal, t_pre = 0.0, t_bal  # tile model prices the launch as a whole: attn_pre carries all of it
    t_att = L * (t_dec_bal + t_pre + excess + split_ovh)
    mixed = bool(pre_q)
    ovh = OVH_PIECEWISE if (mixed or T > 512) else OVH_FULLCG
    # regime label (hypothesised taxonomy)
    sumL_star = m["wbytes"] / (tok_bytes_layer * L)
    mean = sumL / n_dec if n_dec else 0.0
    skew = maxL / mean if mean else 0.0
    occ = n_dec * hkv / NSM
    if is_x:
        regime = "X"  # mixed step whose attention is bound by an unsplit long-KV decode
    elif pre_q and sum(pre_q) >= 0.5 * T:
        regime = "P"
    elif pre_q:
        regime = "M"
    elif n_dec * 1 >= 256:
        regime = "Dc"
    elif occ < 1 and maxL > 4096:
        regime = "Dl"
    elif skew > max(occ, 1.0) and maxL > 2048:
        regime = "Ds"
    elif sumL >= sumL_star:
        regime = "Dk"
    else:
        regime = "Dw"
    comp = dict(gemm=t_gemm, attn_dec=L * t_dec_bal, attn_pre=L * t_pre, strag_excess=L * excess,
                ovh=ovh + L * split_ovh)
    return t_gemm + t_att + ovh, regime, comp


_REASONING = None


def _reasoning_outs(rng, n):
    global _REASONING
    if _REASONING is None:
        import csv
        root = __file__.rsplit("/analysis/", 1)[0]
        with open(f"{root}/data/traces/reasoning_lengths.csv") as f:
            _REASONING = np.array([int(r["output_tokens"]) for r in csv.DictReader(f)])
    return np.clip(rng.choice(_REASONING, n), 1024, 32000)


class _PeakPlanner:
    """Oracle KV planner: future peak usage of admitted requests (each grows 1 token/step until done)."""

    def __init__(self, u: np.ndarray, r: np.ndarray):
        # u: current KV tokens (full prompt for admitted-but-prefilling), r: remaining decode steps
        o = np.argsort(r)
        self.r = r[o].astype(np.int64)
        self.u = u[o].astype(np.float64)
        n = len(self.r)
        # suffix sums over requests with r_i > k
        self.su = np.concatenate([np.cumsum(self.u[::-1])[::-1], [0.0]])
        self.cnt = np.arange(n, -1, -1, dtype=np.float64)
        ks = np.unique(np.concatenate([[0], np.maximum(self.r - 1, 0)]))
        self.ks = ks
        self.base = np.array([self._usage(k) for k in ks])
        self.pre = np.maximum.accumulate(self.base + ks) if len(ks) else np.array([])
        self.suf = np.maximum.accumulate(self.base[::-1])[::-1] if len(ks) else np.array([])

    def _usage(self, k):
        j = np.searchsorted(self.r, k, side="right")  # first with r > k
        return self.su[j] + k * self.cnt[j]

    def fits(self, u, r, cap):
        j = np.searchsorted(self.ks, r - 1, side="left")  # candidates with k < r-1 -> [0, j)
        peak = self._usage(r - 1) + u + r - 1
        if j > 0:
            peak = max(peak, self.pre[j - 1] + u)
        if j < len(self.ks):
            peak = max(peak, self.suf[j] + (u + self.ks[j] if self.ks[j] < r else 0.0))
        return peak <= cap


def simulate(sc: Scenario, policy: str, sched: str = "fcfs", xavoid="off"):
    """policy: attention kernel policy (baseline|static_split|proposed).
    sched: fcfs (vLLM: greedy admission + recompute preemption) | reserve (oracle exact prompt+out
    reservation) | peak (oracle foresight: admit iff future peak KV <= capacity).
    xavoid (step shaping): off | chunk416 (cap each prefill chunk at 416 tokens while a >=8k-KV decode runs:
    keeps FA3 splitting without changing kernels) | static512 (vLLM --max-num-batched-tokens 512: every step
    stays in the full-CG split path) | cg_cap (oracle: total step budget 512 only while a >=4k-KV decode with
    >=256 remaining tokens runs and the prefill backlog is <= 16k tokens)."""
    if xavoid is True:
        xavoid = "chunk416"
    m = MODELS[sc.model]
    rng = np.random.default_rng(sc.seed)
    n_req = int(sc.rate * sc.sim_seconds)
    arr = np.cumsum(rng.exponential(1.0 / sc.rate, n_req))
    is_long = rng.random(n_req) < sc.frac_long
    if sc.prompt_sigma > 0:
        prompts = np.clip(rng.lognormal(math.log(sc.prompt), sc.prompt_sigma, n_req), 16, 8192).astype(np.int64)
    else:
        prompts = np.full(n_req, sc.prompt, dtype=np.int64)
    short = rng.integers(max(1, sc.out_short // 2), sc.out_short * 3 // 2 + 1, n_req)
    longo = _reasoning_outs(rng, n_req) if sc.out_long <= 0 else np.full(n_req, sc.out_long)
    outs = np.where(is_long, longo, short).astype(np.int64)
    cap = m["kvcap"]
    budget, max_seqs = 8192, 1024
    t, nxt = 0.0, 0
    waiting: list[int] = []
    pre_done: dict[int, int] = {}
    admitted: set[int] = set()      # requests holding KV (running or partially prefilled)
    running: list[int] = []
    gen: dict[int, int] = {}
    first_tok, finish = {}, {}
    regime_time: dict[str, float] = {}
    regime_comp: dict[str, dict] = {}
    n_steps = n_preempt = 0
    t_end = sc.sim_seconds
    w_lo, w_hi = 0.4 * t_end, t_end   # steady-state window for regime fractions / throughput
    win_tok = 0
    while t < t_end + 3000:
        while nxt < n_req and arr[nxt] <= t:
            waiting.append(nxt)
            nxt += 1
        if not waiting and not running:
            if nxt >= n_req:
                break
            t = arr[nxt]
            continue
        dec_ids = running
        used = sum(int(prompts[r]) + gen[r] for r in dec_ids) + sum(pre_done.get(r, 0) for r in waiting)
        if sched == "fcfs":
            while dec_ids and used + len(dec_ids) > cap:
                v = dec_ids.pop()
                used -= int(prompts[v]) + gen[v]
                gen.pop(v); pre_done[v] = 0; admitted.discard(v); waiting.insert(0, v); n_preempt += 1
        running = dec_ids
        tok = len(dec_ids)
        long_running = any(int(prompts[r]) + gen[r] >= 8192 for r in dec_ids)
        chunk_cap = 416 if (xavoid == "chunk416" and long_running) else budget
        step_budget = budget
        if xavoid == "static512":
            step_budget = 512
        elif xavoid == "cg_cap" and any(int(prompts[r]) + gen[r] >= 4096 and int(outs[r]) - gen[r] >= 256
                                        for r in dec_ids):
            if sum(int(prompts[r]) - pre_done.get(r, 0) for r in waiting) <= 16384:
                step_budget = 512
        planner = None
        reserved = 0.0
        if sched == "reserve":
            reserved = float(sum(int(prompts[r]) + int(outs[r]) for r in admitted))
        pre_q, pre_ctx, pre_ids = [], [], []
        for rid in list(waiting):
            if tok >= step_budget or len(dec_ids) + len(pre_ids) >= max_seqs:
                break
            if rid not in admitted:
                need = int(prompts[rid]) + int(outs[rid])
                if sched == "fcfs" and used + prompts[rid] > 0.95 * cap:
                    break
                if sched == "reserve" and reserved + need > cap:
                    continue  # oracle planners backfill requests that fit (no head-of-line blocking)
                if sched == "peak":
                    if planner is None:
                        ids = list(admitted)
                        u = np.array([float(prompts[i]) + gen.get(i, 0) for i in ids])
                        rr = np.array([int(outs[i]) - gen.get(i, 0) for i in ids], dtype=np.int64)
                        planner = _PeakPlanner(u, rr)
                    if not planner.fits(float(prompts[rid]), int(outs[rid]), cap):
                        continue
                admitted.add(rid)
                reserved += need
                used += int(prompts[rid])
                planner = None
            done = pre_done.get(rid, 0)
            q = min(int(prompts[rid]) - done, step_budget - tok, chunk_cap)
            if q <= 0:
                break
            pre_q.append(q); pre_ctx.append(done); pre_ids.append(rid)
            tok += q
            if q < int(prompts[rid]) - done:
                break  # chunked: one partial prefill per step (vLLM long_prefill behaviour)
        dec_L = np.array([float(prompts[r]) + gen[r] for r in dec_ids], dtype=np.float64)
        dt, regime, comp = step_cost(m, dec_L, pre_q, pre_ctx, policy, sc.bw_cta)
        t += dt
        n_steps += 1
        if w_lo <= t <= w_hi:
            regime_time[regime] = regime_time.get(regime, 0.0) + dt
            rc = regime_comp.setdefault(regime, dict(gemm=0.0, attn_dec=0.0, attn_pre=0.0, strag_excess=0.0,
                                                     ovh=0.0))
            for k, v in comp.items():
                rc[k] += v
            win_tok += len(dec_ids) + len(pre_ids)
        still = []
        for r in dec_ids:
            gen[r] += 1
            if gen[r] >= outs[r]:
                finish[r] = t
                admitted.discard(r)
            else:
                still.append(r)
        for rid, q in zip(pre_ids, pre_q):
            pre_done[rid] = pre_done.get(rid, 0) + q
            if pre_done[rid] >= prompts[rid]:
                waiting.remove(rid)
                first_tok.setdefault(rid, t)
                gen[rid] = 1
                if outs[rid] <= 1:
                    finish[rid] = t
                    admitted.discard(rid)
                else:
                    still.append(rid)
        running = still
    lo, hi = 0.25 * sc.sim_seconds, 0.75 * sc.sim_seconds
    sel = np.array([r for r in range(n_req) if lo <= arr[r] <= hi and r in finish])
    if len(sel) < 10:
        return None
    ttft = np.array([first_tok[r] - arr[r] for r in sel])
    e2e = np.array([finish[r] - arr[r] for r in sel])
    tpot = np.array([(finish[r] - first_tok[r]) / max(outs[r] - 1, 1) for r in sel])
    lm = is_long[sel]
    tot = sum(regime_time.values())
    return dict(
        steps=n_steps, unfinished=int(n_req - len(finish)), preempt=n_preempt,
        ttft_p50=float(np.median(ttft)), ttft_p99=float(np.percentile(ttft, 99)),
        tpot_long_mean=float(tpot[lm].mean()) if lm.any() else float("nan"),
        tpot_short_mean=float(tpot[~lm].mean()) if (~lm).any() else float("nan"),
        tpot_all_p99=float(np.percentile(tpot, 99)),
        e2e_mean=float(e2e.mean()), e2e_p99=float(np.percentile(e2e, 99)),
        e2e_short_mean=float(e2e[~lm].mean()) if (~lm).any() else float("nan"),
        e2e_long_mean=float(e2e[lm].mean()) if lm.any() else float("nan"),
        out_tok_per_s=float(win_tok / max(tot, 1e-9)), makespan=float(t),
        n_long_sel=int(lm.sum()), n_sel=int(len(sel)),
        regime_frac={k: v / tot for k, v in sorted(regime_time.items())},
        regime_comp={k: {c: x / tot for c, x in d.items()} for k, d in sorted(regime_comp.items())},
        busy_time=tot,
    )


VARIANTS = {  # name: (kernel policy, scheduler, xavoid)
    "B0": ("baseline", "fcfs", "off"),          # vLLM default
    "B1s": ("static_split", "fcfs", "off"),     # best static kernel knob (forced splits)
    "B512": ("baseline", "fcfs", "static512"),  # best static batching knob (--max-num-batched-tokens 512)
    "K": ("proposed", "fcfs", "off"),           # kernel-only regime dispatch (reactive)
    "Bx": ("baseline", "fcfs", "chunk416"),     # batching-only X avoidance (reactive)
    "Bc": ("baseline", "fcfs", "cg_cap"),       # batching-only oracle step shaping
    "S": ("baseline", "peak", "off"),           # scheduling-only oracle KV planning (S3/past-future-like)
    "KS": ("proposed", "peak", "off"),          # proposed: oracle planning + regime dispatch
    "KSc": ("proposed", "peak", "cg_cap"),      # proposed + oracle step shaping
}


def run_one(args):
    sc, var = args
    return asdict(sc), var, simulate(sc, *VARIANTS[var])


def main():
    grid = sys.argv[3] if len(sys.argv) > 3 else "full"
    if grid == "focus":  # calibrated v3: where X should dominate (long KV + steady >=512-token prefills)
        axes = (["Qwen3-8B", "Qwen3-4B"], [(512, 0.0), (1930, 1.0), (4096, 0.5)], [16000, 32000, 0],
                [1 / 256, 1 / 64], [4.0, 8.0, 16.0])
    else:
        axes = (["Qwen3-8B", "Qwen3-4B", "Qwen3-1.7B"], [(512, 0.0), (1930, 1.0)], [4000, 16000, 0],
                [1 / 1024, 1 / 256, 1 / 64], [4.0, 8.0, 16.0, 32.0])
    scen = []
    for model, (prompt, sig), out_long, frac_long, rate in itertools.product(*axes):
        life = 0.03 * (out_long if out_long > 0 else 12000)  # ~long-request lifetime at ~30 ms/token
        scen.append(Scenario(model, prompt, 30, out_long, frac_long, rate, prompt_sigma=sig,
                             sim_seconds=max(240.0, 1.5 * life)))
    variants = sys.argv[4].split(",") if len(sys.argv) > 4 else list(VARIANTS)
    jobs = [(sc, v) for sc in scen for v in variants]
    jobs.sort(key=lambda j: -j[0].rate * j[0].sim_seconds)  # longest first
    out = sys.argv[1] if len(sys.argv) > 1 else "results/pen_and_paper/sim_v2.jsonl"
    res: dict = {}
    with Pool(int(sys.argv[2]) if len(sys.argv) > 2 else 32) as pool:
        for i, (sc, var, r) in enumerate(pool.imap_unordered(run_one, jobs, chunksize=1)):
            res.setdefault(json.dumps(sc, sort_keys=True), {})[var] = r
            if i % 100 == 0:
                print(f"{i}/{len(jobs)}", flush=True)
    with open(out, "w") as f:
        for k, r in res.items():
            f.write(json.dumps(dict(scenario=json.loads(k), **r)) + "\n")
    print("wrote", out, len(res))


if __name__ == "__main__":
    main()
