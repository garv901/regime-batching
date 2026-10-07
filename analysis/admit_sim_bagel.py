"""CPU admission-policy simulator at the BAGEL KV pool, for the mstar admission matrix (no GPU).

  python analysis/admit_sim_bagel.py data/workloads/bagel_burst_v2.jsonl data/workloads/bagel_stag_v2_r*.jsonl

Policies (the four configs of experiments/mstar/admission_matrix.sbatch; KV unit = tokens):
  sum_fifo   admit while sum(prompt + max_tokens) of the admitted requests <= pool; strict FIFO (a head that does not fit blocks)
  sum_bf     same fit test, backfill (skip a head that does not fit)
  peak_fifo  admit while the future peak KV of the admitted set <= pool (requests free KV when done); strict FIFO
  peak_bf    future-peak fit test + backfill
  open       reference only: no limit, so its KV peak is the demand the pool would have to carry
Prompt = input_length + template overhead (20 tokens). A row without `max_tokens` is an ORACLE row: its declared bound is its
output_length. A row with `max_tokens` (experiments/mstar/declare_max_tokens.py: the client declares a bound >= output_length) is
reserved and planned with the DECLARED value (sum: prompt + declared; peak: the request is assumed to grow up to declared),
while it still finishes after output_length tokens and frees its KV then (an emulated EOS). `open` ignores both.

The loop is the one of the throwaway vLLM-study simulator /tmp/admit_sim/admit_sim.py (fcfs and Qwen3-8B step cost
dropped, one set of per-step bookkeeping added); the future-peak test is analysis/pen_and_paper._PeakPlanner.
Step cost is what differs: that simulator prices a decode step with Qwen3-8B (11.7 ms at batch 64, 5,500 tok/s; 147 KB of
KV per token), but BAGEL decodes at ~2,000 tok/s at batch 64 (ITL 29 ms) and keeps 57 KB of KV per token. Here a step costs
  decode_itl(batch) + kv_read(decoding KV tokens) + prefill_tokens / PREFILL_TOK_S:
  decode_itl: linear interpolation of the effective inter-token latency (1 / measured tok/s x batch) at batch 8, 16, 32, 64:
      1,134 / 1,747 / 1,825 / 1,999 tok/s on one H100 (smoke job dec_c8, long_c16, dec_c32, dec_c64; prompts <= 1,000 tokens)
      = 7.05 / 9.2 / 17.5 / 32.0 ms; batch 1 = 5 ms is an ASSUMPTION. Decode is graphed up to batch 64, so a larger set
      decodes in chunks of <= 64, one chunk after the other.
  kv_read: KV tokens of the decoding requests x 57,344 B / BW_KV, BW_KV = 2.85e12 B/s (the in-engine paged-decode fit of
      analysis/pen_and_paper.py for FA3; an ASSUMPTION for BAGEL's FlashInfer). The smoke batches carry < 30k KV tokens, so
      their ITL already holds < 1 ms of this; a full pool adds 5.3 ms.
  prefill: PREFILL_TOK_S = 30,000 tok/s is an ASSUMPTION (7B model, 20-40k tok/s; prefill is graphed to 2,048 tokens and
      eager above). A step prefills at most 8,192 tokens (the old simulator's budget, an assumption about BAGEL).
Absolute times are therefore only as good as these constants; the ORDER of the policies is the result.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pen_and_paper import _PeakPlanner  # noqa: E402

POOL_TOKENS = 2047 * 128  # 2048 pages x 128 tokens; the first step of a real run sees 2047 free pages
OVERHEAD = 20
PREFILL_TOK_S = 30_000.0
BUDGET = 8192  # prefill tokens per step
DECODE_BATCH_CAP = 64
ITL_AT = ([1, 8, 16, 32, 64], [5.0e-3, 7.05e-3, 9.2e-3, 17.5e-3, 32.0e-3])
KV_BYTES_PER_TOKEN = 28 * 2 * 4 * 128 * 2  # 57,344: 28 layers x K,V x 4 kv heads x head_dim 128 x bf16
BW_KV = 2.85e12
SERVER_TIMEOUT_S = 600.0
POLICIES = ["sum_fifo", "sum_bf", "peak_fifo", "peak_bf", "open"]


def decode_time(c: int, kv_tokens: float = 0.0) -> float:
    """Seconds for one token of each of c decoding requests holding kv_tokens of KV (chunks of <= 64)."""
    if c <= 0:
        return 0.0
    full, rest = divmod(c, DECODE_BATCH_CAP)
    itl = full * float(np.interp(DECODE_BATCH_CAP, *ITL_AT)) + (float(np.interp(rest, *ITL_AT)) if rest else 0.0)
    return itl + kv_tokens * KV_BYTES_PER_TOKEN / BW_KV


def load_trace(path: str, overhead: int = OVERHEAD):
    """(arrival, prompt tokens, output tokens, is_long, declared max_tokens); declared == output for a row without max_tokens."""
    rows = sorted((json.loads(line) for line in open(path)), key=lambda r: r["timestamp"])
    arr = np.array([r["timestamp"] for r in rows])
    prompts = np.array([r["input_length"] + overhead for r in rows], dtype=np.int64)
    outs = np.array([r["output_length"] for r in rows], dtype=np.int64)
    decl = np.array([r["output_length"] if r.get("max_tokens") is None else r["max_tokens"] for r in rows], dtype=np.int64)
    if (decl < outs).any():
        raise ValueError(f"{path}: a row declares max_tokens below its output_length (the server would stop it there)")
    is_long = np.array([r.get("cls") == "long" for r in rows])
    return arr, prompts, outs, is_long, decl


def simulate(arr, prompts, outs, is_long, sched: str, cap: float, prefill_tok_s: float = PREFILL_TOK_S, decl=None) -> dict:
    """decl: declared max_tokens per request (what sum/peak reserve and plan with); None = outs (oracle). A request finishes
    after outs[r] tokens whatever decl says."""
    n_req = len(arr)
    decl = outs if decl is None else decl
    if sched == "open":
        base, fifo, cap = "sum", False, float("inf")
    else:
        base, fifo = sched.split("_")[0], sched.endswith("fifo")
    t, nxt = 0.0, 0
    waiting: list[int] = []
    pre_done: dict[int, int] = {}
    admitted: set[int] = set()  # holding KV: prefilling or decoding
    running: list[int] = []  # decoding
    gen: dict[int, int] = {}
    first_tok: dict[int, float] = {}
    finish: dict[int, float] = {}
    busy = conc_acc = peak_kv = 0.0
    while t < arr[-1] + 5000:
        while nxt < n_req and arr[nxt] <= t:
            waiting.append(nxt)
            nxt += 1
        if not waiting and not running:
            if nxt >= n_req:
                break
            t = arr[nxt]
            continue
        dec_ids = running
        tok = len(dec_ids)
        planner = None
        reserved = float(sum(int(prompts[r]) + int(decl[r]) for r in admitted)) if base == "sum" else 0.0
        pre_q, pre_ids = [], []
        for rid in list(waiting):
            if tok >= BUDGET:
                break
            if rid not in admitted:
                need = int(prompts[rid]) + int(decl[rid])
                if base == "sum":
                    ok = reserved + need <= cap
                else:
                    if planner is None:
                        ids = list(admitted)
                        u = np.array([float(prompts[i]) + gen.get(i, 0) for i in ids])
                        rr = np.array([int(decl[i]) - gen.get(i, 0) for i in ids], dtype=np.int64)
                        planner = _PeakPlanner(u, rr)
                    ok = planner.fits(float(prompts[rid]), int(decl[rid]), cap)
                if not ok:
                    if fifo:
                        break  # head of line: nothing passes a request that does not fit
                    continue  # backfill
                admitted.add(rid)
                reserved += need
                planner = None
            done = pre_done.get(rid, 0)
            q = min(int(prompts[rid]) - done, BUDGET - tok)
            if q <= 0:
                break
            pre_q.append(q)
            pre_ids.append(rid)
            tok += q
            if q < int(prompts[rid]) - done:
                break
        kv_dec = sum(int(prompts[r]) + gen[r] for r in dec_ids)
        dt = decode_time(len(dec_ids), kv_dec) + sum(pre_q) / prefill_tok_s
        t += dt
        busy += dt
        conc_acc += len(admitted) * dt
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
        kv = sum(int(prompts[r]) + gen[r] for r in running) + sum(pre_done.get(r, 0) for r in waiting if r in admitted)
        peak_kv = max(peak_kv, kv)
    sel = np.array([r for r in range(n_req) if r in finish])
    lm = is_long[sel]
    e2e = np.array([finish[r] - arr[r] for r in sel])
    ttft = np.array([first_tok[r] - arr[r] for r in sel])

    def stat(a, fn):
        return float(fn(a)) if len(a) else float("nan")

    return dict(
        sched=sched, unfinished=int(n_req - len(sel)), over_600s=int((e2e > SERVER_TIMEOUT_S).sum()),
        short_e2e_mean=stat(e2e[~lm], np.mean), short_e2e_p99=stat(e2e[~lm], lambda a: np.percentile(a, 99)),
        short_ttft_mean=stat(ttft[~lm], np.mean), long_e2e_mean=stat(e2e[lm], np.mean), long_e2e_max=stat(e2e[lm], np.max),
        long_ttft_mean=stat(ttft[lm], np.mean), out_tok_s=int(outs[sel].sum()) / t, makespan=t,
        mean_admitted=conc_acc / max(busy, 1e-9), peak_kv_ratio=peak_kv / POOL_TOKENS,
    )


COLS = [("unfinished", "unfin", 0), ("over_600s", ">600s", 0), ("short_e2e_mean", "short E2E mean", 2),
        ("short_e2e_p99", "short E2E p99", 1), ("short_ttft_mean", "short TTFT mean", 2),
        ("long_e2e_mean", "long E2E mean", 1), ("long_e2e_max", "long E2E max", 1), ("long_ttft_mean", "long TTFT mean", 1),
        ("out_tok_s", "out tok/s", 0), ("makespan", "makespan s", 0), ("mean_admitted", "mean admitted", 1),
        ("peak_kv_ratio", "peak KV / pool", 2)]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("traces", nargs="+", help="timed_trace JSONL workloads")
    ap.add_argument("--policies", nargs="+", default=POLICIES, choices=POLICIES)
    ap.add_argument("--cap-tokens", type=int, default=POOL_TOKENS, help="KV pool in tokens")
    ap.add_argument("--prefill-tok-s", type=float, default=PREFILL_TOK_S)
    a = ap.parse_args(argv)
    for path in a.traces:
        data = load_trace(path)
        decl_note = f", declared max_tokens {data[4].sum() / data[2].sum():.2f}x the output" if (data[4] != data[2]).any() else ""
        print(f"\n### {os.path.basename(path)}  ({len(data[0])} requests, {int(data[3].sum())} long{decl_note})\n")
        print("| policy | " + " | ".join(c[1] for c in COLS) + " |\n|" + "---|" * (len(COLS) + 1))
        for p in a.policies:
            r = simulate(*data[:4], sched=p, cap=a.cap_tokens, prefill_tok_s=a.prefill_tok_s, decl=data[4])
            print(f"| {p} | " + " | ".join(f"{r[k]:.{nd}f}" for k, _, nd in COLS) + " |", flush=True)


if __name__ == "__main__":
    main()
