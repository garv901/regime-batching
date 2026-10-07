"""Per-step batch features, regime classification and baseline-kernel prediction.

Shared by telemetry (baseline + proposed runs) and the regime-aware dispatcher, so both label steps
identically. Thresholds live in RegimeConfig and can be overridden from a JSON file
(env REGIME_CONFIG) once fitted from microbenchmarks.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field, asdict

import numpy as np

NUM_SMS = 132
# Calibrated by S2 attention sweep (results/microbench/attn_sweep.jsonl, H100 SXM, FA3 506341a)
BW_CTA = 58e9      # one FA3 CTA streaming one kv-head of one sequence (B/s)
BW_KV = 2.6e12     # balanced paged-KV decode streaming
F_ATT = 450e12     # prefill attention FLOP/s
BETA_X = 0.8       # unsplit mixed step: t ~= t_strag + BETA_X * t_bal
# Tile model (reports/regimes_study/01_foundations.md): greedy list scheduling of FA3's persistent varlen tiles
# over NUM_SMS CTAs in launch order (decodes first, longest m-block first within a sequence). Replaces the additive
# straggler model for offline analysis. Fit by microbench/validate_tile_model.py --fit decode mixed on F1-F5 (job 1008):
# mixed launches (F3) mean |err| 4.7%, decode-only launches 4-11%.
F_SM_TILE = 5.3e12     # attention FLOP/s of one CTA on a prefill tile
BW_CTA_MIXED = 40.5e9  # decode tile in a launch with kBlockM = 128 (max_q * g > 64)
BW_CTA_DEC = 59.4e9    # decode tile in a decode-only launch (kBlockM = 64)
BW_HBM_ATT = 2.85e12   # floor: all decode KV bytes / HBM bandwidth
TILE_OVH = 8.4e-6      # per tile, mixed launch
LAUNCH_OVH = 6.0e-6    # per mixed launch
DEC_TILE_OVH = 0.0     # per tile, decode-only launch (fit -> 0)
DEC_LAUNCH_OVH = 8.8e-6  # per decode-only launch
COMBINE_OVH = 2.3e-6   # split-KV combine kernel, when any sequence splits
FORCED_SPLITS = (1, 2, 4, 8, 16, 32)


@dataclass
class ModelGeom:
    num_layers: int = 36
    num_q_heads: int = 32
    num_kv_heads: int = 8
    head_dim: int = 128
    weight_bytes_per_step: float = 15.1e9  # GEMM weight bytes streamed per decode step
    kv_dtype_bytes: int = 2

    @property
    def kv_bytes_per_token(self) -> float:  # all layers, K+V
        return self.num_layers * self.num_kv_heads * self.head_dim * 2 * self.kv_dtype_bytes

    @property
    def sumL_star(self) -> float:
        """Total decode KV tokens at which KV bytes == weight bytes per step (attention/GEMM crossover)."""
        return self.weight_bytes_per_step / self.kv_bytes_per_token


@dataclass
class RegimeConfig:
    long_len: int = 2048          # oracle final length >= this -> "long" lane
    prefill_heavy_frac: float = 0.5
    dc_batch: int = 256           # decode batch near GEMM ridge
    dl_min_maxL: int = 4096
    ds_min_maxL: int = 2048
    cg_capture_max: int = 512     # vLLM max_cudagraph_capture_size default on H100
    fa_cg_max_splits: int = 32
    x_struct_min_L: int = 16384   # structural X: shortest "long" decode
    a_min_prefix: int = 16384     # regime A: prefill appended to at least this many cached tokens
    x_cost_ratio: float = 1.2     # regime2 X: default / best-split attention time

    @classmethod
    def load(cls) -> "RegimeConfig":
        p = os.environ.get("REGIME_CONFIG")
        if p and os.path.exists(p):
            with open(p) as f:
                return cls(**{k: v for k, v in json.load(f).items() if k in cls.__dataclass_fields__})
        return cls()


def fa3_static_splits(max_q: int, max_k: int, num_tokens: int, g: int, h_kv: int, head_dim: int = 128,
                      page_size: int = 16, full_cg: bool = True, cfg: RegimeConfig | None = None) -> tuple[int, str]:
    """Replicates vLLM 0.30 FA backend + FA3 (vllm-flash-attention@506341a) split selection for hd128 bf16.

    Returns (num_splits_static, path) where path in {"dyn<k>", "nosplit"}; with static > 1 FA3 computes
    per-sequence dynamic splits proportional to KV length (load-balanced).
    """
    cfg = cfg or RegimeConfig()
    if full_cg and num_tokens <= cfg.cg_capture_max:
        return cfg.fa_cg_max_splits, f"dyn{cfg.fa_cg_max_splits}"
    one_mma_wg = max_q * g <= 64
    k_block_n = 128 if (page_size % 128 != 0) else (128 if one_mma_wg else 128)  # causal -> 128 either way
    k_block_m = 64 if one_mma_wg else 128
    num_n_blocks = math.ceil(max_k / k_block_n)
    num_m_blocks = math.ceil(max_q * g / k_block_m)
    total_mblocks = 1 * h_kv * num_m_blocks  # varlen: batch treated as 1
    if total_mblocks >= 0.8 * NUM_SMS:
        return 1, "nosplit"  # causal -> never the L2-size split
    if num_n_blocks <= 4:
        return 1, "nosplit"
    max_splits = min(128, NUM_SMS, num_n_blocks)
    eff = []
    for s in range(1, max_splits + 1):
        n_waves = total_mblocks * s / NUM_SMS
        eff.append(n_waves / math.ceil(n_waves))
    best = max(eff)
    for s, e in enumerate(eff, start=1):
        if e >= 0.85 * best:
            return s, ("nosplit" if s == 1 else f"dyn{s}")
    return 1, "nosplit"


@dataclass
class StepFeatures:
    num_reqs: int = 0
    n_dec: int = 0
    n_pre: int = 0
    T: int = 0
    pre_tokens: int = 0
    max_q: int = 0
    sumL: float = 0.0
    maxL: int = 0
    minL: int = 0
    meanL: float = 0.0
    p50L: float = 0.0
    cv: float = 0.0
    skew: float = 0.0
    occupancy: float = 0.0
    sumL_frac: float = 0.0     # sumL / sumL_star  (>1 -> KV-bound decode)
    long_share: float = 0.0    # fraction of decode KV bytes in requests whose oracle final len >= long_len
    n_long: int = 0            # decode reqs in the long lane (oracle)
    min_remaining: int = -1    # oracle: steps until the next decode request finishes
    max_ctx_pre: int = 0
    regime: str = ""
    fa3_splits: int = 0
    fa3_path: str = ""
    straggler_risk: float = 0.0  # maxL / sumL (decode KV share of the longest sequence)
    t_att_bal_us: float = 0.0    # predicted per-layer attention time with a balanced kernel
    x_ratio: float = 1.0         # predicted baseline(FA3 heuristic) / balanced attention time per layer
    pre_prefix: int = 0          # longest cached prefix under a prefill chunk (kv - q)
    pre_tiles: int = 0           # FA3 prefill tiles (m-blocks x kv heads) at 1 split
    pre_wave_eff: float = 1.0    # pre_tiles / (NUM_SMS * waves) when FA3 runs 1 split, else 1
    structural_x: bool = False   # T past the CG path, FA3 1 split, prefill present, decode >= x_struct_min_L

    def to_dict(self):
        return asdict(self)


def compute_features(reqs: list[tuple[str, int, int, int, int]], geom: ModelGeom, cfg: RegimeConfig,
                     full_cg: bool = True) -> StepFeatures:
    """reqs: list of (req_id, q_len, kv_len_after_step, oracle_final_len, oracle_remaining_tokens)."""
    f = StepFeatures()
    if not reqs:
        return f
    g = geom.num_q_heads // geom.num_kv_heads
    q = np.fromiter((r[1] for r in reqs), dtype=np.int64, count=len(reqs))
    kv = np.fromiter((r[2] for r in reqs), dtype=np.int64, count=len(reqs))
    fin = np.fromiter((r[3] for r in reqs), dtype=np.int64, count=len(reqs))
    rem = np.fromiter((r[4] for r in reqs), dtype=np.int64, count=len(reqs))
    dec = q == 1
    f.num_reqs = len(reqs)
    f.n_dec = int(dec.sum())
    f.n_pre = f.num_reqs - f.n_dec
    f.T = int(q.sum())
    f.pre_tokens = int(q[~dec].sum())
    f.max_q = int(q.max())
    f.max_ctx_pre = int(kv[~dec].max()) if f.n_pre else 0
    if f.n_dec:
        L = kv[dec]
        f.sumL = float(L.sum())
        f.maxL = int(L.max())
        f.minL = int(L.min())
        f.meanL = float(L.mean())
        f.p50L = float(np.median(L))
        f.cv = float(L.std() / f.meanL) if f.meanL else 0.0
        f.skew = f.maxL / f.meanL if f.meanL else 0.0
        f.occupancy = f.n_dec * geom.num_kv_heads / NUM_SMS
        f.sumL_frac = f.sumL / geom.sumL_star
        longm = fin[dec] >= cfg.long_len
        f.n_long = int(longm.sum())
        f.long_share = float(L[longm].sum() / f.sumL) if f.sumL else 0.0
        r = rem[dec]
        r = r[r >= 0]
        f.min_remaining = int(r.min()) if r.size else -1
        # share of decode KV in the single longest sequence; unsplit FA3 binds on it when
        # maxL/sumL > h_kv*BW_cta/BW_kv (~0.3 for BW_cta~100GB/s; calibrated by microbench)
        f.straggler_risk = f.maxL / f.sumL if f.sumL else 0.0
    max_k = int(kv.max())
    f.fa3_splits, f.fa3_path = fa3_static_splits(f.max_q, max_k, f.T, g, geom.num_kv_heads, geom.head_dim,
                                                 full_cg=full_cg, cfg=cfg)
    tok_b = geom.num_kv_heads * geom.head_dim * 2 * geom.kv_dtype_bytes
    qp, kp = q[~dec].astype(np.float64), kv[~dec].astype(np.float64)
    t_pre = float((4.0 * qp * (kp - qp / 2) * geom.head_dim * geom.num_q_heads).sum()) / F_ATT
    t_bal = f.sumL * tok_b / BW_KV + t_pre
    f.t_att_bal_us = t_bal * 1e6
    if f.n_dec and f.fa3_splits < cfg.fa_cg_max_splits and t_bal > 0:
        t_strag = f.maxL * geom.head_dim * 2 * geom.kv_dtype_bytes / (BW_CTA * max(f.fa3_splits, 1))
        f.x_ratio = max(1.0, (t_strag + BETA_X * t_bal) / t_bal)
    if f.n_pre:
        f.pre_prefix = int((kv[~dec] - q[~dec]).max())
        if f.fa3_path == "nosplit":
            kbm = 64 if f.max_q * g <= 64 else 128
            f.pre_tiles = int(sum(geom.num_kv_heads * math.ceil(int(x) * g / kbm) for x in q[~dec]))
            f.pre_wave_eff = f.pre_tiles / (NUM_SMS * math.ceil(f.pre_tiles / NUM_SMS))
        f.structural_x = bool(f.n_dec and f.fa3_path == "nosplit" and f.maxL >= cfg.x_struct_min_L)
    f.regime = classify(f, geom, cfg)
    return f


def fa3_seq_splits(kv: np.ndarray, mblocks: np.ndarray, s_static: int, h_kv: int) -> np.ndarray:
    """Per-sequence KV splits FA3 uses under a static bound s_static (FA3 prepare_varlen_num_blocks_kernel).

    kBlockN = 128 (paged KV, page 16, is non-TMA); PackGQA, so heads = h_kv. Batches are not sorted
    (vLLM's FA3 build sets varlen_sort_batches = false).
    """
    if s_static <= 1:
        return np.ones_like(kv)
    nblk = np.ceil(kv / 128.0)
    blocks_per_sm = math.ceil(float((nblk * mblocks).sum()) * 1.1 * h_kv / NUM_SMS)
    return np.clip(np.ceil(nblk / blocks_per_sm), 1, s_static).astype(np.int64)


def fa3_tile_time(q: np.ndarray, kv: np.ndarray, s_static: int, g: int, h_kv: int,
                  head_dim: int = 128, ideal: bool = False) -> tuple[float, int, float]:
    """Predicted time of one FA3 varlen launch (one layer) in seconds, its tile count, and SM efficiency.

    q, kv: per-sequence query and KV lengths in launch order (vLLM puts decodes first).
    Efficiency = summed tile time / (NUM_SMS * makespan).
    """
    import heapq
    q = np.asarray(q, dtype=np.int64)
    kv = np.asarray(kv, dtype=np.int64)
    mixed = int(q.max()) * g > 64
    rows = 128 if mixed else 64
    mblocks = np.ceil(q * g / rows).astype(np.int64)
    splits = fa3_seq_splits(kv, mblocks, s_static, h_kv)
    bw_dec = BW_CTA_MIXED if mixed else BW_CTA_DEC
    durs = []
    for qi, ki, mi, si in zip(q.tolist(), kv.tolist(), mblocks.tolist(), splits.tolist()):
        if qi == 1:
            durs += [ki * head_dim * 2 * 2 / si / bw_dec + (TILE_OVH if mixed else DEC_TILE_OVH)] * (h_kv * si)
        else:
            prefix = ki - qi
            seq = [4.0 * rows * (prefix + min(qi, (m + 1) * rows // g)) * head_dim / si / F_SM_TILE + TILE_OVH
                   for m in range(mi)][::-1]
            durs += [d for d in seq for _ in range(si)] * h_kv
    free = [0.0] * NUM_SMS
    end = 0.0
    for d in durs:
        t = heapq.heappop(free) + d
        heapq.heappush(free, t)
        end = max(end, t)
    eff = sum(durs) / (NUM_SMS * end) if end else 1.0
    if ideal:  # perfect packing of the same work: lower bound for any re-tiling or re-splitting fix
        end = sum(durs) / NUM_SMS
    end = max(end, float(kv[q == 1].sum()) * h_kv * head_dim * 2 * 2 / BW_HBM_ATT)
    launch = LAUNCH_OVH if mixed else DEC_LAUNCH_OVH
    return end + launch + (COMBINE_OVH if int(splits.max()) > 1 else 0.0), len(durs), eff


def attn_step_model(q: np.ndarray, kv: np.ndarray, geom: ModelGeom, cfg: RegimeConfig,
                    full_cg: bool = True) -> dict:
    """Offline per-step attention model (per layer, seconds): what vLLM's FA3 path costs and what better choices cost.

    t_default: the split vLLM picks; t_best: best split upper bound for the same launch (C3-like; FA3 treats the
    bound as a cap on its own per-sequence rule, so this cannot split a long prefill); t_ideal: the default launch's
    work packed perfectly over the SMs (bound for re-tiling fixes, H5); t_sep: prefill and decodes as separate
    launches, each with its best split bound (regime-separated counterfactual).
    """
    q = np.asarray(q, dtype=np.int64)
    kv = np.asarray(kv, dtype=np.int64)
    g = geom.num_q_heads // geom.num_kv_heads
    T = int(q.sum())
    s_def, path = fa3_static_splits(int(q.max()), int(kv.max()), T, g, geom.num_kv_heads, geom.head_dim,
                                    full_cg=full_cg, cfg=cfg)
    t_def, n_tiles, eff = fa3_tile_time(q, kv, s_def, g, geom.num_kv_heads, geom.head_dim)
    t_ideal = fa3_tile_time(q, kv, s_def, g, geom.num_kv_heads, geom.head_dim, ideal=True)[0]
    best = min((fa3_tile_time(q, kv, s, g, geom.num_kv_heads, geom.head_dim)[0], s) for s in FORCED_SPLITS)

    def best_of(mask):
        if not mask.any():
            return 0.0
        return min(fa3_tile_time(q[mask], kv[mask], s, g, geom.num_kv_heads, geom.head_dim)[0] for s in FORCED_SPLITS)

    dec = q == 1
    t_sep = best_of(dec) + best_of(~dec)
    n_pre, n_dec = int((~dec).sum()), int(dec.sum())
    pre_prefix = int((kv[~dec] - q[~dec]).max()) if n_pre else 0
    maxL = int(kv[dec].max()) if n_dec else 0
    ratio = t_def / best[0] if best[0] else 1.0
    if n_pre and n_dec and path == "nosplit" and maxL >= cfg.x_struct_min_L and ratio >= cfg.x_cost_ratio:
        regime2 = "X"
    elif n_pre and pre_prefix >= cfg.a_min_prefix:
        regime2 = "A"
    elif n_pre:  # an X shape whose straggler costs < x_cost_ratio is priced as an ordinary mixed step
        regime2 = "P" if int(q[~dec].sum()) >= cfg.prefill_heavy_frac * T else "M"
    else:
        regime2 = None  # decode-only: classify() labels apply unchanged
    return dict(splits_default=s_def, path=path, t_default=t_def, t_best=best[0], s_best=best[1], t_sep=t_sep,
                t_ideal=t_ideal, pack_gain=t_def / t_ideal,
                tiles=n_tiles, sm_eff=eff, x_cost=ratio, sep_gain=t_def / t_sep if t_sep else 1.0,
                pre_prefix=pre_prefix, structural_x=bool(n_pre and n_dec and path == "nosplit"
                                                          and maxL >= cfg.x_struct_min_L), regime2=regime2)


def classify(f: StepFeatures, geom: ModelGeom, cfg: RegimeConfig) -> str:
    """Hypothesised regime taxonomy (validated/refit against measured step times later).

    P  prefill-heavy           M  mixed prefill+decode     X  mixed with unsplit long-KV decode (FA3 straggler)
    Dw decode weight-bound     Dk decode KV-bound          Ds decode skewed (few long dominate KV)
    Dl decode low-occupancy long-context                   Dc decode near GEMM ridge (large B)
    """
    if f.n_pre:
        if f.x_ratio > 1.2:
            return "X"
        if f.pre_tokens >= cfg.prefill_heavy_frac * f.T:
            return "P"
        return "M"
    if f.n_dec >= cfg.dc_batch:
        return "Dc"
    if f.occupancy < 1 and f.maxL >= cfg.dl_min_maxL:
        return "Dl"
    if f.skew > max(f.occupancy, 1.0) and f.maxL >= cfg.ds_min_maxL:
        return "Ds"
    if f.sumL_frac >= 1.0:
        return "Dk"
    return "Dw"


GEOMS = {
    "Qwen/Qwen3-8B": ModelGeom(36, 32, 8, 128, 15.1e9),
    "Qwen/Qwen3-4B": ModelGeom(36, 32, 8, 128, 8.0e9),
    "Qwen/Qwen3-1.7B": ModelGeom(28, 16, 8, 128, 3.4e9),
    "meta-llama/Llama-3.1-8B-Instruct": ModelGeom(32, 32, 8, 128, 15.0e9),
}


def geom_for(model_name: str, hf_config=None) -> ModelGeom:
    """Prefer the HF config (vLLM resolves names to snapshot paths); fall back to the static table."""
    if hf_config is not None:
        try:
            nl = hf_config.num_hidden_layers
            hq = hf_config.num_attention_heads
            hkv = getattr(hf_config, "num_key_value_heads", hq)
            hd = getattr(hf_config, "head_dim", None) or hf_config.hidden_size // hq
            h, inter, vocab = hf_config.hidden_size, hf_config.intermediate_size, hf_config.vocab_size
            per_layer = h * (hq * hd + 2 * hkv * hd) + hq * hd * h + 3 * h * inter
            wbytes = 2.0 * (nl * per_layer + vocab * h)  # bf16 weights streamed per decode step (+lm_head)
            return ModelGeom(nl, hq, hkv, hd, wbytes)
        except AttributeError:
            pass
    for k, v in GEOMS.items():
        name = k.split("/")[-1]
        if model_name.rstrip("/").endswith(name) or f"--{name}/" in model_name:
            return v
    return ModelGeom()
