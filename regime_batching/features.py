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
    f.regime = classify(f, geom, cfg)
    return f


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
