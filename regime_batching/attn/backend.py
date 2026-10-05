"""Regime-aware FA3 dispatch, v1: per-step split policy (component C3).

vLLM 0.30's FA backend passes max_num_splits=32 only when T <= max_cudagraph_capture_size (full-CG range).
Above that, FA3's heuristic treats the varlen batch as a single sequence and disables splitting whenever
a prefill chunk spans >= 0.8*SMs tiles. Long-KV decode rows are then streamed by one CTA per KV head:
this is the X regime, 1.2-4.8x slower attention per layer in the S2 sweep. For X steps only, this builder
raises the split upper bound so FA3's per-sequence dynamic splits balance the long rows. Every other step
is built exactly as in vLLM; forcing splits costs ~8% on non-straggler mixed steps. Full-CG decode steps
(T <= capture size) never reach the override, so captured graphs are unchanged.

Decision inputs are CPU-side metadata only (query_start_loc_cpu, seq_lens_cpu_upper_bound): no GPU sync.
Env:
  REGIME_ATTN_POLICY  x_split (default) | always_split | off
  REGIME_X_THRESH     predicted baseline/balanced attention ratio that triggers splitting (default 1.2)
  REGIME_X_SPLITS     split upper bound for X steps (default: flash_attn_max_num_splits_for_cuda_graph)
  REGIME_X_MIN_GAIN_US  predicted per-layer saving required to split (default 8 us: the in-engine cost of a
                      forced split on a short-KV mixed step was ~0.3 ms/step over 36 layers)
"""
from __future__ import annotations

import os
import time

import math

import numpy as np

from vllm.v1.attention.backends.flash_attn import FlashAttentionBackend, FlashAttentionMetadataBuilder

from regime_batching import state
from regime_batching.features import BETA_X, BW_CTA, BW_KV, F_ATT, NUM_SMS, fa3_static_splits


class RegimeFlashAttentionMetadataBuilder(FlashAttentionMetadataBuilder):
    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self._policy = os.environ.get("REGIME_ATTN_POLICY", "x_split")
        self._x_thresh = float(os.environ.get("REGIME_X_THRESH", "1.2"))
        self._x_splits = int(os.environ.get("REGIME_X_SPLITS", "0")) or \
            self.attention_config.flash_attn_max_num_splits_for_cuda_graph
        self._g = max(self.num_heads_q // max(self.num_heads_kv, 1), 1)
        self._tok_bytes = self.num_heads_kv * self.headdim * 2 * 2  # K+V bf16 per token per layer
        self._min_gain = float(os.environ.get("REGIME_X_MIN_GAIN_US", "8")) * 1e-6
        self._force_splits = 0

    def _decide(self, cam) -> dict:
        t0 = time.perf_counter()
        T = cam.num_actual_tokens
        dec = dict(policy=self._policy, T=T, splits=0, x_ratio=1.0, path="vllm")
        cg_max = self.max_cudagraph_size or 0
        if self._policy == "off" or (self.use_full_cuda_graph and T <= cg_max):
            return dec  # vLLM already passes the CG split bound here
        if self._policy == "always_split":
            dec.update(splits=self._x_splits, path="forced")
            return dec
        seq_cpu = cam.seq_lens_cpu_upper_bound
        max_q = int(cam.max_query_len)
        if seq_cpu is None or max_q == 1:
            return dec
        n = cam.num_reqs
        q = np.diff(cam.query_start_loc_cpu[: n + 1].numpy())
        kv = seq_cpu[:n].numpy()
        is_dec = q == 1
        L = kv[is_dec]
        if L.size == 0:
            return dec
        maxL = int(L.max())
        # FA3 heuristic: varlen batch counts as 1; a chunk with >= 0.8*SMs (m-block x kv-head) tiles never splits
        kbm = 64 if max_q * self._g <= 64 else 128
        if self.num_heads_kv * math.ceil(max_q * self._g / kbm) >= 0.8 * NUM_SMS:
            s = 1
        else:
            s, _ = fa3_static_splits(max_q, int(cam.max_seq_len), T, self._g, self.num_heads_kv, self.headdim,
                                     page_size=self.block_size, full_cg=False)
            if s >= self._x_splits:
                return dec
        qp, kp = q[~is_dec].astype(np.float64), kv[~is_dec]
        t_pre = float((4.0 * qp * (kp - qp / 2)).sum()) * self.headdim * self.num_heads_q / F_ATT
        t_bal = float(L.sum()) * self._tok_bytes / BW_KV + t_pre
        t_strag = maxL * self._tok_bytes / self.num_heads_kv / (BW_CTA * s)
        gain = t_strag - (1.0 - BETA_X) * t_bal  # predicted per-layer saving if the long rows are balanced
        x_ratio = max(1.0, 1.0 + gain / t_bal) if t_bal > 0 else 1.0
        dec.update(x_ratio=round(x_ratio, 3), heur_splits=s, maxL=maxL, max_q=max_q, gain_us=round(gain * 1e6, 1))
        if x_ratio > self._x_thresh and gain > self._min_gain:
            dec.update(splits=self._x_splits, path="x_split")
        dec["decide_us"] = round((time.perf_counter() - t0) * 1e6, 1)
        return dec

    def build(self, common_prefix_len, common_attn_metadata, fast_build: bool = False):
        dec = self._decide(common_attn_metadata)
        self._force_splits = dec["splits"]
        try:
            md = super().build(common_prefix_len, common_attn_metadata, fast_build)
        finally:
            self._force_splits = 0
        if dec["splits"]:
            md.max_num_splits = dec["splits"]
        state.LAST_ATTN_DECISION = dec
        return md

    def _get_scheduler_metadata(self, *, max_num_splits: int, **kw):
        if self._force_splits:
            max_num_splits = self._force_splits
        return super()._get_scheduler_metadata(max_num_splits=max_num_splits, **kw)


class RegimeFlashAttentionBackend(FlashAttentionBackend):
    """FLASH_ATTN with the regime-aware split policy; keeps the FLASH_ATTN name for vLLM's checks."""

    @staticmethod
    def get_builder_cls():
        return RegimeFlashAttentionMetadataBuilder
