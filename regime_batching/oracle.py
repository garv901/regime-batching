"""Oracle KV planner (token units) + helpers to read oracle lengths off vLLM Request objects.

Oracle: every request has ignore_eos and max_tokens == true output length, so the final length
f = num_prompt_tokens + max_tokens is known at arrival.
"""
from __future__ import annotations

import numpy as np


class PeakPlanner:
    """Future peak KV usage of admitted requests (each grows 1 token/step until done, then frees all).

    u: current KV tokens per admitted request (full prompt for admitted-but-prefilling).
    r: remaining decode steps per request.
    pad: per-request slack added to u so token-level checks are block-safe (block_size - 1 covers
         the rounding of any length up to the next block boundary).
    Port of analysis/pen_and_paper.py::_PeakPlanner (the k >= r_new tail is made exact: the original
    added the new request's u to the whole suffix max, a slight over-estimate).
    """

    def __init__(self, u, r, pad: float = 0.0):
        u = np.asarray(u, dtype=np.float64).reshape(-1) + pad
        r = np.asarray(r, dtype=np.int64).reshape(-1)
        self.pad = float(pad)
        o = np.argsort(r, kind="stable")
        self.r = r[o]
        self.u = u[o]
        n = len(self.r)
        self.su = np.concatenate([np.cumsum(self.u[::-1])[::-1], [0.0]])  # suffix sums over r_i > k
        self.cnt = np.arange(n, -1, -1, dtype=np.float64)
        self.ks = np.unique(np.concatenate([[0], np.maximum(self.r - 1, 0)]))
        jk = np.searchsorted(self.r, self.ks, side="right")  # vectorized _usage(k) over all ks
        self.base = self.su[jk] + self.ks * self.cnt[jk]
        self.pre = np.maximum.accumulate(self.base + self.ks)
        self.suf = np.maximum.accumulate(self.base[::-1])[::-1]

    def _usage(self, k):
        j = np.searchsorted(self.r, k, side="right")  # first with r > k
        return self.su[j] + k * self.cnt[j]

    def peak(self) -> float:
        """Peak KV tokens (incl. pad) over the future of the current set (0 if empty)."""
        return float(self.suf[0]) if len(self.r) else 0.0

    def peak_with(self, u, r) -> float:
        u = u + self.pad
        j = np.searchsorted(self.ks, r - 1, side="left")  # candidates with k < r-1 -> [0, j)
        peak = self._usage(r - 1) + u + r - 1
        if j > 0:
            peak = max(peak, self.pre[j - 1] + u)
        j2 = np.searchsorted(self.ks, r, side="left")  # k >= r: the new request is already gone
        if j2 < len(self.ks):
            peak = max(peak, self.suf[j2])
        return float(peak)

    def fits(self, u, r, cap) -> bool:
        return self.peak_with(u, r) <= cap

    def blocks(self, block_size: int) -> int:
        """Peak in blocks, rounding each request up (valid when constructed with pad=0)."""
        return -(-int(self.peak()) // block_size)


def final_len(req) -> int:
    return req.num_prompt_tokens + req.max_tokens


def current_len(req) -> int:
    """Current KV tokens, over-estimated by async-scheduling placeholders (in-flight sampled tokens)."""
    return req.num_prompt_tokens + req.num_output_tokens + req.num_output_placeholders


def remaining(req) -> int:
    """Remaining decode steps (oracle), net of in-flight placeholders; >= 0."""
    return max(0, req.max_tokens - req.num_output_tokens - req.num_output_placeholders)
