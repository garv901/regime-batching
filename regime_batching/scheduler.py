"""Oracle-aware scheduler (C4): peak-KV admission gate with backfill + optional step shaping.

Use with:  --scheduler-cls regime_batching.scheduler.RegimeScheduler  (see plugin.py for env vars).
Subclasses AsyncScheduler so async scheduling stays on (a custom scheduler_cls only logs a warning).
"""
from __future__ import annotations

import os
import time

from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.request_queue import SchedulingPolicy

from regime_batching import state
from regime_batching.oracle import PeakPlanner, current_len, remaining


class RegimeScheduler(AsyncScheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        env = os.environ.get
        self.admit = env("REGIME_SCHED_ADMIT", "peak")
        self.shape = env("REGIME_SCHED_SHAPE", "off")
        assert self.admit in ("peak", "fcfs", "sjf") and self.shape in ("off", "cg_cap")
        self.watermark = float(env("REGIME_SCHED_WATERMARK", "0.02"))
        self.shape_min_kv = int(env("REGIME_SHAPE_MIN_KV", "4096"))
        self.shape_min_remain = int(env("REGIME_SHAPE_MIN_REMAIN", "256"))
        self.shape_max_backlog = int(env("REGIME_SHAPE_MAX_BACKLOG", "16384"))
        cc = self.vllm_config.compilation_config
        self.shape_cap = int(env("REGIME_SHAPE_CAP") or getattr(cc, "max_cudagraph_capture_size", None) or 512)
        pool = self.kv_cache_manager.block_pool
        self.kv_cap_tokens = (pool.num_gpu_blocks - 1) * self.block_size * (1.0 - self.watermark)
        self._calls = 0
        self._writer = None
        d = env("REGIME_TELEMETRY_DIR")
        if d:
            from regime_batching.telemetry import _Writer
            os.makedirs(d, exist_ok=True)
            self._writer = _Writer(os.path.join(d, f"sched_{os.getpid()}.jsonl"))

    # -- admission gate ------------------------------------------------------------------------
    def _gate(self, budget: int):
        """Returns (gated requests, planner peak). Greedy FCFS walk with backfill."""
        waiting = self.waiting
        if not waiting:
            return [], None
        pad = self.block_size - 1
        run = self.running
        u = [current_len(q) for q in run]
        r = [remaining(q) for q in run]
        planner = PeakPlanner(u, r, pad)
        peak0 = planner.peak()
        seats = self.max_num_running_reqs - len(run)
        cands = sorted(waiting) if self.policy == SchedulingPolicy.PRIORITY else waiting
        gated, admitted, new_tok = [], 0, 0
        dirty = False
        for q in cands:
            if admitted >= seats or new_tok >= 2 * budget:
                break  # vLLM cannot admit further this step (seats / token budget with prefix-hit slack)
            uq = q.num_prompt_tokens + q.num_output_tokens
            rq = q.max_tokens - q.num_output_tokens
            if dirty:
                planner = PeakPlanner(u, r, pad)
                dirty = False
            if planner.fits(uq, rq, self.kv_cap_tokens):
                u.append(uq)
                r.append(rq)
                dirty = True
                admitted += 1
                new_tok += max(uq - q.num_computed_tokens, 1)
            else:
                gated.append(q)
        return gated, peak0

    def _restore(self, orig: list, gated: list):
        """Put gated requests back; waiting order == original order, new preemptees stay in front."""
        w = self.waiting
        cur = list(w)
        if self.policy == SchedulingPolicy.PRIORITY:
            for q in gated:
                w.add_request(q)
            return
        gset = set(gated)
        cset = set(cur)
        oset = set(orig)
        front = [q for q in cur if q not in oset]  # preempted during this call (vLLM prepends)
        rest = [q for q in orig if q in gset or q in cset]
        w.remove_requests(cur)
        for q in front + rest:
            w.add_request(q)

    def _sjf_order(self):
        """Ablation: shortest oracle-remaining first (length-aware ordering, no KV planning)."""
        w = self.waiting
        cur = list(w)
        key = [q.max_tokens - q.num_output_tokens for q in cur]
        if all(a <= b for a, b in zip(key, key[1:])):
            return
        srt = [cur[i] for i in sorted(range(len(cur)), key=key.__getitem__)]  # stable
        w.remove_requests(cur)
        for q in srt:
            w.add_request(q)

    # -- step shaping --------------------------------------------------------------------------
    def _want_cap(self) -> bool:
        if not any(q.num_computed_tokens >= self.shape_min_kv and remaining(q) >= self.shape_min_remain
                   for q in self.running):
            return False
        backlog = sum(max(0, q.num_prompt_tokens - q.num_computed_tokens) for q in self.waiting)
        backlog += sum(max(0, q.num_prompt_tokens - q.num_computed_tokens) for q in self.running)
        return backlog <= self.shape_max_backlog

    # -- main ----------------------------------------------------------------------------------
    def schedule(self, throttle_prefills: bool = False):
        self._calls += 1
        orig_budget = self.max_num_scheduled_tokens
        gated, peak = [], None
        orig = None
        if self.admit == "peak" and self.waiting:
            orig = list(self.waiting)
            gated, peak = self._gate(orig_budget)
            if gated:
                self.waiting.remove_requests(gated)
        elif self.admit == "sjf" and len(self.waiting) > 1 and self.policy != SchedulingPolicy.PRIORITY:
            self._sjf_order()
        capped = False
        if self.shape == "cg_cap" and self._want_cap():
            self.max_num_scheduled_tokens = min(orig_budget, self.shape_cap)
            capped = True
        n_wait = len(orig) if orig is not None else len(self.waiting)
        try:
            out = super().schedule(throttle_prefills)
        finally:
            self.max_num_scheduled_tokens = orig_budget
            if gated:
                self._restore(orig, gated)
        rec = {
            "t": time.time(), "call": self._calls, "admit": self.admit, "n_waiting": n_wait,
            "n_gated": len(gated), "gated_ids": [q.request_id for q in gated[:20]],
            "peak_tokens": peak, "cap_tokens": self.kv_cap_tokens, "shape": capped,
            "budget": min(orig_budget, self.shape_cap) if capped else orig_budget,
            "n_new": len(out.scheduled_new_reqs),
            "new_ids": [n.req_id for n in out.scheduled_new_reqs],
            "n_preempt": len(out.preempted_req_ids or ()),
        }
        state.LAST_SCHED_DECISION = rec
        if self._writer is not None:
            self._writer.put(rec)
        return out
