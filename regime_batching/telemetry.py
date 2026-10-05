"""Per-engine-step telemetry ("markers") for baseline and proposed runs.

Enabled with REGIME_TELEMETRY_DIR=<dir>. In the worker process this:
  * derives per-request (q_len, kv_len, oracle final len, oracle remaining) from SchedulerOutput,
  * computes StepFeatures + regime label + predicted baseline FA3 split path,
  * pushes an NVTX range "step=… R=… B=… T=… maxL=… attn=…" around the step (visible in nsys),
  * times the step on the GPU with CUDA events (resolved lazily, no sync),
  * records the cudagraph descriptor actually dispatched and the attention decision (if the regime
    backend is active),
  * appends one JSON line per step to <dir>/steps_<pid>.jsonl from a background thread.
Set REGIME_TELEMETRY_DETAIL=0 to drop the per-request arrays.
"""
from __future__ import annotations

import atexit
import json
import os
import queue
import threading
import time
from collections import deque

import torch

from regime_batching import state
from regime_batching.features import RegimeConfig, compute_features, geom_for

_INSTALLED = False


class _Writer:
    def __init__(self, path: str):
        self.q: queue.Queue = queue.Queue()
        self.path = path
        self.t = threading.Thread(target=self._run, daemon=True)
        self.t.start()
        atexit.register(self.close)

    def _run(self):
        # plain JSONL + frequent flushes: stays readable if the server is killed (compress offline)
        with open(self.path, "a", buffering=1 << 20) as f:
            last = time.monotonic()
            while True:
                try:
                    item = self.q.get(timeout=1.0)
                except queue.Empty:
                    item = ...
                if item is None:
                    break
                if item is not ...:
                    f.write(json.dumps(item, separators=(",", ":")) + "\n")
                if time.monotonic() - last > 2.0:
                    f.flush()
                    last = time.monotonic()

    def put(self, item):
        self.q.put(item)

    def close(self):
        self.q.put(None)
        self.t.join(timeout=10)


class StepTelemetry:
    def __init__(self, vllm_config):
        out_dir = os.environ["REGIME_TELEMETRY_DIR"]
        os.makedirs(out_dir, exist_ok=True)
        self.detail = os.environ.get("REGIME_TELEMETRY_DETAIL", "1") == "1"
        self.nvtx = os.environ.get("REGIME_NVTX", "1") == "1"
        self.writer = _Writer(os.path.join(out_dir, f"steps_{os.getpid()}.jsonl"))
        model = vllm_config.model_config.model
        self.geom = geom_for(model, getattr(vllm_config.model_config, "hf_config", None))
        self.cfg = RegimeConfig.load()
        cc = vllm_config.compilation_config
        self.full_cg = bool(cc.cudagraph_mode is not None and cc.cudagraph_mode.has_full_cudagraphs())
        if cc.max_cudagraph_capture_size:
            self.cfg.cg_capture_max = cc.max_cudagraph_capture_size
        self.step = 0
        self.pending: deque = deque()
        self.open_rec = None
        meta = dict(kind="meta", model=model, pid=os.getpid(), full_cg=self.full_cg,
                    cudagraph_mode=str(cc.cudagraph_mode), capture_max=self.cfg.cg_capture_max,
                    attention_backend=str(getattr(vllm_config, "attention_config", None) and
                                          vllm_config.attention_config.backend),
                    max_num_seqs=vllm_config.scheduler_config.max_num_seqs,
                    max_num_batched_tokens=vllm_config.scheduler_config.max_num_batched_tokens,
                    regime_cfg=self.cfg.__dict__, geom=self.geom.__dict__, t=time.time())
        self.writer.put(meta)

    # ---- request bookkeeping -------------------------------------------------------------
    def _reqs_from_sched(self, so):
        oracle = state.REQ_ORACLE
        for nr in so.scheduled_new_reqs:
            plen = len(nr.prompt_token_ids) if nr.prompt_token_ids is not None else 0
            mt = nr.sampling_params.max_tokens if nr.sampling_params is not None else -1
            oracle[nr.req_id] = (plen, mt if mt is not None else -1)
        rows = []
        nst = so.num_scheduled_tokens
        for nr in so.scheduled_new_reqs:
            q = nst.get(nr.req_id, 0)
            if q <= 0:
                continue
            plen, mt = oracle[nr.req_id]
            rows.append((nr.req_id, q, nr.num_computed_tokens + q, plen + mt, mt))
        cr = so.scheduled_cached_reqs
        n_out = cr.num_output_tokens if len(cr.num_output_tokens) == len(cr.req_ids) else [None] * len(cr.req_ids)
        for rid, nct, no in zip(cr.req_ids, cr.num_computed_tokens, n_out):
            q = nst.get(rid, 0)
            if q <= 0:
                continue
            plen, mt = oracle.get(rid, (0, -1))
            rem = (mt - no) if (mt >= 0 and no is not None) else -1
            rows.append((rid, q, nct + q, plen + mt, rem))
        for rid in so.finished_req_ids:
            oracle.pop(rid, None)
        return rows

    # ---- step hooks ----------------------------------------------------------------------
    def on_execute_start(self, so):
        if self.open_rec is not None:  # previous step had no separate sample_tokens call
            self._close()
        self._drain()
        if so.total_num_scheduled_tokens <= 0:
            return
        rows = self._reqs_from_sched(so)
        feats = compute_features(rows, self.geom, self.cfg, full_cg=self.full_cg)
        state.CURRENT_FEATURES = feats
        state.LAST_ATTN_DECISION = None
        start = torch.cuda.Event(enable_timing=True)
        start.record()
        label = None
        if self.nvtx:
            label = (f"step={self.step} R={feats.regime} B={feats.num_reqs} dec={feats.n_dec} T={feats.T} "
                     f"maxL={feats.maxL} skew={feats.skew:.1f} fa3={feats.fa3_path}")
            torch.cuda.nvtx.range_push(label)
        self.open_rec = dict(kind="step", step=self.step, t0=time.time(), t0m=time.monotonic(), start=start,
                             feats=feats, rows=rows if self.detail else None, nvtx=label is not None)
        self.step += 1

    def on_sample_end(self):
        if self.open_rec is not None:
            self._close()

    def _close(self):
        rec = self.open_rec
        self.open_rec = None
        end = torch.cuda.Event(enable_timing=True)
        end.record()
        if rec["nvtx"]:
            torch.cuda.nvtx.range_pop()
        rec["end"] = end
        rec["cpu_ms"] = (time.monotonic() - rec.pop("t0m")) * 1e3
        rec["cg"] = state.LAST_BATCH_DESC
        rec["attn"] = state.LAST_ATTN_DECISION
        self.pending.append(rec)

    def _drain(self, block: bool = False):
        while self.pending:
            rec = self.pending[0]
            if not block and not rec["end"].query():
                break
            if block:
                rec["end"].synchronize()
            self.pending.popleft()
            gpu_ms = rec.pop("start").elapsed_time(rec.pop("end"))
            feats = rec.pop("feats").to_dict()
            rows = rec.pop("rows")
            rec.pop("nvtx", None)
            out = dict(rec, gpu_ms=gpu_ms, **feats)
            if rows is not None:
                out["reqs"] = [[r[0], r[1], r[2], r[3], r[4]] for r in rows]
            self.writer.put(out)

    def flush(self):
        if self.open_rec is not None:
            self._close()
        self._drain(block=True)


_TEL: StepTelemetry | None = None


def install():
    """Monkey-patch the GPU worker + MRV2 cudagraph dispatch. Safe to call in any process."""
    global _INSTALLED
    if _INSTALLED or not os.environ.get("REGIME_TELEMETRY_DIR"):
        return
    _INSTALLED = True
    from vllm.v1.worker import gpu_worker

    Worker = gpu_worker.Worker
    orig_exec = Worker.execute_model
    orig_sample = Worker.sample_tokens

    def execute_model(self, scheduler_output):
        global _TEL
        if _TEL is None:
            _TEL = StepTelemetry(self.vllm_config)
            atexit.register(_TEL.flush)
        _TEL.on_execute_start(scheduler_output)
        return orig_exec(self, scheduler_output)

    def sample_tokens(self, grammar_output):
        out = orig_sample(self, grammar_output)
        if _TEL is not None:
            _TEL.on_sample_end()
        return out

    Worker.execute_model = execute_model
    Worker.sample_tokens = sample_tokens

    try:
        from vllm.v1.worker.gpu import model_runner as mr

        orig_dispatch = mr.dispatch_cg_and_sync_dp

        def dispatch_cg_and_sync_dp(*args, **kwargs):
            desc, sync = orig_dispatch(*args, **kwargs)
            state.LAST_BATCH_DESC = dict(mode=str(desc.cg_mode).split(".")[-1], num_tokens=desc.num_tokens,
                                         num_reqs=desc.num_reqs)
            return desc, sync

        mr.dispatch_cg_and_sync_dp = dispatch_cg_and_sync_dp
    except Exception as e:  # V1 runner fallback: no descriptor logging
        print(f"[regime_batching] cudagraph dispatch hook not installed: {e}")
