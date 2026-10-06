"""CPU tests for analysis/mstar_admission.py on synthetic case dirs that follow the real producers' schemas.

  uv run --with numpy --with pandas --with matplotlib --with pytest python -m pytest -q tests/test_mstar_admission.py

Producers mirrored here (field for field):
  bench.json          benchmark/trace_replay.py build_bench          (regime-harness worktree)
  steps_*.jsonl       mstar/profile/step_telemetry.py                (meta, step, event, drop, end)
  admission_*.jsonl   mstar/engine/resources/kv/admission_log.py + KVManager._log / _log_release (bench-peak)

Scenario (all times seconds from trace start t0 = 1000.0 on the monotonic clock, 6 requests per run):
  idx  cls    out   offset  ttft  lat(scale 1)
   0   short   10    0.0    0.1    1.0
   1   short   10    1.0    0.2    2.0
   2   short   10    2.0    0.3    3.0
   3   short   10    3.0    0.4    4.0
   4   long  2000    0.5    5.0   60.0
   5   long  2000    1.5    6.0   80.0
Cases: ac r1 (healthy), ac r2 (idx5 timed out after 900 s), peak_bf r1 (scale 0.5, with admission log),
peak_bf r2 (no telemetry at all), peak_bf bagel_control r1 (only a server.log).
Shared server (make_shared_job, the one-server-per-job layout): one server_1/ whose telemetry, admission log and
server.log hold two runs 1000 s apart, symlinked into both case dirs (plus a third case whose client died, and a
second server for a fourth).
"""
import json
import math
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "analysis"))
import mstar_admission as ma  # noqa: E402

T0 = 1000.0
PID = 4242
REQS = [  # (cls, server-side input tokens, output tokens, offset s, ttft s, latency s)
    ("short", 100, 10, 0.0, 0.1, 1.0),
    ("short", 100, 10, 1.0, 0.2, 2.0),
    ("short", 100, 10, 2.0, 0.3, 3.0),
    ("short", 100, 10, 3.0, 0.4, 4.0),
    ("long", 100, 2000, 0.5, 5.0, 60.0),
    ("long", 100, 2000, 1.5, 6.0, 80.0),
]
# handle h of each request on the worker; request 5 reuses request 0's handle after request 0 is removed
HANDLE = [0, 1, 2, 3, 4, 0]
TIMEOUT_S = 900.0


# ------------------------------------------------------------------------------------------------ producers
def _stats(prefix, seconds):  # trace_replay._stats
    a = np.asarray(seconds, dtype=float) * 1e3
    keys = ["mean", "median", "std", "p50", "p90", "p99"]
    if not len(a):
        return {f"{k}_{prefix}_ms": 0.0 for k in keys}
    vals = [a.mean(), np.median(a), a.std(), np.percentile(a, 50), np.percentile(a, 90), np.percentile(a, 99)]
    return {f"{k}_{prefix}_ms": float(v) for k, v in zip(keys, vals)}


def make_bench(run_id, scale=1.0, timeout_idx=None, duration=None, cap_out=None):
    """The dict trace_replay.build_bench returns (same keys, same conventions for failed requests).
    cap_out: the engine stops every request at this many output tokens (BAGEL's max_iters), whatever it asked for."""
    n = len(REQS)
    ok = [i != timeout_idx for i in range(n)]
    starts = [T0 + r[3] for r in REQS]
    out_lens = [min(r[2], cap_out or r[2]) if o else 0 for r, o in zip(REQS, ok)]
    ttfts = [r[4] if o else 0.0 for r, o in zip(REQS, ok)]
    lats = [r[5] * scale if o else 0.0 for r, o in zip(REQS, ok)]
    itls = [[0.02] * (r[2] - 1) if o else [] for r, o in zip(REQS, ok)]
    ends = [s + l for s, l, o in zip(starts, lats, ok) if o]
    if duration is None:
        duration = (TIMEOUT_S + REQS[timeout_idx][3]) if timeout_idx is not None else max(ends) - T0 + 0.5
    # an exception with an empty message (asyncio.TimeoutError): build_bench writes "request failed"
    errors = ["" if o else "request failed" for o in ok]
    completed = sum(ok)
    total_out = sum(out_lens)
    return {
        "date": "20261006-120000", "endpoint_type": "mstar", "backend": "mstar", "label": run_id,
        "model_id": "ByteDance-Seed/BAGEL-7B-MoT", "tokenizer_id": "ByteDance-Seed/BAGEL-7B-MoT",
        "num_prompts": n, "request_rate": "trace", "burstiness": None, "max_concurrency": None,
        "duration": duration, "completed": completed, "failed": n - completed,
        "total_input_tokens": sum(r[1] for r, o in zip(REQS, ok) if o), "total_output_tokens": total_out,
        "request_throughput": completed / duration, "request_goodput": None,
        "output_throughput": total_out / duration, "total_token_throughput": (100 * completed + total_out) / duration,
        "input_lens": [r[1] for r in REQS], "output_lens": out_lens, "ttfts": ttfts, "itls": itls, "latencies": lats,
        "start_times": starts, "errors": errors, "max_concurrent_requests": 5,
        **_stats("ttft", [t for t, o in zip(ttfts, ok) if o]), **_stats("tpot", []), **_stats("itl", []),
        **_stats("e2el", [l for l, o in zip(lats, ok) if o]),
        "cls": [r[0] for r in REQS], "request_ids": [f"{run_id}-{i:06d}" for i in range(n)],
        "trace_input_lens": [r[1] - 20 for r in REQS], "prompt_user_tokens": [r[1] - 20 for r in REQS],
        "intended_output_lens": [r[2] for r in REQS], "intended_start_offsets": [r[3] for r in REQS],
        "intended_start_times": starts, "sched_lag_s": [0.0] * n, "sched_lag_ms_p50": 0.0, "sched_lag_ms_p99": 0.0,
        "sched_lag_ms_max": 0.0, "client_loop_lag_ms_p99": 0.1, "client_loop_lag_ms_max": 0.2,
        "t0_monotonic": T0, "t0_wall": 1.78e9, "template_overhead_tokens": 20, "n_clamped_inputs": 0,
        "n_short_outputs": sum(ol != r[2] for r, ol, o in zip(REQS, out_lens, ok) if o),
        "trace": {"path": "data/workloads/bagel_burst.jsonl", "rows": n, "seed": 0, "timeout_s": TIMEOUT_S},
    }


def step_row(seq, rel, walk, bs, gpu_ms, free, ok=True, node="LLM", admit_err=None):
    tm = T0 + rel
    decode = walk == "decode"
    row = {
        "kind": "step", "seq": seq, "ts": 1.78e9 + rel, "tm": tm, "rank": 0, "node": node, "walk": walk, "bs": bs,
        "nt": bs if decode else 128 * bs, "mode": "graph" if decode else "eager", "batched": True, "ok": ok,
        "cpu_ms": 1.0, "pre_ms": 0.5 if ok else None, "free": free,
    }
    if decode:
        row["bucket"] = [bs, bs]
        row["slot"] = 0
    row["qmax"] = 1 if decode else 128
    row["kvmax"], row["kvsum"] = 300, 300 * bs
    if admit_err:
        row["admit_err"] = admit_err
    row["rids"], row["q"], row["kv"] = list(range(bs)), [row["qmax"]] * bs, [300] * bs
    row["gpu_ms"] = gpu_ms
    if gpu_ms is not None:
        row["gap_ms"] = 0.1
    return row


def event(ev, rel, **fields):
    return {"kind": "event", "ev": ev, "ts": 1.78e9 + rel, "tm": T0 + rel, "rank": 0, **fields}


def telemetry_rows(run_id):
    rid = [f"{run_id}-{i:06d}" for i in range(len(REQS))]
    warm = f"{run_id}-warmup0"
    kv = lambda n: {"kv": n}  # noqa: E731
    meta = [
        {"kind": "meta", "phase": "process", "schema": 1, "rank": 0, "pid": PID, "detail": True, "ts": 1.78e9,
         "tm": T0 - 60, "env": {"MSTAR_STEP_TELEMETRY_DIR": "x"}},
        {"kind": "meta", "phase": "worker", "worker_id": "w0", "rank": 0, "pid": PID, "ts": 1.78e9, "tm": T0 - 60},
        {"kind": "meta", "phase": "engine", "schema": 1, "rank": 0, "pid": PID, "model": "bagel", "device": "cuda:0",
         "gpu_events": True, "ts": 1.78e9, "tm": T0 - 59,
         "nodes": {"LLM": {"buckets": [["decode", 1, 1, "None"]], "decode_bs": [1, 2, 4, 8, 16, 32, 64]}},
         "kv_pools": {"kv": {"max_num_pages": 2048, "page_size": 128, "cpu_offload_pages": 0, "free_at_first_step": 2047}},
         "gpu_mem_free_bytes": 1, "gpu_mem_total_bytes": 2, "gpu_name": "H100"},
    ]
    steps = [  # not in tm order: the file is written as end events resolve
        step_row(5, 5.0, "decode", 9, None, kv(1100), ok=False, admit_err="AdmissionDeferred"),  # refused: no gpu_ms
        step_row(2, 1.0, "prefill_text", 2, 5000.0, kv(2000)),
        step_row(1, -5.0, "prefill_text", 1, 10.0, kv(2047)),  # warmup, before the trace starts
        step_row(4, 3.0, "decode", 6, 15000.0, kv(900)),  # the KV minimum
        step_row(3, 2.0, "decode", 4, 10000.0, kv(1500)),
        step_row(6, 4.0, "decode", 2, 11000.0, kv(1200)),
        step_row(7, 7.0, "image_gen", 1, None, {}, node="vae_decoder"),  # a node without a KV pool, gpu_ms unresolved
    ]
    ev = [
        event("new_request", -6.0, rid=warm, h=0, partition="p0", walk="prefill_text"),
        event("remove_done", -5.5, rid=warm, h=0),
        event("new_request", 0.05, rid=rid[0], h=0, partition="p0", walk="prefill_text"),
        event("new_request", 0.55, rid=rid[4], h=4, partition="p0", walk="prefill_text"),
        event("new_request", 1.05, rid=rid[1], h=1, partition="p0", walk="prefill_text"),
        event("remove_done", 1.2, rid=rid[0], h=0),
        event("new_request", 1.55, rid=rid[5], h=0, partition="p0", walk="prefill_text"),  # recycled handle 0
        event("new_request", 2.05, rid=rid[2], h=2, partition="p0", walk="prefill_text"),
        event("new_request", 3.05, rid=rid[3], h=3, partition="p0", walk="prefill_text"),
        event("hold", 2.5, n=1, h=[4], rid=[rid[4]], backoff_s=0.01),
        event("hold", 3.5, n=2, h=[1, 2], rid=[rid[1], rid[2]], backoff_s=0.01),
        event("admit_fail", 2.6, node="LLM", walk="decode", bs=4, reason="AdmissionDeferred", h=0, rid=rid[0],
              label=None, pages_short=None, resource="kv"),
        event("admit_fail", 2.7, node="LLM", walk="decode", bs=4, reason="AdmissionDeferred", h=4, rid=rid[4],
              label=None, pages_short=None, resource="kv"),
        event("admit_fail", 2.8, node="LLM", walk="decode", bs=4, reason="AdmissionDeferred", h=4, rid=rid[4],
              label=None, pages_short=None, resource="kv"),
        event("admit_fail", 3.2, node="LLM", walk="decode", bs=6, reason="GrantDeferred", h=0, rid=rid[5],
              label="main", pages_short=None, resource="kv"),
        event("admit_fail", 3.3, node="LLM", walk="decode", bs=6, reason="AllocationFailed", h=None, rid=None,
              label="main", pages_short=2, resource="kv"),
        event("offload", 3.4, node="LLM", h=4, rid=rid[4], in_batch=False),
        event("loop_stop", 4.0, node="LLM", walk="decode", h=[0], rid=[rid[0]]),
        event("req_done", -5.4, h=[0], rid=[warm]),
        *[event("req_done", REQS[i][3] + REQS[i][5], h=[HANDLE[i]], rid=[rid[i]]) for i in range(len(REQS))],
    ]
    return meta, steps, ev


def adm_row(event_, rel, h, **fields):  # KVManager._log: base fields + per-event fields
    return {"ts_wall": 1.78e9 + rel, "ts_mono": T0 + rel, "pool": "kv", "event": event_, "rid": h, "fit": "peak",
            "order": "backfill", "supply": 1500, "n_reserved": 3, "n_waiting": 2, **fields}


def admission_rows():
    res = lambda rel, h, waited, us, head=False: adm_row(  # noqa: E731
        "reserve", rel, h, is_head=head, planned_peak=900, plan_capacity=2000, need=8, hit=0, waited_s=waited,
        planner_us=us)
    return [
        res(-5.9, 0, 0.0, 1.0),  # warmup request: before the trace, excluded
        adm_row("release", -5.6, 0, held=1, claim=2, reserved_s=0.3),
        res(0.06, 0, 0.0, 10.0, head=True),  # handle 0 -> request 0 (short)
        adm_row("wait", 0.56, 4, is_head=True, why="peak", planned_peak=3000, plan_capacity=2000, need=70, planner_us=100.0),
        res(1.06, 1, 0.5, 30.0),
        adm_row("release", 1.19, 0, held=1, claim=2, reserved_s=1.13),
        adm_row("wait", 1.58, 0, is_head=False, why="easy", planned_peak=900, plan_capacity=2000, need=70, planner_us=200.0),
        res(2.06, 2, 1.0, 50.0),
        res(3.06, 3, 1.5, 60.0),
        res(4.56, 4, 4.0, 20.0, head=True),  # handle 4 -> request 4 (long)
        res(7.58, 0, 6.0, 40.0),  # handle 0 again: now request 5 (long), not request 0
        adm_row("grant_deferred", 10.0, 4, label="main", pages=1, planner_us=5.0),
        adm_row("grant_deferred", 11.0, 4, label="main", pages=1, planner_us=15.0),
        adm_row("release", 41.6, 0, held=70, claim=70, reserved_s=73.42),  # inside both runs' windows (42 s and 82 s + drain)
    ]


def write_jsonl(path, rows, truncated_tail=False):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
        if truncated_tail:
            f.write('{"kind":"step","seq":99,"tm":')  # server killed mid-write


def make_case(job_dir, workload, rep, *, config, scale=1.0, timeout_idx=None, telemetry=True, admission=False,
              bench=True, truncated_tail=False, cap_out=None):
    d = os.path.join(job_dir, f"{workload}_r{rep}")
    os.makedirs(d, exist_ok=True)
    run_id = f"{config}-{workload}-r{rep}"
    open(os.path.join(d, "server.log"), "w").write("INFO server up\n")
    if bench:
        json.dump(make_bench(run_id, scale, timeout_idx, cap_out=cap_out), open(os.path.join(d, "bench.json"), "w"))
        open(os.path.join(d, "client.log"), "w").write("trace replay: ok\nWrote bench.json\n")
        open(os.path.join(d, "startup_s.txt"), "w").write("71.3\n")
        open(os.path.join(d, "gpu_mem.csv"), "w").write(
            "2026/10/06 12:00:00.100, 0, 70000, 81559, 12\n2026/10/06 12:00:01.100, 0, 71000, 81559, 88\n")
    if telemetry:
        os.makedirs(os.path.join(d, "telemetry"), exist_ok=True)
        meta, steps, ev = telemetry_rows(run_id)
        end = {"kind": "end", "rows": len(steps) + len(ev), "dropped": 0, "ts": 1.78e9, "tm": T0 + 90}
        write_jsonl(os.path.join(d, "telemetry", f"steps_rank0_pid{PID}.jsonl"), [*meta, *steps, *ev, end], truncated_tail)
        # a second process (no engine): meta + end only
        write_jsonl(os.path.join(d, "telemetry", "steps_rank0_pid4243.jsonl"), [
            {**meta[0], "pid": 4243}, {"kind": "end", "rows": 1, "dropped": 0, "ts": 1.78e9, "tm": T0 + 90}])
        if admission:
            write_jsonl(os.path.join(d, "telemetry", f"admission_kv_pid{PID}.jsonl"), admission_rows())
    return d


@pytest.fixture(scope="module")
def root(tmp_path_factory):
    r = str(tmp_path_factory.mktemp("results"))
    ac = os.path.join(r, "12712_ac")
    pb = os.path.join(r, "12714_peak_bf")
    for j in (ac, pb):
        os.makedirs(j)
        open(os.path.join(j, "meta.txt"), "w").write(f"job={os.path.basename(j)} config={os.path.basename(j).split('_', 1)[1]}\n")
    make_case(ac, "bagel_burst", 1, config="ac", truncated_tail=True)
    make_case(ac, "bagel_burst", 2, config="ac", timeout_idx=5)
    make_case(pb, "bagel_burst", 1, config="peak_bf", scale=0.5, admission=True)
    make_case(pb, "bagel_burst", 2, config="peak_bf", scale=0.5, telemetry=False)
    make_case(pb, "bagel_control", 1, config="peak_bf", bench=False, telemetry=False)
    return r


@pytest.fixture(scope="module")
def result(root, tmp_path_factory):
    out = str(tmp_path_factory.mktemp("out"))
    return out, ma.run([root], out)


def cfg(summary, wl, c):
    return summary["workloads"][wl]["configs"][c]


def rep(summary, wl, c, i):
    return cfg(summary, wl, c)["reps"][i]["metrics"]


# ------------------------------------------------------------------------------------------------ discovery
def test_discovery_root_job_and_case_dir(root):
    refs, problems = ma.discover([root])
    assert not problems and len(refs) == 5
    assert {(r.config, r.workload, r.rep) for r in refs} == {
        ("ac", "bagel_burst", 1), ("ac", "bagel_burst", 2), ("peak_bf", "bagel_burst", 1),
        ("peak_bf", "bagel_burst", 2), ("peak_bf", "bagel_control", 1)}
    assert {r.job for r in refs} == {"12712", "12714"}
    job_refs, _ = ma.discover([os.path.join(root, "12714_peak_bf")])
    assert len(job_refs) == 3
    case_refs, _ = ma.discover([os.path.join(root, "12712_ac", "bagel_burst_r2")])
    assert [(r.config, r.workload, r.rep) for r in case_refs] == [("ac", "bagel_burst", 2)]
    # "bagel_stag_r0.5_r1": the repetition is the last _r<N>
    assert ma.CASE_RE.match("bagel_stag_r0.5_r1").groups() == ("bagel_stag_r0.5", "1")
    _, probs = ma.discover([os.path.join(root, "nope")])
    assert probs


# ------------------------------------------------------------------------------------------------ per-case numbers
def test_healthy_case_bench_numbers(result):
    _, s = result
    m = rep(s, "bagel_burst", "ac", 0)
    assert (m["n_requests"], m["completed"], m["failed"], m["timed_out"]) == (6, 6, 0, 0)
    assert m["duration_s"] == pytest.approx(82.0)
    assert m["out_tok_s"] == pytest.approx(4040 / 82.0)
    assert m["short_e2e_mean_s"] == pytest.approx(2.5)
    assert m["short_e2e_p50_s"] == pytest.approx(2.5)
    assert m["short_e2e_p99_s"] == pytest.approx(np.percentile([1, 2, 3, 4], 99))  # 3.97
    assert m["short_ttft_mean_s"] == pytest.approx(0.25)
    assert m["short_ttft_p99_s"] == pytest.approx(0.397)
    assert (m["long_n"], m["long_e2e_mean_s"], m["long_e2e_max_s"], m["long_ttft_mean_s"]) == (2, 70.0, 80.0, 5.5)
    # completions at 1, 3, 5, 7, 60.5, 81.5 with a request outstanding throughout: the longest silence is 7 -> 60.5
    assert m["max_stall_s"] == pytest.approx(53.5)
    assert m["stalled"] is False
    assert m["startup_s"] == 71.3 and m["gpu_mem_peak_mib"] == 71000 and m["gpu_util_mean_pct"] == 50.0


def test_healthy_case_telemetry_numbers(result):
    _, s = result
    m = rep(s, "bagel_burst", "ac", 0)
    assert m["holds"] == 2 and m["held_requests"] == 3 and m["offloads"] == 1
    assert m["admit_fail_total"] == 5
    assert m["admit_fail"] == {"AdmissionDeferred": 3, "GrantDeferred": 1, "AllocationFailed": 1}
    assert m["min_free_kv_pages"] == 900  # the pre-trace warmup step (2047) and the non-KV row do not count
    assert m["kv_pool_pages"] == 2048 and m["kv_key"] == "kv"
    # window = trace start .. 82 s; gpu_ms of the 4 resolved steps in it: 5000 + 10000 + 15000 + 11000 = 41 s
    assert m["gpu_busy_share"] == pytest.approx(41.0 / 82.0)
    assert m["gpu_ms_unresolved"] == 2  # the refused step and the non-KV row
    assert m["decode_bs_mean"] == pytest.approx((4 + 6 + 2) / 3)  # the refused bs=9 decode step is not ok
    assert m["n_steps"] == 6
    assert "grant_deferred_log" not in m  # `ac` has no admission log and is not expected to
    # id join client <-> telemetry
    assert m["join_new_request"] == "6/6"
    assert m["dispatch_lag_median_s"] == pytest.approx(0.05)
    assert m["holds_by_class"] == {"long": 1, "short": 2}
    assert m["admit_fail_by_class"] == {"short": 1, "long": 3}  # the blamed-less AllocationFailed has no rid
    flags = cfg(s, "bagel_burst", "ac")["reps"][0]["flags"]
    assert flags == []  # the truncated last telemetry line and the extra empty-pid file are tolerated
    assert any("unparsable" in n for n in cfg(s, "bagel_burst", "ac")["reps"][0]["notes"])


def test_timed_out_case(result):
    _, s = result
    m = rep(s, "bagel_burst", "ac", 1)
    assert (m["completed"], m["failed"], m["timed_out"]) == (5, 0, 1)
    assert m["duration_s"] == pytest.approx(901.5)
    assert m["out_tok_s"] == pytest.approx(2040 / 901.5)
    assert (m["long_n"], m["long_e2e_mean_s"], m["long_failed"]) == (1, 60.0, 1)
    assert m["max_stall_s"] == pytest.approx(841.0)  # request 5 is outstanding from 60.5 (last completion) to 901.5
    assert m["stalled"] is True
    assert "STALLED" in cfg(s, "bagel_burst", "ac")["reps"][1]["flags"]


def test_missing_telemetry_case(result):
    _, s = result
    r = cfg(s, "bagel_burst", "peak_bf")["reps"][1]
    assert sorted(r["flags"]) == ["NO_ADMLOG", "NO_TELEMETRY"]
    m = r["metrics"]
    assert m["completed"] == 6 and m["short_e2e_mean_s"] == pytest.approx(1.25)  # bench numbers survive
    assert "holds" not in m and "min_free_kv_pages" not in m and "gpu_busy_share" not in m
    assert any("telemetry/ missing" in n for n in r["notes"])


def test_missing_bench_case(result):
    _, s = result
    r = cfg(s, "bagel_control", "peak_bf")["reps"][0] if "bagel_control" in s["workloads"] else None
    assert r is not None
    assert {"NO_BENCH", "NO_TELEMETRY", "NO_START", "NO_ADMLOG"} <= set(r["flags"])
    assert "completed" not in r["metrics"]
    assert cfg(s, "bagel_control", "peak_bf")["mean"]["completed"] is None


def test_no_bench_falls_back_to_server_side_completions(root, tmp_path):
    job = tmp_path / "1_ac"
    make_case(str(job), "bagel_burst", 1, config="ac", bench=False)
    case = ma.analyze_case(ma.discover([str(job)])[0][0])
    assert "NO_BENCH" in case.flags and "completed" not in case.m
    assert case.m["completed_server_side"] == 6  # the warmup request's req_done is excluded
    assert case.m["holds"] == 2 and case.m["min_free_kv_pages"] == 900
    assert len(case.series["completions"]) == 6


def test_admission_log_numbers_and_handle_join(result):
    _, s = result
    m = rep(s, "bagel_burst", "peak_bf", 0)
    # rows after the trace start: 6 reserve + 2 wait + 2 grant_deferred + 2 release (the warmup's two are dropped)
    assert m["adm_events"] == {"reserve": 5 + 1, "wait": 2, "grant_deferred": 2, "release": 2}
    assert m["grant_deferred_log"] == 2
    # planner_us over reserve (10..60), wait (100, 200), grant_deferred (5, 15); the warmup's 1.0 is excluded
    us = sorted([10, 20, 30, 40, 50, 60, 100, 200, 5, 15])
    assert m["planner_us_p50"] == pytest.approx(np.percentile(us, 50)) == pytest.approx(35.0)
    assert m["planner_us_p99"] == pytest.approx(191.0)
    assert m["adm_wait_why"] == {"peak": 1, "easy": 1}
    assert m["adm_fit_order"] == ["peak/backfill"]
    # handle 0 is request 0 (short) at 0.06 s and request 5 (long) at 7.58 s: the join is as-of, per (pid, handle)
    assert m["join_admission_rows"] == "12/12"
    assert m["adm_wait_mean_short_s"] == pytest.approx(0.75)  # waited_s 0.0, 0.5, 1.0, 1.5
    assert m["adm_wait_mean_long_s"] == pytest.approx(5.0)  # requests 4 and 5: 4.0, 6.0
    assert m["adm_wait_p99_long_s"] == pytest.approx(np.percentile([4.0, 6.0], 99))
    assert m["adm_reserved_mean_short_s"] == pytest.approx(1.13) and m["adm_reserved_mean_long_s"] == pytest.approx(73.42)
    assert m["adm_max_waiting"] == 2 and m["adm_max_reserved"] == 3


# ------------------------------------------------------------------------------------------------ table + ratios
def test_aggregation_over_reps_and_ratios(result):
    _, s = result
    ac, pb = cfg(s, "bagel_burst", "ac")["mean"], cfg(s, "bagel_burst", "peak_bf")["mean"]
    assert ac["n"] == 2 and pb["n"] == 2
    assert (ac["completed"], ac["timed_out"]) == (5.5, 0.5)
    assert ac["duration_s"] == pytest.approx((82 + 901.5) / 2)
    assert ac["out_tok_s"] == pytest.approx((4040 / 82 + 2040 / 901.5) / 2)
    assert ac["long_e2e_mean_s"] == pytest.approx(65.0) and ac["long_e2e_max_s"] == pytest.approx(70.0)
    assert ac["holds"] == 2 and ac["admit_fail"]["GrantDeferred"] == 1.0 and ac["admit_fail"]["AdmissionDeferred"] == 3.0
    assert ac["gpu_busy_share"] == pytest.approx((41 / 82 + 41 / 901.5) / 2)
    assert ac["flags"] == {"STALLED": 1} and ac["stalled_reps"] == 1
    # peak_bf: telemetry numbers come from the one rep that has them
    assert pb["holds"] == 2 and pb["min_free_kv_pages"] == 900 and pb["grant_deferred_log"] == 2
    assert pb["flags"] == {"NO_TELEMETRY": 1, "NO_ADMLOG": 1}
    r = s["workloads"]["bagel_burst"]["ratios"]["vs_ac"]["peak_bf"]
    assert r["short_e2e_mean_s"] == pytest.approx(1.25 / 2.5)
    assert r["long_e2e_mean_s"] == pytest.approx(35.0 / 65.0)
    assert r["out_tok_s"] == pytest.approx((4040 / 42) / ac["out_tok_s"])
    assert "ac" not in s["workloads"]["bagel_burst"]["ratios"]["vs_ac"]
    assert s["workloads"]["bagel_burst"]["ratios"]["vs_main"] == {}  # no `main` rows: nothing to compare to


def test_compare_md(result):
    out, _ = result
    md = open(os.path.join(out, "compare.md")).read()
    assert "## bagel_burst" in md and "## bagel_control" in md
    sec = md.split("## bagel_burst")[1].split("\n## ")[0].split("**reps:")[0]  # the bagel_burst main table only
    table = [ln for ln in sec.splitlines() if ln.startswith("| ")]
    split = lambda ln: [c.strip() for c in ln.strip("|").split("|")]  # noqa: E731
    head = split(table[0])
    for col in ("completed", "failed", "timed out", "short E2E mean (s)", "short E2E p99 (s)", "long E2E max (s)",
                "holds", "fail: GrantDeferred", "grant_deferred (adm log)", "min free KV pages", "GPU busy",
                "decode bs", "planner_us p50", "planner_us p99"):
        assert col in head, col
    ctl = split([ln for ln in md.split("## bagel_control")[1].splitlines() if ln.startswith("| ")][0])
    assert ctl == head  # every workload's table has the same columns
    rows = {split(r)[0]: split(r) for r in table[1:]}
    ac, pb, ratio = rows["ac"], rows["peak_bf"], rows["*peak_bf / ac*"]
    cell = lambda row, h: row[head.index(h)]  # noqa: E731
    assert cell(ac, "n") == "2" and cell(ac, "flags") == "STALLED 1/2"
    assert cell(ac, "completed") == "5.5" and cell(ac, "timed out") == "0.5"
    assert cell(ac, "short E2E mean (s)") == "2.50" and cell(ac, "long E2E mean (s)") == "65.0"
    assert cell(ac, "min free KV pages") == "900" and cell(ac, "grant_deferred (adm log)") == "-"
    assert cell(ac, "fail: GrantDeferred") == "1"
    assert "NO_TELEMETRY 1/2" in cell(pb, "flags") and "NO_ADMLOG 1/2" in cell(pb, "flags")
    assert cell(pb, "grant_deferred (adm log)") == "2" and cell(pb, "planner_us p50") == "35.0"
    assert cell(ratio, "short E2E mean (s)") == "0.50" and cell(ratio, "long E2E mean (s)") == "0.54"
    assert "## Data problems" in md and "`12714_peak_bf/bagel_control_r1`" in md and "NO_BENCH" in md
    assert "![bagel_burst completions over time](bagel_burst_completions.png)" in md


def test_summary_json_is_strict_json(result):
    out, s = result
    again = json.load(open(os.path.join(out, "summary.json")))  # allow_nan=False on write: NaN became null
    assert again["workloads"]["bagel_burst"]["configs"]["ac"]["mean"]["n"] == 2
    assert again["workloads"]["bagel_control"]["configs"]["peak_bf"]["mean"]["short_e2e_mean_s"] is None


def test_figures_written(result):
    out, _ = result
    for name in ("completions", "free_kv", "short_e2e_cdf", "long_e2e"):
        p = os.path.join(out, f"bagel_burst_{name}.png")
        assert os.path.getsize(p) > 2000, p
        assert open(p, "rb").read(8) == b"\x89PNG\r\n\x1a\n"
    assert not os.path.exists(os.path.join(out, "bagel_control_completions.png"))  # nothing to draw


def test_cli(root, tmp_path):
    out = str(tmp_path / "o")
    ma.main([os.path.join(root, "12712_ac"), "--out", out])
    assert os.path.exists(os.path.join(out, "compare.md")) and os.path.exists(os.path.join(out, "summary.json"))


# ------------------------------------------------------------------------------------------------ one server, several runs
SHIFT = 1000.0  # case 2 runs this long after case 1 on the same server, on the monotonic and the wall clock alike
SHARED_CFG = "peak_bf"


def _shift(rows, dt):
    return [{**r, **{k: r[k] + dt for k in ("tm", "ts", "ts_mono", "ts_wall") if k in r}} for r in rows]


def _stamp(rel, fmt, hours=0):  # node-local (UTC) text of the instant `rel` s after case 1's trace start
    return datetime.fromtimestamp(1.78e9 + rel + hours * 3600, timezone.utc).strftime(fmt)[:-3]  # %f: 6 digits, keep ms


def _write_bench(d, run_id, dt, t0_wall=True):
    b = make_bench(run_id)
    for k in ("start_times", "intended_start_times"):
        b[k] = [t + dt for t in b[k]]
    b["t0_monotonic"] += dt
    b["t0_wall"] = b["t0_wall"] + dt if t0_wall else None
    json.dump(b, open(os.path.join(d, "bench.json"), "w"))
    open(os.path.join(d, "client.log"), "w").write("trace replay: ok\nWrote bench.json\n")


def make_shared_job(job, *, t0_wall=True, log_hours=0):
    """server_1: case 1 (bagel_burst r1, t0 = T0), case 2 (r2, T0 + SHIFT) and bagel_control r1 (client died, no bench.json);
    server_2: bagel_burst r3 alone. Case 2 differs from case 1 in every number below, so a leak moves one of them."""
    os.makedirs(job)
    open(os.path.join(job, "meta.txt"), "w").write(f"job=12800 config={SHARED_CFG}\n")
    s1, s2 = (os.path.join(job, f"server_{k}") for k in (1, 2))
    os.makedirs(os.path.join(s1, "telemetry"))
    os.makedirs(s2)
    open(os.path.join(s1, "startup_s.txt"), "w").write("71.3\n")
    open(os.path.join(s2, "startup_s.txt"), "w").write("80.5\n")
    r1, r2 = (f"{SHARED_CFG}-bagel_burst-r{i}" for i in (1, 2))
    meta, steps1, ev1 = telemetry_rows(r1)
    _, steps2, ev2 = telemetry_rows(r2)
    # case 1: a hold inside the drain window after its last request (t_end = 82 s) and one just outside it
    ev1 += [event("hold", 85.0, n=1, h=[0], rid=[f"{r1}-000005"], backoff_s=0.01),
            event("hold", 88.0, n=1, h=[0], rid=[f"{r1}-000005"], backoff_s=0.01)]
    for st in steps2:  # case 2 never gets below 1400 free pages (case 1: 900)
        if st["free"]:
            st["free"] = {"kv": st["free"]["kv"] + 500}
    ev2 += [event("hold", 3.6, n=1, h=[3], rid=[f"{r2}-000003"], backoff_s=0.01)] + [
        event("admit_fail", 3.9, node="LLM", walk="decode", bs=4, reason="AllocationFailed", h=None, rid=None,
              label="main", pages_short=1, resource="kv")] * 2
    drop = {"kind": "drop", "n": 3, "ts": 1.78e9 + 10, "tm": T0 + 10}  # the writer queue refused 3 rows (shifted with case 2)
    end = {"kind": "end", "rows": 0, "dropped": 3, "ts": 1.78e9, "tm": T0 + SHIFT + 90}
    write_jsonl(os.path.join(s1, "telemetry", f"steps_rank0_pid{PID}.jsonl"),
                [*meta, *steps1, *ev1, *_shift([*steps2, *ev2, drop], SHIFT), end])
    adm2 = admission_rows() + [adm_row("grant_deferred", 12.0, 4, label="main", pages=1, planner_us=7.0)]
    for r in adm2:
        if "planner_us" in r:
            r["planner_us"] *= 10
    write_jsonl(os.path.join(s1, "telemetry", f"admission_kv_pid{PID}.jsonl"), [*admission_rows(), *_shift(adm2, SHIFT)])
    logts = lambda rel: _stamp(rel, ma.LOG_TS_FMT, log_hours)  # noqa: E731
    err, tb = "ERROR [worker_0] mstar.engine: step failed", "Traceback (most recent call last):"
    open(os.path.join(s1, "server.log"), "w").write("\n".join([
        "INFO:     Started server process [1]", f"{logts(-100)} INFO [api_server] mstar: startup", tb,  # before case 1
        f"{logts(10)} {err}", tb,  # case 1
        f"{logts(500)} {err}", tb,  # between the runs
        f"{logts(SHIFT + 20)} ERROR [worker_0] mstar.engine: CUDA out of memory. Tried to allocate 2 GiB",  # case 2
        f"{logts(SHIFT + 30)} {err}", tb,  # case 2
        f"{logts(SHIFT + 95)} INFO [api_server] mstar: shutdown", ""]))
    open(os.path.join(s2, "server.log"), "w").write(f"{logts(3 * SHIFT)} {err}\n{tb}\n")  # server_2: whole file counts
    gpu = [(-100, 99999, 0), (1, 70000, 10), (40, 71000, 90), (84, 71500, 50), (500, 90000, 5),  # case 1: 1..84 s
           (SHIFT + 1, 72000, 20), (SHIFT + 40, 75000, 60)]
    open(os.path.join(job, "gpu_mem.csv"), "w").write(
        "".join(f"{_stamp(rel, ma.GPU_TS_FMT)}, 0, {mem}, 81559, {util}\n" for rel, mem, util in gpu))
    r3 = f"{SHARED_CFG}-bagel_burst-r3"
    for name, run_id, srv, dt in (("bagel_burst_r1", r1, s1, 0.0), ("bagel_burst_r2", r2, s1, SHIFT),
                                  ("bagel_control_r1", None, s1, 0.0), ("bagel_burst_r3", r3, s2, 3 * SHIFT)):
        d = os.path.join(job, name)
        os.makedirs(d)
        if run_id:
            _write_bench(d, run_id, dt, t0_wall)
        for link in ("telemetry", "server.log"):
            if os.path.exists(os.path.join(srv, link)):
                os.symlink(os.path.join(srv, link), os.path.join(d, link))  # absolute, as the sbatch's ln -s


def analyze(job):
    return {f"{r.workload}_r{r.rep}": ma.analyze_case(r) for r in ma.discover([job])[0]}


def analyze_shared(tmp_path, name, **kw):
    job = str(tmp_path / f"{name}_{SHARED_CFG}")
    make_shared_job(job, **kw)
    return analyze(job)


@pytest.fixture(scope="module")
def shared(tmp_path_factory):
    return analyze_shared(tmp_path_factory.mktemp("shared"), 12800)


def test_shared_telemetry_is_windowed_per_case(shared):
    m1, m2 = shared["bagel_burst_r1"].m, shared["bagel_burst_r2"].m
    # case 1: the single-run numbers plus the hold in the drain window (the one at 88 s is out); case 2's rows are not in
    assert (m1["holds"], m1["held_requests"], m1["offloads"], m1["admit_fail_total"]) == (3, 4, 1, 5)
    assert m1["admit_fail"] == {"AdmissionDeferred": 3, "GrantDeferred": 1, "AllocationFailed": 1}
    assert m1["min_free_kv_pages"] == 900 and m1["n_steps"] == 6 and m1["gpu_busy_share"] == pytest.approx(41.0 / 82.0)
    assert m1["join_new_request"] == "6/6" and m1["holds_by_class"] == {"long": 2, "short": 2}  # none "unmatched"
    assert m1["admit_fail_by_class"] == {"short": 1, "long": 3}
    assert m1["adm_events"] == {"reserve": 6, "wait": 2, "grant_deferred": 2, "release": 2} and m1["grant_deferred_log"] == 2
    assert (m1["planner_us_p50"], m1["planner_us_p99"]) == pytest.approx((35.0, 191.0))
    assert m1["join_admission_rows"] == "12/12" and m1["adm_max_waiting"] == 2
    # case 2: its own extra hold / admit_fail / grant_deferred, its own free pages and planner times (x10), none of case 1's
    assert (m2["holds"], m2["held_requests"], m2["offloads"], m2["admit_fail_total"]) == (3, 4, 1, 7)
    assert m2["admit_fail"] == {"AdmissionDeferred": 3, "GrantDeferred": 1, "AllocationFailed": 3}
    assert m2["min_free_kv_pages"] == 1400 and m2["n_steps"] == 6 and m2["gpu_busy_share"] == pytest.approx(41.0 / 82.0)
    assert m2["join_new_request"] == "6/6" and m2["holds_by_class"] == {"long": 1, "short": 3}
    assert m2["adm_events"] == {"reserve": 6, "wait": 2, "grant_deferred": 3, "release": 2} and m2["grant_deferred_log"] == 3
    us = 10 * np.array([10, 20, 30, 40, 50, 60, 100, 200, 5, 15, 7.0])
    assert (m2["planner_us_p50"], m2["planner_us_p99"]) == pytest.approx((np.percentile(us, 50), np.percentile(us, 99)))
    assert m2["join_admission_rows"] == "13/13"
    # the figure series follow: seconds from each case's own trace start, its own minimum
    for name, low in (("bagel_burst_r1", 900), ("bagel_burst_r2", 1400)):
        ser = shared[name].series
        assert ser["free_t"].min() >= 0 and ser["free_t"].max() <= 90 and ser["free_v"].min() == low
    assert shared["bagel_burst_r1"].flags == set()
    # dropped rows: case 2's window only
    assert m1["telemetry_dropped"] == 0 and m2["telemetry_dropped"] == 3
    assert "TEL_DROPS" not in shared["bagel_burst_r1"].flags and shared["bagel_burst_r2"].flags == {"TEL_DROPS"}


def test_shared_server_resolves_startup_and_index_through_symlinks(shared, result):
    for name in ("bagel_burst_r1", "bagel_burst_r2", "bagel_control_r1"):
        m = shared[name].m
        assert (m["server_index"], m["server_shared_by"]) == (1, 3), name
        assert "NO_START" not in shared[name].flags
    assert shared["bagel_burst_r1"].m["startup_s"] == shared["bagel_burst_r2"].m["startup_s"] == 71.3
    r3 = shared["bagel_burst_r3"].m
    assert (r3["server_index"], r3["server_shared_by"], r3["startup_s"]) == (2, 1, 80.5)
    ctl = shared["bagel_control_r1"]  # no bench.json, but its server started: only NO_BENCH, and no window for the telemetry
    assert ctl.flags == {"NO_BENCH"} and "holds" not in ctl.m and "completed_server_side" not in ctl.m
    assert any("telemetry window is unknown" in n for n in ctl.notes)
    assert ctl.m["startup_s"] == 71.3
    assert "server_index" not in rep(result[1], "bagel_burst", "ac", 0)  # the older layout has none


def test_shared_server_log_is_windowed_per_case(shared):
    c1, c2, c3, ctl = (shared[n] for n in ("bagel_burst_r1", "bagel_burst_r2", "bagel_burst_r3", "bagel_control_r1"))
    assert c1.m["server_error_lines"] == 1 and c2.m["server_error_lines"] == 2  # the Traceback bodies take their line's time
    assert any("in the run window" in n for n in c1.notes) and not any("per server" in n for n in c1.notes + c2.notes)
    assert c3.m["server_error_lines"] == 1 and not any("server.log" in n and "window" in n for n in c3.notes)  # alone: whole file
    assert ctl.m["server_error_lines"] == 5  # no window for it: the whole shared file, said so, and no flag from it
    assert "server.log shared by 3 cases; log counts are per server" in ctl.notes


def test_job_gpu_mem_is_windowed_per_case(shared):
    c1, c2, ctl = (shared[n] for n in ("bagel_burst_r1", "bagel_burst_r2", "bagel_control_r1"))
    # before-the-run (99999) and between-the-runs (90000) samples are in neither window; 84 s is inside the drain
    assert (c1.m["gpu_mem_peak_mib"], c1.m["gpu_util_mean_pct"]) == (71500, 50.0)
    assert (c2.m["gpu_mem_peak_mib"], c2.m["gpu_util_mean_pct"]) == (75000, 40.0)
    assert "gpu_mem_peak_mib" not in ctl.m and any("t0_wall" in n for n in ctl.notes)


def test_shared_server_without_a_trustworthy_wall_clock(tmp_path):
    no_anchor = analyze_shared(tmp_path, 1, t0_wall=False)["bagel_burst_r1"]
    assert no_anchor.m["holds"] == 3 and no_anchor.m["min_free_kv_pages"] == 900  # telemetry uses the monotonic clock
    assert no_anchor.m["server_error_lines"] == 5 and "gpu_mem_peak_mib" not in no_anchor.m
    assert "server.log shared by 3 cases; log counts are per server" in no_anchor.notes
    assert any("t0_wall" in n for n in no_anchor.notes) and no_anchor.flags == set()
    # a log whose clock is 5 h off does not reach any run window: skipped rather than counted
    off = analyze_shared(tmp_path, 2, log_hours=5)
    for name in ("bagel_burst_r1", "bagel_burst_r2"):
        assert "server_error_lines" not in off[name].m and any("do not reach the run window" in n for n in off[name].notes)


# ------------------------------------------------------------------------------------------------ units
def test_truncated_outputs_are_flagged(result, tmp_path):
    _, s = result
    healthy = rep(s, "bagel_burst", "ac", 0)
    assert healthy["truncated"] == 0 and "TRUNCATED" not in cfg(s, "bagel_burst", "ac")["reps"][0]["flags"]
    # the engine cuts both longs (asked 2000) to 1500 tokens; request 5 then times out, so only request 4 is truncated
    job = tmp_path / "5_ac"
    both = make_case(str(job), "bagel_burst", 1, config="ac", telemetry=False, cap_out=1500)
    one = make_case(str(job), "bagel_burst", 2, config="ac", telemetry=False, cap_out=1500, timeout_idx=5)
    refs, _ = ma.discover([str(job)])
    c1, c2 = (ma.analyze_case(r) for r in sorted(refs, key=lambda r: r.rep))
    assert c1.m["truncated"] == 2 and "TRUNCATED" in c1.flags
    assert any("TRUNCATED: 2 of 6" in n and "generated 1500-1500 vs requested 2000-2000" in n for n in c1.notes)
    assert c2.m["truncated"] == 1 and "TRUNCATED" in c2.flags  # a failed request is not a completed one
    # bench.json without intended_output_lens (older client): unknown, not zero, and no flag
    b = json.load(open(os.path.join(both, "bench.json")))
    del b["intended_output_lens"]
    json.dump(b, open(os.path.join(one, "bench.json"), "w"))
    c3 = ma.analyze_case([r for r in refs if r.rep == 2][0])
    assert c3.m["truncated"] != c3.m["truncated"] and "TRUNCATED" not in c3.flags
    # compare.md: the flag in the table, the legend, the note
    out = str(tmp_path / "out")
    ma.run([str(job / "bagel_burst_r1")], out)
    md = open(os.path.join(out, "compare.md")).read()
    assert "TRUNCATED 1/1" in md and "- **TRUNCATED**:" in md and "intended_output_lens" in md
    assert "TRUNCATED: 2 of 6" in md.split("## Data problems")[1]
    assert json.load(open(os.path.join(out, "summary.json")))["workloads"]["bagel_burst"]["configs"]["ac"]["mean"]["truncated"] == 2


def test_max_stall_units():
    # two requests at 0; one completes at 10, the other is outstanding to 100
    assert ma.max_stall([0, 0], [10], [], 100) == (90.0, 10.0)
    # nothing outstanding between 20 and 50: not a stall
    assert ma.max_stall([0, 50], [20, 60], [], 100)[0] == 20.0
    # a failure ends the request but is not progress: 0..30 has no completion
    assert ma.max_stall([0], [], [30], 100)[0] == 30.0
    assert ma.max_stall([], [], [], 100) == (0.0, 0.0)


def test_class_fallbacks():
    b = make_bench("x")
    del b["cls"]
    assert list(ma.request_table(b).cls) == ["short"] * 4 + ["long"] * 2  # intended output length >= 1024
    del b["intended_output_lens"]
    b["output_lens"][1] = 0
    b["errors"][1] = "no output received"
    rt = ma.request_table(b)
    assert list(rt.cls) == ["short", "unknown", "short", "short", "long", "long"]
    assert not rt.ok[1] and not rt.timed_out[1]  # "no output received" is a failure, not a timeout


def test_timeout_classification():
    b = make_bench("x")
    b["errors"] = ["", "Timeout on reading data", "request failed", "no output received", "Server disconnected", ""]
    b["output_lens"] = [10, 0, 0, 0, 0, 2000]
    rt = ma.request_table(b)
    assert list(rt.timed_out) == [False, True, True, False, False, False]
    m = ma.bench_metrics(rt, b, 60.0)
    assert (m["completed"], m["timed_out"], m["failed"]) == (2, 2, 2)


def test_garbage_bench_does_not_crash(tmp_path):
    job = tmp_path / "7_main"
    d = make_case(str(job), "bagel_burst", 1, config="main", bench=False, telemetry=False)
    open(os.path.join(d, "bench.json"), "w").write('{"duration": 1.0, "input_le')  # client killed mid-write
    case = ma.analyze_case(ma.discover([str(job)])[0][0])
    assert "BAD_BENCH" in case.flags and "NO_ADMLOG" not in case.flags  # `main` has no admission log by design


def test_telemetry_without_steps_or_events_does_not_crash(tmp_path):
    job = tmp_path / "9_ac"
    d = make_case(str(job), "bagel_burst", 1, config="ac", telemetry=False)
    os.makedirs(os.path.join(d, "telemetry"))
    write_jsonl(os.path.join(d, "telemetry", "steps_rank0_pid1.jsonl"), [  # server died before its first step
        {"kind": "meta", "phase": "process", "schema": 1, "rank": 0, "pid": 1, "detail": True, "ts": 1.0, "tm": T0 - 1, "env": {}},
        {"kind": "end", "rows": 1, "dropped": 0, "ts": 1.0, "tm": T0}])
    case = ma.analyze_case(ma.discover([str(job)])[0][0])
    assert case.m["completed"] == 6 and case.m["holds"] == 0 and case.m["n_steps"] == 0
    assert case.m["min_free_kv_pages"] != case.m["min_free_kv_pages"]  # NaN: no step rows
    assert case.m["join_new_request"] == "0/6" and any("telemetry has no step rows" in n for n in case.notes)
    assert any("new_request" in n for n in case.notes)  # the failed id join is reported


def test_streaming_reader_skips_garbage(tmp_path):
    p = tmp_path / "x.jsonl"
    p.write_text('{"a":1}\n\nnot json\n{"a":2}\n{"a":')
    bad = [0]
    assert list(ma.iter_jsonl(str(p), bad)) == [{"a": 1}, {"a": 2}] and bad[0] == 2


def test_an_aborted_stream_counts_as_a_server_timeout():
    # the server's 600 s timeout aborts the chunked response; aiohttp reports an incomplete payload
    err = "Response payload is not completed: <TransferEncodingError: 400, message='Not enough data to satisfy transfer length header.'>"
    assert ma.TIMEOUT_RE.search(err)
    assert not ma.TIMEOUT_RE.search("[Errno 104] Connection reset by peer")


# ------------------------------------------------------------------------------------------------ reps, warmup, exclude, host time
def _job(root, name, label=None):
    j = os.path.join(root, name)
    os.makedirs(j)
    open(os.path.join(j, "meta.txt"), "w").write(f"job={name.split('_')[0]} config={label or name.split('_', 1)[1]}\n")
    return j


def test_reps_pool_across_jobs_and_requeued_dirs(tmp_path):
    root = str(tmp_path)
    for name in ("30_ac", "31_ac", "32_ac_rq1"):  # three jobs of one label; the last is a requeued attempt (dir suffix _rq1)
        make_case(_job(root, name), "bagel_burst", 1, config="ac", telemetry=False)
    refs, _ = ma.discover([root])
    assert {r.config for r in refs} == {"ac"} and len(refs) == 3
    assert sorted(r.name for r in refs) == ["30_ac/bagel_burst_r1", "31_ac/bagel_burst_r1", "32_ac_rq1/bagel_burst_r1"]
    s = ma.run([root], str(tmp_path / "out"))
    assert list(s["workloads"]["bagel_burst"]["configs"]) == ["ac"]
    assert s["workloads"]["bagel_burst"]["configs"]["ac"]["mean"]["n"] == 3


def test_warmup_dirs_are_never_cases(tmp_path):
    root = str(tmp_path)
    job = _job(root, "33_ac")
    make_case(job, "bagel_stag_v2_r3", 1, config="ac", telemetry=False)
    # the sbatch's warmup replay: warmup_<workload>[_s<k>]; for this workload the name parses like a case (workload
    # "warmup_bagel_stag_v2", rep 3), so it has to be skipped by name
    make_case(job, "warmup_bagel_stag_v2", 3, config="ac", telemetry=False)
    os.makedirs(os.path.join(job, "warmup_bagel_burst_v2_s2"))
    assert ma.case_match("warmup_bagel_stag_v2_r3") is None and ma.case_match("bagel_stag_v2_r3_r1")
    for target in (root, job):
        refs, probs = ma.discover([target])
        assert [(r.workload, r.rep) for r in refs] == [("bagel_stag_v2_r3", 1)] and not probs, target
    assert ma.discover([os.path.join(job, "warmup_bagel_stag_v2_r3")])[0] == []  # not even when named directly
    assert list(ma.run([root], str(tmp_path / "out"))["workloads"]) == ["bagel_stag_v2_r3"]
    # a warmup dir that links to the server's telemetry does not count as a case sharing that server
    os.makedirs(os.path.join(job, "server_1", "telemetry"))
    for name in ("bagel_stag_v2_r3_r1", "warmup_bagel_stag_v2_r3"):
        os.makedirs(os.path.join(job, name), exist_ok=True)
        os.symlink(os.path.join(job, "server_1", "telemetry"), os.path.join(job, name, "telemetry"))
    assert ma.server_of(os.path.join(job, "bagel_stag_v2_r3_r1")) == (1, 1)


def test_exclude_drops_cases_by_name(tmp_path):
    root = str(tmp_path)
    for job, label in (("34_ac", "ac"), ("35_peak_bf", "peak_bf")):
        j = _job(root, job)
        for r in (1, 2):
            make_case(j, "bagel_burst", r, config=label, telemetry=False)
    out = str(tmp_path / "o1")
    s = ma.run([root], out, exclude=["*/bagel_burst_r1"])
    assert sorted(s["excluded"]) == ["34_ac/bagel_burst_r1", "35_peak_bf/bagel_burst_r1"]
    ac = s["workloads"]["bagel_burst"]["configs"]["ac"]
    assert [r["rep"] for r in ac["reps"]] == [2] and ac["mean"]["n"] == 1
    md = open(os.path.join(out, "compare.md")).read()
    assert "Excluded by `--exclude`: `34_ac/bagel_burst_r1`, `35_peak_bf/bagel_burst_r1`." in md
    s = ma.run([root], str(tmp_path / "o2"), exclude=["34_*", "nothing_matches"])  # a whole job by pattern; a miss is fine
    assert s["excluded"] == ["34_ac/bagel_burst_r1", "34_ac/bagel_burst_r2"] and list(s["workloads"]["bagel_burst"]["configs"]) == ["peak_bf"]
    ma.main([root, "--out", str(tmp_path / "o3"), "--exclude", "34_*", "--exclude", "*_r2"])  # the flag repeats
    s3 = json.load(open(tmp_path / "o3" / "summary.json"))
    assert s3["excluded"] == ["34_ac/bagel_burst_r1", "34_ac/bagel_burst_r2", "35_peak_bf/bagel_burst_r2"]
    assert s3["workloads"]["bagel_burst"]["configs"]["peak_bf"]["mean"]["n"] == 1
    assert ma.run([root], str(tmp_path / "o4"))["excluded"] == []


def _spread_fixture(root):
    # short E2E mean = 2.5 x scale, long E2E mean = 70 x scale, out tok/s = 4040 / (82 x scale): the scale is the noise knob
    for job, label, scale in (("40_ac", "ac", 1.0), ("41_ac", "ac", 1.2),                       # ac: short E2E [2.5, 3.0]
                              ("42_peak_bf", "peak_bf", 0.5), ("43_peak_bf", "peak_bf", 0.55),  # [1.25, 1.375]: apart from ac
                              ("44_noisy", "noisy", 1.1), ("45_noisy", "noisy", 1.3),           # [2.75, 3.25]: overlaps ac
                              ("46_solo", "solo", 0.2)):                                        # one rep, apart from ac
        make_case(_job(root, job), "bagel_burst", 1, config=label, scale=scale, telemetry=False)


def test_spread_and_noise_rule(tmp_path):
    root = str(tmp_path / "r")
    os.makedirs(root)
    _spread_fixture(root)
    out = str(tmp_path / "out")
    s = ma.run([root], out)
    cfgs = s["workloads"]["bagel_burst"]["configs"]
    ac = cfgs["ac"]["mean"]
    assert ac["spread"]["short_e2e_mean_s"] == dict(min=2.5, max=3.0, n=2) and ac["short_e2e_mean_s"] == pytest.approx(2.75)
    assert cfgs["solo"]["mean"]["spread"]["out_tok_s"]["n"] == 1
    rt = s["workloads"]["bagel_burst"]["ratios"]["vs_ac"]
    for k in ("short_e2e_mean_s", "long_e2e_mean_s", "out_tok_s"):
        assert rt["noisy"]["within_noise"][k] is True and rt["peak_bf"]["within_noise"][k] is False, k
        assert rt["solo"]["within_noise"][k] is False and rt["solo"]["single_rep"][k] is True
        assert rt["peak_bf"]["single_rep"][k] is False
    assert rt["peak_bf"]["short_e2e_mean_s"] == pytest.approx((2.5 * 0.525) / 2.75)
    # touching ranges overlap (the rule is <=), a gap does not, a missing metric says nothing
    assert ma.ranges_overlap(dict(min=1, max=2), dict(min=2, max=3)) and not ma.ranges_overlap(dict(min=1, max=2), dict(min=2.01, max=3))
    assert ma.ranges_overlap(dict(min=math.nan, max=math.nan), dict(min=1, max=2)) is None
    full = open(os.path.join(out, "compare.md")).read()
    assert "min–max" in full.split("## bagel_burst")[0] and "within noise" in full.split("## bagel_burst")[0]  # the header explains the rule
    main_tbl, spread_tbl = full.split("## bagel_burst")[1].split("**reps: mean [min–max]**")
    rows = lambda txt: [[c.strip() for c in ln.strip("|").split("|")] for ln in txt.splitlines() if ln.startswith("| ")]  # noqa: E731
    mh, mrows = rows(main_tbl)[0], {r[0]: r for r in rows(main_tbl)[1:]}
    cell = lambda row, h: row[mh.index(h)]  # noqa: E731
    assert cell(mrows["*noisy / ac*"], "short E2E mean (s)").endswith("≈ (within noise)")
    assert cell(mrows["*noisy / ac*"], "out tok/s").endswith("≈ (within noise)")
    assert cell(mrows["*peak_bf / ac*"], "short E2E mean (s)") == "0.48" and cell(mrows["*peak_bf / ac*"], "long E2E mean (s)") == "0.48"
    assert cell(mrows["*solo / ac*"], "short E2E mean (s)") == "0.18 (1 rep)"
    # spread table: mean [min–max], the lone rep without a bracket
    sh, srows = rows(spread_tbl)[0], {r[0]: r for r in rows(spread_tbl)[1:]}
    sc = lambda cfg, h: srows[cfg][sh.index(h)]  # noqa: E731
    assert sc("ac", "n") == "2" and sc("ac", "short E2E mean (s)") == "2.75 [2.50–3.00]"
    assert sc("ac", "out tok/s") == "45.2 [41.2–49.3]" and sc("ac", "long E2E mean (s)") == "77.0 [70.0–84.0]"
    assert sc("solo", "n") == "1" and sc("solo", "short E2E mean (s)") == "0.50"
    assert sc("ac", "host ms/step all") == "-"  # no telemetry in this fixture
    for head in ("short E2E p50 (s)", "short E2E p99 (s)", "short TTFT p99 (s)", "host ms/step [0,50)", "host ms/step [400,inf)"):
        assert head in sh, head


# Host ms per decode step by waiting depth. A phase is (waiting depth, decode steps, gap to the next step in ms); gpu_ms = 2,
# so a phase's host ms is its gap - 2. Depths must not increase from one phase to the next (a queue only drains here).
GPU_MS = 2.0


def host_telemetry(path, phases, *, rids=True, tail=()):
    """steps_rank0 file: 400 requests arrive at t = 0.5 s (handles 1000-1399), then each phase pops handles from the queue
    (its first step carries them in its rids) down to its depth and runs its decode steps. `tail` = (walk, gpu_ms, ok) rows
    after the last phase, 5 ms apart. 40 steps 20 s before the trace start have a host time of 98 ms."""
    queue = list(range(1000, 1400))
    rows = [{"kind": "meta", "phase": "process", "schema": 1, "rank": 0, "pid": PID, "detail": True, "ts": 1.78e9, "tm": T0 - 60, "env": {}}]
    rows += [event("new_request", 0.5, rid=f"x-{h}", h=h, partition="p0", walk="prefill_text") for h in queue]
    seq, t = 0, 1.0
    for i in range(40):
        rows.append(step_row(seq := seq + 1, -20.0 + 0.1 * i, "decode", 1, GPU_MS, {"kv": 1000}))
    rows.sort(key=lambda r: r["tm"])
    if not rids:
        for r in rows:
            r.pop("rids", None)

    def add(walk, gpu_ms, ok, pop=()):
        nonlocal seq
        r = step_row(seq := seq + 1, t, walk, 1, gpu_ms, {"kv": 1000}, ok=ok)
        if rids:
            r["rids"] = [0, *pop]
        else:
            del r["rids"]
        rows.append(r)

    for depth, count, gap_ms in phases:
        pop, queue = queue[:len(queue) - depth], queue[len(queue) - depth:]
        for i in range(count):
            add("decode", GPU_MS, True, pop if i == 0 else ())
            t += gap_ms / 1e3
    for walk, gpu_ms, ok in tail:
        add(walk, gpu_ms, ok)
        t += 0.005
    rows.append({"kind": "end", "rows": len(rows), "dropped": 0, "ts": 1.78e9, "tm": T0 + 90})
    write_jsonl(path, rows)


def host_case(tmp_path, name, phases, **kw):
    job = str(tmp_path / name)
    d = make_case(job, "bagel_burst", 1, config="ac", telemetry=False)
    os.makedirs(os.path.join(d, "telemetry"))
    host_telemetry(os.path.join(d, "telemetry", f"steps_rank0_pid{PID}.jsonl"), phases, **kw)
    return ma.analyze_case(ma.discover([job])[0][0])


FULL_PHASES = [(400, 31, 14.0), (399, 31, 11.0), (200, 31, 11.0), (199, 31, 8.0), (50, 31, 8.0), (49, 31, 5.0), (0, 31, 5.0)]


def test_host_time_by_waiting_depth(tmp_path):
    # depths 400 / 399 / 200 / 199 / 50 / 49 / 0 straddle the bin edges 400, 200, 50 (host ms 12 / 9 / 9 / 6 / 6 / 3 / 3)
    # Not counted: the last step of the first phase (its successor is 0.8 s away: host ~798 ms >= 500), a prefill step, a
    # refused decode step (no gpu_ms), a decode step whose gpu_ms (9) exceeds the 5 ms to its successor (host < 0) and the
    # last row (no successor)
    phases = [(400, 30, 14.0), (400, 1, 800.0)] + FULL_PHASES[1:]
    tail = [("prefill_text", 5.0, True), ("decode", None, False), ("decode", 9.0, True), ("decode", GPU_MS, True)]
    c = host_case(tmp_path, "50_ac", phases, tail=tail)
    m = c.m
    assert (m["host_n_w400p"], m["host_n_w200_400"], m["host_n_w50_200"], m["host_n_w0_50"]) == (30, 62, 62, 62)
    assert m["host_ms_w400p"] == pytest.approx(12.0)    # 30 steps: the fewest that still report a median
    assert m["host_ms_w200_400"] == pytest.approx(9.0)  # depths 399 and 200: 200 belongs to [200, 400)
    assert m["host_ms_w50_200"] == pytest.approx(6.0)   # depths 199 and 50: 50 belongs to [50, 200)
    assert m["host_ms_w0_50"] == pytest.approx(3.0)     # depths 49 and 0
    # 62 x 3, 62 x 6, 62 x 9 and 30 x 12 ms: the median of the 216 is their 108th value, 6
    assert m["host_steps"] == 216 and m["host_ms_p50"] == pytest.approx(6.0)
    assert m["n_steps"] == 7 * 31 + 4  # the 40 steps before the trace start are outside the window, host time or not
    # fewer than 30 steps in a bin: no median (the count stays); the overall median still has them
    c2 = host_case(tmp_path, "51_ac", [(60, 29, 11.0), (0, 40, 5.0)])
    assert c2.m["host_n_w50_200"] == 29 and math.isnan(c2.m["host_ms_w50_200"])
    assert c2.m["host_ms_w0_50"] == pytest.approx(3.0) and c2.m["host_steps"] == 68  # the last step has no successor
    assert c2.m["host_ms_p50"] == pytest.approx(np.median([9.0] * 29 + [3.0] * 39))
    # telemetry without per-step rids gives no queue depth: the overall median stays, the bins are empty
    c3 = host_case(tmp_path, "52_ac", [(0, 40, 5.0)], rids=False)
    assert c3.m["host_ms_p50"] == pytest.approx(3.0) and c3.m["host_n_w0_50"] == 0
    assert all(math.isnan(c3.m[k]) for k in ("host_ms_w0_50", "host_ms_w50_200", "host_ms_w200_400", "host_ms_w400p"))
    # no decode step in the window: NaN, not an error
    c4 = host_case(tmp_path, "53_ac", [])
    assert c4.m["host_steps"] == 0 and math.isnan(c4.m["host_ms_p50"])


def test_host_time_in_the_tables(tmp_path):
    root = str(tmp_path / "r")
    os.makedirs(root)
    for job, phases in (("60_hostcfg", FULL_PHASES), ("61_hostcfg", [(0, 40, 5.0)])):  # the second rep never has a queue
        d = make_case(_job(root, job), "bagel_burst", 1, config="hostcfg", telemetry=False)
        os.makedirs(os.path.join(d, "telemetry"))
        host_telemetry(os.path.join(d, "telemetry", f"steps_rank0_pid{PID}.jsonl"), phases)
    out = str(tmp_path / "out")
    mean = ma.run([root], out)["workloads"]["bagel_burst"]["configs"]["hostcfg"]["mean"]
    # rep 1: 7 x 31 steps, the last has no successor -> median over 216 is 6; rep 2: 40 steps -> 39 of 3 ms
    assert mean["host_ms_p50"] == pytest.approx(4.5) and mean["spread"]["host_ms_p50"] == pytest.approx(dict(min=3.0, max=6.0, n=2))
    assert mean["spread"]["host_ms_w400p"]["n"] == 1 and mean["spread"]["host_ms_w0_50"]["n"] == 2
    md = open(os.path.join(out, "compare.md")).read().split("**reps: mean [min–max]**")[1]
    rows = [[c.strip() for c in ln.strip("|").split("|")] for ln in md.splitlines() if ln.startswith("| ")]
    head, row = rows[0], rows[1]
    assert row[head.index("host ms/step [400,inf)")] == "12.00 (n=1)"  # only one rep has the bin: shown alone, with its n
    assert row[head.index("host ms/step [0,50)")] == "3.00 [3.00–3.00]"
    assert row[head.index("host ms/step all")] == "4.50 [3.00–6.00]"


def test_kv_free_lag(tmp_path):
    # a: loop_stop at 1.0 and again at 2.0 (a later walk), pages back at 2.010 by remove_done: 10 ms from the last stop
    # b: loop_stop 3.0, kv_release 3.002, remove_done 3.5: the release frees first, 2 ms
    # c: loop_stop 4.0, its pages back only after it (a remove_done at 3.9 belongs to an earlier life): no stop before -> left out
    # d: loop_stop with no free in the window: left out. The client's warmup request is left out.
    ev = pd.DataFrame([
        dict(ev="loop_stop", tm=1.0, rid=["a"]), dict(ev="loop_stop", tm=2.0, rid=["a", "warmup0"]),
        dict(ev="remove_done", tm=2.010, rid="a"), dict(ev="remove_done", tm=2.5, rid="warmup0"),
        dict(ev="loop_stop", tm=3.0, rid=["b"]), dict(ev="kv_release", tm=3.002, rid="b"), dict(ev="remove_done", tm=3.5, rid="b"),
        dict(ev="remove_done", tm=3.9, rid="c"), dict(ev="loop_stop", tm=4.0, rid=["c", "d"]),
        dict(ev="new_request", tm=0.5, rid="a"),
    ])
    m = ma.kv_free_lag_metrics(ev)
    assert m["kv_release_n"] == 1 and m["kv_free_lag_n"] == 2
    assert m["kv_free_lag_ms_max"] == pytest.approx(10.0)
    assert m["kv_free_lag_ms_p50"] == pytest.approx(6.0) and m["kv_free_lag_ms_p99"] == pytest.approx(2 + 8 * 0.99)
    # no events, or events without rids: zeros and NaN, not an error
    empty = ma.kv_free_lag_metrics(pd.DataFrame(columns=["ev", "tm", "pid"]))
    assert empty["kv_free_lag_n"] == 0 and math.isnan(empty["kv_free_lag_ms_p50"])


def test_step_gains_table():
    import mstar_step_gains as sg

    def cfg(vals, n=3):  # vals: {metric: (mean, min, max)}
        mean = {k: v[0] for k, v in vals.items()}
        mean["spread"] = {k: dict(min=v[1], max=v[2], n=n) for k, v in vals.items()}
        return dict(mean=mean, reps=[{}] * n)

    s = {"workloads": {"bagel_burst_v2": {"configs": {
        "old": cfg({"short_e2e_mean_s": (2.0, 1.8, 2.2), "out_tok_s": (100.0, 90.0, 110.0)}),
        "new": cfg({"short_e2e_mean_s": (1.0, 0.9, 1.1), "out_tok_s": (105.0, 100.0, 108.0)}),
        "one": cfg({"short_e2e_mean_s": (1.0, 1.0, 1.0)}, n=1),
    }}}}
    t = sg.tables(s, [("new", "old"), ("one", "old"), ("new", "missing")])
    row = next(ln for ln in t.splitlines() if ln.startswith("| new / old"))
    cells = [c.strip() for c in row.strip("|").split("|")]
    assert cells[:3] == ["new / old", "3/3", "0.50"]       # ranges apart: no mark
    assert cells[2 + 4] == "1.05 ≈"                         # out tok/s: ranges overlap
    assert cells[2 + 1] == "-"                              # short E2E p99: neither side has it
    assert "| one / old | 1/3 | 0.50 (1 rep)" in t and "missing" not in t
