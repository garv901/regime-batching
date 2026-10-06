"""KV-admission A/B on mstar (BAGEL text-to-text): compare table, figures and summary.json.

  python analysis/mstar_admission.py <job_dir | results_root | case_dir> [...] \
      --out reports/figures/mstar_admission/<tag>

Input layout (experiments/mstar/admission_matrix.sbatch), one run = one case dir:
  <results_root>/<slurmjob>_<config>/<workload>_r<rep>/{bench.json, client.log, server.log, telemetry/}
  <results_root>/<slurmjob>_<config>/server_<k>/{server.log, startup_s.txt, telemetry/steps_rank*_pid*.jsonl,
      telemetry/admission_*.jsonl (bench-peak configs)}
  <results_root>/<slurmjob>_<config>/{meta.txt, gpu_mem.csv}
One server serves several runs: each case's telemetry/ and server.log are symlinks into its server_<k>/, so cases
share them and everything read from them is cut to the run's window [trace start, last request end + DRAIN_S].
The older layout (real telemetry/, server.log, startup_s.txt, gpu_mem.csv in every case dir) still works.
Every file may be missing; a missing file is reported (flags in the table, notes in compare.md and summary.json),
never an error. Job dirs of the same config are pooled as extra reps.

Writes <out>/compare.md, <out>/summary.json and, per workload, <workload>_{completions,free_kv,short_e2e_cdf,
long_e2e}.png.

Joins (verified in the producers):
  client <-> server telemetry: bench `request_ids[i]` is the id the client pinned ("{run_id}-{index:06d}"); the API
      server uses it unchanged as the wire id, so telemetry `new_request.rid` / `req_done.rid` / `hold.rid` / ... match
      it by string. bench `start_times`, `t0_monotonic` and telemetry `tm` are all time.monotonic() on one node.
  bench <-> server.log, gpu_mem.csv: bench `t0_wall` is time.time() at trace start; log lines (asctime) and nvidia-smi
      stamps are node-local text, read as UTC (the compute nodes run in UTC; a log or csv that does not cover the
      run window is skipped, with a note, rather than trusted).
  admission log <-> telemetry: the KV pool is keyed by the worker's integer handle (the worker passes handles to
      the engine), so admission `rid` is the int `h` of `new_request`, NOT the wire id. Handles are recycled after a
      request is removed and are per worker process, so the join is (pid, h) as-of the row's `ts_mono` against
      `new_request.tm` (admission_<pool>_pid<N>.jsonl pairs with steps_rank*_pid<N>.jsonl).
Request class: bench `cls`; else intended output length (>= 1024 -> long); else actual output length.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone

import matplotlib

matplotlib.use("Agg")
import matplotlib.ticker  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_compare import bench_path, load_bench  # noqa: E402  (reused: bench.json / bench.json.gz lookup + load)

try:  # optional: Rust JSON parser for the multi-10-MB telemetry files
    import orjson

    _loads = orjson.loads
except ImportError:
    _loads = json.loads

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_ORDER = ["main", "ac", "sum_bf", "peak_fifo", "peak_bf"]
ADMISSION_LOG_CONFIGS = {"sum_bf", "peak_fifo", "peak_bf"}  # bench-peak worktree: the only one with admission_*.jsonl
WORKLOAD_ORDER = ["bagel_control", "bagel_burst", "bagel_stag_r0.5", "bagel_stag_r1", "bagel_stag_r1.5", "bagel_stag_r2", "bagel_stag_r3",
                  "bagel_burst_v2", "bagel_stag_v2_r1.5", "bagel_stag_v2_r2.5", "bagel_stag_v2_r3"]
LONG_OUT = 1024  # output tokens: long class (make_workload / trace_replay.LONG_OUTPUT_TOKENS)
STALL_S = 60.0
DRAIN_S = 5.0  # rows (events, admission, log lines, gpu samples) still belong to a run this long after its last request ends
TIMEOUT_RE = re.compile(r"time.?out|timed out", re.I)
# trace_replay.build_bench writes "request failed" for an exception with an empty message, which is what
# asyncio.TimeoutError (the client's --request-timeout) is
EMPTY_MSG_ERROR = "request failed"
SERVER_ERR_RE = re.compile(r"Traceback|CUDA out of memory|OutOfMemory|\bFATAL\b|Fatal error", re.I)
LOG_TS_RE = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d{3})")  # logging asctime; flashinfer's lines start the same way
LOG_TS_FMT, GPU_TS_FMT = "%Y-%m-%d %H:%M:%S,%f", "%Y/%m/%d %H:%M:%S.%f"
SERVER_LINK_RE = re.compile(r"(?:^|/)server_(\d+)/(?:telemetry|server\.log)$")  # the sbatch's symlink targets
# categorical slots 1-5 of the reference palette (blue, orange, aqua, yellow, magenta); fixed per config, validated
# adjacent-pair (lines/bars) with validate_palette.py. Aqua/yellow/magenta are < 3:1 on white, so every chart also
# carries a legend, direct category labels where it can, and the numbers are in compare.md / summary.json.
CONFIG_COLORS = {"main": "#2a78d6", "ac": "#eb6834", "sum_bf": "#1baf7a", "peak_fifo": "#eda100", "peak_bf": "#e87ba4"}
FALLBACK_COLORS = ["#4a3aa7", "#008300", "#e34948", "#898781"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e1e0d9"
REP_STYLES = ["-", "--", ":", "-."]

CASE_RE = re.compile(r"^(?P<wl>.+)_r(?P<rep>\d+)$")
JOB_RE = re.compile(r"^(?P<job>\d+)_(?P<cfg>.+)$")
CASE_FILES = ("bench.json", "bench.json.gz", "client.log", "server.log", "gpu_mem.csv", "startup_s.txt", "telemetry")


# ----------------------------------------------------------------------------------------------- discovery
@dataclass
class CaseRef:
    path: str
    job: str
    config: str
    workload: str
    rep: int
    meta: list[str] = field(default_factory=list)


def _read_meta(job_dir: str) -> tuple[list[str], str | None]:
    p = os.path.join(job_dir, "meta.txt")
    if not os.path.exists(p):
        return [], None
    try:
        lines = [ln.rstrip("\n") for ln in open(p, errors="replace")][:8]
    except OSError:
        return [], None
    m = re.search(r"\bconfig=(\S+)", "\n".join(lines))
    return lines, (m.group(1) if m else None)


def _job_cases(job_dir: str) -> list[CaseRef]:
    base = os.path.basename(os.path.normpath(job_dir))
    jm = JOB_RE.match(base)
    meta, meta_cfg = _read_meta(job_dir)
    job, cfg = (jm.group("job"), jm.group("cfg")) if jm else (base, meta_cfg or base)
    out = []
    for name in sorted(os.listdir(job_dir)):
        cm = CASE_RE.match(name)
        if cm and os.path.isdir(os.path.join(job_dir, name)):
            out.append(CaseRef(os.path.join(job_dir, name), job, cfg, cm.group("wl"), int(cm.group("rep")), meta))
    return out


def discover(paths: list[str]) -> tuple[list[CaseRef], list[str]]:
    """Case dirs under each path: a job dir, a results root of job dirs, or a single case dir."""
    refs: dict[str, CaseRef] = {}
    problems: list[str] = []

    def add(rs):
        for r in rs:
            refs[os.path.realpath(r.path)] = r

    for p in paths:
        p = os.path.normpath(p)
        if not os.path.isdir(p):
            problems.append(f"{p}: not a directory")
            continue
        kids = [k for k in sorted(os.listdir(p)) if os.path.isdir(os.path.join(p, k))]
        base = os.path.basename(p)
        if any(CASE_RE.match(k) for k in kids):  # a job dir
            add(_job_cases(p))
        elif CASE_RE.match(base) and (any(os.path.exists(os.path.join(p, f)) for f in CASE_FILES) or not kids):
            parent = os.path.dirname(p)  # a single case dir; its job dir names the config
            cm = CASE_RE.match(base)
            jm = JOB_RE.match(os.path.basename(parent))
            meta, meta_cfg = _read_meta(parent)
            job, cfg = (jm.group("job"), jm.group("cfg")) if jm else (os.path.basename(parent), meta_cfg or "?")
            add([CaseRef(p, job, cfg, cm.group("wl"), int(cm.group("rep")), meta)])
        else:  # a results root: its children are job dirs
            n0 = len(refs)
            for k in kids:
                add(_job_cases(os.path.join(p, k)))
            if len(refs) == n0:
                problems.append(f"{p}: no case dirs (<workload>_r<rep>) found")
    return sorted(refs.values(), key=lambda r: (r.job, r.workload, r.rep)), problems


# ----------------------------------------------------------------------------------------------- file readers
def iter_jsonl(path: str, bad: list[int]):
    """Rows one at a time (a 100 MB telemetry file is never held whole); `bad[0]` counts unparsable lines.

    A server killed mid-write leaves a truncated last line.
    """
    with open(path, "rb") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield _loads(line)
            except ValueError:
                bad[0] += 1


def _pid_of(path: str) -> int:
    m = re.search(r"_pid(\d+)\.jsonl$", path)
    return int(m.group(1)) if m else -1


@dataclass
class Telemetry:
    steps: pd.DataFrame
    events: pd.DataFrame
    kv_key: str | None
    kv_pool_pages: int | None
    n_files: int
    dropped: int
    drop_marks: list = field(default_factory=list)  # (tm, pid, cumulative rows dropped so far) from `drop` / `end` rows


def load_telemetry(tdir: str, notes: list[str]) -> Telemetry | None:
    files = sorted(glob.glob(os.path.join(tdir, "steps_rank*_pid*.jsonl")))
    if not files:
        notes.append(f"no telemetry/steps_rank*_pid*.jsonl under {tdir}")
        return None
    steps, events, pools, marks = [], [], {}, []
    dropped = 0
    for path in files:
        pid = _pid_of(path)
        bad = [0]
        saw_end, saw_step, file_dropped = False, False, 0
        for r in iter_jsonl(path, bad):
            kind = r.get("kind")
            if kind == "step":
                saw_step = True
                if "tm" in r:  # keep only what the metrics read; the per-row rids/q/kv lists are dropped here
                    steps.append((r["tm"], pid, r.get("node"), r.get("walk"), r.get("bs"), r.get("ok"),
                                  r.get("gpu_ms"), r.get("free")))
            elif kind == "event":
                if "tm" in r:
                    r["pid"] = pid
                    events.append(r)
            elif kind == "meta" and r.get("phase") == "engine":
                pools.update(r.get("kv_pools") or {})
            elif kind == "drop":
                file_dropped = max(file_dropped, r.get("n", 0))
                marks.append((r.get("tm", math.inf), pid, r.get("n", 0)))
            elif kind == "end":
                saw_end = True
                file_dropped = max(file_dropped, r.get("dropped", 0))
                marks.append((r.get("tm", math.inf), pid, r.get("dropped", 0)))
        if bad[0]:
            notes.append(f"{os.path.basename(path)}: {bad[0]} unparsable line(s) skipped")
        dropped += file_dropped
        if saw_step and not saw_end:
            notes.append(f"{os.path.basename(path)}: no `end` row (server killed before its atexit flush; tail may be missing)")
    df = pd.DataFrame(steps, columns=["tm", "pid", "node", "walk", "bs", "ok", "gpu_ms", "free"])
    # `free` is {resource_key: free pages} for nodes that own a KV pool, {} otherwise
    keys = Counter(k for f in df["free"] if f for k in f)
    kv_key = next((k for k in pools if k in keys), None) or ("kv" if "kv" in keys else (keys.most_common(1)[0][0] if keys else None))
    df["free_kv"] = [(f.get(kv_key) if f else None) for f in df["free"]] if kv_key else np.nan
    df["free_kv"] = pd.to_numeric(df["free_kv"], errors="coerce")
    df["gpu_ms"] = pd.to_numeric(df["gpu_ms"], errors="coerce")
    df = df.drop(columns=["free"]).sort_values("tm", kind="stable").reset_index(drop=True)  # rows are not in seq order
    if df.empty:
        notes.append("telemetry has no step rows")
    pool = (pools.get(kv_key) or {}).get("max_num_pages") if kv_key else None
    ev = pd.DataFrame(events) if events else pd.DataFrame(columns=["ev", "tm", "pid"])
    return Telemetry(df, ev, kv_key, pool, len(files), dropped, marks)


def _window(df: pd.DataFrame, t0: float, t_end: float, col: str = "tm") -> pd.DataFrame:
    """Rows of one run, [t0, t_end + DRAIN_S]: the telemetry/admission files are shared by every run on a server."""
    return df[(df[col] >= t0) & (df[col] <= t_end + DRAIN_S)]


def _dropped_by(tel: Telemetry, t: float) -> int:
    """Rows dropped by time t over all processes (the marks are cumulative per pid)."""
    best: dict = {}
    for tm, pid, n in tel.drop_marks:
        if tm <= t:
            best[pid] = max(best.get(pid, 0), n)
    return sum(best.values())


def load_admission(tdir: str, notes: list[str]) -> pd.DataFrame | None:
    files = sorted(glob.glob(os.path.join(tdir, "admission_*.jsonl")))
    if not files:
        return None
    rows = []
    for path in files:
        bad = [0]
        pid = _pid_of(path)
        for r in iter_jsonl(path, bad):
            r["pid"] = pid
            rows.append(r)
        if bad[0]:
            notes.append(f"{os.path.basename(path)}: {bad[0]} unparsable line(s) skipped")
    return pd.DataFrame(rows) if rows else None


def _utc(text: str, fmt: str) -> float:
    """Node-local timestamp text -> epoch seconds. nvidia-smi and the server's asctime print local time; the nodes run UTC."""
    return datetime.strptime(text, fmt).replace(tzinfo=timezone.utc).timestamp()


def read_gpu_mem(path: str, win: tuple[float, float] | None = None) -> dict:
    """nvidia-smi --query-gpu=timestamp,index,memory.used,memory.total,utilization.gpu (csv, noheader, nounits).

    `win` = (lo, hi) epoch seconds keeps only the samples inside it (a job-wide file serves every run).
    """
    mem, util = [], []
    for line in open(path, errors="replace"):
        f = [x.strip() for x in line.split(",")]
        if len(f) >= 5 and f[2].isdigit():
            if win is not None:
                try:
                    if not win[0] <= _utc(f[0], GPU_TS_FMT) <= win[1]:
                        continue
                except ValueError:
                    continue
            mem.append(int(f[2]))
            if f[4].isdigit():
                util.append(int(f[4]))
    return dict(gpu_mem_peak_mib=max(mem) if mem else None, gpu_util_mean_pct=float(np.mean(util)) if util else None)


def scan_server_log(path: str, win: tuple[float, float] | None) -> tuple[list[str], bool]:
    """(Traceback/OOM/fatal lines, whether the log's timestamps reach `win`); with `win`, only lines inside it count.

    A line without a timestamp (a traceback body) takes the last timestamp before it. Only a hit's timestamp is parsed.
    """
    hits, first, last = [], None, None
    for ln in open(path, errors="replace"):
        m = LOG_TS_RE.match(ln)
        if m:
            last = m.group(1)
            first = first or last
        if SERVER_ERR_RE.search(ln) and (win is None or (last and win[0] <= _utc(last, LOG_TS_FMT) <= win[1])):
            hits.append(ln.strip()[:200])
    covered = win is None or bool(first and _utc(first, LOG_TS_FMT) <= win[1] and _utc(last, LOG_TS_FMT) >= win[0])
    return hits, covered


def scan_logs(case: str, notes: list[str], shared_by: int = 1, win: tuple[float, float] | None = None) -> dict:
    """`shared_by` > 1: server.log is one file for that many cases; `win` = the run on the epoch clock, to cut it."""
    info: dict = {}
    cl = os.path.join(case, "client.log")
    if os.path.exists(cl):
        lines = [ln.strip() for ln in open(cl, errors="replace") if ln.strip()]
        info["client_wrote_bench"] = any(ln.startswith("Wrote ") for ln in lines)
        info["client_last_line"] = lines[-1][:200] if lines else ""
    sl = os.path.join(case, "server.log")
    if os.path.exists(sl):
        win = win if shared_by > 1 else None
        if shared_by > 1 and win is None:  # no wall-clock anchor: whole-file counts, per server
            notes.append(f"server.log shared by {shared_by} cases; log counts are per server")
        hits, covered = scan_server_log(sl, win)
        if not covered:
            notes.append("server.log timestamps (read as UTC) do not reach the run window; log counts skipped")
        else:
            info["server_error_lines"] = len(hits)
            if hits:
                notes.append(f"server.log{' in the run window' if win else ''}: {len(hits)} line(s) with Traceback/OOM/fatal; "
                             f"first: {hits[0]}")
    return info


def server_of(case: str) -> tuple[int | None, int]:
    """(k, n): the case's telemetry/ or server.log is a symlink into <job>/server_<k>/, shared by n cases of the job.

    (None, 1) for the older layout, where every case dir holds its own files.
    """
    def index(d: str) -> int | None:
        for name in ("telemetry", "server.log"):
            p = os.path.join(d, name)
            m = SERVER_LINK_RE.search(os.readlink(p)) if os.path.islink(p) else None
            if m:
                return int(m.group(1))
        return None

    k = index(case)
    if k is None:
        return None, 1
    job = os.path.dirname(case)
    return k, sum(1 for x in os.listdir(job) if CASE_RE.match(x) and index(os.path.join(job, x)) == k)


# ----------------------------------------------------------------------------------------------- per request
def request_table(b: dict) -> pd.DataFrame:
    """One row per request, in trace order. Times are seconds from trace start (bench t0_monotonic)."""
    n = len(b["input_lens"])
    out = np.asarray(b["output_lens"], dtype=float)
    lat = np.asarray(b["latencies"], dtype=float)
    ttft = np.asarray(b["ttfts"], dtype=float)
    start_abs = np.asarray(b["start_times"], dtype=float)
    errs = list(b.get("errors") or [""] * n)
    err = np.array([e if e else "" for e in errs], dtype=object)
    ok = (err == "") & (out > 0)
    t0 = b.get("t0_monotonic")
    t0 = float(t0) if t0 is not None else float(start_abs.min() if n else 0.0)
    timed_out = ~ok & np.array([bool(TIMEOUT_RE.search(e)) or e == EMPTY_MSG_ERROR for e in err], dtype=bool)
    want = b.get("intended_output_lens")  # requested output tokens, same trace order as output_lens (trace_replay.build_bench)
    want = np.asarray(want, dtype=float) if want is not None and len(want) == n else np.full(n, np.nan)
    cls = b.get("cls")
    if cls is not None and len(cls) == n:
        cls = np.array([str(c) for c in cls], dtype=object)
    else:  # no cls: intended output length, else the realised one (unknown for a failed request)
        intended = b.get("intended_output_lens")
        if intended is not None and len(intended) == n:
            basis, known = np.asarray(intended, dtype=float), np.ones(n, bool)
        else:
            basis, known = out, ok
        cls = np.where(known, np.where(basis >= LONG_OUT, "long", "short"), "unknown").astype(object)
    rid = b.get("request_ids")
    rid = list(rid) if rid is not None and len(rid) == n else [str(i) for i in range(n)]
    return pd.DataFrame(dict(
        rid=rid, cls=cls, inp=b["input_lens"], out=out, want=want, trunc=ok & (out < want), ttft=ttft, lat=lat, ok=ok,
        timed_out=timed_out, err=err, start_abs=start_abs, start=start_abs - t0, end=np.where(ok, start_abs - t0 + lat, np.nan),
    ))


def max_stall(arrive, finished, other_end, t_end) -> tuple[float, float]:
    """Longest stretch with >= 1 request outstanding and no completion: (seconds, start time).

    `finished` are successful completions (they end a stretch); `other_end` are failures (they stop being
    outstanding but are not progress).
    """
    ev = sorted([(t, 0, "A") for t in arrive] + [(t, 1, "F") for t in other_end] + [(t, 2, "C") for t in finished])
    out, run, best, best_at, prev = 0, 0.0, 0.0, 0.0, 0.0
    for t, _, k in ev:
        if out > 0:
            run += t - prev
        prev = t
        if k == "A":
            out += 1
        else:
            out = max(out - 1, 0)
            if k == "C" or out == 0:
                if run > best:
                    best, best_at = run, t - run
                run = 0.0
    if out > 0:
        run += max(t_end - prev, 0.0)
        if run > best:
            best, best_at = run, t_end - run
    return best, best_at


def _pct(a, q):
    return float(np.percentile(a, q)) if len(a) else math.nan


def _mean(a):
    return float(np.mean(a)) if len(a) else math.nan


def bench_metrics(rt: pd.DataFrame, b: dict, stall_s: float) -> dict:
    n = len(rt)
    ok = rt[rt.ok]
    duration = float(b["duration"]) if b.get("duration") is not None else (float(ok.end.max()) if len(ok) else 0.0)
    timeout_s = (b.get("trace") or {}).get("timeout_s")
    sh, lo = ok[ok.cls == "short"], ok[ok.cls == "long"]
    bad = rt[~rt.ok]
    ends = np.where(bad.timed_out, bad.start + (timeout_s if timeout_s else duration), bad.start)
    stall, stall_at = max_stall(rt.start.values, ok.end.values, np.minimum(ends, duration), duration)
    m = dict(
        n_requests=n, completed=int(rt.ok.sum()), timed_out=int(rt.timed_out.sum()),
        failed=int((~rt.ok & ~rt.timed_out).sum()), duration_s=duration,
        # completed requests that returned fewer tokens than asked (NaN: bench.json has no intended_output_lens)
        truncated=int(rt.trunc.sum()) if rt.want.notna().any() else math.nan,
        out_tok_s=float(ok.out.sum()) / duration if duration > 0 else math.nan,
        short_n=len(sh), short_e2e_mean_s=_mean(sh.lat), short_e2e_p50_s=_pct(sh.lat, 50), short_e2e_p99_s=_pct(sh.lat, 99),
        short_ttft_mean_s=_mean(sh.ttft), short_ttft_p99_s=_pct(sh.ttft, 99),
        long_n=len(lo), long_e2e_mean_s=_mean(lo.lat), long_e2e_max_s=float(lo.lat.max()) if len(lo) else math.nan,
        long_ttft_mean_s=_mean(lo.ttft),
        long_failed=int(((rt.cls == "long") & ~rt.ok).sum()), unknown_class=int((rt.cls == "unknown").sum()),
        max_stall_s=stall, max_stall_at_s=stall_at,
    )
    m["stalled"] = bool(m["timed_out"] > 0 or stall > stall_s)
    return m


# ----------------------------------------------------------------------------------------------- telemetry metrics
def telemetry_metrics(tel: Telemetry, t0: float, t_end: float) -> dict:
    st, ev = tel.steps, _window(tel.events, t0, t_end)
    win = st[(st.tm >= t0) & (st.tm <= t_end)]  # busy share, decode bs and n_steps stop at the last request end
    wall = t_end - t0
    dec = win[win.walk.fillna("").str.startswith("decode") & (win.ok == True)]  # noqa: E712
    kv = _window(st, t0, t_end).free_kv
    holds = ev[ev.ev == "hold"] if len(ev) else ev
    fails = ev[ev.ev == "admit_fail"] if len(ev) else ev
    by_reason = Counter(fails["reason"].fillna("?")) if len(fails) else Counter()
    m = dict(
        holds=int(len(holds)), held_requests=int(holds["n"].sum()) if len(holds) else 0,
        admit_fail_total=int(len(fails)), admit_fail=dict(by_reason),
        offloads=int((ev.ev == "offload").sum()) if len(ev) else 0,
        min_free_kv_pages=float(kv.min()) if kv.notna().any() else math.nan,
        kv_pool_pages=tel.kv_pool_pages, kv_key=tel.kv_key,
        gpu_busy_share=float(win.gpu_ms.sum()) / 1e3 / wall if wall > 0 else math.nan,
        gpu_ms_unresolved=int(win.gpu_ms.isna().sum()), decode_bs_mean=_mean(pd.to_numeric(dec.bs, errors="coerce").dropna()),
        n_steps=int(len(win)), telemetry_dropped=_dropped_by(tel, t_end + DRAIN_S) - _dropped_by(tel, t0),
    )
    return m


def _flat_rids(events: pd.DataFrame, ev: str) -> list[str]:
    out = []
    for v in events.loc[events.ev == ev, "rid"] if len(events) and "rid" in events else []:
        if isinstance(v, (list, tuple)):
            out += [x for x in v if isinstance(x, str)]
        elif isinstance(v, str):
            out.append(v)
    return out


def join_client_telemetry(rt: pd.DataFrame, tel: Telemetry, t0: float, t_end: float) -> dict:
    """Client request ids vs telemetry wire ids; per-class counts of holds, admit failures, offloads."""
    ev = _window(tel.events, t0, t_end)
    nr = ev[ev.ev == "new_request"] if len(ev) else ev
    first = nr.drop_duplicates("rid").set_index("rid").tm if len(nr) else pd.Series(dtype=float)
    seen = rt.rid.isin(first.index)
    lag = (first.reindex(rt.rid[seen]).values - rt.start_abs[seen].values)
    cmap = dict(zip(rt.rid, rt.cls))
    j = dict(
        join_new_request=f"{int(seen.sum())}/{len(rt)}", join_new_request_frac=float(seen.mean()) if len(rt) else math.nan,
        dispatch_lag_median_s=float(np.median(lag)) if len(lag) else math.nan,
    )
    for name, evn in (("holds", "hold"), ("admit_fail", "admit_fail"), ("offload", "offload")):
        j[f"{name}_by_class"] = dict(Counter(cmap.get(r, "unmatched") for r in _flat_rids(ev, evn)))
    return j


def admission_metrics(adm: pd.DataFrame, tel: Telemetry | None, rt: pd.DataFrame | None, t0: float, t_end: float) -> dict:
    a = _window(adm, t0, t_end, "ts_mono").copy()
    a["wire"] = None
    j: dict = {}
    if tel is not None and len(tel.events) and "rid" in a:
        nr = tel.events[tel.events.ev == "new_request"]  # all of them, not the window's: a handle may date from a warmup
        h = pd.to_numeric(a["rid"], errors="coerce")
        is_h = h.notna()
        if is_h.any() and len(nr):
            left = pd.DataFrame(dict(tm=a.loc[is_h, "ts_mono"], pid=a.loc[is_h, "pid"], h=h[is_h].astype("int64")))
            left = left.sort_values("tm", kind="stable")
            right = nr[["tm", "pid", "h", "rid"]].dropna(subset=["h"]).astype({"h": "int64"}).rename(columns={"rid": "wire"})
            mg = pd.merge_asof(left, right.sort_values("tm", kind="stable"), on="tm", by=["pid", "h"], direction="backward")
            a.loc[left.index, "wire"] = mg["wire"].values  # merge_asof keeps the left's (sorted) row order
        a.loc[~is_h, "wire"] = a.loc[~is_h, "rid"].astype(str)  # a string rid is already a wire id
    a = a[~a["wire"].fillna("").str.contains("warmup")]
    ev = a["event"]
    n_ev = Counter(ev)
    pl = a["planner_us"].dropna() if "planner_us" in a else pd.Series(dtype=float)
    m = dict(
        adm_events=dict(n_ev), grant_deferred_log=int(n_ev.get("grant_deferred", 0)),
        planner_us_p50=_pct(pl, 50), planner_us_p99=_pct(pl, 99),
        adm_wait_why=dict(Counter(a.loc[ev == "wait", "why"])) if "why" in a else {},
        adm_max_waiting=int(a.n_waiting.max()) if "n_waiting" in a and len(a) else 0,
        adm_max_reserved=int(a.n_reserved.max()) if "n_reserved" in a and len(a) else 0,
        adm_fit_order=sorted({f"{f}/{o}" for f, o in zip(a.get("fit", []), a.get("order", []))}),
    )
    if rt is not None:
        cmap = dict(zip(rt.rid, rt.cls))
        a["cls"] = a["wire"].map(cmap)
        j["join_admission_rows"] = f"{int(a['cls'].notna().sum())}/{len(a)}"
        j["join_admission_frac"] = float(a["cls"].notna().mean()) if len(a) else math.nan
        for c in ("short", "long"):
            w = a[(ev == "reserve") & (a.cls == c)].get("waited_s", pd.Series(dtype=float)).dropna()
            r = a[(ev == "release") & (a.cls == c)].get("reserved_s", pd.Series(dtype=float)).dropna()
            m[f"adm_wait_mean_{c}_s"], m[f"adm_wait_p99_{c}_s"] = _mean(w), _pct(w, 99)
            m[f"adm_reserved_mean_{c}_s"] = _mean(r)
    m.update(j)
    return m


# ----------------------------------------------------------------------------------------------- one case
@dataclass
class Case:
    ref: CaseRef
    m: dict = field(default_factory=dict)
    flags: set = field(default_factory=set)
    notes: list = field(default_factory=list)
    series: dict = field(default_factory=dict)  # compact arrays for figures

    @property
    def name(self) -> str:
        return f"{self.ref.job}_{self.ref.config}/{self.ref.workload}_r{self.ref.rep}"


def analyze_case(ref: CaseRef, stall_s: float = STALL_S) -> Case:
    c = Case(ref)
    d = ref.path
    b = wall_win = None  # wall_win: the run on the epoch clock, to cut the server log and the job-wide gpu_mem.csv
    if bench_path(d):
        try:
            b = load_bench(d)
            rt = request_table(b)
            bm = bench_metrics(rt, b, stall_s)
            if isinstance(b.get("t0_wall"), (int, float)):
                wall_win = (b["t0_wall"], b["t0_wall"] + bm["duration_s"] + DRAIN_S)
        except (OSError, ValueError, KeyError, TypeError, IndexError) as e:
            c.flags.add("BAD_BENCH")
            c.notes.append(f"bench.json unreadable ({type(e).__name__}: {e})")
            b = None
    else:
        c.flags.add("NO_BENCH")
        c.notes.append("bench.json missing (client crashed, killed by RUN_LIMIT, or the server never started)")
    k, shared = server_of(d)  # k: server_<k>/ holds this case's telemetry and server.log (shared with n-1 other cases)
    sdir = os.path.join(os.path.dirname(d), f"server_{k}") if k is not None else None
    if k is not None:
        c.m["server_index"], c.m["server_shared_by"] = k, shared
    c.m.update(scan_logs(d, c.notes, shared, wall_win))
    if "NO_BENCH" in c.flags and c.m.get("client_last_line"):
        c.notes.append(f"client.log last line: {c.m['client_last_line']}")
    startup = next((p for p in (os.path.join(x, "startup_s.txt") for x in (d, sdir) if x) if os.path.exists(p)), None)
    if startup is None:
        c.notes.append("startup_s.txt missing (the server did not come up)")
        if b is None:
            c.flags.add("NO_START")
    else:
        try:
            c.m["startup_s"] = float(open(startup).read().strip())
        except ValueError:
            pass
    gpu, job_gpu = os.path.join(d, "gpu_mem.csv"), os.path.join(os.path.dirname(d), "gpu_mem.csv")
    if os.path.exists(gpu):
        c.m.update(read_gpu_mem(gpu))
    elif os.path.exists(job_gpu):  # one sampler per job: only the samples of this run
        g = read_gpu_mem(job_gpu, wall_win) if wall_win else {}
        if g.get("gpu_mem_peak_mib") is not None:
            c.m.update(g)
        else:
            c.notes.append("job gpu_mem.csv skipped: " + ("bench.json has no t0_wall to cut it to the run" if wall_win is None
                                                          else "no samples in the run window (timestamps read as UTC)"))
    else:
        c.notes.append("gpu_mem.csv missing")

    tdir = os.path.join(d, "telemetry")
    tel = load_telemetry(tdir, c.notes) if os.path.isdir(tdir) else None
    if tel is None:
        c.flags.add("NO_TELEMETRY")
        if not os.path.isdir(tdir):
            c.notes.append("telemetry/ missing")

    # trace start on the telemetry clock
    t0 = t_end = None
    if b is not None:
        c.m.update(bm)
        t0 = float(b["t0_monotonic"]) if b.get("t0_monotonic") is not None else float(rt.start_abs.min())
        t_end = t0 + c.m["duration_s"]
        if c.m["stalled"]:
            c.flags.add("STALLED")
            c.notes.append(
                f"STALLED: {c.m['timed_out']} request(s) timed out; longest stretch without a completion while requests "
                f"were outstanding {c.m['max_stall_s']:.1f}s, starting at t={c.m['max_stall_at_s']:.1f}s")
        if c.m["truncated"] > 0:
            tr = rt[rt.trunc]
            c.flags.add("TRUNCATED")
            c.notes.append(
                f"TRUNCATED: {c.m['truncated']} of {c.m['completed']} completed request(s) returned fewer output tokens than "
                f"requested (generated {int(tr.out.min())}-{int(tr.out.max())} vs requested {int(tr.want.min())}-"
                f"{int(tr.want.max())}); the engine stopped them early (BAGEL's decode loop ends at 2,049 tokens), so this "
                "run's KV pressure is below what the workload plans")
        ok = rt[rt.ok]
        c.series.update(
            arrivals=np.sort(rt.start.values), completions=np.sort(ok.end.values),
            short_e2e=ok[ok.cls == "short"].lat.values, long_e2e=ok[ok.cls == "long"].lat.values,
        )
    elif tel is not None and shared > 1:  # the first new_request .. last step would span every run on this server
        c.notes.append(f"no bench.json and server_{k} serves {shared} cases: the run's telemetry window is unknown, "
                       "telemetry metrics skipped")
    elif tel is not None and len(tel.events):
        nr = tel.events[(tel.events.ev == "new_request")]
        nr = nr[~nr.rid.fillna("").str.contains("warmup")]
        t0 = float(nr.tm.min()) if len(nr) else (float(tel.steps.tm.min()) if len(tel.steps) else None)
        # the last row of any kind: events are cut to the window as well, and req_done follows the last step
        t_end = max([float(tel.events.tm.max())] + ([float(tel.steps.tm.max())] if len(tel.steps) else []))
    if tel is not None and t0 is not None:
        c.m.update(telemetry_metrics(tel, t0, t_end))
        if b is not None:
            c.m.update(join_client_telemetry(rt, tel, t0, t_end))
            lag = c.m.get("dispatch_lag_median_s")
            if c.m["join_new_request_frac"] < 0.95:
                c.notes.append(f"only {c.m['join_new_request']} client request ids found in telemetry new_request events")
            if lag == lag and not (-0.5 <= lag <= 30):
                c.notes.append(f"median client-start -> worker new_request lag {lag:.2f}s: clocks may not share a domain")
        else:  # completions from the server's own req_done events
            ev = _window(tel.events, t0, t_end)
            done = ev[(ev.ev == "req_done")]
            seen, times = set(), []
            for tm, rids in zip(done.tm, done.rid):
                for r in rids if isinstance(rids, list) else [rids]:
                    if isinstance(r, str) and "warmup" not in r and r not in seen:
                        seen.add(r)
                        times.append(tm - t0)
            c.series["completions"] = np.sort(np.array(times))
            c.m["completed_server_side"] = len(times)
        s = _window(tel.steps, t0, t_end)
        s = s[s.free_kv.notna()]
        if len(s):
            binned = (s.assign(b=np.floor(s.tm - t0).astype(int)).groupby("b").free_kv.min())
            c.series["free_t"], c.series["free_v"] = binned.index.values.astype(float), binned.values
    dropped = c.m.get("telemetry_dropped", tel.dropped if tel is not None and shared <= 1 else 0)  # per run, else per server
    if dropped:
        c.flags.add("TEL_DROPS")
        c.notes.append(f"telemetry dropped {dropped} row(s) (bounded writer queue full)")
    if ref.config in ADMISSION_LOG_CONFIGS:
        adm = load_admission(tdir, c.notes) if os.path.isdir(tdir) else None
        if adm is None:
            c.flags.add("NO_ADMLOG")
            c.notes.append("admission_*.jsonl missing (expected for a bench-peak config)")
        elif t0 is not None:
            c.m.update(admission_metrics(adm, tel, rt if b is not None else None, t0, t_end))
    return c


# ----------------------------------------------------------------------------------------------- aggregation
SCALARS = [
    "n_requests", "completed", "failed", "timed_out", "truncated", "duration_s", "out_tok_s", "max_stall_s",
    "short_e2e_mean_s", "short_e2e_p50_s", "short_e2e_p99_s", "short_ttft_mean_s", "short_ttft_p99_s",
    "long_e2e_mean_s", "long_e2e_max_s", "long_ttft_mean_s", "holds", "held_requests", "admit_fail_total",
    "grant_deferred_log", "min_free_kv_pages", "gpu_busy_share", "decode_bs_mean", "planner_us_p50", "planner_us_p99",
    "offloads", "startup_s", "gpu_util_mean_pct", "gpu_mem_peak_mib", "adm_wait_mean_short_s", "adm_wait_mean_long_s",
    "adm_reserved_mean_short_s", "adm_reserved_mean_long_s", "completed_server_side",
]
KEY_RATIOS = [("short_e2e_mean_s", "short E2E mean"), ("long_e2e_mean_s", "long E2E mean"), ("out_tok_s", "out tok/s")]


def _finite(v) -> bool:
    return v is not None and isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool) and math.isfinite(v)


def aggregate(cases: list[Case]) -> dict:
    """Mean over reps of each metric over the reps that have it (NaN if none); n = reps found."""
    agg: dict = {"n": len(cases)}
    for k in SCALARS:
        vals = [c.m[k] for c in cases if _finite(c.m.get(k))]
        agg[k] = float(np.mean(vals)) if vals else math.nan
    with_tel = [c for c in cases if "admit_fail" in c.m]
    reasons = sorted({r for c in with_tel for r in c.m["admit_fail"]})
    agg["admit_fail"] = {r: float(np.mean([c.m["admit_fail"].get(r, 0) for c in with_tel])) for r in reasons} if with_tel else {}
    agg["has_telemetry"] = bool(with_tel)
    agg["flags"] = dict(Counter(f for c in cases for f in c.flags))
    agg["stalled_reps"] = sum(1 for c in cases if c.m.get("stalled"))
    return agg


def config_sort_key(cfg: str):
    return (CONFIG_ORDER.index(cfg), cfg) if cfg in CONFIG_ORDER else (len(CONFIG_ORDER), cfg)


def workload_sort_key(wl: str):
    return (WORKLOAD_ORDER.index(wl), wl) if wl in WORKLOAD_ORDER else (len(WORKLOAD_ORDER), wl)


def ratios(aggs: dict[str, dict], ref: str) -> dict[str, dict]:
    out = {}
    for cfg, a in aggs.items():
        if cfg == ref or ref not in aggs:
            continue
        r = {}
        for k, _ in KEY_RATIOS:
            num, den = a[k], aggs[ref][k]
            r[k] = num / den if _finite(num) and _finite(den) and den != 0 else math.nan
        out[cfg] = r
    return out


# ----------------------------------------------------------------------------------------------- markdown
def f(v, nd=2, pct=False) -> str:
    if not _finite(v):
        return "-"
    if pct:
        return f"{100 * v:.1f}%"
    if nd == 0:
        return f"{v:.0f}" if abs(v - round(v)) < 1e-9 else f"{v:.1f}"
    return f"{v:.{nd}f}"


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def flags_cell(a: dict) -> str:
    parts = [f"{k} {v}/{a['n']}" for k, v in sorted(a["flags"].items())]
    return ", ".join(parts) if parts else ""


def workload_table(aggs: dict[str, dict], reasons: list[str]) -> str:
    cfgs = sorted(aggs, key=config_sort_key)
    headers = ["config", "n", "flags", "completed", "failed", "timed out", "duration (s)", "out tok/s", "max stall (s)",
               "short E2E mean (s)", "short E2E p50 (s)", "short E2E p99 (s)", "short TTFT mean (s)", "short TTFT p99 (s)",
               "long E2E mean (s)", "long E2E max (s)", "long TTFT mean (s)", "holds"]
    headers += [f"fail: {r}" for r in reasons]
    headers += ["grant_deferred (adm log)", "min free KV pages", "GPU busy", "decode bs", "planner_us p50", "planner_us p99"]
    rows = []
    for cfg in cfgs:
        a = aggs[cfg]
        fails = [f(a["admit_fail"].get(r, 0), 0) if a["has_telemetry"] else "-" for r in reasons]
        rows.append([
            cfg, str(a["n"]), flags_cell(a), f(a["completed"], 0), f(a["failed"], 0), f(a["timed_out"], 0),
            f(a["duration_s"], 1), f(a["out_tok_s"], 1), f(a["max_stall_s"], 1),
            f(a["short_e2e_mean_s"], 2), f(a["short_e2e_p50_s"], 2), f(a["short_e2e_p99_s"], 2),
            f(a["short_ttft_mean_s"], 3), f(a["short_ttft_p99_s"], 3),
            f(a["long_e2e_mean_s"], 1), f(a["long_e2e_max_s"], 1), f(a["long_ttft_mean_s"], 2), f(a["holds"], 0),
            *fails, f(a["grant_deferred_log"], 0), f(a["min_free_kv_pages"], 0), f(a["gpu_busy_share"], pct=True),
            f(a["decode_bs_mean"], 1), f(a["planner_us_p50"], 1), f(a["planner_us_p99"], 1),
        ])
    col = {h: i for i, h in enumerate(headers)}
    for ref in ("ac", "main"):
        for cfg, r in ratios(aggs, ref).items():
            row = [""] * len(headers)
            row[0], row[1] = f"*{cfg} / {ref}*", ""
            row[col["short E2E mean (s)"]] = f(r["short_e2e_mean_s"], 2)
            row[col["long E2E mean (s)"]] = f(r["long_e2e_mean_s"], 2)
            row[col["out tok/s"]] = f(r["out_tok_s"], 2)
            rows.append(row)
    return md_table(headers, rows)


def render_md(groups: dict, aggs: dict, problems: list[str], figs: dict) -> str:
    L = ["# mstar KV-admission A/B (BAGEL text-to-text)", ""]
    L += [
        "Each config row is the mean over its reps (`n` = reps found; a metric a rep lacks is left out of that mean, "
        "`-` = no rep has it). Times in seconds from the client's bench.json; completed-request latencies only "
        "(a failed or timed-out request has no latency, so long E2E is optimistic where `timed out` > 0). "
        "Short/long = bench `cls`. Percentiles are per rep, then averaged.",
        "",
        "- **STALLED**: a request timed out, or no request completed for > 60 s while one was outstanding "
        "(`max stall`; a timed-out request counts as outstanding until its timeout).",
        "- **TRUNCATED**: a completed request returned fewer output tokens than the trace asked for "
        "(`output_lens` < `intended_output_lens`; BAGEL's decode loop stops at 2,049 tokens), so the run's KV pressure is "
        "below the workload's plan; `truncated` in summary.json counts them.",
        "- *timed out* = failed with the client's empty-message error (`request failed`) or a timeout message; "
        "*failed* = every other error.",
        "- holds / fail: / min free KV pages / GPU busy / decode bs come from step telemetry in the run's window "
        "(trace start to last request end + 5 s; the server's telemetry is shared by the runs it serves). "
        "GPU busy = sum of step `gpu_ms` / (trace start to last request end). decode bs = mean `bs` of ok `decode` steps. "
        "fail: columns count `admit_fail` events by `reason`.",
        "- grant_deferred (adm log), planner_us: admission log, bench-peak configs only (`-` for main/ac).",
        "- Ratio rows `X / ref` = X's mean divided by the reference's mean (E2E < 1 and tok/s > 1 favour X).",
        "",
    ]
    # one set of fail: columns for every workload, so the tables line up
    reasons = sorted({r for w in aggs.values() for a in w.values() for r in a["admit_fail"]} | {"GrantDeferred"})
    for wl in sorted(aggs, key=workload_sort_key):
        L += [f"## {wl}", "", workload_table(aggs[wl], reasons), ""]
        for tag, title in (("completions", "completions over time"), ("free_kv", "free KV pages"),
                           ("short_e2e_cdf", "short E2E CDF"), ("long_e2e", "long E2E")):
            if figs.get(wl, {}).get(tag):
                L.append(f"![{wl} {title}]({figs[wl][tag]})")
        L.append("")
    issues = [(c.name, sorted(c.flags), c.notes) for wl in groups for cfg in groups[wl] for c in groups[wl][cfg]
              if c.flags or c.notes]
    L += ["## Data problems", ""]
    if not issues and not problems:
        L.append("None: every case has bench.json, telemetry and (where expected) an admission log.")
    for p in problems:
        L.append(f"- {p}")
    for name, flags, notes in issues:
        L.append(f"- `{name}`" + (f" [{', '.join(flags)}]" if flags else "") + "".join(f"\n  - {n}" for n in notes))
    return "\n".join(L) + "\n"


# ----------------------------------------------------------------------------------------------- figures
def _color(cfg: str) -> str:
    if cfg in CONFIG_COLORS:
        return CONFIG_COLORS[cfg]
    return FALLBACK_COLORS[sum(map(ord, cfg)) % len(FALLBACK_COLORS)]  # deterministic (str hash is salted)


def _style(ax, xlabel, ylabel, title):
    ax.set_xlabel(xlabel, color=INK2)
    ax.set_ylabel(ylabel, color=INK2)
    ax.set_title(title, color=INK, fontsize=10, loc="left")
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=INK2, labelsize=8)


def _label(cfg, cases):
    return cfg + (" (STALLED)" if any("STALLED" in c.flags for c in cases) else "")


def _save(fig, out_dir, name) -> str:
    path = os.path.join(out_dir, name)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return name


def fig_completions(wl, groups, out_dir):
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    drew = False
    arr = next((c.series["arrivals"] for cs in groups.values() for c in cs if "arrivals" in c.series), None)
    if arr is not None and len(arr):
        ax.step(arr, np.arange(1, len(arr) + 1), where="post", color="#898781", lw=1.0, ls=":", label="arrived (trace)")
    for cfg in sorted(groups, key=config_sort_key):
        for i, c in enumerate(groups[cfg]):
            t = c.series.get("completions")
            if t is None or not len(t):
                continue
            ax.step(np.r_[0, t], np.r_[0, np.arange(1, len(t) + 1)], where="post", color=_color(cfg), lw=1.6,
                    ls=REP_STYLES[i % len(REP_STYLES)], label=_label(cfg, groups[cfg]) if i == 0 else None)
            drew = True
    if not drew:
        plt.close(fig)
        return None
    _style(ax, "time since trace start (s)", "cumulative finished requests", f"{wl}: completions over time")
    ax.legend(frameon=False, fontsize=8)
    return _save(fig, out_dir, f"{wl}_completions.png")


def fig_free_kv(wl, groups, out_dir):
    cfgs = [c for c in sorted(groups, key=config_sort_key) if any("free_t" in x.series for x in groups[c])]
    if not cfgs:
        return None
    fig, axes = plt.subplots(len(cfgs), 1, figsize=(7.2, 1.7 * len(cfgs) + 0.9), sharex=True, sharey=True, squeeze=False)
    pool = max((c.m.get("kv_pool_pages") or 0 for cs in groups.values() for c in cs), default=0)
    for ax, cfg in zip(axes[:, 0], cfgs):
        for i, c in enumerate(groups[cfg]):
            if "free_t" in c.series:
                ax.step(c.series["free_t"], c.series["free_v"], where="post", color=_color(cfg), lw=1.2,
                        ls=REP_STYLES[i % len(REP_STYLES)])
        ax.set_ylim(0, pool * 1.05 if pool else None)
        ax.set_ylabel(_label(cfg, groups[cfg]), color=INK, fontsize=8)
        ax.grid(True, color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(colors=INK2, labelsize=8)
    axes[0, 0].set_title(f"{wl}: free KV pages (min per 1 s of step rows)", color=INK, fontsize=10, loc="left")
    axes[-1, 0].set_xlabel("time since trace start (s)", color=INK2)
    return _save(fig, out_dir, f"{wl}_free_kv.png")


def fig_short_cdf(wl, groups, out_dir):
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    drew = False
    for cfg in sorted(groups, key=config_sort_key):
        parts = [c.series["short_e2e"] for c in groups[cfg] if len(c.series.get("short_e2e", []))]
        v = np.sort(np.concatenate(parts)) if parts else np.array([])
        if not len(v):
            continue
        ax.step(np.r_[v[0], v], np.r_[0, np.arange(1, len(v) + 1) / len(v)], where="post", color=_color(cfg), lw=1.6,
                label=_label(cfg, groups[cfg]))
        drew = True
    if not drew:
        plt.close(fig)
        return None
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10, subs=(1, 2, 5)))
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    _style(ax, "short-request E2E (s, log)", "fraction of requests", f"{wl}: short E2E CDF (reps pooled)")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    return _save(fig, out_dir, f"{wl}_short_e2e_cdf.png")


def fig_long_e2e(wl, groups, out_dir):
    cfgs = [c for c in sorted(groups, key=config_sort_key) if any(len(x.series.get("long_e2e", [])) for x in groups[c])]
    if not cfgs:
        return None
    fig, ax = plt.subplots(figsize=(1.4 * len(cfgs) + 3.0, 4.2))
    rng = np.random.default_rng(0)
    for i, cfg in enumerate(cfgs):
        v = np.concatenate([c.series.get("long_e2e", np.array([])) for c in groups[cfg]])
        ax.boxplot([v], positions=[i], widths=0.5, showfliers=False, patch_artist=True,
                   boxprops=dict(facecolor="none", edgecolor=INK2, lw=1.0), medianprops=dict(color=INK, lw=1.4),
                   whiskerprops=dict(color=INK2, lw=1.0), capprops=dict(color=INK2, lw=1.0))
        ax.scatter(i + rng.uniform(-0.16, 0.16, len(v)), v, s=14, color=_color(cfg), edgecolor="#fcfcfb", linewidth=0.6, zorder=3)
    labels = []
    for cfg in cfgs:
        nf = sum(c.m.get("long_failed", 0) for c in groups[cfg])
        n = sum(len(c.series.get("long_e2e", [])) for c in groups[cfg])
        labels.append(f"{cfg}\nn={n}" + (f", {nf} failed" if nf else ""))
    ax.set_xticks(range(len(cfgs)), labels)
    _style(ax, "", "long-request E2E (s)", f"{wl}: long E2E (completed requests, reps pooled)")
    return _save(fig, out_dir, f"{wl}_long_e2e.png")


# ----------------------------------------------------------------------------------------------- json + main
def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return float(o) if math.isfinite(o) else None
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def build_summary(inputs, groups, aggs, problems) -> dict:
    wls = {}
    for wl in sorted(aggs, key=workload_sort_key):
        cfgs = {}
        for cfg in sorted(aggs[wl], key=config_sort_key):
            cfgs[cfg] = dict(
                mean=aggs[wl][cfg],
                reps=[dict(case=c.name, path=c.ref.path, job=c.ref.job, rep=c.ref.rep, flags=sorted(c.flags),
                           notes=c.notes, meta=c.ref.meta, metrics=c.m) for c in groups[wl][cfg]],
            )
        wls[wl] = dict(configs=cfgs, ratios={f"vs_{ref}": ratios(aggs[wl], ref) for ref in ("ac", "main")})
    return _clean(dict(inputs=inputs, problems=problems, stall_threshold_s=STALL_S, workloads=wls))


def run(paths: list[str], out_dir: str, stall_s: float = STALL_S) -> dict:
    refs, problems = discover(paths)
    os.makedirs(out_dir, exist_ok=True)
    groups: dict[str, dict[str, list[Case]]] = {}
    for ref in refs:
        groups.setdefault(ref.workload, {}).setdefault(ref.config, []).append(analyze_case(ref, stall_s))
    aggs = {wl: {cfg: aggregate(cs) for cfg, cs in cfgs.items()} for wl, cfgs in groups.items()}
    figs: dict[str, dict] = {}
    for wl, g in groups.items():
        figs[wl] = dict(completions=fig_completions(wl, g, out_dir), free_kv=fig_free_kv(wl, g, out_dir),
                        short_e2e_cdf=fig_short_cdf(wl, g, out_dir), long_e2e=fig_long_e2e(wl, g, out_dir))
    with open(os.path.join(out_dir, "compare.md"), "w") as fh:
        fh.write(render_md(groups, aggs, problems, figs))
    summary = build_summary(paths, groups, aggs, problems)
    with open(os.path.join(out_dir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1, allow_nan=False)
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("paths", nargs="+", help="job dir(s), results root(s) or case dir(s)")
    ap.add_argument("--out", default=None, help="output dir (default reports/figures/mstar_admission/<first input name>)")
    ap.add_argument("--stall-s", type=float, default=STALL_S, help="STALLED if no completion for this long with requests outstanding")
    a = ap.parse_args(argv)
    out = a.out or os.path.join(ROOT, "reports", "figures", "mstar_admission", os.path.basename(os.path.normpath(a.paths[0])))
    s = run(a.paths, out, a.stall_s)
    n = sum(len(c["reps"]) for w in s["workloads"].values() for c in w["configs"].values())
    print(f"{n} case(s), {len(s['workloads'])} workload(s) -> {out}/compare.md, summary.json, *.png")
    for p in s["problems"]:
        print("problem:", p)


if __name__ == "__main__":
    main()
