#!/usr/bin/env python3
"""Every --interval s, on one fixed idle CPU: CPU time of a fixed Python loop (frequency/IPC) and of a 64 MB numpy pass
(memory/L3), plus wall time. Higher cpu_ms at fixed work = slower host. Appends JSONL to the out file until the stop file exists.

  nohup python3 experiments/mstar/env/speedprobe.py --job 13218 --cpu 35 > /dev/null 2>&1 &      # one per probed CPU

Row: ts, py_cpu_ms, py_wall_ms, np_cpu_ms, np_wall_ms. Pick a CPU that no pool step uses (the probe is pinned to it).
The stop file is checked before each sample, so the probe exits up to --interval s after it appears.
Env defaults: CPU (--cpu), INT (--interval), OUT (--out).
"""
import argparse
import json
import os
import time

import numpy as np

RESULTS = "/data/garv901-55613a/results/mstar"


def loop(n=300_000):
    s = 0
    for i in range(n):
        s += i * i % 7
    return s


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--cpu", type=int, default=int(os.environ.get("CPU", 35)),
                    help="CPU to pin to (default $CPU or %(default)s)")
    ap.add_argument("--job", default="13218", help="pool job id (default %(default)s)")
    ap.add_argument("--results", default=RESULTS, help="results root (default %(default)s)")
    ap.add_argument("--pool", help="pool dir (default <results>/pool/<job>)")
    ap.add_argument("--interval", type=float, default=float(os.environ.get("INT", 10)),
                    help="seconds between samples (default $INT or %(default)s)")
    ap.add_argument("--out", default=os.environ.get("OUT"),
                    help="JSONL file, appended; relative = under the pool dir (default speedprobe_cpu<cpu>.jsonl)")
    ap.add_argument("--stop-file", default="STOP_ENVMON",
                    help="exit when this file exists; relative = under the pool dir (default %(default)s)")
    a = ap.parse_args()
    pool = a.pool or os.path.join(a.results, "pool", a.job)
    out = os.path.join(pool, a.out or f"speedprobe_cpu{a.cpu}.jsonl")
    stop = os.path.join(pool, a.stop_file)

    os.sched_setaffinity(0, {a.cpu})
    arr = np.ones(8 * 1024 * 1024)  # 64 MB
    with open(out, "a") as f:
        while not os.path.exists(stop):
            w0, c0 = time.time(), time.process_time()
            loop()
            c1, w1 = time.process_time(), time.time()
            arr.sum()
            arr.sum()
            c2, w2 = time.process_time(), time.time()
            f.write(json.dumps({"ts": round(w0, 2), "py_cpu_ms": round((c1 - c0) * 1e3, 2),
                                "py_wall_ms": round((w1 - w0) * 1e3, 2), "np_cpu_ms": round((c2 - c1) * 1e3, 2),
                                "np_wall_ms": round((w2 - w1) * 1e3, 2)}) + "\n")
            f.flush()
            time.sleep(a.interval)


if __name__ == "__main__":
    main()
