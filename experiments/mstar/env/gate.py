#!/usr/bin/env python3
"""Holds a pool's queue while outsiders load the node's idle pool CPUs: moves <pool>/queue/*.task to held/ when the 2-min
mean busy share of the envmon group (poolidle) > HOLD, and back when the 10-min mean <= RELEASE and the 2-min mean <= RELEASE2
(and the 10-min window has at least --min-samples rows). Checks every --interval s; exits when the stop file exists.

  HOLD=.06 RELEASE=.035 RELEASE2=.04 nohup python3 experiments/mstar/env/gate.py --job 13218 > <pool>/gate.log 2>&1 &
  python3 experiments/mstar/env/gate.py --job 13218 --dry-run --once      # log what it would do now, move nothing

Reads the last --tail-bytes of the envmon file (rows older than the window are ignored). Env: HOLD, RELEASE, RELEASE2.
"""
import argparse
import datetime as dt
import glob
import json
import os
import shutil
import time

RESULTS = "/data/garv901-55613a/results/mstar"


def log(m):
    print(f"[{dt.datetime.now(dt.timezone.utc):%H:%M:%S}] {m}", flush=True)


def tail(env_file, group, sec, nbytes):
    """(mean of the group's busy share over the last sec s, number of rows) from the last nbytes of the envmon file."""
    rows = []
    with open(env_file, "rb") as f:
        f.seek(0, 2)
        f.seek(max(0, f.tell() - nbytes))
        for line in f.read().decode(errors="ignore").splitlines()[1:]:
            try:
                rows.append(json.loads(line))
            except ValueError:
                pass
    t = time.time()
    v = [r[group] for r in rows if r["ts"] >= t - sec]
    return (sum(v) / len(v), len(v)) if v else (None, 0)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--job", default="13218", help="pool job id (default %(default)s)")
    ap.add_argument("--results", default=RESULTS, help="results root (default %(default)s)")
    ap.add_argument("--pool", help="pool dir (default <results>/pool/<job>)")
    ap.add_argument("--envmon", default="envmon.jsonl",
                    help="envmon file; relative = under the pool dir (default %(default)s)")
    ap.add_argument("--group", default="poolidle", help="envmon CPU group to watch (default %(default)s)")
    ap.add_argument("--stop-file", default="STOP_ENVMON",
                    help="exit when this file exists; relative = under the pool dir (default %(default)s)")
    ap.add_argument("--hold", type=float, default=float(os.environ.get("HOLD", .06)),
                    help="hold when the short-window mean > this (default $HOLD or %(default)s)")
    ap.add_argument("--release", type=float, default=float(os.environ.get("RELEASE", .035)),
                    help="release needs the long-window mean <= this (default $RELEASE or %(default)s)")
    ap.add_argument("--release2", type=float, default=float(os.environ.get("RELEASE2", .04)),
                    help="and the short-window mean <= this (default $RELEASE2 or %(default)s)")
    ap.add_argument("--short-window", type=float, default=120, help="seconds (default %(default)s)")
    ap.add_argument("--long-window", type=float, default=600, help="seconds (default %(default)s)")
    ap.add_argument("--min-samples", type=int, default=250,
                    help="rows the long window needs before a release (default %(default)s)")
    ap.add_argument("--tail-bytes", type=int, default=400_000,
                    help="how much of the envmon file end to read per check (default %(default)s)")
    ap.add_argument("--interval", type=float, default=30, help="seconds between checks (default %(default)s)")
    ap.add_argument("--dry-run", action="store_true", help="log the decisions, move no task")
    ap.add_argument("--once", action="store_true", help="check once and exit")
    a = ap.parse_args()
    pool = a.pool or os.path.join(a.results, "pool", a.job)
    env_file, stop = os.path.join(pool, a.envmon), os.path.join(pool, a.stop_file)
    would = "would " if a.dry_run else ""
    if not a.dry_run:
        os.makedirs(f"{pool}/held", exist_ok=True)

    log(f"gate up: hold>{a.hold} release<={a.release}(10m) and <={a.release2}(2m)")
    while not os.path.exists(stop):
        p2, _ = tail(env_file, a.group, a.short_window, a.tail_bytes)
        p10, n10 = tail(env_file, a.group, a.long_window, a.tail_bytes)
        q, h = sorted(glob.glob(f"{pool}/queue/*.task")), sorted(glob.glob(f"{pool}/held/*.task"))
        if p2 is not None and p2 > a.hold and q:
            if not a.dry_run:
                for t in q:
                    shutil.move(t, f"{pool}/held/")
            log(f"{would}HOLD {len(q)} tasks ({a.group} 2m={p2:.3f} 10m={p10:.3f})")
        elif h and p10 is not None and p2 is not None and n10 >= a.min_samples and p10 <= a.release and p2 <= a.release2:
            if not a.dry_run:
                for t in h:
                    shutil.move(t, f"{pool}/queue/")
            log(f"{would}RELEASE {len(h)} tasks ({a.group} 2m={p2:.3f} 10m={p10:.3f})")
        if a.once:
            break
        time.sleep(a.interval)


if __name__ == "__main__":
    main()
