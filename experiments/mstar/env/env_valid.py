#!/usr/bin/env python3
"""Per measured run of a pool: outsider load on the idle pool CPUs over the run window. Source: envmon (2 s) when it
covers the window, else sar bins (overlap-weighted mean over the window, max of the overlapping bins as p90).
valid = mean <= LIMIT and p90 <= 2 x LIMIT (envmon) or 4 x LIMIT (sar); Azure runs use AZ_LIMIT (rep-1 clean window: 1.7-3.2%).

  LIMIT=15 python3 experiments/mstar/env/env_valid.py --job 13218 --sar /tmp/sar_poolidle.txt --sar-day 2026-10-07 [--tsv]

A run is <results>/<job>*/<task>_<arm>/<run>/bench.json (smoke, diag and warmup runs are skipped). --tsv prints
task, arm, run, mean outsider %, 1 (valid) or 0 per run; summ.py and requeue_env.py use the same rows.
Env: LIMIT (default 15, every workload but Azure), AZ_LIMIT (default 3.5). Make the sar file with sar_poolidle.py.
"""
import argparse
import datetime as dt
import glob
import json
import os
import statistics as stt
import sys

RESULTS = "/data/garv901-55613a/results/mstar"


def add_args(ap, limit=15.0):
    """The flags env_valid.py, summ.py and requeue_env.py share (they all judge the runs of one pool)."""
    ap.add_argument("--results", default=RESULTS, help="results root (default %(default)s)")
    ap.add_argument("--job", default="13218",
                    help="pool job id; its runs are <results>/<job>*/*/bench.json (default %(default)s)")
    ap.add_argument("--pool", help="pool dir (default <results>/pool/<job>)")
    ap.add_argument("--envmon", default="envmon.jsonl",
                    help="envmon file; relative = under the pool dir (default %(default)s)")
    ap.add_argument("--group", default="poolidle", help="envmon CPU group that is the metric (default %(default)s)")
    ap.add_argument("--sar", default="sar_poolidle.txt",
                    help="sar_poolidle.py file, for runs before envmon started; relative = under the pool dir (default %(default)s)")
    ap.add_argument("--sar-day", default="2026-10-07",
                    help="UTC date of the sar day file, its times are HH:MM:SS (default %(default)s)")
    ap.add_argument("--sar-bin", type=int, default=600,
                    help="sar interval in s: a row at T covers (T-bin, T] (default %(default)s)")
    ap.add_argument("--limit", type=float, default=float(os.environ.get("LIMIT", limit)),
                    help="valid if mean outsider %% <= this, non-Azure runs (default $LIMIT or %(default)s)")
    ap.add_argument("--az-limit", type=float, default=float(os.environ.get("AZ_LIMIT", 3.5)),
                    help="the same for runs with 'azure' in the name (default $AZ_LIMIT or %(default)s)")


def pool_dir(a):
    return a.pool or os.path.join(a.results, "pool", a.job)


def evaluate(a):
    """[(task, arm, run, t0, t1, source, mean %, p90 %, own outsider %, valid)] over the finished runs of the pool."""
    pool = pool_dir(a)
    day0 = dt.datetime.strptime(a.sar_day, "%Y-%m-%d").replace(tzinfo=dt.timezone.utc).timestamp()
    sar, env = [], []
    sar_file = os.path.join(pool, a.sar)
    if os.path.exists(sar_file):
        for line in open(sar_file):
            if line.strip():
                h, v = line.split()
                t = list(map(int, h.split(":")))
                sar.append((day0 + t[0] * 3600 + t[1] * 60 + t[2], float(v)))
    else:
        print(f"env_valid: no sar file {sar_file}: runs before envmon started are invalid", file=sys.stderr)
    env_file = os.path.join(pool, a.envmon)
    if os.path.exists(env_file):
        for line in open(env_file):
            try:
                r = json.loads(line)
                env.append((r["ts"], r[a.group] * 100, r["steps"]))
            except ValueError:
                pass

    def metric(task, t0, t1):
        if env and env[0][0] <= t0 + 5:
            v = [p for ts, p, _ in env if t0 <= ts <= t1]
            own = [s["outsider"] * 100 for ts, _, st in env if t0 <= ts <= t1 for s in st.values() if s["task"] == task]
            if v:
                p90 = sorted(v)[int(.9 * (len(v) - 1))]
                return "envmon", stt.mean(v), p90, (stt.mean(own) if own else float("nan"))
        # a sar row at T covers (T - bin, T]: overlap-weighted mean over the run window
        w = [(min(T, t1) - max(T - a.sar_bin, t0), v) for T, v in sar if T > t0 and T - a.sar_bin < t1]
        tot = sum(x for x, _ in w)
        nan = float("nan")
        return "sar", (sum(x * v for x, v in w) / tot if tot else nan), (max(v for _, v in w) if w else nan), nan

    rows = []
    for bj in sorted(glob.glob(f"{a.results}/{a.job}*/*/bench.json")):
        d, run = bj.split("/")[-3:-1]
        task, arm = d.split("_", 1)
        if arm.startswith("smoke") or "diag" in arm or run.startswith("warmup"):
            continue
        b = json.load(open(bj))
        t0 = b["t0_wall"]
        t1 = t0 + b["duration"]
        src, m, p90, own = metric(task, t0, t1)
        lim = a.az_limit if "azure" in run else a.limit   # Azure r1.25 is the host-sensitive workload
        ok = m <= lim and p90 <= (2 * lim if src == "envmon" else 4 * lim)
        rows.append((task, arm, run, t0, t1, src, m, p90, own, ok))
    return rows


def tsv_rows(a):
    """[(task, arm, run, mean outsider % rounded to 2 places, 1 if valid else 0)]: the --tsv rows."""
    return [(r[0], r[1], r[2], round(r[6], 2), int(r[9])) for r in evaluate(a)]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    add_args(ap)
    ap.add_argument("--tsv", action="store_true", help="print task, arm, run, mean outsider %%, valid (1/0) per run")
    a = ap.parse_args()
    if a.tsv:
        for r in tsv_rows(a):
            print("\t".join(map(str, r)))
        return
    rows = evaluate(a)
    for r in rows:
        print(f"{r[0]} {r[1]:16} {r[2]:34} {dt.datetime.fromtimestamp(r[3], dt.timezone.utc):%H:%M} {r[5]:6} "
              f"out={r[6]:5.1f}% p90={r[7]:5.1f} own={r[8]:4.1f} {'OK' if r[9] else 'DIRTY'}")
    print(f"clean {sum(r[9] for r in rows)}/{len(rows)}")


if __name__ == "__main__":
    main()
