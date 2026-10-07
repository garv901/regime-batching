#!/usr/bin/env python3
"""Matrix summary of one pool: per family x variant x arm, short/long/all E2E means and completion over runs, valid runs
(env_valid.py: non-Azure <= LIMIT, Azure <= AZ_LIMIT) vs invalid ones. Prints Markdown tables; writes the per-run cells to JSON.

  LIMIT=15 python3 experiments/mstar/env/summ.py --job 13218 --sar /tmp/sar_poolidle.txt --sar-day 2026-10-07

A run's workload is <family>[_mtx2|_mt2048]_r<rep>; the validity flags are env_valid.py's (same flags and defaults).
"""
import argparse
import collections
import glob
import json
import os
import re
import statistics as st
import sys

from env_valid import add_args, pool_dir, tsv_rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    add_args(ap)
    ap.add_argument("--families", default="pub_azure_conv_r1.25,pub_mooncake_conv_r0.5,bagel_burst_v2,bagel_stag_v3_r5.5",
                    help="comma list of workload families, one table set each (default %(default)s)")
    ap.add_argument("--variants", default="mtx2,mt2048",
                    help="comma list of workload suffixes (_<variant>), one column each after base (default %(default)s)")
    ap.add_argument("--arms", default="main,ac,ac_park,peak_bf_pc_park",
                    help="comma list of arm labels, one row each (default %(default)s)")
    ap.add_argument("--json", help="per-run cells JSON (default <pool>/summ.json)")
    a = ap.parse_args()
    families, variants, ARMS = a.families.split(","), a.variants.split(","), a.arms.split(",")
    cols = ["base"] + variants
    wl_re = re.compile("(" + "|".join(map(re.escape, families)) + ")(_(?:" + "|".join(map(re.escape, variants)) + "))?$")

    envl = {}
    for t, _arm, r, m, ok in tsv_rows(a):
        envl[(t, r)] = (m, ok == 1)
    cells = collections.defaultdict(list)
    skipped = set()
    for bj in sorted(glob.glob(f"{a.results}/{a.job}*/*/bench.json")):
        d, run = bj.split("/")[-3:-1]
        task, arm = d.split("_", 1)
        if (task, run) not in envl:
            continue
        b = json.load(open(bj))
        wl = re.sub(r"_r\d+$", "", run)
        m = wl_re.match(wl)
        if not m:
            skipped.add(wl)
            continue
        fam, var = m.group(1), (m.group(2) or "_base")[1:]
        lat, cls = b["latencies"], b.get("cls") or []
        by = collections.defaultdict(list)
        for l, c in zip(lat, cls):
            if l and l > 0:
                by[c].append(l)
        n = len(b.get("input_lens") or lat)
        errs = collections.Counter(
            "abort" if "payload is not completed" in e else "ttl" if "delivery timed out" in e else "other"
            for e in b["errors"] if e)

        def mean(v):
            return st.mean(v) if v else float("nan")

        cells[(fam, var, arm)].append(dict(
            task=task, env=envl[(task, run)][0], ok=envl[(task, run)][1], done=b["completed"], n=n,
            short=mean(by["short"]), long=mean(by["long"]), all=mean([x for v in by.values() for x in v]),
            tok=b["output_throughput"], errs=dict(errs), t0=b["t0_wall"]))
    if skipped:
        print(f"summ: skipped workloads outside --families/--variants: {' '.join(sorted(skipped))}", file=sys.stderr)
    with open(a.json or os.path.join(pool_dir(a), "summ.json"), "w") as fh:
        json.dump({"|".join(k): v for k, v in cells.items()}, fh)

    def fmt(v, k):
        v = [x[k] for x in v]
        if not v:
            return "-"
        if len(v) == 1:
            return f"{v[0]:.2f}"
        return f"{st.mean(v):.2f} [{min(v):.2f}-{max(v):.2f}]"

    for fam in families:
        print(f"\n### {fam}")
        for metric in ["short", "long", "all"]:
            print(f"\n{metric} E2E mean (s), valid runs: mean [min-max] (n); invalid runs in braces")
            print("| arm | " + " | ".join(cols) + " |\n" + "|---" * (len(cols) + 1) + "|")
            for arm in ARMS:
                row = []
                for var in cols:
                    v = sorted(cells.get((fam, var, arm), []), key=lambda x: x["t0"])
                    ok = [x for x in v if x["ok"]]
                    bad = [x for x in v if not x["ok"]]
                    c = f"{fmt(ok, metric)} ({len(ok)})"
                    if bad:
                        c += " {" + ", ".join(f"{x[metric]:.1f}@{x['env']:.0f}%" for x in bad) + "}"
                    row.append(c)
                print(f"| {arm} | " + " | ".join(row) + " |")
        print("\ncompleted/total per valid run (aborts = main's KV stall)")
        for arm in ARMS:
            row = []
            for var in cols:
                v = [x for x in sorted(cells.get((fam, var, arm), []), key=lambda x: x["t0"]) if x["ok"]]
                row.append(" ".join(f"{x['done']}/{x['n']}" for x in v) or "-")
            print(f"| {arm} | " + " | ".join(row) + " |")


if __name__ == "__main__":
    main()
