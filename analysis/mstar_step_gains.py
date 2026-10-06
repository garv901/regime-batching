"""Per-step gain table from an mstar_admission.py summary.json: for each pair NEW/OLD, NEW's mean / OLD's mean per metric.

    python analysis/mstar_step_gains.py <out>/summary.json ac_park/ac peak_bf/ac peak_bf_pc/peak_bf ... [--md FILE]

One table per workload. A cell is the ratio of the two configs' means over their reps (< 1 is better for times, > 1 for
tok/s), marked ≈ when the two [min–max] rep ranges overlap (mstar_admission.ranges_overlap: rep spread alone could give
the difference), (1 rep) when either side has one rep, and `-` when either side lacks the metric.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mstar_admission import md_table, ranges_overlap, workload_sort_key  # noqa: E402

# (summary key, header, lower is better)
METRICS = [
    ("short_e2e_mean_s", "short E2E mean", True), ("short_e2e_p99_s", "short E2E p99", True),
    ("short_ttft_p99_s", "short TTFT p99", True), ("long_e2e_mean_s", "long E2E mean", True),
    ("out_tok_s", "out tok/s", False), ("host_ms_p50", "host ms/step", True), ("host_ms_w400p", "host ms/step @400+", True),
    ("planner_us_p50", "planner µs p50", True), ("kv_free_lag_ms_p50", "KV free lag p50", True),
    ("kv_free_lag_ms_p99", "KV free lag p99", True),
]


def cell(new: dict, old: dict, k: str) -> str:
    a, b = new["mean"].get(k), old["mean"].get(k)
    if not (isinstance(a, (int, float)) and isinstance(b, (int, float))) or not (math.isfinite(a) and math.isfinite(b)) or b == 0:
        return "-"
    sa, sb = new["mean"]["spread"].get(k), old["mean"]["spread"].get(k)
    s = f"{a / b:.2f}"
    if sa and sb and min(sa["n"], sb["n"]) == 1:
        return s + " (1 rep)"
    return s + (" ≈" if sa and sb and ranges_overlap(sa, sb) else "")


def tables(summary: dict, pairs: list[tuple[str, str]]) -> str:
    out = []
    for wl in sorted(summary["workloads"], key=workload_sort_key):
        cfgs = summary["workloads"][wl]["configs"]
        rows = [[f"{n} / {o}", f"{len(cfgs[n]['reps'])}/{len(cfgs[o]['reps'])}"] + [cell(cfgs[n], cfgs[o], k) for k, _, _ in METRICS]
                for n, o in pairs if n in cfgs and o in cfgs]
        if rows:
            out += [f"### {wl}", "", md_table(["step", "reps"] + [h for _, h, _ in METRICS], rows), ""]
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("summary", help="summary.json written by analysis/mstar_admission.py")
    ap.add_argument("pairs", nargs="+", metavar="NEW/OLD", help="config labels, e.g. peak_bf_pc/peak_bf")
    ap.add_argument("--md", help="also write the tables to this file")
    a = ap.parse_args(argv)
    pairs = []
    for p in a.pairs:
        n, sep, o = p.partition("/")
        if not sep or not n or not o:
            ap.error(f"pair '{p}': use NEW/OLD")
        pairs.append((n, o))
    text = tables(json.load(open(a.summary)), pairs)
    print(text)
    if a.md:
        with open(a.md, "w") as fh:
            fh.write(text + "\n")


if __name__ == "__main__":
    main()
