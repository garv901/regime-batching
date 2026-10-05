"""Mean +- std of per-class speedups over repeats; each repeat is normalized to its own B0.

  python analysis/repeats.py --ref B0 --cases B0 mnbt512 C3 shape full \
      --rep r1=results/runs/s3a_1017/{case} --rep r2=results/runs/s4a_1034/{case}_r2 ... [--md out.md]
"""
from __future__ import annotations

import argparse
import os

import pandas as pd

from bench_compare import bench_path, summarize

METRICS = ["long_tpot", "short_tpot", "short_ttft", "short_e2e", "short_e2e_p99"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="+", required=True)
    ap.add_argument("--ref", default="B0")
    ap.add_argument("--rep", action="append", required=True, help="name=path template with {case}")
    ap.add_argument("--md", default=None)
    a = ap.parse_args()
    rows = []
    for spec in a.rep:
        name, tmpl = spec.split("=", 1)
        ref_dir = tmpl.format(case=a.ref)
        if not bench_path(ref_dir):
            continue
        ref = summarize(ref_dir)
        for c in a.cases:
            d = tmpl.format(case=c)
            if not bench_path(d):
                continue
            s = summarize(d)
            rows.append(dict(rep=name, case=c, **{m: s[m] for m in METRICS},
                             **{m + "_x": ref[m] / s[m] for m in METRICS}))
    t = pd.DataFrame(rows)
    g = t.groupby("case", sort=False)
    out = g.size().rename("n").to_frame()
    for m in METRICS:
        out[m] = g[m].mean().round(1)
    for m in METRICS:
        out[m + "_x"] = g[m + "_x"].mean().round(2).astype(str) + " +- " + g[m + "_x"].std(ddof=1).round(2).astype(str)
    out = out.reset_index()
    pd.set_option("display.width", 250)
    print(out.to_string(index=False))
    if a.md:
        with open(a.md, "w") as f:
            f.write(out.to_markdown(index=False) + "\n")


if __name__ == "__main__":
    main()
