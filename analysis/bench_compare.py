"""Per-class latency table from `vllm bench serve --save-detailed` results.

  python analysis/bench_compare.py results/runs/s3a_1017/{B0,cap2048,mnbt512,shape,full} [--ref B0] [--md out.md]

Long = output_len >= 1024 (oracle class). TPOT = (latency - ttft) / (out - 1); ms.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os

import numpy as np
import pandas as pd


def bench_path(case_dir: str) -> str | None:
    """bench.json, or the bench.json.gz copy kept in git."""
    for name in ("bench.json", "bench.json.gz"):
        p = os.path.join(case_dir, name)
        if os.path.exists(p):
            return p
    return None


def load_bench(case_dir: str) -> dict:
    p = bench_path(case_dir)
    with (gzip.open(p, "rt") if p.endswith(".gz") else open(p)) as f:
        return json.load(f)


def per_request(case_dir: str) -> pd.DataFrame:
    b = load_bench(case_dir)
    d = pd.DataFrame(dict(inp=b["input_lens"], out=b["output_lens"], ttft=b["ttfts"], lat=b["latencies"],
                          start=b["start_times"]))
    d["itl_max"] = [max(x) if x else np.nan for x in b["itls"]]
    d = d[d.out > 0]
    d["tpot"] = (d.lat - d.ttft) / (d.out - 1).clip(lower=1)
    d["long"] = d.out >= 1024
    return d


def summarize(case_dir: str) -> dict:
    d = per_request(case_dir)
    s, l = d[~d.long], d[d.long]
    ms = 1e3
    return dict(
        case=os.path.basename(case_dir.rstrip("/")), n=len(d), n_long=len(l),
        long_tpot=l.tpot.mean() * ms if len(l) else np.nan, long_e2e_s=l.lat.mean() if len(l) else np.nan,
        short_tpot=s.tpot.mean() * ms, short_tpot_p99=s.tpot.quantile(.99) * ms,
        short_ttft=s.ttft.mean() * ms, short_ttft_p99=s.ttft.quantile(.99) * ms,
        short_e2e=s.lat.mean() * ms, short_e2e_p99=s.lat.quantile(.99) * ms,
        itl_max_p99=d.itl_max.quantile(.99) * ms,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cases", nargs="+")
    ap.add_argument("--ref", default=None, help="case name used as the 1.00 reference")
    ap.add_argument("--md", default=None)
    a = ap.parse_args()
    t = pd.DataFrame([summarize(c) for c in a.cases if bench_path(c)])
    ref = a.ref or t.case.iloc[0]
    r = t[t.case == ref].iloc[0]
    for c in ["long_tpot", "short_tpot", "short_ttft", "short_e2e", "short_e2e_p99"]:
        t[c + "_x"] = r[c] / t[c]  # >1 = faster than the reference
    pd.set_option("display.width", 250)
    print(t.round(2).to_string(index=False))
    if a.md:
        with open(a.md, "w") as f:
            f.write(t.round(2).to_markdown(index=False) + "\n")


if __name__ == "__main__":
    main()
