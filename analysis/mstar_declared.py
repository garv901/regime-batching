"""Cost of not knowing the output length: for each arm, metric(declared variant) / metric(oracle base) per workload.

    python analysis/mstar_declared.py <out>/summary.json [--arms ac peak_bf_pc_park ...] [--variants _mtx2 _mt2048] [--md FILE]

summary.json is written by analysis/mstar_admission.py from runs of the oracle workloads (`bagel_burst_v2`, a public
trace `pub_mooncake_conv_<x>`, ...) AND of their declared-max_tokens variants (`<base>_mtx2` = max_tokens 2x the output,
`<base>_mt2048` = 2,048 for every request; experiments/mstar/make_bagel_workloads_v4.sh). The base of a variant is its name
without the suffix. One table per base, one row per (variant, arm) that ran on both. A cell is the variant's mean over its
reps divided by the base's mean for the same arm (< 1 is better for times, > 1 for tok/s, so a time ratio above 1 is what
the unknown output length costs). Marks as in mstar_step_gains.py: ≈ when the two [min-max] rep ranges overlap, (1 rep) when
either side has one rep, `-` when either side lacks the metric. The last column is the variant's declared/generated KV ratio
(mean over its reps): sum(prompt + declared) / sum(prompt + generated) over the completed requests; the base's is 1.00.

The arm-vs-arm ratios ON the variants (ac/main, peak_bf_pc_park/ac, ...) are the ordinary step-gains tables:
    python analysis/mstar_step_gains.py <out>/summary.json ac/main peak_bf_pc_park/ac ... --workloads '*_mtx2' '*_mt2048'
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mstar_admission import (VARIANT_SUFFIXES, config_sort_key, md_table, split_variant,  # noqa: E402
                             workload_sort_key)
from mstar_step_gains import cell  # noqa: E402

# (summary key, header); the same keys as mstar_step_gains.METRICS, the ones that carry the cost of an over-declared bound
METRICS = [
    ("short_e2e_mean_s", "short E2E mean"), ("short_e2e_p99_s", "short E2E p99"),
    ("all_e2e_mean_s", "all E2E mean"), ("all_e2e_p99_s", "all E2E p99"),
    ("short_ttft_p99_s", "short TTFT p99"), ("all_ttft_p99_s", "all TTFT p99"),
    ("long_e2e_mean_s", "long E2E mean"), ("out_tok_s", "out tok/s"),
]


def _decl_kv(cfg: dict) -> str:
    v = cfg["mean"].get("decl_kv_ratio")
    return f"{v:.2f}" if isinstance(v, (int, float)) and math.isfinite(v) else "-"


def tables(summary: dict, arms: list[str] = (), variants: list[str] = VARIANT_SUFFIXES) -> str:
    wls = summary["workloads"]
    bases: dict[str, list[str]] = {}
    for wl in wls:
        base, suf = split_variant(wl)
        if suf and suf in variants:
            bases.setdefault(base, []).append(suf)
    out, orphans = [], []
    for base in sorted(bases, key=workload_sort_key):
        if base not in wls:
            orphans += [base + s for s in bases[base]]
            continue
        rows = []
        for suf in sorted(bases[base], key=variants.index):
            var, old = wls[base + suf]["configs"], wls[base]["configs"]
            for cfg in sorted(set(var) & set(old), key=config_sort_key):
                if arms and cfg not in arms:
                    continue
                rows.append([cfg, suf, f"{len(var[cfg]['reps'])}/{len(old[cfg]['reps'])}"]
                            + [cell(var[cfg], old[cfg], k) for k, _ in METRICS] + [_decl_kv(var[cfg])])
        if rows:
            out += [f"### {base}", "", md_table(["arm", "variant", "reps (var/base)"] + [f"{h} (var/base)" for _, h in METRICS]
                                                + ["declared/generated KV"], rows), ""]
    if orphans:
        out += ["No oracle base run for: " + ", ".join(f"`{w}`" for w in sorted(orphans, key=workload_sort_key)) + ".", ""]
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("summary", help="summary.json written by analysis/mstar_admission.py")
    ap.add_argument("--arms", nargs="+", default=[], metavar="CONFIG", help="only these configs (default: every config run on both)")
    ap.add_argument("--variants", nargs="+", default=list(VARIANT_SUFFIXES), metavar="SUFFIX",
                    help="which of the known variant suffixes to compare with their base (default: %(default)s)")
    ap.add_argument("--md", help="also write the tables to this file")
    a = ap.parse_args(argv)
    text = tables(json.load(open(a.summary)), a.arms, a.variants)
    if not text:
        print(f"no variant workload ({', '.join(a.variants)}) with an oracle base run in {a.summary}", file=sys.stderr)
    print(text)
    if a.md:
        with open(a.md, "w") as fh:
            fh.write(text + "\n")


if __name__ == "__main__":
    main()
