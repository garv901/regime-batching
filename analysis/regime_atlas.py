"""Regime atlas: canonical step shapes -> regime label -> FA3 path -> modelled attention cost (Llama-3.1-8B, H100).

Prices each shape with the FA3 tile model (regime_batching.features.attn_step_model) and with the original additive
model (StepFeatures.x_ratio), so the two can be compared shape by shape.
Usage: python3 analysis/regime_atlas.py [--out reports/regimes_study/figures/atlas.md] [--json FILE]
"""
import argparse, json, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from regime_batching.features import GEOMS, RegimeConfig, attn_step_model, compute_features  # noqa: E402

GEOM = GEOMS["meta-llama/Llama-3.1-8B-Instruct"]
K = 1024

# (group, name, decode KV lengths, prefill chunks as (q, prefix))
SHAPES = [
    ("decode only", "64 x 2k decodes (T <= 512, CUDA graph)", [2 * K] * 64, []),
    ("decode only", "31 x 542 + 1 x 4k (user's 31+1 example)", [542] * 31 + [4 * K], []),
    ("decode only", "3 x 128k decodes", [128 * K] * 3, []),
    ("short prefix (chat)", "512 chunk, 15 short decodes", [542] * 15, [(512, 0)]),
    ("short prefix (chat)", "512 chunk + 15 short + 1 x 32k (original X)", [542] * 15 + [32 * K], [(512, 0)]),
    ("short prefix (chat)", "600 chunk + 4 x 24k (X-heavy workload)", [24 * K] * 4, [(600, 0)]),
    ("short prefix (chat)", "416 chunk + 4 x 24k (just below 417)", [24 * K] * 4, [(416, 0)]),
    ("short prefix (chat)", "2048 chunk + 4 x 24k", [24 * K] * 4, [(2048, 0)]),
    ("long prefix (agent round)", "1k append on 128k, alone", [], [(1 * K, 128 * K)]),
    ("long prefix (agent round)", "1k append on 128k + 3 x 128k decodes", [128 * K] * 3, [(1 * K, 128 * K)]),
    ("long prefix (agent round)", "256 append on 128k + 3 x 128k decodes", [128 * K] * 3, [(256, 128 * K)]),
    ("long prefix (agent round)", "1k append on 32k + 1 x 128k decode", [128 * K], [(1 * K, 32 * K)]),
    ("H5 wave edges (128k prefix)", "512 append on 128k", [], [(512, 128 * K)]),
    ("H5 wave edges (128k prefix)", "513 append on 128k", [], [(513, 128 * K)]),
    ("H5 wave edges (128k prefix)", "544 append on 128k", [], [(544, 128 * K)]),
    ("H5 wave edges (128k prefix)", "1056 append on 128k", [], [(1056, 128 * K)]),
    ("H5 wave edges (128k prefix)", "1057 append on 128k", [], [(1057, 128 * K)]),
    ("re-prefill (H4)", "8k chunk at prefix 0 + 3 x 128k decodes (cold start)", [128 * K] * 3, [(8 * K, 0)]),
    ("re-prefill (H4)", "8k chunk at prefix 64k + 3 x 128k decodes", [128 * K] * 3, [(8 * K, 64 * K)]),
    ("re-prefill (H4)", "2k chunk at prefix 0 + 3 x 128k decodes", [128 * K] * 3, [(2 * K, 0)]),
]


def row(group, name, dec, pre, cfg):
    q = np.array([1] * len(dec) + [x for x, _ in pre], dtype=np.int64)
    kv = np.array(list(dec) + [x + p for x, p in pre], dtype=np.int64)
    reqs = [(f"r{i}", int(a), int(b), int(b) + (4 * K if a == 1 else 0), 4 * K if a == 1 else 0)
            for i, (a, b) in enumerate(zip(q, kv))]
    f = compute_features(reqs, GEOM, cfg, full_cg=True)
    m = attn_step_model(q, kv, GEOM, cfg, full_cg=True)
    return dict(group=group, shape=name, T=int(q.sum()), regime_old=f.regime, x_ratio_old=round(f.x_ratio, 2),
                regime_new=m["regime2"] or f.regime, path=m["path"], splits=m["splits_default"],
                t_default_us=round(m["t_default"] * 1e6, 1), t_best_us=round(m["t_best"] * 1e6, 1),
                s_best=m["s_best"], t_sep_us=round(m["t_sep"] * 1e6, 1), x_cost=round(m["x_cost"], 2),
                sep_gain=round(m["sep_gain"], 2), pack_gain=round(m["pack_gain"], 2), sm_eff=round(m["sm_eff"], 2),
                pre_wave_eff=round(f.pre_wave_eff, 2), structural_x=m["structural_x"])


def to_md(rows):
    cols = ["shape", "T", "path", "splits", "regime_old", "x_ratio_old", "regime_new", "t_default_us", "t_best_us",
            "s_best", "x_cost", "t_sep_us", "sep_gain", "pack_gain", "sm_eff", "structural_x"]
    out, group = [], None
    for r in rows:
        if r["group"] != group:
            group = r["group"]
            out += ["", f"**{group}**", "", "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
        out.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    ap.add_argument("--json")
    a = ap.parse_args()
    cfg = RegimeConfig()
    rows = [row(*s, cfg) for s in SHAPES]
    md = to_md(rows)
    print(md)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(md)
    if a.json:
        Path(a.json).write_text("\n".join(json.dumps(r) for r in rows) + "\n")
