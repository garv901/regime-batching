"""Re-price recorded engine steps with the FA3 tile model and compare regime GPU-time shares, old vs new labels.

For each step with a request list: old label (telemetry `regime`, additive model), new label (attn_step_model regime2,
decode-only steps keep the old label), structural X flag, and per-step attention time over all layers for the
default split, the best split bound (C3-like), separate prefill/decode launches, and perfect packing.
Usage: python3 analysis/relabel_telemetry.py RUN_DIR [RUN_DIR ...] [--out FILE.md] [--jsonl FILE]
"""
import argparse, collections, glob, json, os, sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from regime_batching.features import ModelGeom, RegimeConfig, attn_step_model  # noqa: E402

CFG = RegimeConfig()


def load_steps(run_dir):
    meta, steps = None, []
    for f in sorted(glob.glob(os.path.join(run_dir, "telemetry", "steps_*.jsonl"))):
        for line in open(f):
            r = json.loads(line)
            if r.get("kind") == "meta":
                meta = r
            elif r.get("kind") == "step" and r.get("reqs") and r.get("gpu_ms") is not None \
                    and not r["reqs"][0][0].startswith("_warmup"):
                steps.append(r)
    return meta, steps


def price(args):
    step, geom_d, full_cg = args
    geom = ModelGeom(**geom_d)
    q = np.array([a[1] for a in step["reqs"]], dtype=np.int64)
    kv = np.array([a[2] for a in step["reqs"]], dtype=np.int64)
    if not (q > 1).any():  # decode-only: labels and kernel path unchanged, skip the tile model
        return dict(step=step["step"], gpu_ms=step["gpu_ms"], T=step["T"], regime_old=step["regime"],
                    regime_new=step["regime"], structural_x=False, att_def=None)
    m = attn_step_model(q, kv, geom, CFG, full_cg=full_cg)
    L = geom.num_layers
    return dict(step=step["step"], gpu_ms=step["gpu_ms"], T=step["T"], regime_old=step["regime"],
                regime_new=m["regime2"], structural_x=m["structural_x"], x_cost=m["x_cost"],
                att_def=L * m["t_default"] * 1e3, att_best=L * m["t_best"] * 1e3, att_sep=L * m["t_sep"] * 1e3,
                att_ideal=L * m["t_ideal"] * 1e3, pre_prefix=m["pre_prefix"])


def summarize(name, rows):
    tot = sum(r["gpu_ms"] for r in rows)
    out = [f"### {name}", "", f"{len(rows)} steps, {tot / 1e3:.1f} s GPU time.", ""]
    for key in ("regime_old", "regime_new"):
        share = collections.Counter()
        for r in rows:
            share[r[key]] += r["gpu_ms"]
        out.append(f"- {key}: " + ", ".join(f"{k} {v / tot:.1%}" for k, v in share.most_common()))
    xs = sum(r["gpu_ms"] for r in rows if r["structural_x"])
    xc = sum(r["gpu_ms"] for r in rows if r["regime_new"] == "X")
    priced = [r for r in rows if r["att_def"] is not None]
    loss = lambda k: sum(r["att_def"] - min(r["att_def"], r[k]) for r in priced)  # noqa: E731
    out += [f"- structural X: {xs / tot:.1%} of GPU time; X by cost (default >= 1.2x best split): {xc / tot:.1%}",
            f"- modelled attention time recoverable, as share of all GPU time: best split bound (C3) "
            f"{loss('att_best') / tot:.1%}, separate launches {loss('att_sep') / tot:.1%}, "
            f"perfect packing {loss('att_ideal') / tot:.1%}", ""]
    return "\n".join(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out")
    ap.add_argument("--jsonl")
    ap.add_argument("--procs", type=int, default=32)
    a = ap.parse_args()
    md, allrows = [], []
    with Pool(a.procs) as pool:
        for run in a.runs:
            meta, steps = load_steps(run)
            if not steps:
                print(f"{run}: no steps with request lists")
                continue
            g = meta["geom"]
            geom_d = dict(num_layers=g["num_layers"], num_q_heads=g["num_q_heads"], num_kv_heads=g["num_kv_heads"],
                          head_dim=g["head_dim"], weight_bytes_per_step=g["weight_bytes_per_step"],
                          kv_dtype_bytes=g.get("kv_dtype_bytes", 2))
            rows = pool.map(price, [(s, geom_d, meta.get("full_cg", True)) for s in steps], chunksize=64)
            md.append(summarize(run, rows))
            allrows += [dict(r, run=run) for r in rows]
            print(md[-1], flush=True)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text("\n".join(md))
    if a.jsonl:
        Path(a.jsonl).write_text("\n".join(json.dumps(r) for r in allrows) + "\n")
