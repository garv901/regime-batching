"""Check the FA3 tile model (regime_batching.features.fa3_tile_time) against attention microbench rows.

Usage: python3 microbench/validate_tile_model.py [results/microbench/attn_sweep.jsonl ...] [--fit] [--out FILE]
Each row's batch is rebuilt from its spec name (F1-F5) or its F6 fields; replay rows carry no batch and are skipped.
"""
import argparse, collections, itertools, json, random, re, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import regime_batching.features as F  # noqa: E402

G, HKV, HD = 4, 8, 128
FORCED = {"fa3_s1": 1, "fa3_s2": 2, "fa3_s4": 4, "fa3_s8": 8, "fa3_s16": 16}


def reqs_of(row):
    fam, spec = row["family"], row["spec"]
    if fam == "F6_append":
        dec = [(1, row["L_long"])] * row["n_long"]
        return dec + ([(row["q"], row["P"] + row["q"])] if row.get("P") is not None else [])
    if fam == "F4_steady":
        m = re.match(r"rand_B(\d+)_oL(\d+)k", spec)
        B, oL = int(m.group(1)), int(m.group(2)) * 1024
        rng = random.Random(1000 * oL + B)
        return [(1, rng.randint(512, 512 + oL)) for _ in range(B)]
    out = []
    for n, q, kv in re.findall(r"(\d+)q(\d+)s(\d+)", spec):
        out += [(int(q), int(kv))] * int(n)
    dec = [r for r in out if r[0] == 1]
    return dec + [r for r in out if r[0] > 1]  # decodes first (vLLM order)


REPLAY_GLOBS = ("results/runs/s3a_1017/*/telemetry/steps_*.jsonl,results/runs/s6_1390/C_fa3*/telemetry/steps_*.jsonl,"
                "results/runs/s5a_llama_1040/xh_*/telemetry/steps_*.jsonl")


def replay_reqs(paths=REPLAY_GLOBS, n=40):
    """Rebuild attn_sweep.py --replay-steps batches by name (same selection, seed 0), since rows carry no batch."""
    import glob
    files = []
    for p in paths.split(","):
        files += sorted(glob.glob(p))
    rng = random.Random(0)
    pools = {}
    for f in files:
        for line in open(f):
            if '"kind":"step"' not in line[:20] and '"kind": "step"' not in line[:20]:
                continue
            r = json.loads(line)
            if r["T"] <= 512 or not r.get("reqs") or r["maxL"] < 16384 or r["reqs"][0][0].startswith("_warmup"):
                continue
            pools.setdefault((r["regime"] == "X", f), []).append(r)
    chosen, seen = [], set()
    for want_x in (True, False):
        lists = [v for (x, f), v in sorted(pools.items(), key=lambda kv: kv[0][1]) if x == want_x]
        for v in lists:
            rng.shuffle(v)
        i = 0
        while len(chosen) < n and any(i < len(v) for v in lists):
            for v in lists:
                if i < len(v) and len(chosen) < n:
                    r = v[i]
                    sig = tuple(sorted((a[1], a[2]) for a in r["reqs"]))
                    if sig not in seen:
                        seen.add(sig)
                        chosen.append(r)
            i += 1
    out = {}
    for k, r in enumerate(chosen):
        dec = [(a[1], a[2]) for a in r["reqs"] if a[1] == 1]
        pre = [(a[1], a[2]) for a in r["reqs"] if a[1] > 1]
        out[f"replay{k:02d}_s{r['step']}"] = dec + pre
    return out


def splits_for(row, reqs):
    c = row["config"]
    if c in FORCED:
        return FORCED[c]
    if c == "fa3_auto":
        q, kv = zip(*reqs)
        return F.fa3_static_splits(max(q), max(kv), sum(q), G, HKV, HD, full_cg=False)[0]
    return None


def predict(reqs, s):
    q, kv = (np.array(x) for x in zip(*reqs))
    return F.fa3_tile_time(q, kv, s, G, HKV, HD)[0] * 1e6


def load(paths):
    rows, replay = [], None
    for p in paths:
        if not Path(p).exists():
            print(f"missing {p}, skipped")
            continue
        for line in open(p):
            r = json.loads(line)
            if "ERROR" in r.get("notes", "") or r["config"] not in (*FORCED, "fa3_auto"):
                continue
            if r["family"] == "replay":
                replay = replay if replay is not None else replay_reqs()
                reqs = replay.get(r["spec"])
            else:
                reqs = reqs_of(r)
            if reqs:
                rows.append((r, reqs, splits_for(r, reqs)))
    return rows


def report(rows, out):
    by = collections.defaultdict(list)
    for r, reqs, s in rows:
        p = predict(reqs, s)
        e = p / r["time_us_median"] - 1
        by[(r["family"], "mixed" if any(q > 1 for q, _ in reqs) else "decode", r["config"])].append(e)
        by[(r["family"], "all", "all")].append(e)
    lines = ["| family | batch | config | n | mean abs err | max abs err | mean signed |", "|---|---|---|---|---|---|---|"]
    for k in sorted(by):
        e = np.abs(by[k])
        lines.append(f"| {k[0]} | {k[1]} | {k[2]} | {len(e)} | {e.mean():.1%} | {e.max():.1%} | {np.mean(by[k]):+.1%} |")
    txt = "\n".join(lines)
    print(txt)
    if out:
        Path(out).write_text(txt + "\n")


PARAMS = {"decode": ["BW_CTA_DEC", "BW_HBM_ATT", "DEC_TILE_OVH", "DEC_LAUNCH_OVH", "COMBINE_OVH"],
          "mixed": ["F_SM_TILE", "BW_CTA_MIXED", "TILE_OVH", "LAUNCH_OVH"]}


def fit(rows, kind):
    """Refit one launch kind's constants (log-space Nelder-Mead); each family weighs the same."""
    from scipy.optimize import minimize
    rows = [x for x in rows if any(q > 1 for q, _ in x[1]) == (kind == "mixed")]
    params = PARAMS[kind]
    t = np.array([r["time_us_median"] for r, _, _ in rows])
    grp = collections.defaultdict(list)
    for i, (r, reqs, _) in enumerate(rows):
        grp[(r["family"], any(q > 1 for q, _ in reqs))].append(i)
    w = np.zeros(len(rows))
    for idx in grp.values():
        w[idx] = 1.0 / (len(idx) * len(grp))

    def loss(x):
        for k, v in zip(params, np.exp(x)):
            setattr(F, k, v)
        p = np.array([predict(reqs, s) for _, reqs, s in rows])
        return float((w * np.abs(p / t - 1)).sum())

    x0 = np.log([getattr(F, k) for k in params])
    res = minimize(loss, x0, method="Nelder-Mead", options=dict(maxfev=2500, xatol=1e-3, fatol=1e-5))
    res = minimize(loss, res.x, method="Nelder-Mead", options=dict(maxfev=1500, xatol=1e-3, fatol=1e-5))
    loss(res.x)
    print(f"{kind} refit on {len(rows)} rows, weighted mean abs err {res.fun:.1%}:")
    for k in params:
        print(f"  {k} = {getattr(F, k):.3g}")


def predict_f6(out):
    """Write the model's F6 predictions (same grid as attn_sweep.py F6_append), before the F6 data exists."""
    qs = [417, 448, 512, 513, 544, 640, 768, 1024, 1056, 1057, 1088, 1536, 1584, 1585, 2048, 2112, 2113, 8192]
    decs = [("none", 0, 0), ("1x32k", 1, 32768), ("3x32k", 3, 32768), ("1x128k", 1, 131072), ("3x128k", 3, 131072)]
    with open(out, "w") as f:
        for P in [0, 8192, 32768, 131072]:
            for q in qs:
                for dn, nl, Ll in decs:
                    row = dict(family="F6_append", P=P, q=q, n_long=nl, L_long=Ll, dec_set=dn, spec=f"app_P{P}_q{q}_{dn}")
                    reqs = reqs_of(row)
                    for c in (*FORCED, "fa3_auto"):
                        s = splits_for(dict(config=c), reqs)
                        f.write(json.dumps(dict(row, config=c, splits=s, pred_us=round(predict(reqs, s), 2))) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*", default=["results/microbench/attn_sweep.jsonl", "results/microbench/attn_f6.jsonl"])
    ap.add_argument("--fit", nargs="*", choices=["decode", "mixed"], help="refit these launch kinds, in order")
    ap.add_argument("--out")
    ap.add_argument("--predict-f6", metavar="FILE")
    a = ap.parse_args()
    if a.predict_f6:
        predict_f6(a.predict_f6)
        sys.exit(0)
    rows = load(a.files)
    for kind in a.fit or []:
        fit(rows, kind)
    report(rows, a.out)
