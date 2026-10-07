"""Stage 1b pre-registration: predict every census step before the GPU jobs run.

Whole step (in-engine): gpu_ms = attention (FA3 tile model, 32 layers) + non-attention cost by graph mode:
  FULL (decode-only, T <= capture)        att + max(w, c*T) + dF
  PIECEWISE (prefill, T <= capture)        att + max(w, c*T) + dP
  NONE (eager, T > capture)                max(L, att + max(w, c*T) + dN)    L = CPU launch floor
w, c, dF, dP, L, dN are fit on the earlier Llama-3.1-8B telemetry (job 1040), so the census is out of sample.
Kernel level (F7): FA3 tile model per split config, as in Stage 1.

Usage: python3 analysis/census_predict.py  ->  results/regimes_study/census_pred_pre.jsonl (+ sha256 printed)
"""
import glob, hashlib, json, random, sys, time
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
from census_shapes import kernel_specs, scenarios  # noqa: E402
from regime_batching.features import GEOMS, RegimeConfig, attn_step_model, fa3_static_splits, fa3_tile_time  # noqa: E402

GEOM = GEOMS["meta-llama/Llama-3.1-8B-Instruct"]
L_LAYERS = GEOM.num_layers
FIT_RUNS = [("xh_B0", 512), ("xh_cap2048", 2048), ("bu_B0", 512), ("xh_shape", 512)]
PNAMES = ["w", "c", "dF", "dP", "L", "dN"]
FORCED = {"fa3_s1": 1, "fa3_s2": 2, "fa3_s4": 4, "fa3_s8": 8, "fa3_s16": 16}


def cfg_for(cap):
    c = RegimeConfig()
    c.cg_capture_max = cap
    return c


def att_ms(reqs, cap):
    q = np.array([a for a, _ in reqs]); kv = np.array([b for _, b in reqs])
    return L_LAYERS * attn_step_model(q, kv, GEOM, cfg_for(cap), full_cg=True)["t_default"] * 1e3


def step_model(p, mode, T, att):
    w, c, dF, dP, L, dN = p
    kern = np.maximum(w, c * T)
    return np.where(mode == "FULL", att + kern + dF,
                    np.where(mode == "PIECEWISE", att + kern + dP, np.maximum(L, att + kern + dN)))


def mode_of(T, has_prefill, cap):
    if T > cap:
        return "NONE"
    return "PIECEWISE" if has_prefill else "FULL"


def fit():
    rows = []
    for run, cap in FIT_RUNS:
        rr = []
        for f in sorted(glob.glob(str(ROOT / f"results/runs/s5a_llama_1040/{run}/telemetry/steps_*.jsonl"))):
            for line in open(f):
                r = json.loads(line)
                if r.get("kind") != "step" or not r.get("reqs") or r["reqs"][0][0].startswith("_warmup") \
                        or not r.get("cg"):
                    continue
                rr.append(r)
        random.Random(0).shuffle(rr)
        for r in rr[:1500]:
            reqs = [(a[1], a[2]) for a in r["reqs"]]
            rows.append((r["cg"]["mode"], r["T"], att_ms(reqs, cap), r["gpu_ms"]))
    mode = np.array([r[0] for r in rows]); T = np.array([r[1] for r in rows], float)
    att = np.array([r[2] for r in rows]); y = np.array([r[3] for r in rows])
    res = least_squares(lambda p: step_model(p, mode, T, att) / y - 1, x0=[6, 0.018, 0.2, 2, 15, 1],
                        bounds=([0, 0, -5, -5, 0, -5], [20, 0.1, 10, 20, 60, 20]))
    err = np.abs(step_model(res.x, mode, T, att) / y - 1)
    fit_err = {m: float(err[mode == m].mean()) for m in ("FULL", "PIECEWISE", "NONE")}
    return dict(zip(PNAMES, res.x.tolist())), fit_err, len(rows)


def main():
    p, fit_err, n = fit()
    pv = [p[k] for k in PNAMES]
    out = [dict(kind="fit", params=p, fit_mean_abs_err=fit_err, n_steps=n, runs=FIT_RUNS,
                note="fit on job 1040 telemetry, node load average ~650 then")]
    for cap, config in [(512, "default"), (2048, "cap2048")]:
        for scen, dec, probes, pairs in scenarios():
            d = [(1, L) for L in dec]
            items = [("dec", None, 0, d)] if d else []
            items += [("probe", P, q, d + [(q, P + q)]) for P, q in probes]
            items += [("pair", 128 * 1024, q1 + q2, d + [(q1, 128 * 1024 + q1), (q2, 128 * 1024 + q2)])
                      for q1, q2 in pairs]
            for kind, P, q, reqs in items:
                T = sum(a for a, _ in reqs)
                mode = mode_of(T, any(a > 1 for a, _ in reqs), cap)
                att = att_ms(reqs, cap)
                g = float(step_model(pv, np.array([mode]), np.array([float(T)]), np.array([att]))[0])
                out.append(dict(kind="step", config=config, scen=scen, probe=kind, P=P, q=q, T=T, mode=mode,
                                att_ms=round(att, 3), gpu_ms=round(g, 3)))
    for name, reqs, meta in kernel_specs():
        q, kv = (np.array(x) for x in zip(*reqs))
        for c in ["fa3_auto", "fa3_cg", *FORCED]:
            if c == "fa3_auto":
                s = fa3_static_splits(int(q.max()), int(kv.max()), int(q.sum()), 4, 8, 128, full_cg=False)[0]
            elif c == "fa3_cg":
                s = 32
            else:
                s = FORCED[c]
            out.append(dict(kind="kernel", spec=name, config=c, pred_us=round(fa3_tile_time(q, kv, s, 4, 8)[0] * 1e6, 2)))
    path = ROOT / "results/regimes_study/census_pred_pre.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(json.dumps(r) for r in out) + "\n"
    path.write_text(text)
    print("fit", {k: round(v, 4) for k, v in p.items()}, "fit err", {k: round(v, 3) for k, v in fit_err.items()})
    print(f"wrote {path} rows={len(out)} sha256={hashlib.sha256(text.encode()).hexdigest()} "
          f"at {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}")


if __name__ == "__main__":
    main()
