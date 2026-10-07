"""Stage 1b regime census analysis.

Joins the in-engine census telemetry (experiments/regime_census.py) to its probe log, checks every probe step has
the designed shape (q new tokens, P cached), and prices each step against the alternatives that do the same work:
  cap2048      the same step with CUDA graphs captured up to 2,048 tokens (measured in-engine)
  split        best FA3 split bound for the attention (F7, kernel level), swapped into the default step
  separate     prefill and decodes as two attention launches (F7 fa3_phase_split)
  kernel       FlashInfer or Triton attention (F7)
  cap2048+att  capture 2048 and the best attention option of all
  realign      H5: an aligned chunk now and the remainder in a later step (in-engine probes, prefill alone)
Writes results/regimes_study/census_steps.jsonl (compact; telemetry itself is not committed) and
reports/regimes_study/figures/census_tables.md.
"""
import collections, glob, json, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
from census_shapes import scenarios  # noqa: E402
from regime_batching.features import GEOMS, RegimeConfig, attn_step_model  # noqa: E402

RUN = ROOT / "results/runs/s8_census"
F7 = ROOT / "results/microbench/attn_f7.jsonl"
PRED = ROOT / "results/regimes_study/census_pred_pre.jsonl"
GEOM = GEOMS["meta-llama/Llama-3.1-8B-Instruct"]
NL = GEOM.num_layers
BW, FLOPS = 3.35e12, 989e12  # H100 SXM spec peaks
KV_B = 2 * GEOM.num_kv_heads * GEOM.head_dim * 2  # bytes per token per layer (K+V, bf16)
LIN_PARAMS = 6.98e9  # Llama-3.1-8B linear-layer parameters excluding embeddings (lm_head counted per sequence)
SPLITS = ["fa3_cg", "fa3_s1", "fa3_s2", "fa3_s4", "fa3_s8", "fa3_s16"]
OTHER = ["fi_native", "triton"]
DEC = {n: d for n, d, _, _ in scenarios()}
C_TOKEN = 0.0200  # ms per token of non-attention compute (census_pred_pre fit)


def base(rid):
    return rid.rsplit("-", 1)[0]


def load_config(cfg):
    d = RUN / cfg
    probes, marks = {}, []
    for line in open(d / "probes.jsonl"):
        r = json.loads(line)
        if "rid" in r:
            probes[r["rid"]] = r
        elif r.get("kind") == "mark":
            marks.append(r)
    steps = []
    for f in sorted(glob.glob(str(d / "telemetry/steps_*.jsonl"))):
        for line in open(f):
            r = json.loads(line)
            if r.get("kind") == "step" and r.get("reqs") and not r["reqs"][0][0].startswith("_warmup"):
                steps.append(r)
    return probes, steps


def classify_steps(cfg, probes, steps):
    """-> list of dict rows: probe steps (single/pair), decode-only steps per scenario."""
    out, bad = [], collections.Counter()
    for s in steps:
        reqs = [(base(a[0]), a[1], a[2]) for a in s["reqs"]]
        pre = [r for r in reqs if r[1] > 1]
        dec = [r for r in reqs if r[1] == 1]
        mode = s["cg"]["mode"] if s.get("cg") else "?"
        common = dict(config=cfg, step=s["step"], T=s["T"], mode=mode, gpu_ms=s["gpu_ms"], cpu_ms=s["cpu_ms"],
                      regime_old=s["regime"], n_dec=len(dec), sumL_dec=sum(r[2] for r in dec),
                      maxL_dec=max([r[2] for r in dec], default=0))
        if not pre:
            scen = {r[0].split("|")[1] for r in dec if r[0].startswith("dec|")}
            if len(scen) == 1 and all(r[0].startswith("dec|") for r in dec):
                sc = scen.pop()
                if len(dec) == len(DEC[sc]):
                    out.append(dict(common, kind="dec", scen=sc, P=None, q=0))
            continue
        tags = [probes.get(r[0]) for r in pre]
        if not all(t and t["kind"] in ("probe", "pair") for t in tags):
            continue  # setup steps (decode build-up, prefix warm-up)
        ok = all(r[1] == t["q"] and r[2] == t["P"] + t["q"] for r, t in zip(pre, tags))
        scen = tags[0]["scen"]
        ok &= len(dec) == len(DEC[scen]) and all(r[0].startswith(f"dec|{scen}|") for r in dec)
        if tags[0]["kind"] == "pair":
            ok &= len(pre) == 2
            q, pair = sum(t["q"] for t in tags), tags[0]["pair"]
        else:
            ok &= len(pre) == 1
            q, pair = tags[0]["q"], None
        if not ok:
            bad[(scen, tags[0]["kind"])] += 1
            continue
        out.append(dict(common, kind=tags[0]["kind"], scen=scen, P=tags[0]["P"], q=q, pair=pair, rep=tags[0]["rep"]))
    return out, bad


def shape_key(r):
    return (r["scen"], r["kind"], r["P"], r["q"], tuple(r["pair"]) if r.get("pair") else None)


def reqs_of(scen, kind, P, q, pair):
    dec = [(1, L) for L in DEC[scen]]
    if kind == "dec":
        return dec
    if kind == "pair":
        return dec + [(pair[0], P + pair[0]), (pair[1], P + pair[1])]
    return dec + [(q, P + q)]


def f7_name(scen, kind, P, q, pair):
    if kind == "dec":
        return f"c_{scen}_dec"
    if kind == "pair":
        return f"c_{scen}_pair{pair[0]}+{pair[1]}_P{P}"
    return f"c_{scen}_P{P}_q{q}"


def roofline_ms(reqs):
    kv = sum(b for _, b in reqs)
    nseq = len(reqs)
    T = sum(a for a, _ in reqs)
    mem = (GEOM.weight_bytes_per_step + NL * kv * KV_B) / BW
    att_fl = sum(4 * GEOM.num_q_heads * GEOM.head_dim * a * (b - a + a / 2) for a, b in reqs) * NL
    comp = (2 * LIN_PARAMS * T + 2 * 525e6 * nseq + att_fl) / FLOPS
    return 1e3 * max(mem, comp), "memory" if mem >= comp else "compute"


def med(v):
    return float(np.median(v)) if len(v) else float("nan")


def main():
    import argparse
    global RUN, F7
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=str(RUN))
    ap.add_argument("--f7", default=str(F7))
    ap.add_argument("--out-dir", default=str(ROOT))
    args = ap.parse_args()
    RUN, F7, OUTR = Path(args.run), Path(args.f7), Path(args.out_dir)
    rows, bads = [], {}
    for cfg in ["default", "cap2048"]:
        if not (RUN / cfg / "probes.jsonl").exists():
            print("missing", cfg)
            continue
        probes, steps = load_config(cfg)
        r, bad = classify_steps(cfg, probes, steps)
        rows += r
        bads[cfg] = bad
    (OUTR / "results/regimes_study").mkdir(parents=True, exist_ok=True)
    with open(OUTR / "results/regimes_study/census_steps.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps({k: v for k, v in r.items() if v is not None}) + "\n")

    # medians per (config, shape)
    by = collections.defaultdict(list)
    for r in rows:
        by[(r["config"], shape_key(r))].append(r)
    M = {k: dict(gpu=med([x["gpu_ms"] for x in v]), cpu=med([x["cpu_ms"] for x in v]), n=len(v),
                 spread=(max(x["gpu_ms"] for x in v) / min(x["gpu_ms"] for x in v) - 1) if v else 0,
                 mode=collections.Counter(x["mode"] for x in v).most_common(1)[0][0], T=v[0]["T"],
                 regime_old=collections.Counter(x["regime_old"] for x in v).most_common(1)[0][0])
         for k, v in by.items()}
    att = collections.defaultdict(dict)
    if F7.exists():
        for line in open(F7):
            r = json.loads(line)
            if "ERROR" not in r.get("notes", "") and r.get("time_us_median"):
                att[r["spec"]][r["config"]] = r["time_us_median"] / 1e3  # ms per layer
    pred = collections.defaultdict(dict)
    for line in open(PRED):
        r = json.loads(line)
        if r["kind"] == "step":
            key = (r["scen"], r["probe"], r["P"], r["q"])
            pred[r["config"]][key] = r

    shapes = sorted({k[1] for k in M}, key=lambda s: (list(DEC).index(s[0]),
                                                      s[1] != "dec", s[2] or 0, s[3]))
    table, groups = [], collections.defaultdict(list)
    for sk in shapes:
        scen, kind, P, q, pair = sk
        d, c = M.get(("default", sk)), M.get(("cap2048", sk))
        if not d:
            continue
        reqs = reqs_of(scen, kind, P, q, list(pair) if pair else None)
        T = sum(a for a, _ in reqs)
        rl, bound_kind = roofline_ms(reqs)
        a = att.get(f7_name(scen, kind, P, q, list(pair) if pair else None), {})
        alts = {}
        if c:
            alts["cap2048"] = c["gpu"]
        def_att_cfg = "fa3_cg" if T <= 512 else "fa3_auto"
        cap_att_cfg = "fa3_cg" if T <= 2048 else "fa3_auto"
        if a.get(def_att_cfg):
            best_split = min([a[x] for x in SPLITS + ["fa3_auto"] if x in a])
            alts["split"] = d["gpu"] - NL * (a[def_att_cfg] - best_split)
            if "fa3_phase_split" in a:
                alts["separate"] = d["gpu"] - NL * (a[def_att_cfg] - a["fa3_phase_split"])
            oth = [a[x] for x in OTHER if x in a]
            if oth:
                alts["kernel"] = d["gpu"] - NL * (a[def_att_cfg] - min(oth))
            best_all = min(v for k_, v in a.items())
            if c and a.get(cap_att_cfg):
                alts["cap2048+att"] = c["gpu"] - NL * (a[cap_att_cfg] - best_all)
        realign_marginal = None
        if scen == "none" and kind == "probe" and q in (513, 544, 1057):
            al, rem = (512, q - 512) if q < 1024 else (1056, q - 1056)
            remq = min([x for x in (16, 32, 64) if x >= rem], default=None)
            m_al, m_rem = M.get(("default", (scen, "probe", P, al, None))), M.get(("default", (scen, "probe", P, remq, None)))
            if m_al and m_rem:  # conservative: the remainder runs as its own step
                alts["realign"] = m_al["gpu"] + m_rem["gpu"]
            ar = att.get(f"c_none_P{P}_q{remq}", {}).get("fa3_cg")
            if m_al and ar:  # marginal: the remainder rides in a step that runs anyway (attention + per-token GEMM)
                realign_marginal = m_al["gpu"] + NL * ar + C_TOKEN * remq
        best_k = min(alts, key=alts.get) if alts else None
        best = min(alts.values()) if alts else d["gpu"]
        loss = d["gpu"] / best if best < d["gpu"] else 1.0
        src = best_k if loss >= 1.05 else "none"
        p = None if (kind == "pair" and q == 1056) else pred.get("default", {}).get((scen, kind, P, q))  # 2 pairs sum to 1056
        row = dict(scen=scen, kind=kind, P=P, q=q, pair=pair, T=T, mode=d["mode"], regime_old=d["regime_old"],
                   gpu_def=d["gpu"], cpu_def=d["cpu"], spread=d["spread"], n=d["n"],
                   gpu_cap=c["gpu"] if c else None, mode_cap=c["mode"] if c else None,
                   att_def=NL * a[def_att_cfg] if a.get(def_att_cfg) else None, roofline=rl, bound=bound_kind,
                   eff=rl / d["gpu"], alts=alts, best=best, loss=loss, src=src,
                   pred_def=p["gpu_ms"] if p else None,
                   realign_marginal=realign_marginal)
        if kind != "dec":
            q_ = np.array([x for x, _ in reqs]); kv_ = np.array([y for _, y in reqs])
            row["regime2"] = attn_step_model(q_, kv_, GEOM, RegimeConfig(), full_cg=True)["regime2"]
        table.append(row)
        groups[src].append(row)

    md = ["# Stage 1b regime census tables", "",
          "Whole step = one engine step of Llama-3.1-8B on one H100 (GPU time from CUDA events around the step, "
          "median of 3 repeats). Attention alternatives are swapped in from the F7 kernel census (32 layers). "
          "roofline = max(weights + KV bytes at 3.35 TB/s, FLOPs at 989 TFLOPS); eff = roofline / measured.", ""]
    md += ["## Data checks", ""]
    for cfg, bad in bads.items():
        md.append(f"- {cfg}: {sum(1 for r in rows if r['config'] == cfg)} probe/decode steps kept; "
                  f"rejected (shape mismatch): {dict(bad) if bad else 'none'}")
    sp = [r["spread"] for r in table if r["n"] >= 2]
    md.append(f"- repeat spread (max/min - 1, default config): median {med(sp):.1%}, "
              f"share of shapes within 5%: {np.mean([x <= 0.05 for x in sp]):.0%}")
    pe = collections.defaultdict(list)
    for r in table:
        if r["pred_def"]:
            pe[r["mode"]].append(abs(r["pred_def"] / r["gpu_def"] - 1))
    md.append("- pre-registered whole-step model, mean abs error (default config): " +
              ", ".join(f"{k} {np.mean(v):.1%} (n={len(v)})" for k, v in pe.items()))
    md += ["", "## Every census step", "",
           "| scenario | kind | P | q | T | mode | regime | default ms | cap2048 ms | attention ms | roofline ms (bound) | eff | best alternative | loss |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in table:
        md.append(f"| {r['scen']} | {r['kind'] if not r['pair'] else 'pair ' + '+'.join(map(str, r['pair']))} | "
                  f"{r['P'] if r['P'] is not None else ''} | {r['q']} | {r['T']} | {r['mode']} | "
                  f"{r.get('regime2') or r['regime_old']} | {r['gpu_def']:.1f} | "
                  f"{(f'{r['gpu_cap']:.1f}' if r['gpu_cap'] else '')} | {(f'{r['att_def']:.1f}' if r['att_def'] else '')} | "
                  f"{r['roofline']:.1f} ({r['bound'][0]}) | {r['eff']:.0%} | {r['src']} | {r['loss']:.2f} |")
    md += ["", "## Steps grouped by the alternative that recovers the most", "",
           "| best alternative | steps | median loss | max loss | scenarios |", "|---|---|---|---|---|"]
    for k, v in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        md.append(f"| {k} | {len(v)} | {med([x['loss'] for x in v]):.2f} | {max(x['loss'] for x in v):.2f} | "
                  f"{', '.join(sorted({x['scen'] for x in v}))} |")
    out = OUTR / "reports/regimes_study/figures/census_tables.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(md) + "\n")
    with open(OUTR / "results/regimes_study/census_priced.jsonl", "w") as f:
        for r in table:
            f.write(json.dumps({k: (list(v) if isinstance(v, tuple) else v) for k, v in r.items()}) + "\n")
    print("\n".join(md[:12]))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
