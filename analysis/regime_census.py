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
        cap_differs = c and 512 < T <= 2048  # elsewhere capture 2048 runs the same path (measured: ratio 1.000)
        if cap_differs:
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
            if cap_differs and a.get(cap_att_cfg):
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

    # where the time goes: attention (F7, 32 layers), eager excess (default - cap2048, net of the attention change),
    # the rest is GEMMs and other non-attention kernels
    for r in table:
        a = att.get(f7_name(r["scen"], r["kind"], r["P"], r["q"], list(r["pair"]) if r["pair"] else None), {})
        eager = None
        if r["mode"] == "NONE" and r["mode_cap"] and r["mode_cap"] != "NONE":
            d_att = NL * (a["fa3_auto"] - a["fa3_cg"]) if a.get("fa3_auto") and a.get("fa3_cg") else 0.0
            eager = max(0.0, r["gpu_def"] - r["gpu_cap"] - d_att)
        r["eager_ms"] = eager
        r["nonatt_ms"] = r["gpu_def"] - r["att_def"] - (eager or 0) if r["att_def"] is not None else None
        r["label"] = r.get("regime2") or r["regime_old"]
        if r["label"] == "A" and max(DEC[r["scen"]], default=0) >= RegimeConfig().x_struct_min_L:
            r["label"] = "A+L"  # long-prefix append sharing the launch with a long decode (census refinement)
        r["size"] = "<=512" if r["T"] <= 512 else ("513-2048" if r["T"] <= 2048 else ">2048")
        r["kernel_move"] = (NL * (a["fa3_cg"] - min(a.values())) / r["gpu_def"]
                            if r["kind"] == "dec" and a.get("fa3_cg") else None)
    regimes = collections.defaultdict(list)
    for r in table:
        regimes[(r["label"], r["size"])].append(r)
    md_reg = ["", "## Regimes: where the time goes and what recovers it", "",
              "Shares are medians over the shapes in the row, of the default-config step. eager = default minus "
              "capture-2048 time for the same step, net of the attention change (measured only for 513-2048 "
              "tokens). fix: the alternative that recovers the most, by count of shapes with loss >= 1.05.", "",
              "| regime | step tokens | shapes | scenarios | attention | eager | non-attention | eff (median) | "
              "loss median (max) | fix |", "|---|---|---|---|---|---|---|---|---|---|"]
    for (lab, size), v in sorted(regimes.items(), key=lambda kv: (kv[0][0], ["<=512", "513-2048", ">2048"].index(kv[0][1]))):
        sh = lambda key: med([x[key] / x["gpu_def"] for x in v if x.get(key) is not None])
        fixes = collections.Counter(x["src"] for x in v if x["src"] != "none")
        fx = ", ".join(f"{k} ({n})" for k, n in fixes.most_common()) or "none"
        f_att, f_eag, f_non = sh("att_def"), sh("eager_ms"), sh("nonatt_ms")
        md_reg.append(f"| {lab} | {size} | {len(v)} | {', '.join(sorted({x['scen'] for x in v}))} | "
                      f"{'' if np.isnan(f_att) else f'{f_att:.0%}'} | {'' if np.isnan(f_eag) else f'{f_eag:.0%}'} | "
                      f"{'' if np.isnan(f_non) else f'{f_non:.0%}'} | {med([x['eff'] for x in v]):.0%} | "
                      f"{med([x['loss'] for x in v]):.2f} ({max(x['loss'] for x in v):.2f}) | {fx} |")

    bk = ["<1.05", "1.05-1.15", "1.15-1.5", ">=1.5"]
    bucket = lambda x: bk[0] if x < 1.05 else bk[1] if x < 1.15 else bk[2] if x < 1.5 else bk[3]
    md_reg += ["", "Regime label vs measured loss (shapes):", "", "| label | " + " | ".join(bk) + " |",
               "|---|" + "---|" * len(bk)]
    for lab in sorted({r["label"] for r in table}):
        cnt = collections.Counter(bucket(r["loss"]) for r in table if r["label"] == lab)
        md_reg.append(f"| {lab} | " + " | ".join(str(cnt.get(b, 0)) for b in bk) + " |")

    # decode-only: cost per token is the lever
    md_dec = ["", "## Decode-only steps", "",
              "| decode set | B | KV tokens | step ms | ms per token | roofline ms | eff | attention share | "
              "best attention kernel vs FA3 (whole step) |", "|---|---|---|---|---|---|---|---|---|"]
    for r in table:
        if r["kind"] != "dec":
            continue
        B, kv = len(DEC[r["scen"]]), sum(DEC[r["scen"]])
        md_dec.append(f"| {r['scen']} | {B} | {kv:,} | {r['gpu_def']:.1f} | {r['gpu_def'] / B:.3f} | {r['roofline']:.1f} | "
                      f"{r['eff']:.0%} | {'' if r['att_def'] is None else f'{r['att_def'] / r['gpu_def']:.0%}'} | "
                      f"{'' if r['kernel_move'] is None else f'-{r['kernel_move']:.1%}'} |")
    md_dec += ["", "Long-context decode-only attention per layer (kernel level, no in-engine counterpart):", "",
               "| spec | FA3 (graph, bound 32) ms | best config | best ms | FA3 / best |", "|---|---|---|---|---|"]
    for spec in sorted((s for s in att if s.startswith("c_dec_")), key=lambda s: (int(s.split("_")[2].split("x")[1][:-1]), int(s.split("_")[2].split("x")[0]))):
        a = att[spec]
        if a.get("fa3_cg"):
            bk = min(a, key=a.get)
            md_dec.append(f"| {spec} | {a['fa3_cg']:.3f} | {bk} | {a[bk]:.3f} | {a['fa3_cg'] / a[bk]:.2f} |")

    # H5, revised criterion: same tokens, total attention time, measured (F7); plus the in-engine whole step
    LONGP = max(P for _, _, pr, _ in scenarios() for P, _ in pr)
    md_h5 = ["", "## H5: the aligned chunk now and the remainder later (revised criterion, attention time, measured)", "",
             "now = the chunk in one step (default config: split bound 32 if the step is <= 512 tokens, else FA3's own "
             "choice); later = the aligned chunk plus the remainder (rounded up to the measured grid) in a step of <= 512 "
             "tokens. SM efficiency from the tile model. Whole step: in-engine default config, remainder as its own step "
             "(conservative) or riding in a step that runs anyway (marginal).", "",
             "| P | q | aligned + remainder | SM eff (now) | attention now ms/layer | attention later ms/layer | gain | "
             "whole step now ms | realign (own step) ms | realign (marginal) ms |", "|---|---|---|---|---|---|---|---|---|---|"]
    h5_rows = []
    for P in [8192, 32768, LONGP]:
        for q, al in [(513, 512), (544, 512), (1057, 1056)]:
            remq = min(x for x in (16, 32, 64, 96, 128) if x >= q - al)
            a_q, a_al, a_rm = (att.get(f"c_none_P{P}_q{x}", {}) for x in (q, al, remq))
            c_q, c_al = ("fa3_cg" if x <= 512 else "fa3_auto" for x in (q, al))
            if not (a_q.get(c_q) and a_al.get(c_al) and a_rm.get("fa3_cg")):
                continue
            now, later = a_q[c_q], a_al[c_al] + a_rm["fa3_cg"]
            eff = attn_step_model(np.array([q]), np.array([P + q]), GEOM, RegimeConfig(), full_cg=True)["sm_eff"]
            tr = next((x for x in table if (x["scen"], x["kind"], x["P"], x["q"]) == ("none", "probe", P, q)), None)
            h5_rows.append(dict(P=P, q=q, al=al, remq=remq, eff=eff, now=now, later=later, gain=now / later))
            md_h5.append(f"| {P} | {q} | {al} + {remq} | {eff:.0%} | {now:.3f} | {later:.3f} | {now / later:.2f} | "
                         f"{tr['gpu_def']:.1f} | {tr['alts'].get('realign', float('nan')):.1f} | "
                         f"{tr['realign_marginal'] or float('nan'):.1f} |" if tr else
                         f"| {P} | {q} | {al} + {remq} | {eff:.0%} | {now:.3f} | {later:.3f} | {now / later:.2f} | | | |")
    md_h5 += ["", "Other work filling the idle wave: two appends in one launch vs each in its own launch (kernel level).", "",
              "| pair | P | tiles | together ms/layer | apart ms/layer | apart / together |", "|---|---|---|---|---|---|"]
    for P in [32768, LONGP]:
        for q1, q2 in [(512, 512), (544, 512), (513, 513), (528, 528), (1056, 32)]:
            a_p = att.get(f"c_none_pair{q1}+{q2}_P{P}", {})
            parts = [att.get(f"c_none_P{P}_q{x}", {}).get("fa3_cg" if x <= 512 else "fa3_auto") for x in (q1, q2)]
            if a_p.get("fa3_auto") and all(parts):
                tiles = 8 * (-(-q1 // 32) + -(-q2 // 32))
                md_h5.append(f"| {q1}+{q2} | {P} | {tiles} | {a_p['fa3_auto']:.3f} | {sum(parts):.3f} | "
                             f"{sum(parts) / a_p['fa3_auto']:.2f} |")

    # the four qualitative predictions, checked
    md_pc = ["", "## Qualitative predictions, checked", ""]
    km = [r["kernel_move"] for r in table if r["kernel_move"] is not None]
    if km:
        md_pc.append(f"1. Decode-only: the best attention kernel moves the whole step by at most {max(km):.1%} "
                     f"(median {med(km):.1%}, {len(km)} decode sets); predicted <= 5%.")
    ec = [(r["gpu_def"] / r["gpu_cap"], r) for r in table if r["size"] == "513-2048" and r["gpu_cap"]]
    if ec:
        md_pc.append(f"2. Steps of 513-2048 tokens: capture 2048 is faster by {min(e for e, _ in ec):.2f}-"
                     f"{max(e for e, _ in ec):.2f}x (median {med([e for e, _ in ec]):.2f}x, {len(ec)} shapes); "
                     f"eager excess median {med([r['eager_ms'] for _, r in ec if r['eager_ms'] is not None]):.1f} ms.")
    xs = [r for r in table if r["label"] == "X"]
    if xs:
        xc = [r for r in xs if r["size"] != ">2048" and r["gpu_cap"]]
        md_pc.append(f"3. X steps: {len(xs)} shapes, loss median {med([r['loss'] for r in xs]):.2f}; "
                     + (f"of the {len(xc)} at <= 2,048 tokens, capture 2048 alone reaches the best alternative "
                        f"within 5% for {sum(r['gpu_cap'] <= 1.05 * r['best'] for r in xc)}; best-alternative kinds: "
                        f"{dict(collections.Counter(r['src'] for r in xs))}." if xc else ""))
    rem = [att.get(f"c_none_P{LONGP}_q{x}", {}).get("fa3_cg") for x in (16, 32, 64)]
    if all(rem):
        md_pc.append(f"4. Remainder on a {LONGP:,}-token prefix in a step of <= 512 tokens: 16 / 32 / 64 tokens cost "
                     f"{rem[0]:.3f} / {rem[1]:.3f} / {rem[2]:.3f} ms of attention per layer (predicted <= 0.25 ms); "
                     + (f"H5 gain (attention, same tokens) {min(h['gain'] for h in h5_rows):.2f}-"
                        f"{max(h['gain'] for h in h5_rows):.2f}x over {len(h5_rows)} shapes." if h5_rows else ""))

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
    md += md_pc + md_reg + md_dec + md_h5
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
