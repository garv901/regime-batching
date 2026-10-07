"""Summarize attn_f6.jsonl + attn_replay.jsonl into ATTN_F6.md."""
import json, math, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

R = Path(__file__).resolve().parent.parent / "results/microbench"
FORCED = ["fa3_s1", "fa3_s2", "fa3_s4", "fa3_s8", "fa3_s16"]
DECS = ["none", "1x32k", "3x32k", "1x128k", "3x128k"]


def load(p):
    rows = [json.loads(l) for l in open(p)] if Path(p).exists() else []
    return [r for r in rows if "time_us_median" in r]


def main():
    rows = load(R / "attn_f6.jsonl")
    T = defaultdict(dict)  # spec -> cfg -> us
    meta = {}
    for r in rows:
        T[r["spec"]][r["config"]] = r["time_us_median"]
        meta[r["spec"]] = r
    pre = [s for s in T if meta[s]["q"]]
    Ps = sorted({meta[s]["P"] for s in pre}); Qs = sorted({meta[s]["q"] for s in pre})
    out = ["# F6: long-prefix append attention (1 layer, H100, CUDA-graph replay, L2 flush subtracted)\n",
           f"Specs: {len(T)} ({len(pre)} append + decode-only refs); rows: {len(rows)}. Times in us.\n",
           "default = fa3_auto (vLLM above the CUDA-graph capture size). best = min over forced splits "
           "fa3_s1/2/4/8/16. gain = default / best.\n"]
    def name(P, q, d): return f"app_P{P}_q{q}_{d}"
    def best(t):
        c = {k: t[k] for k in FORCED if k in t}
        k = min(c, key=c.get); return k, c[k]
    # (a)
    out.append("## (a) default vs best forced split\n")
    allg = []
    for P in Ps:
        for d in DECS:
            out.append(f"### P={P}, decodes={d}\n\n| q | tiles | default | def splits | best | best cfg | gain |\n|---|---|---|---|---|---|---|")
            for q in Qs:
                t = T.get(name(P, q, d))
                if not t or "fa3_auto" not in t: continue
                k, b = best(t); g = t["fa3_auto"] / b
                allg.append((g, P, q, d, k, t["fa3_auto"], b))
                out.append(f"| {q} | {meta[name(P,q,d)]['tiles']} | {t['fa3_auto']:.0f} | {meta[name(P,q,d)].get('default_splits')} | {b:.0f} | {k} | {g:.3f} |")
            out.append("")
    if allg:
        allg.sort(reverse=True)
        out.append("Top gains: " + "; ".join(f"{g:.2f}x (P={P},q={q},dec={d},{k}: {a:.0f}->{b:.0f}us)" for g, P, q, d, k, a, b in allg[:8]) + "\n")
        out.append(f"Mean gain over all append specs: {np.mean([a[0] for a in allg]):.3f}; fraction with gain>1.05: {np.mean([a[0] > 1.05 for a in allg]):.2f}\n")
    # (b)
    out.append("## (b) straggler hiding\n")
    out.append("For each (P, q, decode set) and config c in {default, s1}: t_mix = time(decodes+prefill), t_pre = prefill alone (same config), "
               "t_dec = decodes alone (same config, from app_decalone_*). max = max(t_pre,t_dec), sum = t_pre+t_dec. "
               "h = (sum - t_mix)/(sum - max): 1 = perfectly hidden (=max), 0 = additive (=sum).\n")
    for cfg in ["fa3_auto", "fa3_s1"]:
        out.append(f"### config {cfg}\n\n| P | dec | q | t_pre | t_dec | t_mix | max | sum | h |\n|---|---|---|---|---|---|---|---|---|")
        hs = defaultdict(list)
        for P in Ps:
            for d in DECS[1:]:
                td = T.get(f"app_decalone_{d}", {}).get(cfg)
                for q in Qs:
                    tp = T.get(name(P, q, "none"), {}).get(cfg); tm = T.get(name(P, q, d), {}).get(cfg)
                    if None in (td, tp, tm): continue
                    mx, sm = max(tp, td), tp + td
                    h = (sm - tm) / (sm - mx) if sm - mx > 1e-9 else float("nan")
                    hs[(P, d)].append((h, tm / mx, tm / sm))
                    if q in (512, 513, 1057, 2048, 8192):
                        out.append(f"| {P} | {d} | {q} | {tp:.0f} | {td:.0f} | {tm:.0f} | {mx:.0f} | {sm:.0f} | {h:.2f} |")
        out.append("\nMedian over q: t_mix/max, t_mix/sum\n\n| P | dec | median t_mix/max | median t_mix/sum | median h |\n|---|---|---|---|---|")
        for (P, d), v in sorted(hs.items()):
            a = np.array(v)
            out.append(f"| {P} | {d} | {np.median(a[:,1]):.2f} | {np.median(a[:,2]):.2f} | {np.nanmedian(a[:,0]):.2f} |")
        out.append("")
    # (c)
    out.append("## (c) wave quantization: implied efficiency vs tile count\n")
    out.append("Prefill alone (decodes=none), config fa3_s1 (one split, tiles = 8*ceil(q/32), 132 SMs). "
               "waves = tiles/132; ideal = waves*(time/ceil(waves))... efficiency eff = (tiles/132 / ceil(tiles/132)) is the "
               "wave-quantization ceiling; 'meas' = (time per tile-wave-unit) normalized by the best (lowest time/tiles) in that P: "
               "meas_eff = min_q(t/tiles) / (t/tiles) (the q-dependent part of causal work is not removed, so compare shape not level; "
               "t/tiles uses per-tile work which grows with q only through the causal diagonal, small when P is large).\n")
    for P in Ps:
        pts = [(meta[name(P, q, 'none')]["tiles"], q, T[name(P, q, 'none')].get("fa3_s1")) for q in Qs if T.get(name(P, q, 'none'), {}).get("fa3_s1")]
        if not pts: continue
        base = min(t / ti for ti, q, t in pts)
        out.append(f"### P={P}\n\n| q | tiles | waves | ceil | wave-eff | t(s1) | meas_eff | t(s2)/t(s1) | t(s4)/t(s1) | t(s8)/t(s1) |\n|---|---|---|---|---|---|---|---|---|---|")
        for ti, q, t in pts:
            w = ti / 132; tt = T[name(P, q, 'none')]
            out.append(f"| {q} | {ti} | {w:.2f} | {math.ceil(w)} | {w/math.ceil(w):.2f} | {t:.0f} | {base/(t/ti):.2f} | " +
                       " | ".join(f"{tt[k]/t:.2f}" if k in tt else "-" for k in ["fa3_s2", "fa3_s4", "fa3_s8"]) + " |")
        out.append("")
    # (d)
    rp = load(R / "attn_replay.jsonl")
    out.append("## (d) replay of telemetry steps\n")
    if rp:
        Tr = defaultdict(dict); mr = {}
        for r in rp:
            Tr[r["spec"]][r["config"]] = r["time_us_median"]; mr[r["spec"]] = r
        out.append("| step | regime | T | n_dec | n_pre | maxL | max_q | default splits | default us | s1 us | best | best us | gain |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        gains = []; tot_d = tot_b = 0
        for s, t in sorted(Tr.items()):
            if "fa3_auto" not in t: continue
            k, b = best(t); g = t["fa3_auto"] / b; gains.append(g); tot_d += t["fa3_auto"]; tot_b += b
            m = mr[s]
            out.append(f"| {m.get('src_step')} | {m.get('regime')} | {m['num_actual_tokens']} | {m['B_dec']} | {m['n_prefill']} | {m['maxL']} | {m['prefill_q']} | {m.get('default_splits')} | {t['fa3_auto']:.0f} | {t.get('fa3_s1', float('nan')):.0f} | {k} | {b:.0f} | {g:.3f} |")
        gains = np.array(gains)
        out.append(f"\nSteps: {len(gains)}. Mean per-step gain {gains.mean():.3f}, median {np.median(gains):.3f}, max {gains.max():.3f}; "
                   f"time-weighted gain {tot_d/tot_b:.3f}; steps with gain>1.05: {(gains>1.05).sum()}. "
                   "Note: best-of-5 per step is an oracle (includes noise selection bias of ~1%).\n")
    open(R / "ATTN_F6.md", "w").write("\n".join(out)); print("wrote", R / "ATTN_F6.md")


if __name__ == "__main__":
    main()
