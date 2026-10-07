"""Stage 1 tables from the F6 microbench: H5 wave efficiency, and X at short vs long prefixes (measured and model).

Inputs: results/microbench/attn_f6.jsonl (job 1444), results/microbench/attn_f6_pred_pre.jsonl (model, written
before job 1444). Output: reports/regimes_study/figures/f6_tables.md
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CFGS = ["fa3_auto", "fa3_cg", "fa3_s1", "fa3_s2", "fa3_s4", "fa3_s8", "fa3_s16"]
PS = [0, 8192, 32768, 131072]
PS_H5 = PS[1:]  # at P = 0 short causal tiles are overhead-bound; the wave effect is not separable there
QS = [417, 448, 512, 513, 544, 640, 768, 1024, 1056, 1057, 1088, 1536, 1584, 1585, 2048, 2112, 2113, 8192]
ALIGNED = [512, 1056, 2112]  # 128, 264, 528 tiles: <= 1, 2, 4 full waves on 132 SMs


def work(P, q):
    return q * (P + q / 2)


def main():
    meas = {}
    for line in open(ROOT / "results/microbench/attn_f6.jsonl"):
        r = json.loads(line)
        meas[(r["spec"], r["config"])] = r["time_us_median"]
    pred = {}
    for line in open(ROOT / "results/microbench/attn_f6_pred_pre.jsonl"):
        r = json.loads(line)
        pred[(r["spec"], r["config"])] = r["pred_us"]
    t = lambda s, c="fa3_auto": meas[(s, c)]  # noqa: E731
    best = lambda s: min(t(s, c) for c in CFGS)  # noqa: E731
    out = ["# F6 tables (Stage 1)", "",
           "Per attention layer, us, H100, Llama-3.1-8B geometry. default = vLLM's FA3 path above 512 tokens "
           "(`fa3_auto`). best = min over split bounds 1-32 and the CUDA-graph path. model = tile model written "
           "before the run.", "",
           "## H5: prefill alone, efficiency vs the best wave-aligned size (512, 1056 or 2112 tokens)", "",
           "efficiency = (us per unit of attention work at the best aligned size) / (us per unit at q), with work = "
           "q x (P + q/2) (causal). Model in brackets.", "",
           "| q | tiles | " + " | ".join(f"P={P}" for P in PS_H5) + " |", "|---|---|" + "---|" * len(PS_H5)]
    ref = {}
    for P in PS_H5:
        ref[P] = min(t(f"app_P{P}_q{q}_none") / work(P, q) for q in ALIGNED)
        ref[("m", P)] = min(pred[(f"app_P{P}_q{q}_none", "fa3_auto")] / work(P, q) for q in ALIGNED)
    for q in QS:
        cells = []
        for P in PS_H5:
            s = f"app_P{P}_q{q}_none"
            e = ref[P] / (t(s) / work(P, q))
            em = ref[("m", P)] / (pred[(s, "fa3_auto")] / work(P, q))
            gain = t(s) / best(s)
            cells.append(f"{e:.0%} ({em:.0%})" + (f", split {gain:.2f}x" if gain >= 1.05 else ""))
        out.append(f"| {q} | {8 * -(-q // 32)} | " + " | ".join(cells) + " |")
    out += ["", "No split bound recovers the loss (the 'split' note appears only where the best bound beats the "
            "default by >= 5%).", "",
            "## X: prefill chunk + long decodes, default vs best split bound vs separate launches", "",
            "sep = prefill alone (default) + decodes alone (best). Model in brackets.", "",
            "| P | q | decodes | default | best bound | default / best | sep | default / sep |",
            "|---|---|---|---|---|---|---|---|"]
    dec_alone = {d: min(t(f"app_decalone_{d}", c) for c in CFGS) for d in ["1x32k", "3x32k", "1x128k", "3x128k"]}
    for P in PS:
        for q in [513, 1024, 2048, 8192]:
            for d in ["1x128k", "3x128k"]:
                s = f"app_P{P}_q{q}_{d}"
                sep = t(f"app_P{P}_q{q}_none") + dec_alone[d]
                pd, ps_ = pred[(s, "fa3_auto")], pred[(f"app_P{P}_q{q}_none", "fa3_auto")]
                pb = min(pred[(s, c)] for c in ["fa3_auto", "fa3_s1", "fa3_s2", "fa3_s4", "fa3_s8", "fa3_s16"])
                out.append(f"| {P} | {q} | {d} | {t(s):.0f} ({pd:.0f}) | {best(s):.0f} ({pb:.0f}) | "
                           f"{t(s) / best(s):.2f} ({pd / pb:.2f}) | {sep:.0f} | {t(s) / sep:.2f} |")
    (ROOT / "reports/regimes_study/figures").mkdir(parents=True, exist_ok=True)
    (ROOT / "reports/regimes_study/figures/f6_tables.md").write_text("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
