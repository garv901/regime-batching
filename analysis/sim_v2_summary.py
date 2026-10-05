"""Summarise the pen-and-paper gain search: python analysis/sim_v2_summary.py [sim.jsonl] [summary.csv]."""
import json
import sys

import numpy as np
import pandas as pd

pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40); pd.set_option("display.max_rows", 200)
VARS = ["B0", "B1s", "K", "Bx", "S", "KS"]
rows = []
for l in open(sys.argv[1] if len(sys.argv) > 1 else "results/pen_and_paper/sim_v2.jsonl"):
    d = json.loads(l); sc = d["scenario"]
    r = dict(model=sc["model"], prompt=f'{sc["prompt"]}{"~" if sc["prompt_sigma"] > 0 else ""}',
             out_long=sc["out_long"] or "reason", f=round(1 / sc["frac_long"]), rate=sc["rate"])
    if not all(d.get(v) for v in VARS):
        continue
    for v in VARS:
        x = d[v]
        r.update({f"{v}_e2e": x["e2e_mean"], f"{v}_e2eS": x["e2e_short_mean"], f"{v}_e2eL": x["e2e_long_mean"],
                  f"{v}_thr": x["out_tok_per_s"], f"{v}_ttft99": x["ttft_p99"], f"{v}_pre": x["preempt"],
                  f"{v}_X": x["regime_frac"].get("X", 0), f"{v}_tpotL": x["tpot_long_mean"], f"{v}_unf": x["unfinished"]})
    rows.append(r)
df = pd.DataFrame(rows)
df["stable_B0"] = (df.B0_ttft99 < 10) & (df.B0_pre == 0)
df["stable_S"] = df.S_ttft99 < 10
df["K_vs_B0_e2e"] = df.B0_e2e / df.K_e2e - 1
df["K_vs_B0_tpotL"] = df.B0_tpotL / df.K_tpotL - 1
df["B1s_vs_B0_e2e"] = df.B0_e2e / df.B1s_e2e - 1
df["Bx_vs_B0_e2e"] = df.B0_e2e / df.Bx_e2e - 1
for m in ["e2e", "e2eS", "e2eL", "tpotL"]:
    df[f"KS_vs_S_{m}"] = df[f"S_{m}"] / df[f"KS_{m}"] - 1
df["KS_vs_S_thr"] = df.KS_thr / df.S_thr - 1
df["S_vs_B0_e2eS"] = df.B0_e2eS / df.S_e2eS - 1
df["KS_vs_B0_e2eS"] = df.B0_e2eS / df.KS_e2eS - 1
df.to_csv(sys.argv[2] if len(sys.argv) > 2 else "results/pen_and_paper/sim_v2_summary.csv", index=False)

q = lambda s, qs=(.5, .9, 1): {k: round(v, 3) for k, v in s.quantile(list(qs)).items()}
st, s2 = df[df.stable_B0], df[df.stable_S]
print(f"scenarios {len(df)}; B0-stable {len(st)}; S-stable {len(s2)}")
print("K vs B0 (B0-stable)     e2e:", q(st.K_vs_B0_e2e), " long TPOT:", q(st.K_vs_B0_tpotL))
print("B1s vs B0 (B0-stable)   e2e:", q(st.B1s_vs_B0_e2e, (.01, .5, .9, 1)))
print("Bx vs B0 (B0-stable)    e2e:", q(st.Bx_vs_B0_e2e, (.01, .5, .9, 1)))
print("KS vs S (S-stable)  e2e:", q(s2.KS_vs_S_e2e), " shortE2E:", q(s2.KS_vs_S_e2eS), " longTPOT:", q(s2.KS_vs_S_tpotL),
      " longE2E:", q(s2.KS_vs_S_e2eL), " thr:", q(s2.KS_vs_S_thr))
cols = ["model", "prompt", "out_long", "f", "rate", "B0_X", "S_X", "K_vs_B0_e2e", "KS_vs_S_e2eS", "KS_vs_S_tpotL",
        "KS_vs_S_e2eL", "KS_vs_S_thr", "S_vs_B0_e2eS", "B0_pre", "B0_ttft99", "S_ttft99"]
print("\nTop 25 by kernel gain on top of oracle scheduling (long-request TPOT):")
print(s2.sort_values("KS_vs_S_tpotL", ascending=False)[cols].head(25).round(3).to_string(index=False))
print("\nTop 12 by K vs B0 mean E2E (B0 stable):")
print(st.sort_values("K_vs_B0_e2e", ascending=False)[cols].head(12).round(3).to_string(index=False))
print("\nPer model (S-stable):")
print(s2.groupby("model").agg(tpotL_med=("KS_vs_S_tpotL", "median"), tpotL_max=("KS_vs_S_tpotL", "max"),
                              e2eS_max=("KS_vs_S_e2eS", "max"), thr_max=("KS_vs_S_thr", "max"),
                              X_med=("S_X", "median"), X_max=("S_X", "max")).round(3))
print("\nPer prompt/out_long (S-stable), median long-TPOT gain:")
print(s2.pivot_table(index=["prompt", "out_long"], columns="model", values="KS_vs_S_tpotL", aggfunc="median").round(3))
u = df[~df.stable_B0]
print("\nB0-unstable n=", len(u), " S vs B0 short-E2E speedup x:", q(u.S_vs_B0_e2eS + 1, (.1, .5, .9)),
      " B0 preemptions median:", u.B0_pre.median())
