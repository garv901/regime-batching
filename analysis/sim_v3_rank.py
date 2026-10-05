"""Rank simulated variants against the best static vLLM config (B0 / B1s forced splits / B512 static 512 budget).

  python analysis/sim_v3_rank.py results/pen_and_paper/sim_v3_focus.jsonl results/pen_and_paper/sim_v3b_focus.jsonl
"""
import json
import sys

import pandas as pd

pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)
VARS = ["B0", "B1s", "B512", "K", "Bx", "Bc", "S", "KS", "KSc"]
STATICS = ["B0", "B1s", "B512"]
res = {}
for f in sys.argv[1:]:
    for l in open(f):
        d = json.loads(l); k = json.dumps(d.pop("scenario"), sort_keys=True); res.setdefault(k, {}).update(d)
rows = []
for k, v in res.items():
    sc = json.loads(k)
    r = dict(model=sc["model"][5:], prompt=f'{sc["prompt"]}{"~" if sc["prompt_sigma"] > 0 else ""}',
             out=sc["out_long"] or "rsn", f=round(1 / sc["frac_long"]), rate=sc["rate"])
    for var in VARS:
        x = v.get(var)
        if not x:
            continue
        r.update({var + "_e2eS": x["e2e_short_mean"], var + "_tpL": x["tpot_long_mean"], var + "_t99": x["ttft_p99"],
                  var + "_pre": x["preempt"], var + "_X": x["regime_frac"].get("X", 0)})
    rows.append(r)
d = pd.DataFrame(rows).dropna(subset=[s + "_e2eS" for s in VARS if s + "_e2eS" in rows[0]])
d["bestS_e2eS"] = d[[s + "_e2eS" for s in STATICS]].min(axis=1)
d["bestS_cfg"] = d[[s + "_e2eS" for s in STATICS]].idxmin(axis=1).str.replace("_e2eS", "")
for v in ["K", "Bc", "S", "KS", "KSc"]:
    d[v + "_vs_best"] = d.bestS_e2eS / d[v + "_e2eS"]
    d[v + "_vs_B0"] = d.B0_e2eS / d[v + "_e2eS"]
q = lambda s: {p: round(s.quantile(p), 3) for p in (.1, .5, .9, 1)}
print("scenarios", len(d), " best static:", d.bestS_cfg.value_counts().to_dict())
for v in ["K", "Bc", "S", "KS", "KSc"]:
    print(f"{v:4s} short-E2E speedup vs best static {q(d[v + '_vs_best'])}   vs B0 {q(d[v + '_vs_B0'])}")
st = d[(d.B0_t99 < 10) & (d.B0_pre == 0)]
print("B0-stable n =", len(st))
for v in ["K", "Bc", "KSc"]:
    print(f"{v:4s} vs best static (B0-stable) {q(st[v + '_vs_best'])}")
cols = ["model", "prompt", "out", "f", "rate", "bestS_cfg", "B0_X", "B0_pre", "B512_t99", "KSc_t99", "K_vs_best",
        "Bc_vs_best", "S_vs_best", "KS_vs_best", "KSc_vs_best", "KSc_vs_B0"]
print(d.sort_values("KSc_vs_best", ascending=False)[cols].head(25).round(2).to_string(index=False))
print("worst:"); print(d.sort_values("KSc_vs_best")[cols].head(8).round(2).to_string(index=False))
d.to_csv("results/pen_and_paper/sim_v3_merged.csv", index=False)
