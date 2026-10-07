#!/usr/bin/env python3
"""Build fa2-vs-fa3 prefill and decode tables from saved ServingStudio kernel rows."""
import json, csv, collections, sys
D = "results/servingstudio/kernels/"
def load(f):
    d = json.load(open(D + f)); c = d["columns"]
    return [dict(zip(c, r)) for r in d["rows"]], d["provenance"]

R, prov = load("prefill_rows.json")
cell = collections.defaultdict(dict)
for r in R: cell[(r["prefix_len"], r["append_len"])][r["backend"]] = r
apps = sorted({a for _, a in cell})
PT = [0, 8192, 32768, 65536]; AT = [128, 256, 512, 1024, 2048, 8192]
rows = []
for A in AT:
    a = min(apps, key=lambda x: abs(x - A))
    ps = sorted(p for p, aa in cell if aa == a)
    for P in PT:
        p = min(ps, key=lambda x: abs(x - P))
        c = cell[(p, a)]; f2, f3 = c["fa2"], c["fa3"]
        rows.append(dict(target_prefix=P, target_append=A, prefix_len=p, append_len=a,
            fa2_ms=f2["time_ms"], fa3_ms=f3["time_ms"], fa2_tflops=f2["tflops"], fa3_tflops=f3["tflops"],
            fa2_over_fa3=f2["time_ms"] / f3["time_ms"], faster="fa2" if f2["time_ms"] < f3["time_ms"] else "fa3",
            exact_cell=(p == P and a == A), run_at=prov[f2["provenance"]]["profiler_run_at"]))
with open("results/servingstudio/kernels/prefill_table.csv", "w", newline="") as f:
    w = csv.DictWriter(f, rows[0].keys()); w.writeheader(); w.writerows(rows)
# full join of every measured cell
full = []
for (p, a), c in sorted(cell.items()):
    f2, f3 = c["fa2"], c["fa3"]
    full.append(dict(prefix_len=p, append_len=a, fa2_ms=f2["time_ms"], fa3_ms=f3["time_ms"], fa2_tflops=f2["tflops"],
                     fa3_tflops=f3["tflops"], fa2_over_fa3=f2["time_ms"] / f3["time_ms"]))
with open("results/servingstudio/kernels/prefill_all_cells.csv", "w", newline="") as f:
    w = csv.DictWriter(f, full[0].keys()); w.writeheader(); w.writerows(full)

md = ["| target P | target A | measured (prefix, append) | fa2 ms | fa3 ms | fa2 TFLOPS | fa3 TFLOPS | fa2/fa3 time | faster |", "|---|---|---|---|---|---|---|---|---|"]
for r in rows:
    md.append(f"| {r['target_prefix']} | {r['target_append']} | ({r['prefix_len']}, {r['append_len']}){'' if r['exact_cell'] else ' *'} | {r['fa2_ms']:.4f} | {r['fa3_ms']:.4f} | {r['fa2_tflops']:.0f} | {r['fa3_tflops']:.0f} | {r['fa2_over_fa3']:.2f} | {r['faster']} |")
open("results/servingstudio/kernels/prefill_table.md", "w").write("\n".join(md) + "\n")
wins = [x for x in full if x["fa2_over_fa3"] < 1]
print("cells", len(full), "fa2 faster in", len(wins))
bucket = collections.Counter()
for x in full:
    pb = "P<=1k" if x["prefix_len"] <= 1024 else "1k<P<=8k" if x["prefix_len"] <= 8192 else "8k<P<=32k" if x["prefix_len"] <= 32768 else "P>32k"
    bucket[(pb, x["fa2_over_fa3"] < 1)] += 1
print(sorted(bucket.items()))
json.dump([w for w in wins], open("results/servingstudio/kernels/prefill_fa2_wins.json", "w"))

# decode
Rd, provd = load("decode_rows.json")
dc = collections.defaultdict(dict)
for r in Rd: dc[(r["batch_size"], r["total_tokens"])][r["backend"]] = r
TT = [8192, 32768, 131072, 262144, 524288, 1048576]
dr = []
for b in [1, 2, 4, 8]:
    for t in TT:
        c = dc.get((b, t))
        if not c: continue
        d = dict(batch_size=b, total_tokens=t)
        for be in ["fa3", "fa2", "fa2_cudagraph"]:
            d[be + "_ms"] = c[be]["time_ms"]; d[be + "_GBps"] = c[be]["memory_bandwidth_gbps"]
        d["best"] = min(["fa3", "fa2", "fa2_cudagraph"], key=lambda k: d[k + "_ms"])
        dr.append(d)
with open("results/servingstudio/kernels/decode_table.csv", "w", newline="") as f:
    w = csv.DictWriter(f, dr[0].keys()); w.writeheader(); w.writerows(dr)
md = ["| batch | total_tokens | fa3 ms | fa2 ms | fa2_cudagraph ms | fa3 GB/s | fa2 GB/s | fa2_cg GB/s | best |", "|---|---|---|---|---|---|---|---|---|"]
for d in dr:
    md.append(f"| {d['batch_size']} | {d['total_tokens']} | {d['fa3_ms']:.4f} | {d['fa2_ms']:.4f} | {d['fa2_cudagraph_ms']:.4f} | {d['fa3_GBps']:.0f} | {d['fa2_GBps']:.0f} | {d['fa2_cudagraph_GBps']:.0f} | {d['best']} |")
open("results/servingstudio/kernels/decode_table.md", "w").write("\n".join(md) + "\n")
