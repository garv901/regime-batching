#!/usr/bin/env python3
import json, csv
D = "results/servingstudio/"
pred = json.load(open(D + "prediction.json")); names = json.load(open(D + "predict_case_names.json"))
req = json.load(open(D + "predict_request.json"))
rows = {}
for i, (nm, c) in enumerate(zip(names, pred["cases"])):
    s = c["sections"][0]; nodes = s["nodes"]
    leaf = [n for n in nodes if n["kind"] == "leaf"]
    g = sum(n["total_ms"] for n in leaf if "(single_gemm)" in n["label"])
    pf = sum(n["total_ms"] for n in leaf if n["slot"] == 4); dc = sum(n["total_ms"] for n in leaf if n["slot"] == 5)
    tot = s["total_ms"]
    sb = s["slot_backend"]
    ext = s["coverage"].get("extrapolated", [])
    rows[nm] = dict(case_id=i, case=nm, total_ms=tot, attn_prefill_ms=pf, attn_decode_ms=dc, gemm_ms=g,
                    other_ms=tot - pf - dc - g, prefill_backend=sb[4], decode_backend=sb[5],
                    gemm_backends="/".join(sorted({b for j, b in enumerate(sb) if j in (2, 6, 8, 10, 12) and b})),
                    extrapolated=json.dumps(s["coverage"]), root_total_ms=nodes[0]["total_ms"])
# derived mixed-case arithmetic
import re
for nm, r in rows.items():
    m = re.match(r"pf\[(\d+),(\d+)\]\+(\d)x126976dec", nm)
    if m:
        P, A, n = m.groups()
        pf_only = rows[f"pf[{P},{A}]"]["total_ms"]; dec_only = rows[f"dec{n}x126976"]["total_ms"]
        r["pf_only_ms"] = pf_only; r["dec_only_ms"] = dec_only
        r["mixed_minus_sum"] = r["total_ms"] - (pf_only + dec_only); r["mixed_over_sum"] = r["total_ms"] / (pf_only + dec_only)
        r["mixed_over_dec_only"] = r["total_ms"] / dec_only; r["mixed_over_pf_only"] = r["total_ms"] / pf_only
    m = re.match(r"(cold\[\d+,8192\])\+3x126976dec", nm)
    if m:
        pf_only = rows[m.group(1)]["total_ms"]; dec_only = rows["dec3x126976"]["total_ms"]
        r["pf_only_ms"] = pf_only; r["dec_only_ms"] = dec_only
        r["mixed_minus_sum"] = r["total_ms"] - (pf_only + dec_only); r["mixed_over_sum"] = r["total_ms"] / (pf_only + dec_only)
        r["mixed_over_dec_only"] = r["total_ms"] / dec_only; r["mixed_over_pf_only"] = r["total_ms"] / pf_only
keys = ["case_id", "case", "total_ms", "attn_prefill_ms", "attn_decode_ms", "gemm_ms", "other_ms", "prefill_backend", "decode_backend", "gemm_backends", "extrapolated", "root_total_ms", "pf_only_ms", "dec_only_ms", "mixed_minus_sum", "mixed_over_sum", "mixed_over_dec_only", "mixed_over_pf_only"]
with open(D + "predict_table.csv", "w", newline="") as f:
    w = csv.DictWriter(f, keys, restval=""); w.writeheader(); w.writerows(rows.values())
fm = lambda x: "" if x in ("", None) else (f"{x:.3f}" if isinstance(x, float) else str(x))
md = ["| case | total ms | attn prefill | attn decode | GEMM | other | prefill be | decode be | extrapolated slots |", "|---|---|---|---|---|---|---|---|---|"]
for r in rows.values():
    ex = json.loads(r["extrapolated"]); exs = ", ".join(f"{k}:{v}" for k, v in ex.items()) or "none"
    md.append(f"| {r['case']} | {r['total_ms']:.3f} | {r['attn_prefill_ms']:.3f} | {r['attn_decode_ms']:.3f} | {r['gemm_ms']:.3f} | {r['other_ms']:.3f} | {r['prefill_backend'] or '-'} | {r['decode_backend'] or '-'} | {exs} |")
open(D + "predict_table.md", "w").write("\n".join(md) + "\n")
md = ["| mixed case | mixed ms | prefill-only ms | decode-only ms | mixed - (pf+dec) ms | mixed/(pf+dec) | mixed/decode-only |", "|---|---|---|---|---|---|---|"]
for r in rows.values():
    if "mixed_over_sum" in r:
        md.append(f"| {r['case']} | {r['total_ms']:.3f} | {r['pf_only_ms']:.3f} | {r['dec_only_ms']:.3f} | {r['mixed_minus_sum']:.3f} | {r['mixed_over_sum']:.3f} | {r['mixed_over_dec_only']:.2f} |")
open(D + "predict_mixed_table.md", "w").write("\n".join(md) + "\n")
print(max(abs(r["total_ms"] - r["root_total_ms"]) for r in rows.values()))
