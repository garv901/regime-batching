#!/usr/bin/env python3
"""Build the Stage-1 timing-predict request (Llama-3.1-8B/llama3_dense_tp, tp_size=1, H200)."""
import json
cases, names = [], []
def add(name, pairs=None, dec=None):
    g = {}
    if pairs: g["prefill_chunk_pairs"] = pairs
    if dec: g["decode_kv_lens"] = dec
    cases.append({"groups": [g]}); names.append(name)
for P in (8192, 32768, 126976):
    for A in (256, 530, 1024, 2048, 8192 if P < 126976 else 4096):  # 126976+8192 > 131072, so 4096 for P=126976
        add(f"pf[{P},{A}]", [[P, A]])
for P in (32768, 126976):
    for A in (530, 1024, 8192 if P < 126976 else 4096):  # same cap
        for n in (1, 3): add(f"pf[{P},{A}]+{n}x126976dec", [[P, A]], [126976] * n)
for L in (126976, 32768):
    for n in (1, 3, 8): add(f"dec{n}x{L}", None, [L] * n)
add("cold[0,8192]", [[0, 8192]]); add("cold[8192,8192]", [[8192, 8192]])
add("cold[0,8192]+3x126976dec", [[0, 8192]], [126976] * 3); add("cold[8192,8192]+3x126976dec", [[8192, 8192]], [126976] * 3)
req = {"preset": "Llama-3.1-8B/llama3_dense_tp", "params": {"tp_size": 1}, "cases": cases, "analyze": True}
json.dump(req, open("results/servingstudio/predict_request.json", "w"))
json.dump(names, open("results/servingstudio/predict_case_names.json", "w"))
print(len(cases))
