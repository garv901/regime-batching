#!/usr/bin/env python3
"""Convert a TraceLab multi-round CSV (id,input_len,output_len,arrival_time,round_idx,
tool_wait_after_ms,prefix_len) to ServingStudio text-generation-session-execution-v2.
A round 0 with prefix_len>0 is folded into input_len (API rejects a cached first round).
Sessions are truncated at the first round with prefix+input+output > max-model-len
(sessions whose round 0 overflows are dropped). Sessions are visited in seeded-shuffled
order and taken whole (after truncation) until --max-rounds is reached; the last
session is cut to fit. arrival_time_ms = session arrival (k*arrival-gap-ms) for every round of a session."""
import argparse, csv, random
from collections import OrderedDict

ap = argparse.ArgumentParser()
ap.add_argument("--in", dest="inp", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--max-rounds", type=int, default=50)
ap.add_argument("--max-model-len", type=int, default=131072)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--arrival-gap-ms", type=float, default=20000.0, help="spacing between kept sessions (0.05 sessions/s)")
a = ap.parse_args()

sess = OrderedDict()
with open(a.inp) as f:
    for r in csv.DictReader(f):
        sess.setdefault(r["id"], []).append(r)
ids = list(sess)
random.Random(a.seed).shuffle(ids)

out, n = [], 0
for sid in ids:
    rows = sorted(sess[sid], key=lambda r: int(r["round_idx"]))
    keep = []
    for r in rows:
        if int(r["prefix_len"]) + int(r["input_len"]) + int(r["output_len"]) > a.max_model_len:
            break
        keep.append(r)
    if not keep:
        continue
    keep = keep[: a.max_rounds - n]
    out.append((sid, keep))
    n += len(keep)
    if n >= a.max_rounds:
        break

# renumber sessions, shift arrivals so earliest session arrives at 0, sort by arrival
out.sort(key=lambda t: float(t[1][0]["arrival_time"]))
with open(a.out, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["request_id", "session_id", "round_idx", "arrival_time_ms", "prefix_len",
                "input_len", "output_len", "tool_wait_after_ms"])
    for k, (sid, rows) in enumerate(out):
        for i, r in enumerate(rows):
            last = i == len(rows) - 1
            pre, inp = int(r["prefix_len"]), int(r["input_len"])
            if i == 0 and pre > 0:  # ServingStudio: a session's first round has no cached context
                pre, inp = 0, inp + pre
            w.writerow([f"session_s{k}_round_{i:06d}", f"s{k}", i, round(k * a.arrival_gap_ms, 3),
                        pre, inp, int(r["output_len"]),
                        0 if last else round(float(r["tool_wait_after_ms"]), 3)])
print(f"{len(out)} sessions, {n} rounds -> {a.out}")
