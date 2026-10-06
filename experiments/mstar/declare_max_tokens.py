#!/usr/bin/env python3
"""Copy a timed trace and give every row a declared ``max_tokens`` that differs from its generated ``output_length``.

The v1-v3 traces are oracle traces: the replay client sends ``max_tokens = output_length`` and ignores EOS, so the server
knows each request's exact output length. A row with a ``max_tokens`` field instead declares that bound to the server
(what KV admission reserves for) while the request still stops after ``output_length`` tokens (an emulated EOS, see
``benchmark/trace_replay.py`` in the regime-harness worktree). Rows keep every other field unchanged.

Modes (CAP = 2048 by default: BAGEL's MAX_OUTPUT_TOKENS, mstar/model/base.py, which is also what the server assumes when a
client sends no max_tokens at all):
  --mode x<k>     max_tokens = min(CAP, max(output_length, ceil(k * output_length)))   e.g. x2: a predictor with a 2x margin
  --mode fixed    max_tokens = max(output_length, CAP)                                   one bound for every request (no info)

  python3 experiments/mstar/declare_max_tokens.py data/workloads/bagel_burst_v2.jsonl data/workloads/bagel_burst_v2_mt2048.jsonl --mode fixed
"""
import argparse
import json
import math
import sys


def declared(n_out: int, mode: str, cap: int) -> int:
    if mode == "fixed":
        return max(n_out, cap)
    if mode.startswith("x"):
        return min(cap, max(n_out, math.ceil(float(mode[1:]) * n_out)))
    raise ValueError(f"unknown mode {mode!r}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--mode", required=True, help="x<k> (e.g. x2) or fixed")
    ap.add_argument("--cap", type=int, default=2048)
    ap.add_argument("--id-offset", type=int, default=0,
                    help="add this to every hash_id, so a variant replayed on the same server as its base (or another variant) "
                         "builds different prompt tokens and cannot hit the prefix cache the other run left behind")
    a = ap.parse_args()
    n = over = 0
    with open(a.src) as f, open(a.dst, "w") as g:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            out = int(row["output_length"])
            if out > a.cap:
                sys.exit(f"{a.src}: output_length {out} > cap {a.cap}; the server could not generate it")
            row["max_tokens"] = declared(out, a.mode, a.cap)
            if a.id_offset and row.get("hash_ids"):
                row["hash_ids"] = [h + a.id_offset for h in row["hash_ids"]]
            n += 1
            over += row["max_tokens"] > out
            g.write(json.dumps(row) + "\n")
    print(f"{a.dst}: {n} rows, {over} with max_tokens > output_length (mode {a.mode}, cap {a.cap})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
