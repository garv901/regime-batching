"""Annotate workload .meta.json files with their KV pressure relative to the BAGEL KV pool.

Final KV of a request = input_length + TEMPLATE_OVERHEAD + output_length tokens (the text template that BAGEL wraps
around a text-to-text prompt adds 20 tokens, measured with the BAGEL tokenizer; see README). Ratios are to the pool in
tokens (max_num_pages * page_size). Fails if any request exceeds max_seq_len.

  python experiments/mstar/kv_report.py --pool 262144 data/workloads/bagel_*.jsonl
"""
import argparse
import json

ap = argparse.ArgumentParser()
ap.add_argument("files", nargs="+")
ap.add_argument("--pool", type=int, required=True, help="KV pool in tokens = max_num_pages * page_size")
ap.add_argument("--overhead", type=int, default=20, help="BAGEL text-to-text template tokens added to each prompt")
ap.add_argument("--max-seq-len", type=int, default=32768)
ap.add_argument("--itl-ms", type=float, default=0.0,
                help="if >0, also estimate the peak in-flight KV if every request were admitted on arrival and decoded "
                     "at this fixed inter-token latency (a fluid model: no queueing, no prefill time)")
args = ap.parse_args()


def peak_inflight(rows, itl_s):
    """Peak over time of sum_i KV_i(t), KV_i growing 1 token per itl_s from in+overhead to in+overhead+out."""
    ev = sorted({r["timestamp"] + k * itl_s for r in rows for k in (0, r["output_length"])})
    best = 0.0
    for t in ev:  # KV is piecewise linear, so its maximum sits on an arrival or a completion
        tot = 0.0
        for r in rows:
            d = (t - r["timestamp"]) / itl_s
            if 0 <= d <= r["output_length"]:
                tot += r["input_length"] + args.overhead + d
        best = max(best, tot)
    return best

for path in args.files:
    rows = [json.loads(line) for line in open(path)]
    per = {}
    worst = 0
    for r in rows:
        kv = r["input_length"] + args.overhead + r["output_length"]
        worst = max(worst, kv)
        c = per.setdefault(r["cls"], {"n": 0, "final_kv_tokens": 0})
        c["n"] += 1
        c["final_kv_tokens"] += kv
    assert worst <= args.max_seq_len, f"{path}: request needs {worst} > max_seq_len {args.max_seq_len}"
    total = sum(c["final_kv_tokens"] for c in per.values())
    for c in per.values():
        c["ratio_to_pool"] = round(c["final_kv_tokens"] / args.pool, 3)
    kv = dict(pool_tokens=args.pool, template_overhead=args.overhead, max_seq_len=args.max_seq_len,
              max_request_kv=worst, per_class=per, total_final_kv_tokens=total,
              total_ratio_to_pool=round(total / args.pool, 3))
    if args.itl_ms > 0:
        kv[f"peak_inflight_kv_ratio_itl{args.itl_ms:g}ms"] = round(peak_inflight(rows, args.itl_ms / 1e3) / args.pool, 3)
    mp = path.replace(".jsonl", ".meta.json")
    meta = json.load(open(mp))
    meta["kv"] = kv
    json.dump(meta, open(mp, "w"), indent=1)
    print(path, json.dumps({k: v for k, v in kv.items() if k == "max_request_kv" or k.endswith("_to_pool") or "peak" in k}),
          {k: (v["n"], v["ratio_to_pool"]) for k, v in per.items()})
