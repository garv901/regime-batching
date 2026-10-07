"""Annotate workload .meta.json files with their KV pressure relative to the BAGEL KV pool.

Final KV of a request = input_length + TEMPLATE_OVERHEAD + output_length tokens (the text template that BAGEL wraps
around a text-to-text prompt adds 20 tokens, measured with the BAGEL tokenizer; see README). Ratios are to the pool in
tokens (max_num_pages * page_size). Fails if any request exceeds max_seq_len.

Rows that carry a declared `max_tokens` (experiments/mstar/declare_max_tokens.py; the request still stops after
`output_length` tokens) also get the DECLARED figures next to the generated ones, in the same `kv` block (a row without
`max_tokens` counts as declared == output_length; a file with no `max_tokens` at all gets none of them):
  per class and total: declared_final_kv_tokens / declared_ratio_to_pool  (sum of prompt + overhead + declared)
                       decl_kv_ratio                                       (declared / generated final KV, same sums)
  decl_ratio_mean                                                          (mean over requests of declared / output_length)
  peak_declared_inflight_kv_ratio_itl<X>ms   the fluid peak below with every request running for its DECLARED length
                                             (the KV a planner must be ready for if nothing stops early)
  peak_reserved_inflight_kv_ratio_itl<X>ms   the fluid peak of the summed reservation: each in-flight request holds
                                             prompt + overhead + declared from arrival until it finishes after output_length

  python experiments/mstar/kv_report.py --pool 262144 data/workloads/bagel_*.jsonl
"""
import argparse
import json
from bisect import bisect_left, bisect_right

ap = argparse.ArgumentParser()
ap.add_argument("files", nargs="+")
ap.add_argument("--pool", type=int, required=True, help="KV pool in tokens = max_num_pages * page_size")
ap.add_argument("--overhead", type=int, default=20, help="BAGEL text-to-text template tokens added to each prompt")
ap.add_argument("--max-seq-len", type=int, default=32768)
ap.add_argument("--max-declared", type=int, default=2048,
                help="largest valid declared max_tokens (BAGEL's MAX_OUTPUT_TOKENS: the server cannot generate more)")
ap.add_argument("--itl-ms", type=float, default=0.0,
                help="if >0, also estimate the peak in-flight KV if every request were admitted on arrival and decoded "
                     "at this fixed inter-token latency (a fluid model: no queueing, no prefill time)")
args = ap.parse_args()


def peak_inflight(rows, itl_s, n_key="output_length", reserved_key=None):
    """Peak over time of sum_i KV_i(t), request i in flight from its arrival for n_key * itl_s seconds.

    KV_i grows 1 token per itl_s from in+overhead to in+overhead+n (reserved_key None), or is the constant
    in+overhead+row[reserved_key] (a reservation held for the whole flight). The sum is piecewise linear (or a step
    function), so its maximum sits on an arrival or a completion; each such time only looks at the rows that arrived
    within the longest flight before it (rows are sorted by arrival, so the sum runs in file order for a sorted file).
    """
    rows = sorted(rows, key=lambda r: r["timestamp"])
    arr = [r["timestamp"] for r in rows]
    span = max(r[n_key] for r in rows) * itl_s + 1e-3
    ev = sorted({r["timestamp"] + k * itl_s for r in rows for k in (0, r[n_key])})
    best = 0.0
    for t in ev:
        tot = 0.0
        for r in rows[bisect_left(arr, t - span):bisect_right(arr, t)]:
            d = (t - r["timestamp"]) / itl_s
            if 0 <= d <= r[n_key]:
                tot += r["input_length"] + args.overhead + (d if reserved_key is None else r[reserved_key])
        best = max(best, tot)
    return best


for path in args.files:
    rows = [json.loads(line) for line in open(path)]
    has_decl = any(r.get("max_tokens") is not None for r in rows)
    per = {}
    worst = 0
    for r in rows:
        kv = r["input_length"] + args.overhead + r["output_length"]
        worst = max(worst, kv)
        c = per.setdefault(r["cls"], {"n": 0, "final_kv_tokens": 0})
        c["n"] += 1
        c["final_kv_tokens"] += kv
        if has_decl:
            decl = r["output_length"] if r.get("max_tokens") is None else r["max_tokens"]
            r["declared"] = decl
            assert r["output_length"] <= decl <= args.max_declared, \
                f"{path}: declared max_tokens {decl} not in [output_length {r['output_length']}, {args.max_declared}]"
            dkv = r["input_length"] + args.overhead + decl
            assert dkv <= args.max_seq_len, f"{path}: declared KV {dkv} > max_seq_len {args.max_seq_len}"
            c["declared_final_kv_tokens"] = c.get("declared_final_kv_tokens", 0) + dkv
    assert worst <= args.max_seq_len, f"{path}: request needs {worst} > max_seq_len {args.max_seq_len}"
    total = sum(c["final_kv_tokens"] for c in per.values())
    for c in per.values():
        c["ratio_to_pool"] = round(c["final_kv_tokens"] / args.pool, 3)
        if has_decl:
            c["declared_ratio_to_pool"] = round(c["declared_final_kv_tokens"] / args.pool, 3)
            c["decl_kv_ratio"] = round(c["declared_final_kv_tokens"] / c["final_kv_tokens"], 3)
    kv = dict(pool_tokens=args.pool, template_overhead=args.overhead, max_seq_len=args.max_seq_len,
              max_request_kv=worst, per_class=per, total_final_kv_tokens=total,
              total_ratio_to_pool=round(total / args.pool, 3))
    if has_decl:
        dtotal = sum(c["declared_final_kv_tokens"] for c in per.values())
        kv.update(declared_total_final_kv_tokens=dtotal, declared_total_ratio_to_pool=round(dtotal / args.pool, 3),
                  decl_kv_ratio=round(dtotal / total, 3),
                  decl_ratio_mean=round(sum(r["declared"] / r["output_length"] for r in rows) / len(rows), 3))
    if args.itl_ms > 0:
        itl = args.itl_ms / 1e3
        kv[f"peak_inflight_kv_ratio_itl{args.itl_ms:g}ms"] = round(peak_inflight(rows, itl) / args.pool, 3)
        if has_decl:
            kv[f"peak_declared_inflight_kv_ratio_itl{args.itl_ms:g}ms"] = round(peak_inflight(rows, itl, "declared") / args.pool, 3)
            kv[f"peak_reserved_inflight_kv_ratio_itl{args.itl_ms:g}ms"] = round(
                peak_inflight(rows, itl, "output_length", "declared") / args.pool, 3)
    mp = path.replace(".jsonl", ".meta.json")
    meta = json.load(open(mp))
    meta["kv"] = kv
    json.dump(meta, open(mp, "w"), indent=1)
    print(path, json.dumps({k: v for k, v in kv.items() if k == "max_request_kv" or k.endswith("_to_pool") or "peak" in k or k.startswith("decl")}),
          {k: (v["n"], v["ratio_to_pool"]) + ((v["declared_ratio_to_pool"],) if has_decl else ()) for k, v in per.items()})
