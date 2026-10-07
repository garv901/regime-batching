#!/usr/bin/env python3
"""Characterise the public-trace workloads (pub_*.jsonl) and run the admission simulator on them (CPU only).

  python3 experiments/mstar/public_trace_report.py data/workloads/pub_*_r?*.jsonl            # characterisation table
  python3 experiments/mstar/public_trace_report.py --sim data/workloads/pub_mooncake_conv_r0.5.jsonl
  python3 experiments/mstar/public_trace_report.py --sim --prefix-hits data/workloads/pub_mooncake_conv_r0.5.jsonl
  python3 experiments/mstar/public_trace_report.py --write-meta data/workloads/pub_azure_conv_r1.jsonl   # variants' meta
  python3 experiments/mstar/public_trace_report.py --pick-window mooncake --rates 0.5 0.75 1 [--centre 1500]   # window choice

--pick-window SOURCE: for every window centre C on a 10-s grid, take the windows [C - 300 r, C + 300 r) (600 r original seconds,
as make_public_workloads.sh does) and score each by the RMS of the relative deviations of eight statistics from the whole-trace
value (request rate, input mean/p50/p90, output mean/p50, final-KV mean, final-KV demand in tokens/s; after the clamps of
convert_public_trace.py). It prints the best centres by mean score over --rates, then, for --centre C (default: the best),
each window's statistics beside the whole trace's, with the percentile rank of the statistic among all windows of that length.

Characterisation: requests, span, rate, input/output/final-KV percentiles, in-file prefix reuse, clamps (from .meta.json),
declared/generated KV for variants (rows with max_tokens), and the average demand per second over the file's output span
(window length / r; the last request arrives a little before the end of it): decode tokens/s against the BAGEL decode capacity
(~2,000 tok/s at batch 64), prefill tokens/s (input + 20-token template) against the assumed 20-40k tok/s.

--sim runs analysis/admit_sim_bagel.py (policies sum_fifo, sum_bf, peak_fifo, peak_bf, open; BAGEL step cost, pool 2047 pages
x 128). It reports the "short" class (cls == short: final KV <= the file's p50) and ALL requests: the simulator splits E2E by a
mask, so it is called twice, once with mask = (cls != short), once with an all-false mask. Only oracle files (max_tokens ==
output_length): the simulator in this tree does not read a declared max_tokens, so a variant is refused.

--prefix-hits (with --sim) is the optimistic bound for a trace with shared prefixes: the simulator has no prefix cache, so it
prefills and holds KV for every prompt token. With this flag each request's prompt is cut by the tokens that an EARLIER
request of the file already holds (leading hash ids seen before, rounded down to the server's 128-token pages), as if every
hit were served from the cache: no prefill compute and no new KV pages for them, no eviction, no wait for the earlier request's
prefill to finish. The truth lies between the plain run and this one.
"""
import argparse
import json
import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "analysis"))

OVERHEAD = 20
HASH_BLOCK = 16
PAGE = 128
DECODE_CAP_TOK_S = 2000.0  # decode saturates near 2,000 tok/s at batch 64
POOL = 262144


def percentile(vals, q):
    s = sorted(vals)
    k = (len(s) - 1) * q / 100.0
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def read_rows(path):
    rows = [json.loads(line) for line in open(path) if line.strip()]
    rows.sort(key=lambda r: r["timestamp"])  # stable, like load_trace
    return rows


def reused_tokens(rows):
    """Per row: tokens covered by leading hash ids that an earlier row already has, rounded down to whole 128-token pages."""
    seen, out = set(), []
    for r in rows:
        ids = r.get("hash_ids") or []
        k = 0
        for h in ids:
            if h not in seen:
                break
            k += 1
        out.append(min(r["input_length"], k * HASH_BLOCK) // PAGE * PAGE)
        seen.update(ids)
    return out


def load_meta(path):
    mp = path.replace(".jsonl", ".meta.json")
    return json.load(open(mp)) if os.path.exists(mp) else {}


def describe(path):
    rows = read_rows(path)
    meta = load_meta(path)
    n = len(rows)
    ins = [r["input_length"] for r in rows]
    outs = [r["output_length"] for r in rows]
    kv = [i + OVERHEAD + o for i, o in zip(ins, outs)]
    span = meta.get("window", {}).get("out_span_s") or rows[-1]["timestamp"]
    reuse = sum(reused_tokens(rows)) / sum(ins)
    d = dict(path=os.path.basename(path), n=n, span=span, last=rows[-1]["timestamp"], rate=n / span,
             in_p50=percentile(ins, 50), in_p90=percentile(ins, 90), in_max=max(ins),
             out_p50=percentile(outs, 50), out_p90=percentile(outs, 90), out_max=max(outs),
             kv_mean=sum(kv) / n, kv_p50=percentile(kv, 50), kv_p90=percentile(kv, 90), kv_max=max(kv),
             reuse_page=reuse, decode_tok_s=sum(outs) / span, prefill_tok_s=sum(i + OVERHEAD for i in ins) / span,
             clamps=meta.get("clamps", {}), n_tiny_in=sum(i <= OVERHEAD for i in ins), cls=meta.get("cls_counts", {}),
             fluid20=meta.get("kv", {}).get("peak_inflight_kv_ratio_itl20ms"),
             total_kv_over_P=sum(kv) / POOL)
    if "max_tokens" in rows[0]:
        dec = [r["max_tokens"] for r in rows]
        d["declared_kv"] = sum(i + OVERHEAD + m for i, m in zip(ins, dec))
        d["generated_kv"] = sum(kv)
        d["declared_over_generated"] = d["declared_kv"] / d["generated_kv"]
        d["max_tokens_over_output"] = sum(dec) / sum(outs)
        d["n_over_declared"] = sum(m > o for m, o in zip(dec, outs))
    return d


def print_describe(paths):
    cols = ["file", "requests", "span s", "req/s", "in p50/p90/max", "out p50/p90/max", "final KV mean/p50/p90/max",
            "prefix reuse", "clamped out / in rows; input <= 20", "decode tok/s (% of 2k)", "prefill tok/s (% of 30k)", "fluid peak/P (20 ms)",
            "declared/generated KV"]
    print("| " + " | ".join(cols) + " |\n|" + "---|" * len(cols))
    for p in paths:
        d = describe(p)
        c = d["clamps"]
        print("| " + " | ".join([
            d["path"].replace(".jsonl", ""), str(d["n"]), f"{d['span']:.0f}", f"{d['rate']:.2f}",
            f"{d['in_p50']:.0f} / {d['in_p90']:.0f} / {d['in_max']}", f"{d['out_p50']:.0f} / {d['out_p90']:.0f} / {d['out_max']}",
            f"{d['kv_mean']:.0f} / {d['kv_p50']:.0f} / {d['kv_p90']:.0f} / {d['kv_max']}", f"{100 * d['reuse_page']:.1f}%",
            f"{c.get('output_clamped', '?')} / {c.get('input_clamped', '?')}; {d['n_tiny_in']}",
            f"{d['decode_tok_s']:.0f} ({100 * d['decode_tok_s'] / DECODE_CAP_TOK_S:.0f}%)",
            f"{d['prefill_tok_s']:.0f} ({100 * d['prefill_tok_s'] / 30000:.0f}%)",
            f"{d['fluid20']}", f"{d['declared_over_generated']:.2f}" if "declared_over_generated" in d else "-"]) + " |")


def write_meta(bases):
    for base in bases:
        for suffix, mode in (("_mtx2", "x2"), ("_mt2048", "fixed")):
            vp = base.replace(".jsonl", suffix + ".jsonl")
            meta = load_meta(base)
            vrows, brows = read_rows(vp), read_rows(base)
            assert len(vrows) == len(brows)
            d = describe(vp)
            offs = {v["hash_ids"][0] - b["hash_ids"][0] for v, b in zip(vrows, brows) if v.get("hash_ids")}
            assert len(offs) == 1, offs
            assert all(v["output_length"] == b["output_length"] and v["input_length"] == b["input_length"]
                       and v["timestamp"] == b["timestamp"] for v, b in zip(vrows, brows))
            meta["path"] = os.path.relpath(vp, REPO) if os.path.abspath(vp).startswith(REPO) else vp
            meta["variant_of"] = os.path.basename(base)
            meta["declared"] = dict(
                mode=mode, cap=2048, id_offset_vs_base=offs.pop(), n_over_declared=d["n_over_declared"],
                declared_kv_tokens=d["declared_kv"], generated_kv_tokens=d["generated_kv"],
                declared_over_generated=round(d["declared_over_generated"], 3),
                max_tokens_over_output=round(d["max_tokens_over_output"], 3),
                note="declared KV = input + 20 + max_tokens summed over all rows; the server stops each request after output_length tokens")
            json.dump(meta, open(vp.replace(".jsonl", ".meta.json"), "w"), indent=1)
            print(f"{vp}: declared/generated KV {d['declared_over_generated']:.3f}, {d['n_over_declared']} rows over-declared")


def run_sim(paths, policies, prefix_hits):
    import numpy as np
    import admit_sim_bagel as sim

    for p in paths:
        rows = read_rows(p)
        if "max_tokens" in rows[0] and any(r["max_tokens"] != r["output_length"] for r in rows):
            print(f"\n### {os.path.basename(p)}: declared max_tokens, which analysis/admit_sim_bagel.py cannot use in this "
                  f"tree; skipped (simulate the oracle file)")
            continue
        arr = np.array([r["timestamp"] for r in rows])
        ins = np.array([r["input_length"] for r in rows], dtype=np.int64)
        outs = np.array([r["output_length"] for r in rows], dtype=np.int64)
        if prefix_hits:
            ins = np.maximum(ins - np.array(reused_tokens(rows), dtype=np.int64), 1)
        prompts = ins + OVERHEAD
        short = np.array([r["cls"] == "short" for r in rows])
        tag = " with every in-file prefix hit served from the cache (optimistic bound)" if prefix_hits else ""
        print(f"\n### {os.path.basename(p)}{tag}  ({len(rows)} requests, {int(short.sum())} short; "
              f"{prompts.sum() / rows[-1]['timestamp']:.0f} prefill tok/s offered)\n")
        print("| policy | unfin | >600s | short E2E mean (p99) | all E2E mean (p99) | short TTFT mean | out tok/s | makespan s "
              "| mean admitted | peak KV / P |\n|" + "---|" * 10)
        for pol in policies:
            a = sim.simulate(arr, prompts, outs, ~short, sched=pol, cap=sim.POOL_TOKENS)  # "short" = cls short
            b = sim.simulate(arr, prompts, outs, np.zeros(len(rows), dtype=bool), sched=pol, cap=sim.POOL_TOKENS)  # all
            print(f"| {pol} | {a['unfinished']} | {a['over_600s']} | {a['short_e2e_mean']:.2f} ({a['short_e2e_p99']:.1f}) | "
                  f"{b['short_e2e_mean']:.2f} ({b['short_e2e_p99']:.1f}) | {a['short_ttft_mean']:.2f} | {a['out_tok_s']:.0f} | "
                  f"{a['makespan']:.0f} | {a['mean_admitted']:.1f} | {a['peak_kv_ratio']:.2f} |", flush=True)


WINDOW_STATS = ["rate", "in_mean", "in_p50", "in_p90", "out_mean", "out_p50", "kv_mean", "kv_demand"]
PREC = {k: (".2f" if k == "rate" else ".1f") for k in WINDOW_STATS}


def window_stats(rows, length):
    """The eight statistics of the rows (dicts with t, inp, out) of a window of `length` original seconds."""
    ins, outs = [r["inp"] for r in rows], [r["out"] for r in rows]
    kv = [i + OVERHEAD + o for i, o in zip(ins, outs)]
    return dict(rate=len(rows) / length, in_mean=sum(ins) / len(ins), in_p50=percentile(ins, 50), in_p90=percentile(ins, 90),
                out_mean=sum(outs) / len(outs), out_p50=percentile(outs, 50), kv_mean=sum(kv) / len(kv),
                kv_demand=sum(kv) / length)


def pick_window(source, rates, centre):
    import bisect

    import convert_public_trace as cv

    src = os.path.join(REPO, "data", "public_traces", cv.DEFAULT_SRC[source])
    rows = cv.read_mooncake(src) if source == "mooncake" else cv.read_azure(src)
    rows.sort(key=lambda r: r["t"])
    for r in rows:  # the clamps of convert_public_trace.py
        r["out"] = min(r["output_length"], 2000)
        r["inp"] = min(r["input_length"], 32768 - OVERHEAD - r["out"])
    ts = [r["t"] for r in rows]
    t_last = ts[-1]
    whole = window_stats(rows, t_last)

    def win(start, length):
        return rows[bisect.bisect_left(ts, start):bisect.bisect_left(ts, start + length)]

    def score(s):
        return (sum(((s[k] - whole[k]) / whole[k]) ** 2 for k in WINDOW_STATS) / len(WINDOW_STATS)) ** 0.5

    half = 300.0 * max(rates)
    centres = [c for c in range(int(half), int(t_last - half) + 1, 10)]
    table = []
    for c in centres:
        sc = [score(window_stats(win(c - 300 * r, 600 * r), 600 * r)) for r in rates]
        table.append((sum(sc) / len(sc), c, sc))
    table.sort()
    print(f"{source}: {len(rows)} rows, last request {t_last:.0f} s; whole trace: "
          + ", ".join(f"{k} {whole[k]:.2f}" for k in WINDOW_STATS))
    print("\nbest centres by mean score over rates " + str(rates) + ":\n\n| centre s | mean score | score per rate |\n|---|---|---|")
    for m, c, sc in table[:6]:
        print(f"| {c} | {m:.3f} | " + ", ".join(f"{x:.3f}" for x in sc) + " |")
    c = centre if centre is not None else table[0][1]
    print(f"\ncentre {c} s: window statistics (percentile rank among the windows of that length on the 10-s grid)\n")
    print("| r | window s | start | " + " | ".join(WINDOW_STATS) + " | score |\n|" + "---|" * (len(WINDOW_STATS) + 4))
    print("| whole | %.0f | 0 | " % t_last + " | ".join(f"{whole[k]:{PREC[k]}}" for k in WINDOW_STATS) + " | 0 |")
    for r in rates:
        length = 600 * r
        s = window_stats(win(c - 300 * r, length), length)
        allw = [window_stats(win(st, length), length) for st in range(0, int(t_last - length) + 1, 10)]
        ranks = {k: 100.0 * sum(w[k] < s[k] for w in allw) / len(allw) for k in WINDOW_STATS}
        print(f"| {r:g} | {length:.0f} | {c - 300 * r:g} | " + " | ".join(f"{s[k]:{PREC[k]}} (p{ranks[k]:.0f})" for k in WINDOW_STATS)
              + f" | {score(s):.3f} |")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="*")
    ap.add_argument("--sim", action="store_true", help="run the admission simulator instead of the characterisation")
    ap.add_argument("--prefix-hits", action="store_true", help="with --sim: cut every prompt by its in-file prefix hits")
    ap.add_argument("--policies", nargs="+", default=["sum_fifo", "sum_bf", "peak_fifo", "peak_bf", "open"])
    ap.add_argument("--write-meta", action="store_true",
                    help="given BASE files, fill the .meta.json of their _mtx2 / _mt2048 variants (path, declared block)")
    ap.add_argument("--pick-window", choices=["mooncake", "azure_conv"], help="score window centres (see above)")
    ap.add_argument("--rates", nargs="+", type=float, default=[0.5, 0.75, 1.0], help="with --pick-window")
    ap.add_argument("--centre", type=float, help="with --pick-window: print the windows of this centre")
    a = ap.parse_args()
    if a.pick_window:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        pick_window(a.pick_window, a.rates, a.centre)
        return
    if not a.files:
        ap.error("files required")
    if a.write_meta:
        write_meta(a.files)
    elif a.sim:
        run_sim(a.files, a.policies, a.prefix_hits)
    else:
        print_describe(a.files)


if __name__ == "__main__":
    main()
