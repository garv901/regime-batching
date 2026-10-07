"""Serving-run analysis for Stages 2-3: regime occupancy, H1-H5 metrics and matched per-request latency, per case.

Definitions (fixed before the Stage 2 data was read):
- Labels. Prefill-bearing steps get the tile-model label (attn_step_model regime2: X, A, P, M) at the case's capture
  size; an A step with a decode of >= 16k context in the same launch becomes A+L. Decode-only steps keep the engine's
  label (Dc, Dl, Ds, Dk, Dw).
- Occupancy: step share and measured GPU-time share per label.
- H1: X_cost = X label (modelled default attention >= 1.2x the best split bound) and modelled excess (default - best,
  all layers) >= 1 ms; its share of GPU time. Also: structural X share, and modelled attention time recoverable by the
  best split bound and by separate prefill/decode launches, as shares of GPU time.
- H2: launch-bound steps above 512 tokens (T > 512 and cpu_ms >= 0.9 gpu_ms), share of GPU time; also the share of
  GPU time in eager steps (graph mode NONE).
- H3: decode-weighted time = sum over steps of gpu_ms x decodes in the step; the share held by prefill-bearing steps.
- H5: trace-weighted FA3 prefill efficiency = sum t_ideal / sum t_default over prefill-bearing steps (tile model: the
  launch's tiles packed perfectly over the SMs vs the default launch).
- Served prefix hits: tokens already computed when a request is first scheduled, over its input length. With
  --wl GROUP=WORKLOAD.jsonl (hash_ids, 512-token blocks) also the unlimited-cache estimate for the same requests.
- Latency: bench.json per request, in dataset order (the same order in every case; checked on input lengths). Each
  case against its group's `<group>_default` (group = case name up to the first `_`): mean, p50, p99 of TTFT, TPOT,
  ITL, E2E and the median per-request ratio.

Usage: python3 analysis/stage_regimes.py RUN_ROOT [RUN_ROOT ...] [--wl mc10=data/workloads/chat_mc10.jsonl]
           [--out FILE.md] [--json FILE.json] [--steps FILE.jsonl] [--procs 16] [--cases smoke ...]
"""
import argparse, collections, glob, json, os, sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from regime_batching.features import ModelGeom, RegimeConfig, attn_step_model  # noqa: E402

DECODE_LABELS = ["Dc", "Dl", "Ds", "Dk", "Dw"]
LABELS = ["X", "A+L", "A", "P", "M"] + DECODE_LABELS
PREFILL_LABELS = {"X", "A+L", "A", "P", "M"}


def load_case(case_dir):
    meta, steps = None, []
    for f in sorted(glob.glob(os.path.join(case_dir, "telemetry", "steps_*.jsonl"))):
        for line in open(f):
            r = json.loads(line)
            if r.get("kind") == "meta":
                meta = r
            elif r.get("kind") == "step" and r.get("reqs") and r.get("gpu_ms") is not None \
                    and not r["reqs"][0][0].startswith("_warmup"):
                steps.append(r)
    bench = json.load(open(os.path.join(case_dir, "bench.json"))) if os.path.exists(f"{case_dir}/bench.json") else None
    return meta, steps, bench


def price(args):
    s, geom_d, cfg_d, full_cg = args
    geom, cfg = ModelGeom(**geom_d), RegimeConfig(**cfg_d)
    q = np.array([a[1] for a in s["reqs"]], dtype=np.int64)
    kv = np.array([a[2] for a in s["reqs"]], dtype=np.int64)
    dec = q == 1
    row = dict(step=s["step"], t0=s.get("t0"), T=s["T"], gpu_ms=s["gpu_ms"], cpu_ms=s.get("cpu_ms"),
               mode=(s.get("cg") or {}).get("mode"), n_dec=int(dec.sum()), n_pre=int((~dec).sum()),
               max_dec_kv=int(kv[dec].max()) if dec.any() else 0,
               c3=(s.get("attn") or {}).get("path") == "x_split")
    if not (~dec).any():
        return dict(row, label=s["regime"], x_struct=False, x_cost=False, att_def=None)
    m = attn_step_model(q, kv, geom, cfg, full_cg=full_cg)
    label = m["regime2"]
    if label == "A" and row["max_dec_kv"] >= cfg.x_struct_min_L:
        label = "A+L"
    L = geom.num_layers
    att = {k: L * m[t] * 1e3 for k, t in [("att_def", "t_default"), ("att_best", "t_best"), ("att_sep", "t_sep"),
                                           ("att_ideal", "t_ideal")]}
    return dict(row, label=label, x_struct=m["structural_x"], x_ratio=m["x_cost"], pre_prefix=m["pre_prefix"],
                x_cost=label == "X" and att["att_def"] - att["att_best"] >= 1.0, **att)


def pct(a, p):
    return float(np.percentile(a, p)) if len(a) else float("nan")


def bench_requests(b):
    """Per request (dataset order): ttft, tpot, e2e in ms; itls flattened per request."""
    out = []
    for i in range(len(b["input_lens"])):
        ok = not b["errors"][i]
        ttft, lat, n = b["ttfts"][i], b["latencies"][i], b["output_lens"][i]
        out.append(dict(ok=ok, inp=b["input_lens"][i], out=n, ttft=ttft * 1e3 if ok else None,
                        tpot=(lat - ttft) / (n - 1) * 1e3 if ok and n > 1 else None,
                        e2e=lat * 1e3 if ok else None, itl=[x * 1e3 for x in b["itls"][i]] if ok else []))
    return out


def latency_summary(reqs):
    d = {}
    for k in ("ttft", "tpot", "e2e"):
        v = np.array([r[k] for r in reqs if r[k] is not None])
        d[k] = dict(mean=float(v.mean()), p50=pct(v, 50), p99=pct(v, 99))
    itl = np.concatenate([np.array(r["itl"]) for r in reqs if r["itl"]] or [np.zeros(0)])
    d["itl"] = dict(mean=float(itl.mean()), p50=pct(itl, 50), p99=pct(itl, 99))
    return d


def matched(base, other):
    """Ratios other / base over requests completed in both (same dataset index)."""
    if [r["inp"] for r in base] != [r["inp"] for r in other]:
        return None
    both = [(a, b) for a, b in zip(base, other) if a["ok"] and b["ok"]]
    res = dict(n=len(both))
    for k in ("ttft", "tpot", "e2e"):
        pairs = [(a[k], b[k]) for a, b in both if a[k] and b[k]]
        x, y = np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])
        res[k] = dict(mean=float(y.mean() / x.mean()), p50=pct(y, 50) / pct(x, 50), p99=pct(y, 99) / pct(x, 99),
                      per_req_median=float(np.median(y / x)))
    ia = np.concatenate([np.array(a["itl"]) for a, b in both if a["itl"]])
    ib = np.concatenate([np.array(b["itl"]) for a, b in both if b["itl"]])
    res["itl"] = dict(mean=float(ib.mean() / ia.mean()), p50=pct(ib, 50) / pct(ia, 50), p99=pct(ib, 99) / pct(ia, 99))
    return res


def req_index(rid):
    p = rid.split("-")  # cmpl-bench-<hex>-<index>-<n>-<hex>
    try:
        return int(p[3])
    except (IndexError, ValueError):
        return None


def served_hits(steps, reqs):
    first = {}
    for s in steps:
        for a in s["reqs"]:
            first.setdefault(a[0], a[2] - a[1])
    hit = inp = n_hit = 0
    for rid, h in first.items():
        i = req_index(rid)
        if i is None or i >= len(reqs):
            continue
        hit += h
        inp += reqs[i]["inp"]
        n_hit += h > 0
    return dict(hit_tokens=hit, input_tokens=inp, share=hit / inp if inp else float("nan"), requests_with_hit=n_hit,
                requests=len(first))


def unlimited_cache_estimate(path, n, block=512):
    seen, hit, inp = set(), 0, 0
    for line in open(path).readlines()[:n]:
        r = json.loads(line)
        h = r.get("hash_ids") or []
        lead = 0
        for x in h:
            if x not in seen:
                break
            lead += 1
        hit += min(lead * block, r["input_length"] - 1)
        inp += r["input_length"]
        seen.update(h)
    return dict(share=hit / inp, hit_tokens=hit, input_tokens=inp)


def case_metrics(rows):
    gpu = sum(r["gpu_ms"] for r in rows)
    by = collections.defaultdict(lambda: [0, 0.0])
    for r in rows:
        by[r["label"]][0] += 1
        by[r["label"]][1] += r["gpu_ms"]
    pre = [r for r in rows if r["att_def"] is not None]
    share = lambda f: sum(r["gpu_ms"] for r in rows if f(r)) / gpu  # noqa: E731
    dw = sum(r["gpu_ms"] * r["n_dec"] for r in rows)
    sum_def = sum(r["att_def"] for r in pre)
    tb = collections.Counter()
    for r in pre:
        tb["<=512" if r["T"] <= 512 else "513-2048" if r["T"] <= 2048 else ">2048"] += r["gpu_ms"]
    return dict(
        steps=len(rows), gpu_s=gpu / 1e3,
        labels={k: dict(steps=v[0], step_share=v[0] / len(rows), gpu_share=v[1] / gpu) for k, v in by.items()},
        prefill_gpu_share=share(lambda r: r["label"] in PREFILL_LABELS),
        prefill_gpu_by_T={k: v / gpu for k, v in tb.items()},
        long_decode_step_gpu_share=share(lambda r: r["max_dec_kv"] >= 16384),
        h1=dict(x_struct=share(lambda r: r["x_struct"]), x_label=share(lambda r: r["label"] == "X"),
                x_cost=share(lambda r: r["x_cost"]),
                recoverable_best_split=sum(r["att_def"] - min(r["att_def"], r["att_best"]) for r in pre) / gpu,
                recoverable_sep_launch=sum(r["att_def"] - min(r["att_def"], r["att_sep"]) for r in pre) / gpu,
                c3_forced_steps=sum(r["c3"] for r in rows), c3_forced_gpu=share(lambda r: r["c3"])),
        h2=dict(launch_bound_T512=share(lambda r: r["T"] > 512 and r["cpu_ms"] is not None
                                        and r["cpu_ms"] >= 0.9 * r["gpu_ms"]),
                eager=share(lambda r: r["mode"] == "NONE")),
        h3=dict(prefill_decode_weighted=sum(r["gpu_ms"] * r["n_dec"] for r in rows if r["att_def"] is not None) / dw
                if dw else float("nan")),
        h5=dict(prefill_eff=sum(r["att_ideal"] for r in pre) / sum_def if sum_def else float("nan"),
                recoverable_perfect_pack=sum(r["att_def"] - r["att_ideal"] for r in pre) / gpu,
                modelled_att_share=sum_def / gpu),
    )


def fmt(x, p=1):
    return "n/a" if x is None or x != x else f"{100 * x:.{p}f}%"


def markdown(res):
    out = []
    cases = list(res)
    out += ["## Occupancy: GPU-time share by label (step share in brackets)", "",
            "| case | steps | GPU s | " + " | ".join(LABELS) + " |", "|---|---|---|" + "---|" * len(LABELS)]
    for c in cases:
        m = res[c]["steps"]
        cell = lambda k: (f"{fmt(m['labels'][k]['gpu_share'])} ({fmt(m['labels'][k]['step_share'])})"  # noqa: E731
                          if k in m["labels"] else "-")
        out.append(f"| {c} | {m['steps']} | {m['gpu_s']:.1f} | " + " | ".join(cell(k) for k in LABELS) + " |")
    out += ["", "## H1-H5 step metrics (shares of measured GPU time unless noted)", "",
            "| case | prefill-bearing | by T <=512 / 513-2048 / >2048 | steps with a >=16k decode | X shape | X label "
            "| X_cost | recoverable: split bound / separate launches | C3 forced | launch-bound T>512 | eager "
            "| H3 decode-weighted prefill share | H5 prefill eff. | H5 perfect-pack recoverable |",
            "|---|" + "---|" * 13]
    for c in cases:
        m = res[c]["steps"]
        b = m["prefill_gpu_by_T"]
        out.append(
            f"| {c} | {fmt(m['prefill_gpu_share'])} | {fmt(b.get('<=512', 0))} / {fmt(b.get('513-2048', 0))} / "
            f"{fmt(b.get('>2048', 0))} | {fmt(m['long_decode_step_gpu_share'])} | {fmt(m['h1']['x_struct'])} | "
            f"{fmt(m['h1']['x_label'])} | {fmt(m['h1']['x_cost'])} | {fmt(m['h1']['recoverable_best_split'], 2)} / "
            f"{fmt(m['h1']['recoverable_sep_launch'], 2)} | {m['h1']['c3_forced_steps']} ({fmt(m['h1']['c3_forced_gpu'])}) | "
            f"{fmt(m['h2']['launch_bound_T512'], 2)} | {fmt(m['h2']['eager'])} | {fmt(m['h3']['prefill_decode_weighted'])} | "
            f"{fmt(m['h5']['prefill_eff'])} | {fmt(m['h5']['recoverable_perfect_pack'], 2)} |")
    out += ["", "## Latency (ms, all requests) and served prefix hits", "",
            "| case | completed | TTFT mean / p50 / p99 | TPOT mean / p50 / p99 | ITL p50 / p99 | E2E mean / p99 "
            "| served hit share (unlimited-cache estimate) |", "|---|" + "---|" * 6]
    for c in cases:
        r = res[c]
        if not r.get("latency"):
            continue
        l, h = r["latency"], r["hits"]
        est = f" ({fmt(r['hit_estimate']['share'])})" if r.get("hit_estimate") else ""
        out.append(f"| {c} | {r['completed']} | {l['ttft']['mean']:.1f} / {l['ttft']['p50']:.1f} / {l['ttft']['p99']:.1f} | "
                   f"{l['tpot']['mean']:.2f} / {l['tpot']['p50']:.2f} / {l['tpot']['p99']:.2f} | "
                   f"{l['itl']['p50']:.2f} / {l['itl']['p99']:.2f} | {l['e2e']['mean']:.0f} / {l['e2e']['p99']:.0f} | "
                   f"{fmt(h['share'])}{est} |")
    out += ["", "## Matched requests: case / group default (ratio; < 1 is faster)", "",
            "| case | n | TTFT mean / p99 / per-request median | TPOT mean / p99 / per-request median | ITL p50 / p99 "
            "| E2E mean |", "|---|" + "---|" * 5]
    for c in cases:
        mt = res[c].get("matched")
        if not mt:
            continue
        f3 = lambda d: f"{d['mean']:.3f} / {d['p99']:.3f} / {d['per_req_median']:.3f}"  # noqa: E731
        out.append(f"| {c} | {mt['n']} | {f3(mt['ttft'])} | {f3(mt['tpot'])} | {mt['itl']['p50']:.3f} / "
                   f"{mt['itl']['p99']:.3f} | {mt['e2e']['mean']:.3f} |")
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="+")
    ap.add_argument("--wl", action="append", default=[], help="GROUP=workload.jsonl (hash_ids) for the hit estimate")
    ap.add_argument("--cases", nargs="*", help="only these case names")
    ap.add_argument("--out")
    ap.add_argument("--json")
    ap.add_argument("--steps")
    ap.add_argument("--procs", type=int, default=16)
    a = ap.parse_args()
    wl = dict(x.split("=", 1) for x in a.wl)
    res, reqs_by_case, step_rows = {}, {}, []
    with Pool(a.procs) as pool:
        for root in a.roots:
            for case_dir in sorted(glob.glob(os.path.join(root, "*", "telemetry"))):
                case_dir = os.path.dirname(case_dir)
                case = os.path.basename(case_dir)
                if (a.cases and case not in a.cases) or (not a.cases and case == "smoke"):
                    continue
                key = case if case not in res else f"{os.path.basename(root.rstrip('/'))}:{case}"
                meta, steps, bench = load_case(case_dir)
                if not steps:
                    print(f"{case}: no steps")
                    continue
                g = meta["geom"]
                geom_d = dict(num_layers=g["num_layers"], num_q_heads=g["num_q_heads"],
                              num_kv_heads=g["num_kv_heads"], head_dim=g["head_dim"],
                              weight_bytes_per_step=g["weight_bytes_per_step"], kv_dtype_bytes=g.get("kv_dtype_bytes", 2))
                cfg_d = {k: v for k, v in (meta.get("regime_cfg") or {}).items()
                         if k in RegimeConfig.__dataclass_fields__}
                cfg_d["cg_capture_max"] = meta["capture_max"]
                rows = pool.map(price, [(s, geom_d, cfg_d, meta.get("full_cg", True)) for s in steps], chunksize=32)
                r = dict(case=case, root=root, capture_max=meta["capture_max"],
                         max_num_batched_tokens=meta.get("max_num_batched_tokens"), steps=case_metrics(rows))
                if bench:
                    reqs = bench_requests(bench)
                    reqs_by_case[key] = reqs
                    r.update(completed=bench["completed"], failed=bench["failed"], duration_s=bench["duration"],
                             latency=latency_summary(reqs), hits=served_hits(steps, reqs))
                    grp = case.split("_")[0]
                    if grp in wl:
                        r["hit_estimate"] = unlimited_cache_estimate(wl[grp], len(reqs))
                res[key] = r
                if a.steps:
                    step_rows += [dict(x, case=key) for x in rows]
                print(f"{key}: {len(rows)} steps priced", flush=True)
    for key, r in res.items():  # baseline: the group's default case from the same run root
        base = next((k for k, v in res.items() if v["case"] == f"{r['case'].split('_')[0]}_default"
                     and v["root"] == r["root"]), None)
        if base and key != base and key in reqs_by_case and base in reqs_by_case:
            r["matched"] = matched(reqs_by_case[base], reqs_by_case[key])
            if r["matched"] is None:
                print(f"{key}: input lengths differ from {base}; no matched comparison")
    md = markdown(res)
    print(md)
    for path, text in [(a.out, md), (a.json, json.dumps(res, indent=1)),
                       (a.steps, "\n".join(json.dumps(x) for x in step_rows) + "\n")]:
        if path:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text(text)


if __name__ == "__main__":
    main()
