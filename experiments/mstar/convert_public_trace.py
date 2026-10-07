#!/usr/bin/env python3
"""Convert a window of a public production trace into a BAGEL timed-trace JSONL (data/workloads/ format).

Sources (download them with experiments/mstar/fetch_public_traces.sh into data/public_traces/):
  mooncake    Mooncake (Kimi) FAST'25 conversation trace: JSONL, timestamp in ms, input_length, output_length, hash_ids
              (prefix-chained block ids, 512 tokens per block, FAST25-release/README.md).
  azure_conv  Azure LLM inference trace 2023, conversation: CSV TIMESTAMP, ContextTokens, GeneratedTokens (no prefix info).

Output row (one per request, sorted by timestamp): timestamp (s), input_length, output_length, cls, src, hash_ids.
A <out>.meta.json is written next to it in the shape make_workload.py writes (plus provenance, window, clamps, class
thresholds); experiments/mstar/kv_report.py adds the "kv" block afterwards.

Window and rate, precisely. Original time t is seconds since the first request of the source file. The window is
[--start, --start + D) of the ORIGINAL trace, with D = --duration, or D = --span * --rate when --span is given. A request in
the window gets output timestamp (t - start) / rate, so --rate r multiplies the request rate by r (r = 0.5: half the rate,
twice the span) and the output spans D / r seconds. --span S therefore makes every output span ~S seconds whatever r is
(D = S * r original seconds). Rows are kept as they are; only their arrival times are scaled.

Clamps (BAGEL limits: MAX_OUTPUT_TOKENS 2048, max_seq_len 32768, 20-token chat template; every clamp is counted and printed):
  output_length  <- min(output_length, --max-output)                        (default 2000)
  input_length   <- min(input_length, max_seq_len - overhead - output_length)  (overhead 20, max_seq_len 32768)
so input + 20 + output <= 32768 for every row. Rows with input_length < 1 or output_length < 1 are an error (as in
benchmark/trace_replay.py load_trace, which also rejects timestamp < 0).

hash_ids (16 tokens per id, benchmark/trace_replay.py HASH_BLOCK):
  mooncake    each 512-token block id b becomes the 32 ids b * 32 + j (j = 0..31) + --id-offset, in order, truncated to
              ceil(clamped input_length / 16) ids. Two requests that share their first k blocks share their first k * 32 ids.
  azure_conv  no prefix information: every request gets its own ids, --id-offset + AZURE_ID_BASE + 2048 * (row index in the
              source file) + j, so no two requests share a prefix, the ids of one request are the same in every window, and
              they never meet a Mooncake id. (Rows without hash_ids would build their prompt from the row index, so a base
              file and a declare_max_tokens.py variant replayed on one server would share tokens; ids avoid that, because
              declare_max_tokens.py --id-offset shifts them.)
  --id-offset (default 3,000,000,000) keeps every id far above the small ids of make_workload.py's synthetic traces.

cls, per OUTPUT file, on the final KV = input_length + 20 + output_length of each row (after the clamps):
  "short" if final KV <= the file's 50th percentile; "long" if final KV >= its 90th percentile; otherwise "mid".
(percentiles with linear interpolation, as numpy's default.)

  python3 experiments/mstar/convert_public_trace.py mooncake --start 1350 --span 600 --rate 0.5 \\
      --out data/workloads/pub_mooncake_conv_r0.5.jsonl
"""
import argparse
import csv
import hashlib
import json
import math
import os
import sys
from datetime import datetime, timezone

HASH_BLOCK = 16  # tokens per hash id (benchmark/trace_replay.py)
MOONCAKE_BLOCK = 512  # tokens per Mooncake hash id (FAST25-release/README.md)
IDS_PER_BLOCK = MOONCAKE_BLOCK // HASH_BLOCK  # 32
AZURE_ID_BASE = 100_000_000  # Azure ids start here above --id-offset; Mooncake ids end below ~6 million above it
AZURE_ID_STRIDE = 2048  # ids reserved per Azure request: 32768 / 16
PAGE = 128  # the server's prefix-cache page (tokens)
DEFAULT_SRC = {"mooncake": "mooncake_conversation.jsonl", "azure_conv": "azure_conv_2023.csv"}
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def percentile(sorted_vals, q):
    """q-th percentile (0-100) of a sorted list, linear interpolation (numpy's default)."""
    k = (len(sorted_vals) - 1) * q / 100.0
    lo = math.floor(k)
    hi = min(lo + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (k - lo)


def read_mooncake(path):
    """Rows of {t (s), idx, input_length, output_length, blocks}, in file order."""
    rows = []
    with open(path) as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                o = json.loads(line)
                rows.append(dict(t=float(o["timestamp"]) / 1000.0, idx=len(rows), input_length=int(o["input_length"]),
                                 output_length=int(o["output_length"]), blocks=[int(b) for b in o["hash_ids"]]))
            except (KeyError, TypeError, ValueError) as e:
                raise ValueError(f"{path}:{lineno}: bad Mooncake row ({e!r}): {line[:120]}") from e
    return rows


def parse_azure_time(s):
    """'2023-11-16 18:15:46.6805900' -> epoch-like seconds (fraction kept to 7 digits)."""
    whole, _, frac = s.strip().partition(".")
    base = datetime.strptime(whole, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc).timestamp()  # only differences are used
    return base + (float("0." + frac) if frac else 0.0)


def read_azure(path):
    rows = []
    with open(path, newline="") as f:
        rd = csv.DictReader(f)
        for lineno, o in enumerate(rd, 2):
            try:
                rows.append(dict(t=parse_azure_time(o["TIMESTAMP"]), idx=len(rows), input_length=int(o["ContextTokens"]),
                                 output_length=int(o["GeneratedTokens"])))
            except (KeyError, TypeError, ValueError) as e:
                raise ValueError(f"{path}:{lineno}: bad Azure row ({e!r}): {o}") from e
    t0 = min(r["t"] for r in rows)
    for r in rows:
        r["t"] -= t0
    return rows


def expand_ids(blocks, n_ids, offset):
    """Mooncake: 512-token block ids -> 16-token ids b * 32 + j + offset, truncated to n_ids."""
    out = []
    for b in blocks:
        base = b * IDS_PER_BLOCK + offset
        out.extend(range(base, base + IDS_PER_BLOCK))
        if len(out) >= n_ids:
            break
    return out[:n_ids]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", choices=sorted(DEFAULT_SRC))
    ap.add_argument("--src", help="source file (default data/public_traces/<standard name> under the repo root)")
    ap.add_argument("--out", required=True, help="output timed-trace JSONL; <out>.meta.json is written next to it")
    ap.add_argument("--start", type=float, required=True, help="window start, seconds of the ORIGINAL trace")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--duration", type=float, help="window length, seconds of the ORIGINAL trace")
    g.add_argument("--span", type=float, help="wanted output span in seconds: the window is span * rate original seconds")
    ap.add_argument("--rate", type=float, default=1.0,
                    help="request-rate multiplier r: output timestamp = (t - start) / r (default 1)")
    ap.add_argument("--max-output", type=int, default=2000, help="clamp for output_length (BAGEL MAX_OUTPUT_TOKENS is 2048)")
    ap.add_argument("--max-seq-len", type=int, default=32768)
    ap.add_argument("--overhead", type=int, default=20, help="BAGEL chat-template tokens added to every prompt")
    ap.add_argument("--id-offset", type=int, default=3_000_000_000, help="added to every hash id")
    ap.add_argument("--allow-partial", action="store_true", help="allow a window that runs past the last request")
    a = ap.parse_args(argv)

    if a.rate <= 0 or a.start < 0:
        sys.exit("need --rate > 0 and --start >= 0")
    window = a.duration if a.duration is not None else a.span * a.rate
    src = a.src or os.path.join(REPO, "data", "public_traces", DEFAULT_SRC[a.source])
    rows = read_mooncake(src) if a.source == "mooncake" else read_azure(src)
    if not rows:
        sys.exit(f"{src}: empty")
    t_last = max(r["t"] for r in rows)
    if a.start + window > t_last + 1e-6 and not a.allow_partial:
        sys.exit(f"window [{a.start}, {a.start + window}) runs past the last request at {t_last:.1f} s (--allow-partial)")
    with open(src, "rb") as f:
        src_sha = hashlib.sha256(f.read()).hexdigest()

    win = sorted((r for r in rows if a.start <= r["t"] < a.start + window), key=lambda r: (r["t"], r["idx"]))
    if not win:
        sys.exit("no request in the window")
    max_total = a.max_seq_len - a.overhead
    clamp = dict(output_clamped=0, output_tokens_cut=0, input_clamped=0, input_tokens_cut=0)
    out_rows = []
    for r in win:
        n_in, n_out = r["input_length"], r["output_length"]
        if n_in < 1 or n_out < 1:
            raise ValueError(f"source row {r['idx']}: need input_length >= 1 and output_length >= 1, got {n_in}, {n_out}")
        if n_out > a.max_output:
            clamp["output_clamped"] += 1
            clamp["output_tokens_cut"] += n_out - a.max_output
            n_out = a.max_output
        if n_in > max_total - n_out:
            clamp["input_clamped"] += 1
            clamp["input_tokens_cut"] += n_in - (max_total - n_out)
            n_in = max_total - n_out
        n_ids = math.ceil(n_in / HASH_BLOCK)
        if a.source == "mooncake":
            if len(r["blocks"]) * MOONCAKE_BLOCK < r["input_length"]:
                raise ValueError(f"source row {r['idx']}: {len(r['blocks'])} blocks cannot hold {r['input_length']} tokens")
            ids = expand_ids(r["blocks"], n_ids, a.id_offset)
        else:
            base = a.id_offset + AZURE_ID_BASE + AZURE_ID_STRIDE * r["idx"]
            ids = list(range(base, base + n_ids))
        ts = round((r["t"] - a.start) / a.rate, 6)
        if ts < 0:
            raise ValueError(f"source row {r['idx']}: negative timestamp {ts}")
        out_rows.append(dict(timestamp=ts, input_length=n_in, output_length=n_out, src=a.source, hash_ids=ids))

    kvs = sorted(r["input_length"] + a.overhead + r["output_length"] for r in out_rows)
    p50, p90 = percentile(kvs, 50), percentile(kvs, 90)
    cls_counts = {"short": 0, "mid": 0, "long": 0}
    for r in out_rows:
        kv = r["input_length"] + a.overhead + r["output_length"]
        r["cls"] = "short" if kv <= p50 else ("long" if kv >= p90 else "mid")
        cls_counts[r["cls"]] += 1

    # prefix reuse inside the output file: tokens of each row covered by leading ids that an earlier row already has
    seen = set()
    reused = reused_pages = 0
    for r in out_rows:
        k = 0
        for h in r["hash_ids"]:
            if h not in seen:
                break
            k += 1
        tok = min(r["input_length"], k * HASH_BLOCK)
        reused += tok
        reused_pages += tok // PAGE * PAGE
        seen.update(r["hash_ids"])

    # validate as benchmark/trace_replay.py load_trace does, and that the rows are sorted
    prev = -1.0
    for i, r in enumerate(out_rows):
        if r["timestamp"] < 0 or r["input_length"] < 1 or r["output_length"] < 1 or r["timestamp"] < prev:
            raise ValueError(f"output row {i} invalid: {r['timestamp']} {r['input_length']} {r['output_length']}")
        if r["input_length"] + a.overhead + r["output_length"] > a.max_seq_len or r["output_length"] > a.max_output:
            raise ValueError(f"output row {i} breaks a limit after clamping")
        prev = r["timestamp"]

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        for r in out_rows:
            f.write(json.dumps({k: r[k] for k in ("timestamp", "input_length", "output_length", "cls", "src", "hash_ids")},
                               separators=(",", ":")) + "\n")

    ins = sorted(r["input_length"] for r in out_rows)
    outs = sorted(r["output_length"] for r in out_rows)
    span = out_rows[-1]["timestamp"]
    meta = dict(
        path=os.path.relpath(a.out, REPO) if os.path.abspath(a.out).startswith(REPO) else a.out,
        n=len(out_rows), duration=span, rate=len(out_rows) / max(span, 1e-9),
        in_p50=percentile(ins, 50), out_p50=percentile(outs, 50), out_mean=sum(outs) / len(outs), out_max=outs[-1],
        cls_counts=cls_counts, total_out_tokens=sum(outs), total_in_tokens=sum(ins),
        source=dict(name=a.source, file=os.path.basename(src), sha256=src_sha, rows=len(rows), last_request_s=t_last),
        window=dict(start_s=a.start, length_s=window, rate_multiplier=a.rate, out_span_s=window / a.rate,
                    rule="out timestamp = (t - start) / rate; window = [start, start + length) of the original trace"),
        clamps=dict(max_output=a.max_output, max_seq_len=a.max_seq_len, overhead=a.overhead, **clamp),
        cls_rule=dict(basis="final KV = input_length + overhead + output_length of the output file",
                      short="<= p50", long=">= p90", mid="otherwise", p50_kv=p50, p90_kv=p90),
        hash_ids=dict(id_offset=a.id_offset, tokens_per_id=HASH_BLOCK,
                      prefix_reuse_token_frac=reused / sum(ins), prefix_reuse_page128_token_frac=reused_pages / sum(ins)),
    )
    with open(a.out.replace(".jsonl", ".meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(f"{a.out}: {len(out_rows)} requests, window [{a.start:g}, {a.start + window:g}) s, rate x{a.rate:g} -> span {span:.1f} s, "
          f"{len(out_rows) / span:.2f} req/s")
    print(f"  clamps: output > {a.max_output}: {clamp['output_clamped']} rows ({clamp['output_tokens_cut']} tokens cut); "
          f"input > {max_total} - output: {clamp['input_clamped']} rows ({clamp['input_tokens_cut']} tokens cut)")
    print(f"  final KV p50 {p50:.0f}, p90 {p90:.0f}; cls {cls_counts}; prefix reuse {100 * reused / sum(ins):.1f}% of input tokens "
          f"({100 * reused_pages / sum(ins):.1f}% at 128-token pages)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
