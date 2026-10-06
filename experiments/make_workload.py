"""Generate oracle workloads in vLLM `timed_trace` JSONL format.

Each line: {"timestamp": s_from_start, "input_length": n, "output_length": m, "hash_ids": [...unique...],
            "cls": "short"|"long"|..., "src": ...}
`vllm bench serve --dataset-name timed_trace` replays timestamps, forces ignore_eos and sets
max_tokens = output_length -> the engine sees the exact output length (oracle) via max_tokens.
hash_ids are unique per request (16 tokens per id, the default chunk) -> random prompts, no prefix reuse.

Families:
  bimodal   : Poisson arrivals; short outputs ~U[s_lo,s_hi], long outputs fixed or from reasoning CDF.
  mix       : short requests resampled from a production trace (lengths) + long from reasoning lengths.
  replay    : a production trace replayed (timestamps rescaled to a target rate), lengths as-is.
  waves     : the motivating example -- every --wave-period s a wave of --wave-size requests arrives
              together: (wave-size - 1) short outputs ~U[short-lo, short-hi] + 1 long output.
"""
from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np
import pandas as pd

TR = os.path.join(os.path.dirname(__file__), "..", "data", "traces")
HASH_CHUNK = 16


def _load_lengths(name: str) -> pd.DataFrame:
    if name == "reasoning":
        df = pd.read_csv(os.path.join(TR, "reasoning_lengths.csv"))
        return df.rename(columns={c: c.lower() for c in df.columns})[["input_tokens", "output_tokens"]]
    if name.startswith("azure"):
        fn = {"azure2023_code": "AzureLLMInferenceTrace_code.csv", "azure2023_conv": "AzureLLMInferenceTrace_conv.csv",
              "azure2024_code": "AzureLLMInferenceTrace_code_1week.csv",
              "azure2024_conv": "AzureLLMInferenceTrace_conv_1week.csv"}[name]
        nrows = 2_000_000 if "2024" in name else None
        df = pd.read_csv(os.path.join(TR, fn), nrows=nrows)
        df = df.rename(columns={"ContextTokens": "input_tokens", "GeneratedTokens": "output_tokens",
                                "TIMESTAMP": "timestamp"})
        return df
    if name.startswith("burstgpt"):
        df = pd.read_csv(os.path.join(TR, f"BurstGPT_{name[-1]}.csv"), nrows=2_000_000)
        df = df.rename(columns={"Request tokens": "input_tokens", "Response tokens": "output_tokens",
                                "Timestamp": "timestamp"})
        return df[df.output_tokens > 0]
    if name.startswith("mooncake"):
        kind = name.split("_", 1)[1]
        df = pd.read_json(os.path.join(TR, f"{kind}_trace.jsonl"), lines=True)
        return df.rename(columns={"input_length": "input_tokens", "output_length": "output_tokens"})
    raise ValueError(name)


def write(reqs: list[dict], path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    reqs.sort(key=lambda r: r["timestamp"])
    base = 1
    with open(path, "w") as f:
        for r in reqs:
            n_ids = math.ceil(r["input_length"] / HASH_CHUNK)
            r["hash_ids"] = list(range(base, base + n_ids))
            base += n_ids
            f.write(json.dumps(r) + "\n")
    s = pd.DataFrame(reqs)
    meta = dict(path=path, n=len(s), duration=float(s.timestamp.max()), rate=len(s) / max(s.timestamp.max(), 1e-9),
                in_p50=float(s.input_length.median()), out_p50=float(s.output_length.median()),
                out_mean=float(s.output_length.mean()), out_max=int(s.output_length.max()),
                cls_counts=s["cls"].value_counts().to_dict(),
                total_out_tokens=int(s.output_length.sum()), total_in_tokens=int(s.input_length.sum()))
    with open(path.replace(".jsonl", ".meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(json.dumps(meta))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("family", choices=["bimodal", "mix", "replay", "waves"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--rate", type=float, default=8.0, help="mean arrivals/s")
    ap.add_argument("--duration", type=float, default=120.0)
    ap.add_argument("--burstiness", type=float, default=1.0, help="gamma shape (1=Poisson)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--prompt", type=int, default=512)
    ap.add_argument("--prompt-sigma", type=float, default=0.0, help="lognormal sigma for prompt length")
    ap.add_argument("--prompt-lo", type=int, default=0, help="if >0: prompts ~U[prompt-lo, prompt] (overrides sigma)")
    ap.add_argument("--long-prompt", type=int, default=0, help="if >0: prompt length of long requests (long context)")
    ap.add_argument("--long-prompt-lo", type=int, default=0,
                    help="if >0 (needs --long-prompt): long prompts ~U[long-prompt-lo, long-prompt] instead of fixed")
    ap.add_argument("--n-long", type=int, default=0, help="if >0: exactly this many long requests (with --long-first)")
    ap.add_argument("--short-lo", type=int, default=16)
    ap.add_argument("--short-hi", type=int, default=48)
    ap.add_argument("--long", default="4000", help="int or 'reasoning'")
    ap.add_argument("--frac-long", type=float, default=1 / 32)
    ap.add_argument("--short-src", default="azure2024_code")
    ap.add_argument("--src", default="azure2023_conv")
    ap.add_argument("--max-in", type=int, default=8192)
    ap.add_argument("--max-out", type=int, default=32768)
    ap.add_argument("--max-model-len", type=int, default=40960)
    ap.add_argument("--long-first", action="store_true", help="inject the long requests at t=0 (transient study)")
    ap.add_argument("--long-at", type=float, default=-1.0, help="if >=0: long requests arrive at this time (burst)")
    ap.add_argument("--wave-size", type=int, default=32)
    ap.add_argument("--wave-period", type=float, default=2.0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    n = int(args.rate * args.duration)
    gaps = rng.gamma(args.burstiness, 1.0 / (args.rate * args.burstiness), n)
    ts = np.cumsum(gaps) - gaps[0]
    reqs = []
    if args.family == "waves":
        for w in range(int(args.duration / args.wave_period)):
            t = w * args.wave_period
            for i in range(args.wave_size):
                is_l = i == args.wave_size - 1
                out = int(args.long) if is_l else int(rng.integers(args.short_lo, args.short_hi + 1))
                reqs.append(dict(timestamp=round(t + 1e-4 * i, 6), input_length=args.prompt, output_length=out,
                                 cls="long" if is_l else "short", src="waves"))
        write(reqs, args.out)
        return
    if args.family in ("bimodal", "mix"):
        is_long = rng.random(n) < args.frac_long
        if args.n_long > 0:
            is_long[:] = False
            is_long[rng.choice(n, args.n_long, replace=False)] = True
        if args.family == "bimodal":
            if args.prompt_lo > 0:
                ins = rng.integers(args.prompt_lo, args.prompt + 1, n)
            elif args.prompt_sigma > 0:
                ins =np.clip(rng.lognormal(math.log(args.prompt), args.prompt_sigma, n).astype(int), 16, args.max_in)
            else:
                ins = np.full(n, args.prompt)
            short_out = rng.integers(args.short_lo, args.short_hi + 1, n)
        else:
            sdf = _load_lengths(args.short_src).sample(n, replace=True, random_state=args.seed)
            ins = np.clip(sdf.input_tokens.to_numpy(), 16, args.max_in)
            short_out = np.clip(sdf.output_tokens.to_numpy(), 1, args.max_out)
        if args.long == "reasoning":
            rdf = _load_lengths("reasoning").sample(n, replace=True, random_state=args.seed + 1)
            long_out = np.clip(rdf.output_tokens.to_numpy(), 1024, args.max_out)
        else:
            long_out = np.full(n, int(args.long))
        if args.long_prompt_lo > 0:  # drawn last, and only here, so every other flag set keeps its random stream
            assert args.long_prompt >= args.long_prompt_lo, "--long-prompt-lo needs --long-prompt >= it"
            long_in = rng.integers(args.long_prompt_lo, args.long_prompt + 1, n)
        else:
            long_in = np.full(n, args.long_prompt)
        for i in range(n):
            out = int(long_out[i] if is_long[i] else short_out[i])
            inp = int(long_in[i]) if (is_long[i] and args.long_prompt > 0) else ins[i]
            inp = int(min(inp, args.max_model_len - out - 1))
            t = float(ts[i])
            if is_long[i] and args.long_first:
                t = 0.0
            elif is_long[i] and args.long_at >= 0:
                t = args.long_at + 0.01 * int(is_long[:i].sum())
            reqs.append(dict(timestamp=round(t, 6), input_length=inp, output_length=out,
                             cls="long" if is_long[i] else "short", src=args.family))
    else:
        df = _load_lengths(args.src)
        if "timestamp" in df.columns and len(df) > 1:
            t = pd.to_datetime(df.timestamp, errors="coerce") if df.timestamp.dtype == object else df.timestamp
            t = (t - t.min()).dt.total_seconds() if hasattr(t, "dt") else (t - t.min()).astype(float)
            df = df.assign(t=t.to_numpy()).sort_values("t")
            native_rate = len(df) / max(df.t.max(), 1e-9)
            df["t"] = df.t * (native_rate / args.rate)
            df = df[df.t <= args.duration]
        else:
            df = df.sample(n, replace=True, random_state=args.seed).assign(t=ts)
        for _, r in df.iterrows():
            out = int(min(max(r.output_tokens, 1), args.max_out))
            inp = int(min(max(r.input_tokens, 16), args.max_in, args.max_model_len - out - 1))
            reqs.append(dict(timestamp=round(float(r.t), 6), input_length=inp, output_length=out,
                             cls="long" if out >= 1024 else "short", src=args.src))
    write(reqs, args.out)


if __name__ == "__main__":
    main()
