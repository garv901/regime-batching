"""C3 correctness: greedy outputs with the regime split policy vs stock vLLM, on X-heavy steps.

One long-context request (prompt --long-prompt tokens) decodes while a new --short-prompt-token request
is added every engine step. Every such step is mixed with T > 512 and a >=512-token prefill chunk, which
puts it in the X regime. Run twice (REGIME_ATTN unset / =1) and compare token ids + logprobs.

  python experiments/c3_correctness.py --out results/c3/base.json
  REGIME_ATTN=1 python experiments/c3_correctness.py --out results/c3/regime.json
  python experiments/c3_correctness.py --compare results/c3/base.json results/c3/regime.json
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np


def run(args):
    from vllm import EngineArgs, LLMEngine, SamplingParams
    from vllm.inputs import TokensPrompt

    kw = {}
    if args.cudagraph_mode:  # stock-vLLM alternative config, used as the numerical noise floor
        kw["compilation_config"] = {"cudagraph_mode": args.cudagraph_mode}
    eng = LLMEngine.from_engine_args(EngineArgs(model=args.model, max_model_len=args.max_model_len, seed=0,
                                                enable_prefix_caching=False, **kw))
    rng = np.random.default_rng(0)
    vocab = 100000

    def prompt(n):
        return TokensPrompt(prompt_token_ids=rng.integers(1000, vocab, n).tolist())

    greedy = lambda n: SamplingParams(temperature=0.0, max_tokens=n, logprobs=1, ignore_eos=True)
    eng.add_request("long0", prompt(args.long_prompt), greedy(args.long_out))
    outs, n_added, steps, t0 = {}, 0, 0, time.time()
    while eng.has_unfinished_requests():
        if steps >= 3 and n_added < args.n_short:  # long request is decoding by now
            eng.add_request(f"s{n_added}", prompt(args.short_prompt), greedy(args.short_out))
            n_added += 1
        for o in eng.step():
            if o.finished:
                c = o.outputs[0]
                outs[o.request_id] = dict(tokens=list(c.token_ids),
                                          logprobs=[float(lp[t].logprob) for t, lp in zip(c.token_ids, c.logprobs)])
        steps += 1
    print(f"done: {len(outs)} requests, {steps} steps, {time.time() - t0:.1f}s, REGIME_ATTN={os.environ.get('REGIME_ATTN')}")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(outs, f)


def compare(a_path, b_path):
    a, b = json.load(open(a_path)), json.load(open(b_path))
    n_same, first_div, max_lp, n_tok = 0, [], 0.0, 0
    for rid in sorted(a):
        ta, tb = a[rid]["tokens"], b[rid]["tokens"]
        if ta == tb:
            n_same += 1
            d = np.abs(np.array(a[rid]["logprobs"]) - np.array(b[rid]["logprobs"]))
            max_lp = max(max_lp, float(d.max()))
        else:
            first_div.append((rid, next(i for i, (x, y) in enumerate(zip(ta, tb)) if x != y), len(ta)))
        n_tok += len(ta)
    res = dict(requests=len(a), identical=n_same, max_logprob_diff_identical=max_lp, divergent=first_div[:20])
    print(json.dumps(res, indent=1))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--max-model-len", type=int, default=40960)
    ap.add_argument("--long-prompt", type=int, default=16000)
    ap.add_argument("--long-out", type=int, default=200)
    ap.add_argument("--short-prompt", type=int, default=700)
    ap.add_argument("--short-out", type=int, default=32)
    ap.add_argument("--n-short", type=int, default=100)
    ap.add_argument("--cudagraph-mode", default=None, help="e.g. PIECEWISE (noise-floor reference)")
    ap.add_argument("--out")
    ap.add_argument("--compare", nargs=2)
    a = ap.parse_args()
    compare(*a.compare) if a.compare else run(a)
