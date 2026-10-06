"""Write the long text-only prompts for the S5 mixed load: one prompt per line, each an exact token count drawn from
U[lo, hi] with the Qwen3-Omni tokenizer (mstar adds ~25 template tokens on top).

Random lowercase words that are one BPE token with a leading space, cut to the exact length, so no two prompts share a
prefix (the Thinker has prefix caching) and the prefill cost is the same under both policies.
Run on the login node, CPU only, offline:
    HF_HUB_OFFLINE=1 <mstar venv>/bin/python make_long_prompts.py --out FILE
"""
import argparse
import glob
import random

from transformers import AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--n", type=int, default=64)
ap.add_argument("--lo", type=int, default=1000)
ap.add_argument("--hi", type=int, default=2000)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument(
    "--tokenizer",
    default=(glob.glob("/data/garv901-55613a/hf/hub/models--Qwen--Qwen3-Omni-30B-A3B-Instruct/snapshots/*") or [""])[0],
)
args = ap.parse_args()

tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
rng = random.Random(args.seed)
vocab = [w[1:] for w in tok.get_vocab() if w.startswith("Ġ") and w[1:].isalpha() and w[1:].islower()]
lens = []
with open(args.out, "w") as f:
    for _ in range(args.n):
        want = rng.randint(args.lo, args.hi)
        ids = tok.encode(" ".join(rng.choices(vocab, k=want)), add_special_tokens=False)[:want]
        text = tok.decode(ids)
        lens.append(len(tok.encode(text, add_special_tokens=False)))
        f.write(text.replace("\n", " ") + "\n")
print(f"wrote {args.n} prompts to {args.out}: tokens min={min(lens)} max={max(lens)} mean={sum(lens) / len(lens):.0f}")
