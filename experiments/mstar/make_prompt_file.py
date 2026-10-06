"""Write a one-prompt-per-line text file for `python -m benchmark.runner --request-txt-file`.

Each prompt is random English-like words trimmed to a token count drawn from U[lo, hi] with the BAGEL tokenizer,
so the smoke test can exercise the same prompt lengths as the workloads (default: the 450-1000 token short class).
Run with the mstar venv python (it has the tokenizer): .venv/bin/python make_prompt_file.py --out FILE
"""
import argparse
import random

from mstar.model.bagel.components.tokenization import BagelTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--n", type=int, default=64)
ap.add_argument("--lo", type=int, default=450)
ap.add_argument("--hi", type=int, default=1000)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--model", default="ByteDance-Seed/BAGEL-7B-MoT")
args = ap.parse_args()

tok = BagelTokenizer.from_pretrained(args.model)
rng = random.Random(args.seed)
# words that are one BPE token with a leading space ("Ġword"), so n words ~ n tokens
vocab = [w[1:] for w in tok.get_vocab() if w.startswith("\u0120") and w[1:].isalpha() and w[1:].islower()]
lens = []
with open(args.out, "w") as f:
    for _ in range(args.n):
        want = rng.randint(args.lo, args.hi)
        text = tok.decode(tok.encode(" ".join(rng.choices(vocab, k=want)))[:want])  # exact-length prefix
        lens.append(len(tok.encode(text)))  # retokenized length (can differ by a token at the cut)
        f.write(text.replace("\n", " ") + "\n")
print(f"wrote {args.n} prompts to {args.out}: tokens min={min(lens)} max={max(lens)} mean={sum(lens)/len(lens):.0f}")
