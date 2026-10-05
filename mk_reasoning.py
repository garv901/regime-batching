import csv, itertools
from datasets import load_dataset
from transformers import AutoTokenizer
tok=AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
ds=load_dataset("open-thoughts/OpenThoughts-114k",split="train",streaming=True)
N=20000
with open("reasoning_lengths.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["input_tokens","output_tokens"])
    for r in itertools.islice(ds,N):
        q=r["conversations"][0]["value"]; a=r["conversations"][1]["value"]
        w.writerow([len(tok(q).input_ids),len(tok(a).input_ids)])
