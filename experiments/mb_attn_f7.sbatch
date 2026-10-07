#!/bin/bash
#SBATCH -J mb_attn_f7
#SBATCH -p main
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH -o logs/%x_%j.out
# Stage 1b kernel-level regime census (F7): census shapes x FA3 split bounds, separate launches, FlashInfer, Triton.
cd /mnt/storage/garv901/regime-batching
source .venv/bin/activate
export HF_HUB_OFFLINE=1
nvidia-smi --query-gpu=index,name,memory.used --format=csv; uptime
C=fa3_auto,fa3_cg,fa3_s1,fa3_s2,fa3_s4,fa3_s8,fa3_s16,fa3_phase_split,fi_native,triton
# smoke: three specs (prefill alone, long-prefix + decodes, a pair), abort on error
for f in c_none_P131072_q1057 c_1x128k_P131072_q1024 c_none_pair544+512_P131072; do
  python microbench/attn_sweep.py --families F7_census --spec-filter $f --configs $C \
      --out results/microbench/attn_f7_smoke.jsonl || exit 1
done
if grep -q ERROR results/microbench/attn_f7_smoke.jsonl; then echo SMOKE_ERRORS; grep ERROR results/microbench/attn_f7_smoke.jsonl | head; fi
python microbench/attn_sweep.py --families F7_census --configs $C --out results/microbench/attn_f7.jsonl --resume
uptime; echo "done $(date)"
