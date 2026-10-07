#!/bin/bash
#SBATCH -J mb_attn_f6
#SBATCH -p main
#SBATCH --gres=gpu:1
#SBATCH --time=01:15:00
#SBATCH -o logs/%x_%j.out
cd /mnt/storage/garv901/regime-batching
source .venv/bin/activate
export HF_HUB_OFFLINE=1
nvidia-smi --query-gpu=index,name,memory.used --format=csv
C=fa3_auto,fa3_cg,fa3_s1,fa3_s2,fa3_s4,fa3_s8,fa3_s16
# smoke: a handful of specs, abort on error
python microbench/attn_sweep.py --families F6_append --spec-filter app_P131072_q1057 --configs $C --out results/microbench/attn_f6_smoke.jsonl || exit 1
if grep -q ERROR results/microbench/attn_f6_smoke.jsonl; then echo SMOKE_ERRORS; exit 1; fi
python microbench/attn_sweep.py --replay-steps 'results/runs/s3a_1017/*/telemetry/steps_*.jsonl,results/runs/s6_1390/C_fa3*/telemetry/steps_*.jsonl,results/runs/s5a_llama_1040/xh_*/telemetry/steps_*.jsonl' --replay-n 40 --configs $C --out results/microbench/attn_replay.jsonl --resume
python microbench/attn_sweep.py --families F6_append --configs $C --out results/microbench/attn_f6.jsonl --resume
