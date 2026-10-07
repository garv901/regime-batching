#!/bin/bash
#SBATCH -J s10_ultralong
#SBATCH -p main
#SBATCH --gres=gpu:1
#SBATCH --time=00:35:00
#SBATCH -o logs/%x_%j.out
# Stage 3 model check: Llama-3.1-Nemotron-8B-UltraLong-1M (Llama-3.1-8B architecture, RoPE table to 1,073,152).
# 1) prompts past Llama-3.1's 131,072-position limit run (140k, 250k random tokens) and a 200k passkey is retrieved;
# 2) the Stage 1b default-config census on this model, to compare whole-step times with Llama-3.1-8B (job 1452).
cd /mnt/storage/garv901/regime-batching
source .venv/bin/activate
export HF_HUB_OFFLINE=1 PYTHONHASHSEED=0 VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 VLLM_ENABLE_V1_MULTIPROCESSING=0
unset VLLM_API_KEY OPENAI_API_KEY
OUT=results/runs/s10_ultralong_${SLURM_JOB_ID}
nvidia-smi --query-gpu=index,name,memory.used --format=csv; uptime
REGIME_TELEMETRY_DIR=$OUT/default/telemetry python experiments/regime_census.py --config default --out $OUT/default \
    --model nvidia/Llama-3.1-Nemotron-8B-UltraLong-1M-Instruct --dtype bfloat16 \
    --long-check 140000,250000 --passkey 200000 || { echo PROBE_FAILED; exit 1; }
uptime; echo "done $(date)"
