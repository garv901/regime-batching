#!/bin/bash
#SBATCH -J s1b_c3
#SBATCH -p main
#SBATCH --gres=gpu:1
#SBATCH --time=01:15:00
#SBATCH -o logs/%x_%j.out
# S1b: (1) C3 split-policy correctness on X-heavy steps, (2) B0 vs C3 on X-stress + the motivating
# 31-short/1-long waves, (3) rerun FlashInfer-XQA smoke case.
cd "${SLURM_SUBMIT_DIR:-.}"
PORT=$((20000 + SLURM_JOB_ID % 10000))
OUT=results/runs/s1b_$SLURM_JOB_ID
EXTRA_ENV=""
source experiments/lib_serve.sh
nvidia-smi --query-gpu=index,name,memory.used --format=csv

echo "=== [$(date +%T)] C3 correctness"
mkdir -p $OUT/c3
REGIME_TELEMETRY_DIR=$OUT/c3/tel_base python experiments/c3_correctness.py --out $OUT/c3/base.json 2>&1 | tail -3
gpu_drain
REGIME_ATTN=1 REGIME_TELEMETRY_DIR=$OUT/c3/tel_regime python experiments/c3_correctness.py --out $OUT/c3/regime.json 2>&1 | tail -3
python experiments/c3_correctness.py --compare $OUT/c3/base.json $OUT/c3/regime.json | tee $OUT/c3/compare.json
gpu_drain

for WL in xstress_16k user_waves_4k user_waves_16k; do
    EXTRA_ENV=""            run_case ${WL}_B0 data/workloads/$WL.jsonl 960
    EXTRA_ENV="REGIME_ATTN=1" run_case ${WL}_C3 data/workloads/$WL.jsonl 960
done
EXTRA_ENV="" run_case fi_xqa data/workloads/smoke_bimodal4k.jsonl 240 --attention-backend FLASHINFER
echo "done $(date)"
