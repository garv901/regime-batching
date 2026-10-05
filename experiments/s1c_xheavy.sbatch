#!/bin/bash
#SBATCH -J s1c_xheavy
#SBATCH -p main
#SBATCH --gres=gpu:1
#SBATCH --time=01:25:00
#SBATCH -o logs/%x_%j.out
# S1c: (1) C3 correctness with full logs, (2) B0 vs C3 (min-gain gated split policy) on X-heavy long-context
# workloads (4 long-KV requests at t=0 + 24 req/s of 450-1000-token prompts), Qwen3-8B then Qwen3-4B.
cd "${SLURM_SUBMIT_DIR:-.}"
PORT=$((20000 + SLURM_JOB_ID % 10000))
OUT=results/runs/s1c_$SLURM_JOB_ID
EXTRA_ENV=""
source experiments/lib_serve.sh
nvidia-smi --query-gpu=index,name,memory.used --format=csv

echo "=== [$(date +%T)] C3 correctness"
mkdir -p $OUT/c3
REGIME_TELEMETRY_DIR=$OUT/c3/tel_base python experiments/c3_correctness.py --out $OUT/c3/base.json \
    > $OUT/c3/base.log 2>&1
echo "base rc=$?"; tail -2 $OUT/c3/base.log
gpu_drain
REGIME_ATTN=1 REGIME_TELEMETRY_DIR=$OUT/c3/tel_regime python experiments/c3_correctness.py --out $OUT/c3/regime.json \
    > $OUT/c3/regime.log 2>&1
rc=$?; echo "regime rc=$rc"
if [ $rc -ne 0 ]; then grep -nE "Error|error|Traceback" $OUT/c3/regime.log | head -30; fi
[ -f $OUT/c3/regime.json ] && python experiments/c3_correctness.py --compare $OUT/c3/base.json $OUT/c3/regime.json \
    | tee $OUT/c3/compare.json
gpu_drain

for M in Qwen/Qwen3-8B Qwen/Qwen3-4B; do
    MODEL=$M; tag=$(basename $M)
    for WL in xheavy_lc24k xheavy_lc16k; do
        [ "$tag" = "Qwen3-4B" ] && [ "$WL" = "xheavy_lc16k" ] && continue
        EXTRA_ENV=""              run_case ${tag}_${WL}_B0 data/workloads/$WL.jsonl 2400
        EXTRA_ENV="REGIME_ATTN=1" run_case ${tag}_${WL}_C3 data/workloads/$WL.jsonl 2400
    done
done
echo "done $(date)"
