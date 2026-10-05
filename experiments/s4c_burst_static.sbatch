#!/bin/bash
#SBATCH -J s4c_bstatic
#SBATCH -p main
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH -o logs/%x_%j.out
# S4c: best-static memory-pressure baselines on the lburst80_8k burst (default max_num_seqs = 1024).
# Capping concurrency is the obvious static knob against preemption storms: fewer seats -> less KV
# overcommit, but shorts then queue behind the 80 longs for seats. Compare against s4b peak/full.
cd "${SLURM_SUBMIT_DIR:-.}"
PORT=$((20000 + SLURM_JOB_ID % 10000))
OUT=results/runs/s4c_$SLURM_JOB_ID
EXTRA_ENV=""
source experiments/lib_serve.sh
nvidia-smi --query-gpu=index,name,memory.used --format=csv
WL=data/workloads/lburst80_8k.jsonl; N=4800
for S in 32 64 128 256; do
    EXTRA_ENV="" run_case mns$S $WL $N --max-num-seqs $S
done
echo "done $(date)"
