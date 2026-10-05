#!/bin/bash
#SBATCH -J s4b_burst
#SBATCH -p main
#SBATCH --gres=gpu:1
#SBATCH --time=01:10:00
#SBATCH -o logs/%x_%j.out
# S4b: memory-pressure burst. 16 req/s short stream for 300 s + 80 long (8000-token output) requests at t=30 s:
# peak KV demand ~1.5x the 451k-token cache. vLLM admits greedily and preempts (recompute); the oracle
# peak-KV planner admits only what fits until completion and backfills shorts.
cd "${SLURM_SUBMIT_DIR:-.}"
PORT=$((20000 + SLURM_JOB_ID % 10000))
OUT=results/runs/s4b_$SLURM_JOB_ID
EXTRA_ENV=""
source experiments/lib_serve.sh
nvidia-smi --query-gpu=index,name,memory.used --format=csv
WL=data/workloads/lburst80_8k.jsonl; N=4800
SCHED="--scheduler-cls regime_batching.scheduler.RegimeScheduler"
EXTRA_ENV="" run_case B0 $WL $N
EXTRA_ENV="REGIME_SCHED_ADMIT=peak REGIME_SCHED_SHAPE=off" run_case peak $WL $N $SCHED
EXTRA_ENV="REGIME_ATTN=1 REGIME_SCHED_ADMIT=peak REGIME_SCHED_SHAPE=cg_cap" run_case full $WL $N $SCHED
echo "done $(date)"
