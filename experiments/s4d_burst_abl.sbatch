#!/bin/bash
#SBATCH -J s4d_babl
#SBATCH -p main
#SBATCH --gres=gpu:1
#SBATCH --time=00:50:00
#SBATCH -o logs/%x_%j.out
# S4d: burst ablation + repeat. sjf = oracle shortest-remaining-first ordering without KV planning (what
# length-aware schedulers such as SSJF/LTR do, given perfect lengths); B0/peak second repeat of s4b.
cd "${SLURM_SUBMIT_DIR:-.}"
PORT=$((20000 + SLURM_JOB_ID % 10000))
OUT=results/runs/s4d_$SLURM_JOB_ID
EXTRA_ENV=""
source experiments/lib_serve.sh
nvidia-smi --query-gpu=index,name,memory.used --format=csv
WL=data/workloads/lburst80_8k.jsonl; N=4800
SCHED="--scheduler-cls regime_batching.scheduler.RegimeScheduler"
EXTRA_ENV="REGIME_SCHED_ADMIT=sjf REGIME_SCHED_SHAPE=off" run_case sjf $WL $N $SCHED
EXTRA_ENV="" run_case B0_r2 $WL $N
EXTRA_ENV="REGIME_SCHED_ADMIT=peak REGIME_SCHED_SHAPE=off" run_case peak_r2 $WL $N $SCHED
echo "done $(date)"
