#!/bin/bash
#SBATCH -J s5b_qwen4b
#SBATCH -p main
#SBATCH --gres=gpu:1
#SBATCH --time=01:25:00
#SBATCH -o logs/%x_%j.out
# S5: replicate the two headlines on Qwen/Qwen3-4B -- X-heavy step shaping vs best static (s3a matrix) and the
# memory-pressure burst (oracle peak-KV admission vs default; s4b).
cd "${SLURM_SUBMIT_DIR:-.}"
PORT=$((20000 + SLURM_JOB_ID % 10000))
OUT=results/runs/s5b_qwen4b_$SLURM_JOB_ID
MODEL=Qwen/Qwen3-4B
EXTRA_ENV=""
source experiments/lib_serve.sh
nvidia-smi --query-gpu=index,name,memory.used --format=csv
SCHED="--scheduler-cls regime_batching.scheduler.RegimeScheduler"
WL=data/workloads/xheavy_lc24k.jsonl; N=2400
EXTRA_ENV=""  run_case xh_B0       $WL $N
EXTRA_ENV=""  run_case xh_cap2048  $WL $N -cc '{"max_cudagraph_capture_size":2048}'
EXTRA_ENV=""  run_case xh_mnbt512  $WL $N --max-num-batched-tokens 512
EXTRA_ENV="REGIME_SCHED_ADMIT=fcfs REGIME_SCHED_SHAPE=cg_cap" run_case xh_shape $WL $N $SCHED
EXTRA_ENV="REGIME_ATTN=1 REGIME_SCHED_ADMIT=peak REGIME_SCHED_SHAPE=cg_cap" run_case xh_full $WL $N $SCHED
WL=data/workloads/lburst80_8k.jsonl; N=4800
EXTRA_ENV=""  run_case bu_B0       $WL $N
EXTRA_ENV="REGIME_SCHED_ADMIT=peak REGIME_SCHED_SHAPE=off" run_case bu_peak $WL $N $SCHED
# Ablation on Qwen3-8B: reactive shaping (cap whenever a >=4k-KV decode runs; ignores oracle remaining tokens).
MODEL=Qwen/Qwen3-8B; WL=data/workloads/xheavy_lc24k.jsonl; N=2400
EXTRA_ENV="REGIME_SCHED_ADMIT=fcfs REGIME_SCHED_SHAPE=cg_cap REGIME_SHAPE_MIN_REMAIN=0" run_case q8_shape_reactive $WL $N $SCHED
echo "done $(date)"
