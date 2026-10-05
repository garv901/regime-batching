#!/bin/bash
#SBATCH -J s3a_static
#SBATCH -p main
#SBATCH --gres=gpu:1
#SBATCH --time=01:10:00
#SBATCH -o logs/%x_%j.out
# S3a: best-static vLLM baselines vs regime-aware step shaping on the X-heavy long-context workload.
# X steps (T > 512) leave both the full-CG split path and CUDA graphs entirely (eager, CPU-launch bound
# on this node). Static fixes: raise the capture size, or cap the step at 512 tokens. Proposed: cap only
# while a long-KV decode with oracle-remaining work runs (C4 cg_cap), optionally + C3 + peak admission.
cd "${SLURM_SUBMIT_DIR:-.}"
PORT=$((20000 + SLURM_JOB_ID % 10000))
OUT=results/runs/s3a_$SLURM_JOB_ID
EXTRA_ENV=""
source experiments/lib_serve.sh
nvidia-smi --query-gpu=index,name,memory.used --format=csv
WL=data/workloads/xheavy_lc24k.jsonl; N=2400
SCHED="--scheduler-cls regime_batching.scheduler.RegimeScheduler"

EXTRA_ENV=""  run_case B0       $WL $N
EXTRA_ENV=""  run_case cap2048  $WL $N -cc '{"max_cudagraph_capture_size":2048}'
EXTRA_ENV=""  run_case mnbt512  $WL $N --max-num-batched-tokens 512
EXTRA_ENV="REGIME_SCHED_ADMIT=fcfs REGIME_SCHED_SHAPE=cg_cap" \
              run_case shape    $WL $N $SCHED
EXTRA_ENV="REGIME_ATTN=1 REGIME_SCHED_ADMIT=peak REGIME_SCHED_SHAPE=cg_cap" \
              run_case full     $WL $N $SCHED
echo "done $(date)"
