#!/bin/bash
#SBATCH -J s1_smoke
#SBATCH -p main
#SBATCH --gres=gpu:1
#SBATCH --time=00:50:00
#SBATCH -o logs/%x_%j.out
# S1 smoke: serve Qwen3-8B under each attention backend with telemetry, replay a 60 s oracle workload.
cd "${SLURM_SUBMIT_DIR:-.}"
PORT=$((20000 + SLURM_JOB_ID % 10000))
OUT=results/runs/s1_smoke_$SLURM_JOB_ID
EXTRA_ENV=""
source experiments/lib_serve.sh
nvidia-smi --query-gpu=index,name,memory.used --format=csv
WL=data/workloads/smoke_bimodal4k.jsonl
run_case fa3_default   $WL 240
run_case fi_xqa        $WL 240 --attention-backend FLASHINFER
run_case fi_native     $WL 240 -ac '{"backend":"FLASHINFER","use_trtllm_attention":false}'
run_case triton        $WL 240 --attention-backend TRITON_ATTN
run_case fa3_piecewise $WL 240 -cc '{"cudagraph_mode":"PIECEWISE"}'
run_case fa3_mix       data/workloads/smoke_mix.jsonl 480
echo "done $(date)"
