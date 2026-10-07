#!/bin/bash
#SBATCH -J s9_chat_az
#SBATCH -p main
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH -o logs/%x_%j.out
# Stage 2 (chat baseline), ordinary chat: Azure 2024 conversation trace replayed on Llama-3.1-8B (no prefix reuse, no
# context >= 16k). Same 3,010 requests at 6.3 and 12.6 req/s; fixes at the higher load; default repeated at the end.
cd "${SLURM_SUBMIT_DIR:-.}"
PORT=$((20000 + SLURM_JOB_ID % 10000))
OUT=results/runs/s9_chat_az_$SLURM_JOB_ID
MODEL=meta-llama/Llama-3.1-8B-Instruct MAX_MODEL_LEN=131072
EXTRA_ENV="" CLIENT_ARGS="" CLIENT_TIMEOUT=900
source experiments/lib_serve.sh
unset VLLM_API_KEY OPENAI_API_KEY
nvidia-smi --query-gpu=index,name,memory.used --format=csv; uptime
SCHED="--scheduler-cls regime_batching.scheduler.RegimeScheduler"
# smoke: 40 requests must complete and reach the telemetry
run_case smoke data/workloads/chat_az14.jsonl 40
python - "$OUT/smoke" <<'PY' || { echo SMOKE_FAILED; exit 1; }
import glob, json, sys
b = json.load(open(f"{sys.argv[1]}/bench.json"))
n = sum(1 for f in glob.glob(f"{sys.argv[1]}/telemetry/steps_*.jsonl") for l in open(f) if '"kind": "step"' in l)
print("smoke completed", b.get("completed"), "steps", n)
raise SystemExit(0 if b.get("completed") == 40 and n > 50 else 1)
PY
EXTRA_ENV="" run_case az7_default  data/workloads/chat_az7.jsonl  3010
EXTRA_ENV="" run_case az14_default data/workloads/chat_az14.jsonl 3010
EXTRA_ENV="" run_case az14_cap2048 data/workloads/chat_az14.jsonl 3010 -cc '{"max_cudagraph_capture_size":2048}'
EXTRA_ENV="" run_case az14_mnbt512 data/workloads/chat_az14.jsonl 3010 --max-num-batched-tokens 512
EXTRA_ENV="REGIME_SCHED_ADMIT=fcfs REGIME_SCHED_SHAPE=cg_cap" run_case az14_shape data/workloads/chat_az14.jsonl 3010 $SCHED
EXTRA_ENV="" run_case az14_default_rep data/workloads/chat_az14.jsonl 3010
uptime; echo "done $(date)"
