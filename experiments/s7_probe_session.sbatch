#!/bin/bash
#SBATCH -J s7_probe
#SBATCH -p main
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH -o logs/%x_%j.out
# S7: probe TraceLab session_runner against vLLM 0.30 (Llama-3.1-8B, max-model-len 262144, prefix caching).
# Server runs WITHOUT VLLM_API_KEY on 127.0.0.1 (session_runner has no auth option).
cd "${SLURM_SUBMIT_DIR:-.}"
ROOT=$PWD
source "$ROOT/.venv/bin/activate"
export PYTHONHASHSEED=0 HF_HUB_OFFLINE=1 VLLM_LOGGING_LEVEL=INFO
PORT=$((20000 + SLURM_JOB_ID % 10000))
OUT=results/runs/s7_probe; mkdir -p "$OUT/telemetry" logs
MODEL=meta-llama/Llama-3.1-8B-Instruct
R=third_party/TraceLab/replay
TOK=$(ls /mnt/storage/garv901/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/*/tokenizer.json | head -1)
nvidia-smi --query-gpu=index,name,memory.used --format=csv

srun --overlap --gres=gpu:1 -n1 env -u VLLM_API_KEY -u OPENAI_API_KEY VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
    REGIME_TELEMETRY_DIR="$ROOT/$OUT/telemetry" \
    vllm serve $MODEL --host 127.0.0.1 --port "$PORT" --max-model-len 262144 \
    --enable-prefix-caching --enable-prompt-tokens-details --seed 0 > "$OUT/server.log" 2>&1 &
SERVER=$!
t=0
until curl -sf "http://127.0.0.1:$PORT/health" >/dev/null; do
    kill -0 "$SERVER" 2>/dev/null || { echo "server died"; tail -40 "$OUT/server.log"; exit 1; }
    sleep 5; t=$((t+5)); [ $t -ge 900 ] && { echo "health timeout"; kill $SERVER; exit 1; }
done
echo "server healthy after ${t}s"

timeout 720 srun --overlap --gres=none -n1 $R/target/release/session_runner \
    --trace $R/examples/multi_session_large.csv --text-file data/corpus/enwik9 --tokenizer "$TOK" \
    --model $MODEL --base-url "http://127.0.0.1:$PORT/v1" --stream-idle-timeout-secs 600 \
    --max-model-len 262144 --max-active-sessions 8 \
    --summary-path "$OUT/summary.json" --log-path "$OUT/session_runner.jsonl" > "$OUT/client.log" 2>&1
echo "client rc=$?"
kill -INT $SERVER; sleep 10; kill $SERVER 2>/dev/null; wait
echo "done $(date)"
