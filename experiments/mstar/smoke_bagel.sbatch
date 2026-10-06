#!/bin/bash
#SBATCH -J mstar_smoke
#SBATCH -p team1
#SBATCH --gres=gpu:1
#SBATCH --time=00:55:00
#SBATCH -o /data/garv901-55613a/logs/mstar/%x_%j.out
# Smoke test: mstar serve bagel (single GPU, SHM) from the pristine bench-main worktree, then mstar's own client
# (benchmark.runner, text_to_text, --ignore-eos). Submit from the study repo root:
#   sbatch experiments/mstar/smoke_bagel.sbatch
# Phases (closed loop): seq (1 stream), dec_c* (decode capacity, tiny prompts), short_c* (450-1000 token prompts,
# 16-64 token outputs = the workloads' short class), long_c16 (2000-token outputs). Then a warm restart for startup time.
cd "${SLURM_SUBMIT_DIR:-/shared/home/garv901-55613a/regime-batching/regime-batching}"
source /data/mstar/env-laude.sh
export HF_HOME=/data/garv901-55613a/hf
PORT=$((20000 + SLURM_JOB_ID % 10000))
OUT=${OUT:-/data/garv901-55613a/logs/mstar/smoke_$SLURM_JOB_ID}
source experiments/mstar/lib_mstar.sh
mkdir -p "$OUT"
echo "job $SLURM_JOB_ID on $(hostname), GPUs: ${CUDA_VISIBLE_DEVICES:-?}, mstar $(git -C "$MSTAR_DIR" rev-parse --short HEAD) ($MSTAR_DIR)"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv

# the KV pool the server will build (main does not log it)
(cd "$MSTAR_DIR" && HF_HUB_OFFLINE=1 .venv/bin/python -I "$OLDPWD/experiments/mstar/kv_probe.py" 2>&1 | tail -2) | tee "$OUT/kv_probe.txt"
# prompts with 450-1000 tokens (the short class), deterministic
HF_HUB_OFFLINE=1 "$MSTAR_DIR/.venv/bin/python" -I experiments/mstar/make_prompt_file.py --out "$OUT/short_450_1000.txt" --n 64 | tail -1
PF=$OUT/short_450_1000.txt

start_gpu_sampler "$OUT/gpu_mem.csv"
C="$OUT/cold"; mkdir -p "$C"
echo "=== [$(date +%T)] cold start"
if ! start_server "$C"; then   # retry once (the first attempt warms the compile caches)
    echo "!!! server failed, retrying once"; tail -30 "$C/server.log"; stop_server $SRV
    mv "$C/server.log" "$C/server_attempt1.log"
    if ! start_server "$C"; then
        echo "!!! server failed again"; tail -60 "$C/server.log"; stop_server $SRV; stop_gpu_sampler; exit 1
    fi
fi
echo "cold startup_s=$SERVER_STARTUP_S" | tee "$OUT/startup.txt"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader | sed 's/^/after-start gpu mem: /'

phase() {  # <name> [runner args...]
    local n=$1; shift
    echo "=== [$(date +%T)] phase $n"
    run_client "$C" --request-type text_to_text --profiling-type closed_loop --ignore-eos --output-dir "$C/out_$n" "$@" \
        > "$C/client_$n.log" 2>&1
    echo "client rc=$?"
    grep -E "Requests :|TTFT|E2E|ITL|Text tokens|Prompt tokens|Throughput|wall time|Error" "$C/client_$n.log"
}
O=--output-len-min; X=--output-len-max
phase seq        --max-concurrency 1  --num-requests 5   --num-warmup 2 $O 128 $X 128
phase dec_c8     --max-concurrency 8  --num-requests 16  --num-warmup 1 $O 128 $X 128
phase dec_c32    --max-concurrency 32 --num-requests 64  --num-warmup 1 $O 128 $X 128
phase dec_c64    --max-concurrency 64 --num-requests 128 --num-warmup 1 $O 128 $X 128
phase short_c4   --max-concurrency 4  --num-requests 48  --num-warmup 1 --dataset text --request-txt-file "$PF" $O 16 $X 64
phase short_c16  --max-concurrency 16 --num-requests 96  --num-warmup 1 --dataset text --request-txt-file "$PF" $O 16 $X 64
phase short_c64  --max-concurrency 64 --num-requests 192 --num-warmup 1 --dataset text --request-txt-file "$PF" $O 16 $X 64
phase long_c16   --max-concurrency 16 --num-requests 16  --num-warmup 0 --dataset text --request-txt-file "$PF" $O 2000 $X 2000
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader | sed 's/^/after-clients gpu mem: /'
stop_server $SRV

echo "=== [$(date +%T)] warm restart (startup time with warm compile caches)"
W="$OUT/warm"; mkdir -p "$W"
if start_server "$W"; then echo "warm startup_s=$SERVER_STARTUP_S" | tee -a "$OUT/startup.txt"; fi
stop_server $SRV
stop_gpu_sampler
echo "done $(date); outputs in $OUT"
