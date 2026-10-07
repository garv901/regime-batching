#!/bin/bash
# Shared helpers for server+client benchmark jobs (sourced from sbatch scripts).
# Usage: run_case <name> <workload.jsonl> <num_prompts> [extra vllm serve args...]
# Env: MODEL, OUT (run root), PORT, EXTRA_ENV (space-separated VAR=val applied to the server only),
#      CLIENT_ARGS (extra `vllm bench serve` args, e.g. --timed-trace-chunk-hash-size 512), CLIENT_TIMEOUT (s)
set -uo pipefail
ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source "$ROOT/.venv/bin/activate"
set -a; [ -f "$ROOT/.env" ] && source "$ROOT/.env"; set +a
export PYTHONHASHSEED=0 HF_HUB_OFFLINE=1 VLLM_LOGGING_LEVEL=${VLLM_LOGGING_LEVEL:-INFO}
: "${MODEL:=Qwen/Qwen3-8B}" "${MAX_MODEL_LEN:=40960}"

wait_health() {  # <pid> <port> <timeout_s>
    local t=0
    until curl -sf "http://127.0.0.1:$2/health" >/dev/null; do
        kill -0 "$1" 2>/dev/null || return 1
        sleep 3; t=$((t + 3))
        if [ $t -ge "$3" ]; then return 1; fi
    done
    return 0
}

gpu_drain() {  # wait until our GPU(s) hold < 2 GiB (previous server fully gone)
    for _ in $(seq 60); do
        local used; used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)
        [ "${used:-0}" -lt 2048 ] && return 0
        sleep 2
    done
    echo "warn: GPU memory still ${used} MiB"; nvidia-smi --query-compute-apps=pid,used_memory --format=csv
}

stop_server() {  # <pid>: SIGINT the process group, escalate to SIGKILL
    kill -INT -- -"$1" 2>/dev/null
    for _ in $(seq 60); do kill -0 "$1" 2>/dev/null || break; sleep 1; done
    kill -9 -- -"$1" 2>/dev/null; wait "$1" 2>/dev/null
    gpu_drain
}

start_server() {  # <case_dir> [vllm serve args...]; sets SRV
    local d=$1; shift
    gpu_drain
    setsid env $EXTRA_ENV REGIME_TELEMETRY_DIR="$d/telemetry" \
        vllm serve "$MODEL" --host 127.0.0.1 --port "$PORT" --max-model-len "$MAX_MODEL_LEN" --seed 0 \
        "$@" >> "$d/server.log" 2>&1 &
    SRV=$!
    wait_health $SRV "$PORT" 900
}

run_case() {
    local name=$1 wl=$2 n=$3; shift 3
    local d="$OUT/$name"; mkdir -p "$d/telemetry"
    echo "=== [$(date +%T)] case $name  wl=$(basename "$wl") n=$n env=[$EXTRA_ENV] args=$* client=[${CLIENT_ARGS:-}] load=[$(cut -d' ' -f1-3 /proc/loadavg)]"
    if ! start_server "$d" "$@"; then
        echo "!!! server failed for $name, retrying once"; tail -5 "$d/server.log"; stop_server $SRV
        rm -f "$d"/telemetry/*
        if ! start_server "$d" "$@"; then
            echo "!!! server failed again for $name"; tail -40 "$d/server.log"; stop_server $SRV; return 1
        fi
    fi
    OPENAI_API_KEY="${VLLM_API_KEY:-}" ${CLIENT_TIMEOUT:+timeout $CLIENT_TIMEOUT} vllm bench serve --backend vllm --model "$MODEL" --base-url "http://127.0.0.1:$PORT" \
        --dataset-name timed_trace --dataset-path "$wl" --num-prompts "$n" \
        --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99 \
        --save-result --save-detailed --result-dir "$d" --result-filename bench.json ${CLIENT_ARGS:-} \
        > "$d/client.log" 2>&1
    echo "client rc=$? load=[$(cut -d' ' -f1-3 /proc/loadavg)]"; grep -E "Successful|throughput|Mean TTFT|P99 TTFT|Mean TPOT|P99 TPOT|P99 E2EL" "$d/client.log"
    stop_server $SRV
    ls -la "$d/telemetry"
}
