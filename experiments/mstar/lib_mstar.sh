#!/bin/bash
# Shared helpers for mstar server+client benchmark jobs (sourced from sbatch scripts).
# Port of experiments/lib_serve.sh (vLLM) to `mstar serve`.
#
# Usage:  source experiments/mstar/lib_mstar.sh
#         start_server <case_dir> [extra `mstar serve` args...]   # sets SRV, returns 0 when /health is up
#         run_client   <case_dir> [benchmark.runner args...]
#         stop_server  $SRV
#
# Env (all optional except PORT):
#   PORT          server port (required)
#   MSTAR_DIR     mstar worktree that has its own .venv   (default: bench-main = pristine origin/main)
#   MSTAR_MODEL   model name for `mstar serve` / benchmark.runner   (default: bagel)
#   MSTAR_CONFIG  YAML, absolute or relative to MSTAR_DIR   (default: configs/bagel_single_gpu.yaml)
#   TENSOR_PROTO  --tensor-comm-protocol                    (default: SHM)
#   EXTRA_ENV     space-separated VAR=val applied to the SERVER only, e.g.
#                 "MSTAR_STEP_TELEMETRY_DIR=/x MSTAR_KV_ADMISSION_FIT=1 MSTAR_KV_ADMISSION_ORDER=lpt"
#                 (applied after the defaults below, so it can override them)
#   HEALTH_TIMEOUT  seconds to wait for /health             (default: 1500; cold start compiles + captures graphs)
#   LOG_STATS     1 = add --log-stats-file (turns ON worker profiling, enable_prof; default 0 so baselines run unprofiled)
#   HF_HOME       HF cache with the weights                 (default: /data/garv901-55613a/hf)
set -uo pipefail
[ -n "${CUDA_HOME:-}" ] || source /data/mstar/env-laude.sh   # shared CUDA 12.9 toolkit + JIT cache dirs
export HF_HOME="${HF_HOME:-/data/garv901-55613a/hf}"
export PYTHONUNBUFFERED=1
# The server runs with HF_HUB_OFFLINE=1 (weights are cached). The client does NOT: transformers 4.57 AutoTokenizer
# (trust_remote_code) calls the hub API even for a cached repo and fails offline, which only costs the per-token ITL.
: "${MSTAR_DIR:=/shared/home/garv901-55613a/mstar-worktrees/bench-main}"
: "${MSTAR_MODEL:=bagel}" "${MSTAR_CONFIG:=configs/bagel_single_gpu.yaml}" "${TENSOR_PROTO:=SHM}"
: "${HEALTH_TIMEOUT:=1500}" "${EXTRA_ENV:=}" "${LOG_STATS:=0}"

wait_health() {  # <pid> <port> <timeout_s>; the API server only binds after all workers finished setup + capture
    local t=0
    until curl -sf "http://127.0.0.1:$2/health" >/dev/null; do
        kill -0 "$1" 2>/dev/null || return 1
        sleep 3; t=$((t + 3))
        if [ $t -ge "$3" ]; then return 1; fi
    done
    return 0
}

_our_gpus() { echo "${CUDA_VISIBLE_DEVICES:-${SLURM_JOB_GPUS:-${SLURM_STEP_GPUS:-}}}"; }

gpu_drain() {  # wait until OUR GPU(s) hold < 2 GiB (previous server fully gone); never looks at other users' GPUs
    local g; g=$(_our_gpus); local sel=(); [ -n "$g" ] && sel=(-i "$g")
    local used=0
    for _ in $(seq 60); do
        used=$(nvidia-smi "${sel[@]}" --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)
        [[ $used =~ ^[0-9]+$ ]] || return 0          # no nvidia-smi / NVML (login node): nothing to wait for
        [ "$used" -lt 2048 ] && return 0
        sleep 2
    done
    echo "warn: GPU memory still ${used} MiB"; nvidia-smi "${sel[@]}" --query-compute-apps=pid,used_memory --format=csv
}

stop_server() {  # <pid>: SIGTERM the process group (conductor + workers are children), escalate to SIGKILL.
    # SIGTERM, not SIGINT: bash starts background jobs with SIGINT ignored, and children inherit that.
    # uvicorn installs handlers for both signals, so the API server still shuts down gracefully.
    kill -TERM -- -"$1" 2>/dev/null
    for _ in $(seq 90); do kill -0 "$1" 2>/dev/null || break; sleep 1; done
    kill -9 -- -"$1" 2>/dev/null; wait "$1" 2>/dev/null
    rm -rf "${SOCK_DIR:-/nonexistent}" "${UPLOAD_DIR:-/nonexistent}"
    gpu_drain
}

start_server() {  # <case_dir> [mstar serve args...]; sets SRV; returns 0 iff /health came up
    local d=$1; shift
    mkdir -p "$d/telemetry"
    gpu_drain
    local cfg=$MSTAR_CONFIG; [[ $cfg = /* ]] || cfg=$MSTAR_DIR/$cfg
    # The default socket prefix /tmp/mstar_<user>/ is shared by every job of this user on the node
    # (other agents run mstar servers too), so give each server its own ZMQ socket dir and upload dir.
    SOCK_DIR=/tmp/mstar_${SLURM_JOB_ID:-$$}_$PORT/; UPLOAD_DIR=/tmp/mstar_up_${SLURM_JOB_ID:-$$}_$PORT/
    SERVER_T0=$(date +%s.%N)
    setsid env -C "$MSTAR_DIR" HF_HUB_OFFLINE=1 MSTAR_STEP_TELEMETRY_DIR="$d/telemetry" $EXTRA_ENV \
        "$MSTAR_DIR/.venv/bin/mstar" serve "$MSTAR_MODEL" --config "$cfg" --host 127.0.0.1 --port "$PORT" \
        --tensor-comm-protocol "$TENSOR_PROTO" --socket-path-prefix "$SOCK_DIR" --upload-dir "$UPLOAD_DIR" \
        $([ "$LOG_STATS" = 1 ] && echo --log-stats-file "$d/server_stats.log") \
        "$@" >> "$d/server.log" 2>&1 &
    SRV=$!
    wait_health $SRV "$PORT" "$HEALTH_TIMEOUT"
    local rc=$?
    SERVER_STARTUP_S=$(awk -v a="$SERVER_T0" -v b="$(date +%s.%N)" 'BEGIN{printf "%.1f", b-a}')
    [ $rc -eq 0 ] && echo "server up after ${SERVER_STARTUP_S}s (port $PORT, pid $SRV)"
    return $rc
}

run_client() {  # <case_dir> [benchmark.runner args...]; runs mstar's own client from the worktree, same node
    local d=$1; shift
    mkdir -p "$d/cache"
    # cwd must be the worktree (benchmark/ is not part of the installed package).
    env -C "$MSTAR_DIR" "$MSTAR_DIR/.venv/bin/python" -m benchmark.runner \
        --url "http://127.0.0.1:$PORT" --model "$MSTAR_MODEL" --local-cache "$d/cache" "$@"
}

start_gpu_sampler() {  # <csv>: 1 Hz memory timeline of our GPU(s); sets SMI_PID
    local g; g=$(_our_gpus); local sel=(); [ -n "$g" ] && sel=(-i "$g")
    nvidia-smi "${sel[@]}" --query-gpu=timestamp,index,memory.used,memory.total,utilization.gpu \
        --format=csv,noheader,nounits -l 1 > "$1" 2>&1 &
    SMI_PID=$!
}
stop_gpu_sampler() { kill "${SMI_PID:-0}" 2>/dev/null; wait "${SMI_PID:-0}" 2>/dev/null; }
