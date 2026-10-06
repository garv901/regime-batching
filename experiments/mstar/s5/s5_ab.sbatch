#!/bin/bash
#SBATCH -J mstar_s5
#SBATCH -p team1,guest
#SBATCH -w node1
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH -o /data/garv901-55613a/logs/mstar/%x_%j.out
#SBATCH --open-mode=append
# S5 A/B: round-robin vs deadline-aware micro-scheduling on the colocated Qwen3-Omni worker
# (configs/qwen3omni_colocated.yaml: Thinker, Talker, Code2Wav and both encoders on rank 0, one MicroScheduler).
#   sbatch experiments/mstar/s5/s5_ab.sbatch                               # from the study repo root
#   ARMS="deadline_w1000" RES=... sbatch --export=ALL experiments/mstar/s5/s5_ab.sbatch   # an extra arm into the same run dir
# One server per arm (round_robin, deadline), every case against it. Arms:
#   round_robin        MSTAR_SCHED_POLICY=round_robin (today's behaviour)
#   deadline           MSTAR_SCHED_POLICY=deadline, MSTAR_SCHED_MAX_WAIT_MS=200
#   deadline_w<ms>     the same with that aging guard, e.g. deadline_w1000
#   deadline_pa[_w<ms>]  deadline + MSTAR_SCHED_PIPELINE_AWARE=1 (Thinker decode carries the stream's deadline; the chain
#                      yields on urgency, not priority): the fix for what the first `deadline` arm showed
#   round_robin_<x>    another round_robin server (same env), to measure server-to-server noise
# Per server: one throwaway warm-up run (a mixed one: audio conc 8 + text conc 8), then REPS rounds of the four cases
#   a4  audio only, conc 4         a8  audio only, conc 8
#   b4  a4 + text-only long prompts at conc 8      b8  a8 + the same text load
# The client is experiments/mstar/s5/mixed_client.py (benchmark.request.OurSystem underneath).
# Restart-safe (-p guest can be preempted and requeued): the results dir does not depend on the job id, a case is done
# once its client.json exists, an arm with every case done is skipped, and a restarted job starts a fresh server for the
# arms that remain (and warms it up again: the first run on a fresh server is ~40% worse).
# -w node1: /data (CUDA toolkit env, the Omni weights) is node1's local disk; node2/4/5 have none of it.
cd "${SLURM_SUBMIT_DIR:-/shared/home/garv901-55613a/regime-batching/regime-batching}"
S5=experiments/mstar/s5
source /data/mstar/env-laude.sh
export HF_HOME=/data/garv901-55613a/hf
export MSTAR_DIR=${MSTAR_DIR:-/shared/home/garv901-55613a/mstar-worktrees/s5-stage-deadline}
# The server runs this worktree's code in the cu129 venv: it has flash-attn (the Qwen3-Omni vision encoder needs it; the
# worktree's own .venv does not, and is cu132 against the node's CUDA 12.9 toolkit). PYTHONPATH (set per arm below) makes
# `import mstar` resolve to $MSTAR_DIR instead of the venv's editable install of another worktree.
export MSTAR_VENV=${MSTAR_VENV:-/data/garv901-55613a/venvs/mstar-cu129}
export MSTAR_MODEL=qwen3_omni MSTAR_CONFIG=configs/qwen3omni_colocated.yaml
export HEALTH_TIMEOUT=${HEALTH_TIMEOUT:-2400}
: "${RES:=$PWD/results/mstar/s5/run1}" "${ARMS:=round_robin deadline}" "${REPS:=2}"
: "${CASE_TIMEOUT:=1500}" "${WARMUP_TIMEOUT:=1200}"
PROMPTS=$PWD/results/mstar/s5/prompts_long_1k2k.txt
export PORT=$((20000 + ${SLURM_JOB_ID:-$$} % 10000))
for need in /data/mstar/env-laude.sh "$HF_HOME/hub/models--Qwen--Qwen3-Omni-30B-A3B-Instruct/snapshots" "$PROMPTS" "$MSTAR_VENV/bin/mstar" "$MSTAR_DIR/.venv/bin/python"; do
    [ -e "$need" ] || { echo "preflight: $need missing on $(hostname)"; exit 3; }
done
source experiments/mstar/lib_mstar.sh
PYTHONPATH="$MSTAR_DIR" "$MSTAR_VENV/bin/python" -c "
import os, sys, flash_attn, mstar
assert os.path.realpath(mstar.__file__).startswith(os.path.realpath(sys.argv[1])), mstar.__file__
print('preflight: mstar from', mstar.__file__, 'flash_attn', flash_attn.__version__)" "$MSTAR_DIR" \
    || { echo "preflight: the server venv cannot import flash_attn and the worktree's mstar"; exit 3; }
mkdir -p "$RES"
{
    echo "job=${SLURM_JOB_ID:-?} host=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-?} restart=${SLURM_RESTART_COUNT:-0} date=$(date -Is)"
    echo "server=$MSTAR_DIR @ $(git -C "$MSTAR_DIR" rev-parse --short HEAD) ($(git -C "$MSTAR_DIR" rev-parse --abbrev-ref HEAD))"
    echo "arms=$ARMS reps=$REPS config=$MSTAR_CONFIG server venv=$MSTAR_VENV"
} | tee -a "$RES/meta.txt"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv

# case name -> client flags
case_args() {
    case $1 in
        a4) echo "--audio-conc 4 --audio-requests 24" ;;
        a8) echo "--audio-conc 8 --audio-requests 32" ;;
        b4) echo "--audio-conc 4 --audio-requests 24 --text-conc 8" ;;
        b8) echo "--audio-conc 8 --audio-requests 32 --text-conc 8" ;;
        warmup) echo "--audio-conc 8 --audio-requests 16 --text-conc 8" ;;
    esac
}
arm_env() {  # server-side env of an arm
    # MSTAR_CODE2WAV_MAX_GRAPH_BS: the bs=32 Code2Wav graph holds ~7 GiB, and with every Omni node on one 80 GB GPU the
    # other CUDA graphs then fail to capture and the first request OOMs (job 13068). Same value in every arm.
    local common="MSTAR_STEP_TELEMETRY_DETAIL=0 PYTHONPATH=$MSTAR_DIR MSTAR_CODE2WAV_MAX_GRAPH_BS=${CODE2WAV_MAX_GRAPH_BS:-8}"
    case $1 in
        round_robin|round_robin_*) echo "$common MSTAR_SCHED_POLICY=round_robin" ;;   # round_robin_b: a second server, for the noise floor
        deadline) echo "$common MSTAR_SCHED_POLICY=deadline MSTAR_SCHED_MAX_WAIT_MS=200" ;;
        deadline_pa) echo "$common MSTAR_SCHED_POLICY=deadline MSTAR_SCHED_MAX_WAIT_MS=200 MSTAR_SCHED_PIPELINE_AWARE=1" ;;
        deadline_pa_w*) echo "$common MSTAR_SCHED_POLICY=deadline MSTAR_SCHED_MAX_WAIT_MS=${1#deadline_pa_w} MSTAR_SCHED_PIPELINE_AWARE=1" ;;
        deadline_w*) echo "$common MSTAR_SCHED_POLICY=deadline MSTAR_SCHED_MAX_WAIT_MS=${1#deadline_w}" ;;
        *) return 1 ;;
    esac
}
run_case() {  # <arm> <case> <rep> <timeout_s> <outdir>
    local arm=$1 c=$2 rep=$3 tmo=$4 d=$5
    mkdir -p "$d"
    ln -sfn "$SDIR/telemetry" "$d/telemetry"
    echo "=== [$(date +%T)] $arm $c rep $rep -> $d"
    timeout "$tmo" env PYTHONPATH="$MSTAR_DIR" HF_HUB_OFFLINE=1 "$MSTAR_DIR/.venv/bin/python" "$S5/mixed_client.py" \
        --url "http://127.0.0.1:$PORT" --out "$d/client.json" --tag "${arm}-${c}-r$rep" \
        --text-prompts "$PROMPTS" $(case_args "$c") > "$d/client.log" 2>&1
    echo "client rc=$? ($(date +%T)): $(tail -1 "$d/client.log")"
}
server_ok() { kill -0 "$SRV" 2>/dev/null && curl -sf -m 10 "http://127.0.0.1:$PORT/health" > /dev/null; }

SRV=
for arm in $ARMS; do
    EXTRA_ENV=$(arm_env "$arm") || { echo "unknown arm $arm"; continue; }
    export EXTRA_ENV
    todo=()
    for rep in $(seq "$REPS"); do
        for c in a4 a8 b4 b8; do
            [ -f "$RES/$arm/${c}_r$rep/client.json" ] || todo+=("$c:$rep")
        done
    done
    if [ "${#todo[@]}" -eq 0 ]; then echo "arm $arm: every case is done, skipping"; continue; fi
    k=1; while [ -e "$RES/$arm/server_$k" ]; do k=$((k + 1)); done   # a requeue starts a new server dir
    SDIR=$RES/$arm/server_$k; mkdir -p "$SDIR"
    echo "=== [$(date +%T)] arm $arm: ${#todo[@]} cases to run, env: $EXTRA_ENV -> $SDIR"
    start_gpu_sampler "$SDIR/gpu_mem.csv"
    if ! start_server "$SDIR"; then
        echo "server for $arm failed to start; see $SDIR/server.log"; tail -30 "$SDIR/server.log"
        stop_server $SRV; stop_gpu_sampler; SRV=; continue
    fi
    echo "$SERVER_STARTUP_S" > "$SDIR/startup_s.txt"
    grep -m1 -i "scheduling policy" "$SDIR/server.log" > "$SDIR/policy_line.txt" 2>/dev/null
    # a graph bucket that failed to capture is served eagerly, which would make every number below unrepresentative
    grep -c "Failed to capture CUDA graph" "$SDIR/server.log" > "$SDIR/capture_failures.txt" 2>/dev/null
    echo "CUDA graph capture failures in this server: $(cat "$SDIR/capture_failures.txt")"
    run_case "$arm" warmup 0 "$WARMUP_TIMEOUT" "$SDIR/warmup"
    sleep 10
    for item in "${todo[@]}"; do
        c=${item%%:*}; rep=${item##*:}
        server_ok || { echo "server unhealthy before $arm $c r$rep; skipping the rest of this arm"; break; }
        run_case "$arm" "$c" "$rep" "$CASE_TIMEOUT" "$RES/$arm/${c}_r$rep"
        [ -f "$RES/$arm/${c}_r$rep/client.json" ] || echo "case $arm $c r$rep did not finish"
        sleep 10   # let the previous case's requests drain before the next one
    done
    stop_server $SRV; SRV=
    stop_gpu_sampler
done
python3 "$S5/analyze.py" --run "$RES" > "$RES/analysis.md" 2> "$RES/analysis.err" || echo "analysis failed; see $RES/analysis.err"
echo "=== [$(date +%T)] done: $RES"
