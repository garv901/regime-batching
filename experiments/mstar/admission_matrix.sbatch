#!/bin/bash
#SBATCH -J mstar_adm
#SBATCH -p team1
#SBATCH --gres=gpu:1
#SBATCH --time=03:30:00
#SBATCH -o /data/garv901-55613a/logs/mstar/%x_%j.out
# KV admission A/B on BAGEL text-to-text: one admission config per job, every workload, fresh server per run.
# Submit from the study repo root, one job per config:
#   for c in main ac sum_bf peak_fifo peak_bf; do sbatch --export=ALL,CONFIG=$c experiments/mstar/admission_matrix.sbatch; done
# Configs:
#   main       origin/main + telemetry                      (worktree regime-harness)
#   ac         origin/admission-control + telemetry         (bench-ac): summed reservation, strict FIFO
#   sum_bf     ours, MSTAR_KV_ADMISSION_FIT=sum  ORDER=backfill  (bench-peak)
#   peak_fifo  ours, MSTAR_KV_ADMISSION_FIT=peak ORDER=fifo      (bench-peak)
#   peak_bf    ours, MSTAR_KV_ADMISSION_FIT=peak ORDER=backfill  (bench-peak): EASY backfill + grant guard
# The client is always the trace-replay client from regime-harness, so every config is driven identically.
cd "${SLURM_SUBMIT_DIR:-/shared/home/garv901-55613a/regime-batching/regime-batching}"
source /data/mstar/env-laude.sh
export HF_HOME=/data/garv901-55613a/hf
WT=/shared/home/garv901-55613a/mstar-worktrees
: "${CONFIG:?set CONFIG}"
: "${WORKLOADS:=bagel_control bagel_burst bagel_stag_r0.5 bagel_stag_r1 bagel_stag_r2 bagel_burst}"
: "${REQ_TIMEOUT:=900}" "${RUN_LIMIT:=2400}"
case $CONFIG in
    main)      MSTAR_DIR=$WT/regime-harness; ADM_ENV= ;;
    ac)        MSTAR_DIR=$WT/bench-ac;       ADM_ENV= ;;
    sum_bf)    MSTAR_DIR=$WT/bench-peak;     ADM_ENV="MSTAR_KV_ADMISSION_FIT=sum MSTAR_KV_ADMISSION_ORDER=backfill" ;;
    peak_fifo) MSTAR_DIR=$WT/bench-peak;     ADM_ENV="MSTAR_KV_ADMISSION_FIT=peak MSTAR_KV_ADMISSION_ORDER=fifo" ;;
    peak_bf)   MSTAR_DIR=$WT/bench-peak;     ADM_ENV="MSTAR_KV_ADMISSION_FIT=peak MSTAR_KV_ADMISSION_ORDER=backfill" ;;
    *) echo "unknown CONFIG=$CONFIG"; exit 2 ;;
esac
CLIENT_DIR=$WT/regime-harness
EXTRA_ENV="$ADM_ENV ${EXTRA_ENV:-}"
PORT=$((20000 + SLURM_JOB_ID % 10000))
OUT=${OUT:-/data/garv901-55613a/results/mstar/${SLURM_JOB_ID}_$CONFIG}
source experiments/mstar/lib_mstar.sh
mkdir -p "$OUT"
{
    echo "job=$SLURM_JOB_ID host=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-?} config=$CONFIG"
    echo "server=$MSTAR_DIR @ $(git -C "$MSTAR_DIR" rev-parse --short HEAD) ($(git -C "$MSTAR_DIR" rev-parse --abbrev-ref HEAD))"
    echo "client=$CLIENT_DIR @ $(git -C "$CLIENT_DIR" rev-parse --short HEAD)"
    echo "extra_env=$EXTRA_ENV workloads=$WORKLOADS req_timeout=$REQ_TIMEOUT"
} | tee "$OUT/meta.txt"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv

declare -A seen
for wl in $WORKLOADS; do
    rep=$(( ${seen[$wl]:-0} + 1 )); seen[$wl]=$rep
    case_dir=$OUT/${wl}_r$rep
    mkdir -p "$case_dir"
    echo "=== [$(date +%T)] $CONFIG $wl rep $rep -> $case_dir"
    start_gpu_sampler "$case_dir/gpu_mem.csv"
    if ! start_server "$case_dir"; then
        echo "server failed to start; see $case_dir/server.log"; tail -30 "$case_dir/server.log"
        stop_server $SRV; stop_gpu_sampler; continue
    fi
    echo "$SERVER_STARTUP_S" > "$case_dir/startup_s.txt"
    timeout "$RUN_LIMIT" env -C "$CLIENT_DIR" HF_HUB_OFFLINE=1 "$CLIENT_DIR/.venv/bin/python" -m benchmark.runner \
        --url "http://127.0.0.1:$PORT" --model bagel --trace "$PWD/data/workloads/$wl.jsonl" \
        --bench-json "$case_dir/bench.json" --request-timeout "$REQ_TIMEOUT" --num-warmup 2 \
        --trace-run-id "${CONFIG}-${wl}-r$rep" --local-cache "$case_dir/cache" > "$case_dir/client.log" 2>&1
    echo "client rc=$? ($(date +%T))"; tail -12 "$case_dir/client.log"
    stop_server $SRV
    stop_gpu_sampler
    rm -rf "$case_dir/cache"
done
echo "=== [$(date +%T)] done: $OUT"
