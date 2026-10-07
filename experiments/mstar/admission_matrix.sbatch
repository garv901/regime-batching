#!/bin/bash
#SBATCH -J mstar_adm
#SBATCH -p team1
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH -o /data/garv901-55613a/logs/mstar/%x_%j.out
#SBATCH --open-mode=append
# KV admission A/B on BAGEL text-to-text: one admission config per job, every workload, fresh server per run.
# Submit from the study repo root, one job per config:
#   for c in main ac sum_bf peak_fifo peak_bf; do sbatch --export=ALL,CONFIG=$c experiments/mstar/admission_matrix.sbatch; done
# Configs:
#   main       origin/main + telemetry                      (worktree regime-harness)
#   ac         origin/admission-control + telemetry         (bench-ac): summed reservation, strict FIFO
#   sum_bf     ours, MSTAR_KV_ADMISSION_FIT=sum  ORDER=backfill  (bench-peak)
#   peak_fifo  ours, MSTAR_KV_ADMISSION_FIT=peak ORDER=fifo      (bench-peak)
#   peak_bf    ours, MSTAR_KV_ADMISSION_FIT=peak ORDER=backfill  (bench-peak): EASY backfill + grant guard
#   ac_fast, peak_bf_fast  ac / peak_bf plus the incremental evictable set (bench-ac-fast / bench-peak-fast): same
#              decisions, less host time per admission check; kept apart so the policy A/B stays clean
#   next       worktree bench-next, no admission env of its own: an arm on it differs only by EXTRA_ENV (+ LABEL)
# Partitions: team1 (node1) only by default: a guest node (2, 4, 5) cannot run this until it is staged (see below).
# With -p team1,guest (sbatch -p, or SBATCH_OPTS in submit_matrix.sh) the job starts on whichever can run it first (guest = node1,2,4,5, and a
# guest job can be preempted and REQUEUED). A requeued job restarts this script from the top: with the default OUT it writes a
# new dir <job>_<label>_rq<SLURM_RESTART_COUNT>, so it never overwrites its earlier one (the analysis pools it with its label),
# and --open-mode=append keeps both attempts in the slurm log. A guest node (2, 4, 5) has no /data of its own to match node1's
# (no env-laude.sh, no BAGEL weights, no results/): the preflight below ends the job at once there. Check with probe_guest.sbatch.
# Arms on one config (several jobs, repeats, interleaved arms): experiments/mstar/submit_matrix.sh.
# The client is always the trace-replay client from regime-harness, so every config is driven identically.
# WORKLOADS (default below) is the v1 set. The v2 set has every requested output <= 2048 (BAGEL stops a request at 2,049
# tokens) and the longs carry their KV in the prompt; see experiments/mstar/README.md. --export=ALL passes the list on:
#   export WORKLOADS="bagel_control bagel_burst_v2 bagel_stag_v2_r1.5 bagel_stag_v2_r2.5 bagel_stag_v2_r3 bagel_burst_v2"
#   for c in main ac sum_bf peak_fifo peak_bf; do sbatch --export=ALL,CONFIG=$c experiments/mstar/admission_matrix.sbatch; done
# WARMUP (default bagel_burst_v2; empty = none): one throwaway replay of that workload on each fresh server, before the measured
# workloads, into $OUT/warmup_<wl>/ (warmup_<wl>_s<k> for the k-th server of the job). On a fresh server the first burst is ~40% worse
# in short-request E2E in every arm (first-run effect, not the policy); the analysis ignores warmup_* dirs. It costs ~6 min per server.
# Time: 6 workloads take ~45 min (v1 jobs: 51-77 min with the 6-32 min startup); each run ends by RUN_LIMIT=2400 s (a v2 run
# needs <= 1,500 s even if its last request waits out the 900 s client timeout). --time=03:00:00 leaves room for two server restarts.
cd "${SLURM_SUBMIT_DIR:-/shared/home/garv901-55613a/regime-batching/regime-batching}"
source /data/mstar/env-laude.sh
export HF_HOME=/data/garv901-55613a/hf
WT=/shared/home/garv901-55613a/mstar-worktrees/regime-batching   # the regime-batching worktrees
# bench-next and regime-harness (the v4 matrix runs from them) move under $WT after that matrix; until then they are here:
WT_OLD=/shared/home/garv901-55613a/mstar-worktrees
: "${CONFIG:?set CONFIG}"
: "${WORKLOADS:=bagel_control bagel_burst bagel_stag_r1 bagel_stag_r1.5 bagel_stag_r2 bagel_burst}"
: "${REQ_TIMEOUT:=900}" "${RUN_LIMIT:=2400}" "${WARMUP=bagel_burst_v2}"   # no colon: WARMUP= (empty) turns the warmup off
case $CONFIG in
    main)      MSTAR_DIR=$WT_OLD/regime-harness; ADM_ENV= ;;
    ac)        MSTAR_DIR=$WT/bench-ac;       ADM_ENV= ;;
    sum_bf)    MSTAR_DIR=$WT/bench-peak;     ADM_ENV="MSTAR_KV_ADMISSION_FIT=sum MSTAR_KV_ADMISSION_ORDER=backfill" ;;
    peak_fifo) MSTAR_DIR=$WT/bench-peak;     ADM_ENV="MSTAR_KV_ADMISSION_FIT=peak MSTAR_KV_ADMISSION_ORDER=fifo" ;;
    peak_bf)   MSTAR_DIR=$WT/bench-peak;     ADM_ENV="MSTAR_KV_ADMISSION_FIT=peak MSTAR_KV_ADMISSION_ORDER=backfill" ;;
    ac_fast)   MSTAR_DIR=$WT/bench-ac-fast;  ADM_ENV= ;;
    peak_bf_fast) MSTAR_DIR=$WT/bench-peak-fast; ADM_ENV="MSTAR_KV_ADMISSION_FIT=peak MSTAR_KV_ADMISSION_ORDER=backfill" ;;
    next)      MSTAR_DIR=$WT_OLD/bench-next; ADM_ENV= ;;
    *) echo "unknown CONFIG=$CONFIG"; exit 2 ;;
esac
CLIENT_DIR=$WT_OLD/regime-harness
EXTRA_ENV="$ADM_ENV ${EXTRA_ENV:-}"
PORT=$((20000 + SLURM_JOB_ID % 10000))
# LABEL names the arm in the results dir and meta.txt (default CONFIG); set it when only EXTRA_ENV differs
LABEL=${LABEL:-$CONFIG}
RQ=${SLURM_RESTART_COUNT:-0}   # > 0: slurm requeued this job (a guest-partition preemption); keep the earlier attempt's dir
OUT=${OUT:-/data/garv901-55613a/results/mstar/${SLURM_JOB_ID}_$LABEL$([ "$RQ" -gt 0 ] && echo "_rq$RQ")}
# Preflight: fail in seconds, not after a 1 h /health timeout, on a node that cannot run this. /shared (repo, worktrees, venvs) is
# NFS from node1 on every node, but /data (env-laude.sh + CUDA toolkit, the HF weights, results/) is node-local: node2/4/5 have
# none of it (probe_guest.sbatch, jobs 13056-13058). A guest node needs those staged first; see experiments/mstar/README.md.
for need in /data/mstar/env-laude.sh "$(dirname "$OUT")" "$HF_HOME/hub/models--ByteDance-Seed--BAGEL-7B-MoT/snapshots"; do
    [ -e "$need" ] || { echo "preflight: $need missing on $(hostname); this node cannot run the matrix"; exit 3; }
done
source experiments/mstar/lib_mstar.sh
# POOL_PIN=1 (set by pool_sbatch.sh): the tasks of a gpu_pool.sbatch job share the pool's CPUs, so a busy arm (hundreds of
# waiters) takes CPU time from the arms on the other GPUs; in the v4 smoke that delayed a server's request intake by 17-24 s.
# Pin this task, the server and client it starts included, to one core (both hyperthreads) of the pool, picked by the GPU index:
# the 2 CPUs a job of its own gets (cpu=2), never shared. No pin if the pool has fewer cores than that index.
CPUS=
if [ "${POOL_PIN:-0}" = 1 ]; then
    cores=() seen=" "
    for r in $(sed -n 's/^Cpus_allowed_list:[[:space:]]*//p' /proc/self/status | tr ',' ' '); do
        for ((c = ${r%-*}; c <= ${r#*-}; c++)); do
            g=$(cat /sys/devices/system/cpu/cpu$c/topology/thread_siblings_list)
            [[ $seen == *" $g "* ]] || { seen+="$g "; cores+=("$g"); }
        done
    done
    slot=${CUDA_VISIBLE_DEVICES:-0}
    if [[ $slot =~ ^[0-9]+$ ]] && [ "$slot" -lt ${#cores[@]} ]; then
        CPUS=${cores[$slot]}
        taskset -acp "$CPUS" $$ > /dev/null || CPUS=
    fi
    [ -n "$CPUS" ] || echo "POOL_PIN: no core for GPU slot $slot among ${#cores[@]} cores; not pinned"
fi
mkdir -p "$OUT"
{
    echo "job=$SLURM_JOB_ID host=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-?} config=$LABEL cpus=${CPUS:-job}"
    echo "server=$MSTAR_DIR @ $(git -C "$MSTAR_DIR" rev-parse --short HEAD) ($(git -C "$MSTAR_DIR" rev-parse --abbrev-ref HEAD))"
    echo "client=$CLIENT_DIR @ $(git -C "$CLIENT_DIR" rev-parse --short HEAD)"
    echo "extra_env=$EXTRA_ENV workloads=$WORKLOADS warmup=${WARMUP:-none} req_timeout=$REQ_TIMEOUT restart=$RQ"
} | tee "$OUT/meta.txt"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv

# One server per job: BAGEL compiles its prefill graphs with max-autotune at every start (15-22 min, the compile
# cache barely helps), so the server is restarted only when a run leaves it unhealthy or with timed-out requests.
# Telemetry and server.log live in $OUT/server_<k>/ and are symlinked into each case dir; the analysis restricts
# telemetry to each run's trace window.
server_ok() { kill -0 "$SRV" 2>/dev/null && curl -sf -m 10 "http://127.0.0.1:$PORT/health" > /dev/null; }
run_timed_out() {  # <bench.json>: any request that failed. A stalled server also fails requests with "Response payload is not
    # completed" (its 600 s abort) and "HTTP 500: result delivery timed out", which the old timeout-only test missed: the
    # next workload then ran on the broken server (every main stall in Phase 1c, the v4 calibration's Azure runs).
    python3 -c "import json,sys; b=json.load(open(sys.argv[1])); sys.exit(0 if any(b['errors']) else 1)" "$1" 2>/dev/null
}
run_trace() {  # <workload> <case_dir> <trace_run_id>: replay data/workloads/<workload>.jsonl; the same call for warmup and measured runs
    timeout "$RUN_LIMIT" env -C "$CLIENT_DIR" HF_HUB_OFFLINE=1 "$CLIENT_DIR/.venv/bin/python" -m benchmark.runner \
        --url "http://127.0.0.1:$PORT" --model bagel --trace "$PWD/data/workloads/$1.jsonl" \
        --bench-json "$2/bench.json" --request-timeout "$REQ_TIMEOUT" --num-warmup 2 \
        --trace-run-id "$3" --local-cache "$2/cache" > "$2/client.log" 2>&1
    echo "client rc=$? ($(date +%T))"; tail -6 "$2/client.log"
    rm -rf "$2/cache"
}
HEALTH_TIMEOUT=${HEALTH_TIMEOUT_JOB:-3600}
SRV=; nsrv=0
declare -A seen
start_gpu_sampler "$OUT/gpu_mem.csv"
for wl in $WORKLOADS; do
    rep=$(( ${seen[$wl]:-0} + 1 )); seen[$wl]=$rep
    case_dir=$OUT/${wl}_r$rep
    mkdir -p "$case_dir"
    if [ -z "$SRV" ] || ! server_ok; then
        [ -n "$SRV" ] && { echo "server unhealthy; restarting"; stop_server $SRV; }
        nsrv=$((nsrv + 1)); sdir=$OUT/server_$nsrv
        echo "=== [$(date +%T)] start server $nsrv -> $sdir"
        if ! start_server "$sdir"; then
            echo "server failed to start; see $sdir/server.log"; tail -30 "$sdir/server.log"
            stop_server $SRV; SRV=; continue
        fi
        echo "$SERVER_STARTUP_S" > "$sdir/startup_s.txt"
        if [ -n "$WARMUP" ]; then  # the run id carries "warmup", which the analysis already drops from the admission log
            wdir=$OUT/warmup_$WARMUP$([ "$nsrv" -gt 1 ] && echo "_s$nsrv"); mkdir -p "$wdir"
            echo "=== [$(date +%T)] $LABEL warmup $WARMUP -> $wdir (server $nsrv)"
            run_trace "$WARMUP" "$wdir" "${LABEL}-warmup-${WARMUP}"
            sleep 15
        fi
    fi
    ln -sfn "$sdir/telemetry" "$case_dir/telemetry"; ln -sf "$sdir/server.log" "$case_dir/server.log"
    echo "=== [$(date +%T)] $LABEL $wl rep $rep -> $case_dir (server $nsrv)"
    run_trace "$wl" "$case_dir" "${LABEL}-${wl}-r$rep"
    if run_timed_out "$case_dir/bench.json"; then
        echo "run had failed requests; restarting the server before the next workload"
        stop_server $SRV; SRV=
    fi
    sleep 15   # let completed requests' REMOVE_REQUESTs drain before the next trace
done
[ -n "$SRV" ] && stop_server $SRV
stop_gpu_sampler
echo "=== [$(date +%T)] done: $OUT"
