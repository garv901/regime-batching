#!/bin/bash
# Drop-in for `sbatch --parsable --export=ALL [opts] <script>` (SBATCH= of submit_matrix.sh) that queues the task on a running
# gpu_pool.sbatch job instead of submitting a Slurm job. Prints the task id (the "job id" the results dir is named after).
#   POOL_JOB=<pool job id> SBATCH=experiments/mstar/pool_sbatch.sh ... experiments/mstar/submit_matrix.sh ...
# The task carries the variables admission_matrix.sbatch reads: CONFIG LABEL EXTRA_ENV WORKLOADS WARMUP and, when set,
# REQ_TIMEOUT RUN_LIMIT HEALTH_TIMEOUT_JOB plus every MSTAR_* variable of the caller's environment.
set -euo pipefail
: "${POOL_JOB:?set POOL_JOB to the job id of a running gpu_pool.sbatch}"
POOL=/data/garv901-55613a/results/mstar/pool/$POOL_JOB
[ -d "$POOL/queue" ] || { echo "pool_sbatch: $POOL is not a pool dir (is job $POOL_JOB running?)" >&2; exit 1; }
script=${!#}
[[ $script == *admission_matrix.sbatch ]] || { echo "pool_sbatch: only runs admission_matrix.sbatch, got $script" >&2; exit 1; }
exec 9> "$POOL/.lock"; flock 9
n=$(( $(ls "$POOL"/queue "$POOL"/running "$POOL"/done 2>/dev/null | grep -c '\.task$') + 1 ))
id=$(printf '%s%03d' "$POOL_JOB" "$n")
tmp=$POOL/.$id.tmp
{
    for v in CONFIG LABEL EXTRA_ENV WORKLOADS WARMUP REQ_TIMEOUT RUN_LIMIT HEALTH_TIMEOUT_JOB; do
        [ -n "${!v+x}" ] && printf '%s=%q\n' "$v" "${!v}"
    done
    while IFS='=' read -r k _; do printf '%s=%q\n' "$k" "${!k}"; done < <(env | grep -E '^MSTAR_[A-Z0-9_]*=' || true)
} > "$tmp"
mv "$tmp" "$POOL/queue/$id.task"
echo "$id"
