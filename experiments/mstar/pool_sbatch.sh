#!/bin/bash
# Drop-in for `sbatch --parsable --export=ALL [opts] <script>` (SBATCH= of submit_matrix.sh) that queues the task on a running
# gpu_pool.sbatch job instead of submitting a Slurm job. Prints the task id (the "job id" the results dir is named after).
#   POOL_JOB=<pool job id> SBATCH=experiments/mstar/pool_sbatch.sh ... experiments/mstar/submit_matrix.sh ...
# The task carries the variables admission_matrix.sbatch reads: CONFIG LABEL EXTRA_ENV WORKLOADS WARMUP and, when set,
# REQ_TIMEOUT RUN_LIMIT HEALTH_TIMEOUT_JOB STEP_CPUS STEP_MEM (a job-step pool: this task's CPUs and memory) plus every MSTAR_*
# variable of the caller's environment, and POOL_PIN (default 1).
set -euo pipefail
: "${POOL_JOB:?set POOL_JOB to the job id of a running gpu_pool.sbatch}"
POOL=/data/garv901-55613a/results/mstar/pool/$POOL_JOB
[ -d "$POOL/queue" ] || { echo "pool_sbatch: $POOL is not a pool dir (is job $POOL_JOB running?)" >&2; exit 1; }
script=${!#}
[[ $script == *admission_matrix.sbatch ]] || { echo "pool_sbatch: only runs admission_matrix.sbatch, got $script" >&2; exit 1; }
exec 9> "$POOL/.lock"; flock 9
# count the tasks in every subdir (queue, running, done, and any set aside by hand), so an id is never reused
n=$(( $(find "$POOL" -mindepth 2 -maxdepth 2 -name '*.task' 2>/dev/null | wc -l) + 1 ))
id=$(printf '%s%03d' "$POOL_JOB" "$n")
tmp=$POOL/.$id.tmp
{
    for v in CONFIG LABEL EXTRA_ENV WORKLOADS WARMUP REQ_TIMEOUT RUN_LIMIT HEALTH_TIMEOUT_JOB STEP_CPUS STEP_MEM; do
        [ -n "${!v+x}" ] && printf '%s=%q\n' "$v" "${!v}"
    done
    while IFS='=' read -r k _; do printf '%s=%q\n' "$k" "${!k}"; done < <(env | grep -E '^MSTAR_[A-Z0-9_]*=' || true)
    printf 'POOL_PIN=%q\n' "${POOL_PIN:-1}"   # admission_matrix.sbatch pins the task to one core of the pool (POOL_PIN=0: no pin)
} > "$tmp"
mv "$tmp" "$POOL/queue/$id.task"
echo "$id"
