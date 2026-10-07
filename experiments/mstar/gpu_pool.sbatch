#!/bin/bash
#SBATCH -J mstar_pool
#SBATCH -p team1
#SBATCH --gres=gpu:8
#SBATCH -c 16
#SBATCH --mem=128G
#SBATCH --time=16:00:00
#SBATCH -o /data/garv901-55613a/logs/mstar/%x_%j.out
# Holds GPUs and runs queued admission_matrix.sbatch tasks on them, one task per GPU, so a matrix starts at once instead
# of waiting in the Slurm queue. Same footprint per task as an admission_matrix job (1 GPU, 2 CPUs, 16 GB).
#   sbatch experiments/mstar/gpu_pool.sbatch                      # prints the pool's job id, e.g. 13200
#   POOL_JOB=13200 SBATCH=experiments/mstar/pool_sbatch.sh REPS=3 WORKLOADS="..." experiments/mstar/submit_matrix.sh -f arms.txt
#   touch /data/garv901-55613a/results/mstar/pool/13200/STOP     # the pool exits when its queue is empty and its tasks are done
# STEP_CPUS=<n> (sbatch -c >= 8n+2 --mem >= 8*STEP_MEM --export=ALL,STEP_CPUS=n,STEP_MEM=20G): each task runs as its own job step
# (srun --exact): Slurm gives it n CPUs, STEP_MEM and one GPU in its own cgroup, which no process of the task can leave. Without it
# (STEP_CPUS=0) the tasks share the pool's CPUs: in the v4 matrix (pool 13181) all 16 were 76-100% busy, the per-task core pin did not
# hold (server and client processes widened their affinity to the pool's CPUs), and burst_v2 warmups ran at 185 out tok/s against
# 647 for the same arm in a job of its own. Steps keep the task's processes inside its CPUs, whatever affinity they set.
# A task file may set its own STEP_CPUS/STEP_MEM (pool_sbatch.sh passes them). Use 2 CPUs per task for admission runs: with 4 or 8
# the API server misses its 15 s result-delivery TTL under ac and main and the server stalls (queue_v4b.sh has the numbers).
# A task is <pool dir>/queue/<id>.task (VAR=value lines, written by pool_sbatch.sh). <id> = <pool job id><3-digit seq>, a number,
# so the sbatch's OUT (<id>_<label>), PORT (20000 + id % 10000) and socket dirs stay unique per task. Tasks start in id order.
REPO=${SLURM_SUBMIT_DIR:-/shared/home/garv901-55613a/regime-batching/regime-batching}
POOL=/data/garv901-55613a/results/mstar/pool/$SLURM_JOB_ID
LOGS=/data/garv901-55613a/logs/mstar
mkdir -p "$POOL"/{queue,running,done} "$LOGS"
IFS=, read -ra GPUS <<< "${CUDA_VISIBLE_DEVICES:?no GPUs in this allocation}"
echo "pool $SLURM_JOB_ID on $(hostname): gpus ${GPUS[*]}, dir $POOL ($(date +%T))"
STEP_CPUS=${STEP_CPUS:-0} STEP_MEM=${STEP_MEM:-20G}
[ "$STEP_CPUS" -gt 0 ] && echo "tasks run as job steps: $STEP_CPUS CPUs, $STEP_MEM, 1 GPU each"
declare -A busy   # gpu -> pid of its task
while :; do
    for g in "${!busy[@]}"; do
        kill -0 "${busy[$g]}" 2>/dev/null || { wait "${busy[$g]}"; echo "[$(date +%T)] gpu $g free (rc=$?)"; unset "busy[$g]"; }
    done
    for g in "${GPUS[@]}"; do
        [ -n "${busy[$g]:-}" ] && continue
        t=$(ls "$POOL"/queue/*.task 2>/dev/null | sort | head -1)
        [ -n "$t" ] || break
        id=$(basename "$t" .task)
        mv "$t" "$POOL/running/" || continue
        (
            set -a; source "$POOL/running/$id.task"; set +a
            if [ "$STEP_CPUS" -gt 0 ]; then   # Slurm picks the step's GPU (CUDA_VISIBLE_DEVICES=0 inside); the step's cgroup replaces the pin
                srun --exact -n1 -c "$STEP_CPUS" --mem="$STEP_MEM" --gres=gpu:1 --cpu-bind=none --job-name="$id" \
                    env -u SLURM_RESTART_COUNT SLURM_JOB_ID="$id" SLURM_SUBMIT_DIR="$REPO" POOL_PIN=0 \
                    bash "$REPO/experiments/mstar/admission_matrix.sbatch" >> "$LOGS/mstar_adm_$id.out" 2>&1
            else
                export CUDA_VISIBLE_DEVICES=$g SLURM_JOB_ID=$id SLURM_SUBMIT_DIR=$REPO
                unset SLURM_RESTART_COUNT
                bash "$REPO/experiments/mstar/admission_matrix.sbatch" >> "$LOGS/mstar_adm_$id.out" 2>&1
            fi
            echo "rc=$?" >> "$POOL/running/$id.task"; mv "$POOL/running/$id.task" "$POOL/done/"
        ) &
        busy[$g]=$!
        echo "[$(date +%T)] gpu $g <- task $id ($(grep -E '^LABEL=' "$POOL/running/$id.task"))"
    done
    if [ -e "$POOL/STOP" ] && [ "${#busy[@]}" -eq 0 ] && [ -z "$(ls "$POOL"/queue/ 2>/dev/null)" ]; then
        echo "pool $SLURM_JOB_ID: STOP and idle, exiting ($(date +%T))"; break
    fi
    sleep 20
done
