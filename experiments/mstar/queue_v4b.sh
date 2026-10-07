#!/bin/bash
# Queue the v4 + public-trace matrix on pool 13181: 4 arms (arms_v4.txt), 4 workload families (oracle base, _mtx2, _mt2048 on
# one server), 3 reps of ac/ac_park/peak_bf_pc_park, main 1/1/2/3 reps (burst_v2, stag_v3_r5.5, mooncake r0.5, azure r1.25).
# The workload order rotates per rep; main is queued first in each rep (its tasks are the longest: stalls + restarts).
# Two smoke tasks go first: they check the request intake under AC vs main, in job steps of 8 CPUs.
set -euo pipefail
export POOL_PIN=0   # pool 13218 runs each task as a job step (8 CPUs, own cgroup): no pin
cd "$(dirname "$0")/../.."
export POOL_JOB=13218 SBATCH=experiments/mstar/pool_sbatch.sh MANIFEST=/data/garv901-55613a/results/mstar/manifest_v4b.tsv
DRY=${DRY:-}
S=./experiments/mstar/submit_matrix.sh
AC_ENV="MSTAR_KV_ADMISSION_FIT=sum MSTAR_KV_ADMISSION_ORDER=fifo MSTAR_KV_PLAN_CACHE=0 MSTAR_SCHED_PARK=0 MSTAR_KV_RELEASE_AT_COMPLETION=0"

# smoke (WARMUP off, as the first smoke)
REPS=1 WARMUP= WORKLOADS="smoke_eos smoke_res_decl smoke_res_oracle smoke_res_decl" $S $DRY "smoke_ac_step next $AC_ENV" "smoke_main_step main"

# The matrix goes in only after both smoke tasks are done: they build the per-venv FlashInfer caches, so no matrix task compiles
# next to a running benchmark. DRY: no wait.
P=/data/garv901-55613a/results/mstar/pool/$POOL_JOB
while [ -z "$DRY" ] && [ "$(ls "$P"/queue "$P"/running 2>/dev/null | grep -c '\.task$')" -gt 0 ]; do sleep 30; done
[ -z "$DRY" ] && { echo "smoke done ($(date +%T)):"; grep -h '^rc=' "$P"/done/*.task; }

# the arms of arms_v4.txt other than main, one quoted word each
mapfile -t ARMS < <(/usr/bin/grep -vE '^[[:space:]]*(#|$)|^main[[:space:]]' experiments/mstar/arms_v4.txt)
[ ${#ARMS[@]} -eq 3 ] || { echo "expected 3 non-main arms, got ${#ARMS[@]}"; exit 1; }

fam() {  # <family> <rep>: the family's three workloads, rotated by rep
    local b=$1 r=$2 w=("$1" "$1_mtx2" "$1_mt2048")
    echo "${w[$(( (r - 1) % 3 ))]} ${w[$(( r % 3 ))]} ${w[$(( (r + 1) % 3 ))]}"
}
declare -A MAIN_REPS=([bagel_burst_v2]=1 [bagel_stag_v3_r5.5]=1 [pub_mooncake_conv_r0.5]=2 [pub_azure_conv_r1.25]=3)
FAMS="pub_azure_conv_r1.25 pub_mooncake_conv_r0.5 bagel_burst_v2 bagel_stag_v3_r5.5"
for rep in 1 2 3; do
    for f in $FAMS; do
        [ "$rep" -le "${MAIN_REPS[$f]}" ] && REPS=1 WORKLOADS="$(fam $f $rep)" $S $DRY "main main"
    done
    for f in $FAMS; do
        REPS=1 WORKLOADS="$(fam $f $rep)" $S $DRY "${ARMS[@]}"
    done
done
