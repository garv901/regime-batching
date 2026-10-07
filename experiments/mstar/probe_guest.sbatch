#!/bin/bash
#SBATCH -J probe_guest
#SBATCH -p guest
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH -o /data/garv901-55613a/logs/mstar/%x_%j.out
# One-off, CPU only (no GPU): can a guest node (node2, node4, node5) run admission_matrix.sbatch? On node1 /data and
# /shared are one local NVMe (xfs); this prints whether the other nodes see the same files. One job per node:
#   for n in node2 node4 node5; do sbatch -w $n --export=ALL,PROBE_HOLD=420 experiments/mstar/probe_guest.sbatch; done
# The -o log is written on the node: if /data is node-local there, the login node never sees it. So the report is also copied to
# /tmp/probe_guest_<job>.out on the node and its one-line summary is put in the job comment (`sacct -j <id> -o Comment%400`);
# PROBE_HOLD=<s> keeps the job alive that long at the end, so `srun --jobid <id> --overlap cat /tmp/probe_guest_<id>.out` works.
# If Slurm cannot open the -o file (job fails at once), resubmit with -o /tmp/probe_guest_%j.out.
PY=/shared/home/garv901-55613a/mstar-worktrees/regime-batching/bench-peak-fast/.venv/bin/python
HF=/data/garv901-55613a/hf
RES=/data/garv901-55613a/results/mstar
exec > >(tee "/tmp/probe_guest_${SLURM_JOB_ID:-$$}.out") 2>&1
SUM="$(hostname)"; add() { SUM="$SUM; $*"; }
echo "== hostname: $(hostname)  job=${SLURM_JOB_ID:-?} node=${SLURM_NODELIST:-?} user=$(whoami)"
echo "== df -h /data /shared"; df -h /data /shared 2>&1
echo "== mounts"; grep -E " (/data|/shared|/shared/home) " /proc/mounts
add "data=$(df --output=source,fstype /data 2>/dev/null | tail -1 | tr -s ' ' ':') shared=$(df --output=source,fstype /shared 2>/dev/null | tail -1 | tr -s ' ' ':')"
echo "== paths"
for p in /data/mstar/env-laude.sh "$HF" "$PY" "$HOME/.cache/flashinfer" "$RES" "$RES/13029_ac"; do
    if [ -e "$p" ]; then echo "EXISTS  $p"; add "ok:${p#/data/garv901-55613a/}"; else echo "MISSING $p"; add "MISSING:${p#/data/garv901-55613a/}"; fi
done
echo "(results/mstar/13029_ac exists on node1: if it is visible here, this node sees node1's /data)"
snap=$(ls -d "$HF"/hub/models--ByteDance-Seed--BAGEL-7B-MoT/snapshots/*/ 2>/dev/null | head -1)
if [ -n "$snap" ] && [ -e "${snap}ema.safetensors" ]; then echo "EXISTS  BAGEL snapshot $snap ($(ls "$snap" | wc -l) files, ema.safetensors present)"; add "bagel:ok"
else echo "MISSING BAGEL snapshot under $HF/hub/models--ByteDance-Seed--BAGEL-7B-MoT/snapshots/"; add "bagel:MISSING"; fi
echo "== source /data/mstar/env-laude.sh"
if out=$(source /data/mstar/env-laude.sh && [ -x "$CUDA_HOME/bin/nvcc" ] && echo "$CUDA_HOME/bin/nvcc" 2>&1); then echo "OK  sourced; nvcc at $out"; add "env:ok"
else echo "FAIL  source env-laude.sh (or no nvcc under CUDA_HOME): $out"; add "env:FAIL"; fi
echo "== venv imports ($PY)"
out=$("$PY" -c "import mstar, torch, flashinfer; print(torch.__version__)" 2>&1); rc=$?
if [ $rc -eq 0 ]; then ver=$(echo "$out" | tail -1); echo "OK  import mstar, torch, flashinfer; torch $ver"; add "import:ok torch $ver"
else echo "FAIL  import (rc=$rc)"; echo "$out" | tail -3; add "import:FAIL rc=$rc"; fi
echo "== $RES writable?"
tmp=$RES/.probe_$(hostname)_${SLURM_JOB_ID:-$$}
if touch "$tmp" 2>/dev/null && rm -f "$tmp"; then echo "OK  touch + rm in $RES"; add "results:writable"; else echo "FAIL  cannot write in $RES"; add "results:NOT writable"; fi
echo "== done"
scontrol update JobId="${SLURM_JOB_ID:-0}" Comment="$SUM" 2>&1 | head -2
sleep "${PROBE_HOLD:-0}"
