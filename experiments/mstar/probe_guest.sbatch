#!/bin/bash
#SBATCH -J probe_guest
#SBATCH -p guest
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH -o /data/garv901-55613a/logs/mstar/%x_%j.out
# One-off, CPU only (no GPU): can a guest node (node2, node4, node5) run admission_matrix.sbatch? On node1 /data and
# /shared are one local NVMe (xfs); this prints whether the other nodes see the same files. One job per node:
#   for n in node2 node4 node5; do sbatch -w $n experiments/mstar/probe_guest.sbatch; done
# If /data is not visible, Slurm cannot open the -o file above and the job fails at once; resubmit with
#   sbatch -w <node> -o /tmp/probe_guest_%j.out experiments/mstar/probe_guest.sbatch    (the log then stays on that node)
PY=/shared/home/garv901-55613a/mstar-worktrees/bench-peak-fast/.venv/bin/python
HF=/data/garv901-55613a/hf
RES=/data/garv901-55613a/results/mstar
echo "== hostname: $(hostname)  job=${SLURM_JOB_ID:-?} node=${SLURM_NODELIST:-?} user=$(whoami)"
echo "== df -h /data /shared"; df -h /data /shared 2>&1
echo "== paths"
for p in /data/mstar/env-laude.sh "$HF" "$PY" "$HOME/.cache/flashinfer" "$RES"; do
    [ -e "$p" ] && echo "EXISTS  $p" || echo "MISSING $p"
done
snap=$(ls -d "$HF"/hub/models--ByteDance-Seed--BAGEL-7B-MoT/snapshots/*/ 2>/dev/null | head -1)
if [ -n "$snap" ] && [ -e "${snap}ema.safetensors" ]; then echo "EXISTS  BAGEL snapshot $snap ($(ls "$snap" | wc -l) files, ema.safetensors present)"
else echo "MISSING BAGEL snapshot under $HF/hub/models--ByteDance-Seed--BAGEL-7B-MoT/snapshots/"; fi
echo "== source /data/mstar/env-laude.sh"
if (source /data/mstar/env-laude.sh && [ -n "${CUDA_HOME:-}" ] && [ -x "$CUDA_HOME/bin/nvcc" ]); then echo "OK  sourced; nvcc at $CUDA_HOME/bin/nvcc"
else echo "FAIL  source env-laude.sh (or no nvcc under CUDA_HOME)"; fi
echo "== venv imports ($PY)"
out=$("$PY" -c "import mstar, torch, flashinfer; print(torch.__version__)" 2>&1); rc=$?
[ $rc -eq 0 ] && echo "OK  import mstar, torch, flashinfer; torch $out" || { echo "FAIL  import (rc=$rc)"; echo "$out" | tail -3; }
echo "== $RES writable?"
tmp=$RES/.probe_$(hostname)_${SLURM_JOB_ID:-$$}
if touch "$tmp" 2>/dev/null && rm -f "$tmp"; then echo "OK  touch + rm in $RES"; else echo "FAIL  cannot write in $RES"; fi
echo "== done"
