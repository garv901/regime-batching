#!/bin/bash
#SBATCH -J s4a_repeat
#SBATCH -p main
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH -o logs/%x_%j.out
# S4a: (1) numerical noise floor: stock vLLM FULL_AND_PIECEWISE vs stock PIECEWISE vs C3 split policy;
# (2) two more repeats of the X-heavy 24k comparison (B0, static 512 budget, C3, shaping, full).
cd "${SLURM_SUBMIT_DIR:-.}"
PORT=$((20000 + SLURM_JOB_ID % 10000))
OUT=results/runs/s4a_$SLURM_JOB_ID
EXTRA_ENV=""
source experiments/lib_serve.sh
nvidia-smi --query-gpu=index,name,memory.used --format=csv
mkdir -p $OUT/c3
for v in base piecewise regime; do
    a=""; e=""
    [ $v = piecewise ] && a="--cudagraph-mode PIECEWISE"
    [ $v = regime ] && e="REGIME_ATTN=1"
    gpu_drain
    env $e python experiments/c3_correctness.py $a --out $OUT/c3/$v.json > $OUT/c3/$v.log 2>&1
    echo "c3 $v rc=$?"
done
for v in piecewise regime; do
    echo "== base vs $v"; python experiments/c3_correctness.py --compare $OUT/c3/base.json $OUT/c3/$v.json | tee $OUT/c3/cmp_$v.json | head -4
done

WL=data/workloads/xheavy_lc24k.jsonl; N=2400
SCHED="--scheduler-cls regime_batching.scheduler.RegimeScheduler"
for rep in 2 3; do
    EXTRA_ENV="" run_case B0_r$rep      $WL $N
    EXTRA_ENV="" run_case mnbt512_r$rep $WL $N --max-num-batched-tokens 512
    EXTRA_ENV="REGIME_ATTN=1" run_case C3_r$rep $WL $N
    EXTRA_ENV="REGIME_SCHED_ADMIT=fcfs REGIME_SCHED_SHAPE=cg_cap" run_case shape_r$rep $WL $N $SCHED
    EXTRA_ENV="REGIME_ATTN=1 REGIME_SCHED_ADMIT=peak REGIME_SCHED_SHAPE=cg_cap" run_case full_r$rep $WL $N $SCHED
done
echo "done $(date)"
