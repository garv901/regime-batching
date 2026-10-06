#!/bin/bash
# Regenerate the v3 BAGEL workloads in data/workloads/ (bagel_burst_v3, bagel_stag_v3_r{R}); the v1 and v2 files stay as they are.
#   bash experiments/mstar/make_bagel_workloads_v3.sh
#   POOL=<tokens> V3_RATES="4.5 5.5 6.5" ITL_MS_LIST="15 20" W=<dir> bash experiments/mstar/make_bagel_workloads_v3.sh
# Why v3: in v1/v2 every long has the same output (v2: 2,000), so the longs of a trace start and finish in lockstep and a
# summed reservation (prompt + max_tokens for the whole life) and a future-peak fit test see nearly the same set of
# requests. In v3 every long's output is ~U[256, 2000] (<= 2,000 < 2,049, so requested == generated, as in v2): the longs
# finish at different times and the KV that a running long still has to GROW by differs per request. Shorts are the v1/v2
# class (prompt 450-1000, output 16-64, cls "short"). POOL, OVH (20-token template), MAXLEN as in make_bagel_workloads_v2.sh.
#
# How the parameters were picked (analysis/admit_sim_bagel.py, CPU; README "Workloads v3" has the tables): only a burst with
# GROWTH-DOMINATED longs separates the future-peak policies from the summed ones; with the long prompt ~U[4000, 12000] a
# long's KV is ~88% prompt, so summing prompt + max_tokens over-books by only ~12% and peak_bf is NOT better than sum_bf.
#  - burst: long prompt ~U[1000, 4000] (90 longs = 1.20 x POOL of final KV), so a long's growth is ~31% of its final KV on
#    average (20-67% per long);
#  - staggered: long prompt ~U[4000, 12000] as required. This family is the "prompt-dominated" control: it is expected to
#    show little or no gain from peak over sum (see README).
set -euo pipefail
cd "$(dirname "$0")/../.."
POOL=${POOL:-262144}; OVH=20; MAXLEN=32768
MML=$((MAXLEN - OVH))
V3_RATES=${V3_RATES:-"4.5 5.5 6.5"}
STAG_DUR=${STAG_DUR:-600}
ITL_MS_LIST=${ITL_MS_LIST:-"15 20"}
LONG_OUT=2000      # < 2,049 with margin; long outputs ~U[LONG_OUT_LO, LONG_OUT]
LONG_OUT_LO=256
W=${W:-data/workloads}
mkdir -p "$W"
mk() { uv run --no-project --with numpy --with pandas python experiments/make_workload.py "$@" --max-model-len $MML > /dev/null; }

# bagel_burst_v3: same structure as bagel_burst_v2 (shorts at 9/s for 300 s, a burst of longs at t = 30 s, seed 0), but 90
# longs with prompt ~U[1000, 4000] and output ~U[256, 2000]: mean final KV 3.5k each (as drawn), 315k = 1.20x POOL in total.
mk bimodal --out $W/bagel_burst_v3.jsonl --rate 9 --duration 300 --seed 0 \
   --prompt-lo 450 --prompt 1000 --short-lo 16 --short-hi 64 \
   --long $LONG_OUT --long-lo $LONG_OUT_LO --long-prompt 4000 --long-prompt-lo 1000 --n-long 90 --long-at 30

# bagel_stag_v3_r{R}: Poisson arrivals at R req/s for STAG_DUR s, each request long with probability 0.15; a long has a
# prompt ~U[4000, 12000] and an output ~U[256, 2000] (mean final KV 9.1k). Rates 4.5 / 5.5 / 6.5 req/s: the first has
# little queueing (the low-pressure rung), the other two are where the pool binds (in the simulator the unlimited "open"
# policy would hold 1.3 / 4.7 / 8.2 x POOL; the fluid estimate below stays ~0.8-1.0 x because it keeps the decode ITL fixed
# at 15-20 ms), decode demand 0.9-1.0k tok/s, trace makespan 610-750 s in the simulator.
for R in $V3_RATES; do
  mk bimodal --out $W/bagel_stag_v3_r${R}.jsonl --rate $R --duration $STAG_DUR --seed 1 \
     --prompt-lo 450 --prompt 1000 --short-lo 16 --short-hi 64 \
     --long $LONG_OUT --long-lo $LONG_OUT_LO --long-prompt 12000 --long-prompt-lo 4000 --frac-long 0.15
done

# Fluid estimate of the peak in-flight KV (every request admitted on arrival, fixed decode ITL) per ITL in ITL_MS_LIST.
# kv_report.py rebuilds the "kv" block of a .meta.json on every call, so the file keeps the LAST ITL of the list only (as the
# v2 files do: itl20ms); each call prints its own. Only the v3 files are annotated.
for itl in $ITL_MS_LIST; do
  python3 experiments/mstar/kv_report.py --pool $POOL --overhead $OVH --max-seq-len $MAXLEN --itl-ms $itl \
     $W/bagel_burst_v3.jsonl $W/bagel_stag_v3_r*.jsonl
done
