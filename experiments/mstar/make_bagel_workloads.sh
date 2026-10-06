#!/bin/bash
# Regenerate the BAGEL workloads in data/workloads/ (timed_trace JSONL, same format as lburst80_8k.jsonl).
#   bash experiments/mstar/make_bagel_workloads.sh            # defaults below
#   POOL=<tokens> STAG_RATES="0.5 1 2" bash experiments/mstar/make_bagel_workloads.sh
# POOL = max_num_pages * page_size of the BAGEL "kv" resource (default 2048 * 128; see experiments/mstar/README.md).
# BAGEL's text-to-text template adds OVH=20 tokens to every prompt; max_seq_len is 32768. make_workload.py clamps
# input_length to MAX_MODEL_LEN - output - 1, so MAX_MODEL_LEN = 32768 - OVH leaves room for the template.
set -euo pipefail
cd "$(dirname "$0")/../.."
POOL=${POOL:-262144}; OVH=20; MAXLEN=32768
MML=$((MAXLEN - OVH))
STAG_RATES=${STAG_RATES:-"0.5 1 2"}
STAG_DUR=${STAG_DUR:-600}
mk() { uv run --no-project --with numpy --with pandas python experiments/make_workload.py "$@" --max-model-len $MML > /dev/null; }
W=data/workloads

# bagel_burst: lburst80_8k scaled to the BAGEL pool. 80 long x (512 + 8000) = 681k vs the Qwen3-8B pool of 451k (1.51x).
# Here 46 long x (512 + 20 + 8000) = 392k vs POOL 262k (1.50x). Shorts: 9/s (16/s x 262k/451k), prompt 450-1000, out 16-64.
mk bimodal --out $W/bagel_burst.jsonl --rate 9 --duration 300 --seed 0 \
   --prompt-lo 450 --prompt 1000 --short-lo 16 --short-hi 64 \
   --long 8000 --long-prompt 512 --n-long 46 --long-at 30

# bagel_stag_r{R}: Poisson arrivals at R req/s for STAG_DUR s, each request long with probability 0.15 (same long class
# as the burst: prompt 512, output 8000), so longs arrive staggered instead of in one burst.
for R in $STAG_RATES; do
  mk bimodal --out $W/bagel_stag_r${R}.jsonl --rate $R --duration $STAG_DUR --seed 1 \
     --prompt-lo 450 --prompt 1000 --short-lo 16 --short-hi 64 \
     --long 8000 --long-prompt 512 --frac-long 0.15
done

# bagel_control: no KV pressure. 136 short + 8 long (prompt 512, output 1024) over ~60 s; total final KV <= 0.5 x POOL.
mk bimodal --out $W/bagel_control.jsonl --rate 2.4 --duration 60 --seed 2 \
   --prompt-lo 450 --prompt 1000 --short-lo 16 --short-hi 64 \
   --long 1024 --long-prompt 512 --n-long 8

# ITL_MS: assumed fixed decode inter-token latency for the fluid peak-in-flight-KV estimate (replace with the measured one)
python3 experiments/mstar/kv_report.py --pool $POOL --overhead $OVH --max-seq-len $MAXLEN --itl-ms ${ITL_MS:-20} $W/bagel_*.jsonl
