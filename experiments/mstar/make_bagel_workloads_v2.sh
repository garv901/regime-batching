#!/bin/bash
# Regenerate the v2 BAGEL workloads in data/workloads/ (bagel_burst_v2, bagel_stag_v2_r{R}); the v1 files stay as they are.
#   bash experiments/mstar/make_bagel_workloads_v2.sh
#   POOL=<tokens> V2_RATES="1.5 2.5 3" ITL_MS_LIST="15 20" bash experiments/mstar/make_bagel_workloads_v2.sh
# Why v2: BAGEL's decode loop is built with max_iters = MAX_OUTPUT_TOKENS = 2048 (mstar/model/base.py), so a request
# stops at 2,049 output tokens (1 from prefill + 2,048 loop iterations) whatever max_tokens it asks for. The v1 longs
# (output 8,000) were cut to 2,049 tokens and used ~0.45x of the KV pool instead of the planned 1.5x. In v2 every
# requested output is <= 2,048, so requested == generated, and the longs carry their KV in the PROMPT instead.
# POOL = max_num_pages * page_size of the BAGEL "kv" resource (2048 * 128). Template overhead OVH = 20 tokens per prompt,
# max_seq_len 32768 (see make_bagel_workloads.sh). Shorts are the v1 class: prompt 450-1000, output 16-64.
set -euo pipefail
cd "$(dirname "$0")/../.."
POOL=${POOL:-262144}; OVH=20; MAXLEN=32768
MML=$((MAXLEN - OVH))
V2_RATES=${V2_RATES:-"1.5 2.5 3"}
STAG_DUR=${STAG_DUR:-600}
ITL_MS_LIST=${ITL_MS_LIST:-"15 20"}
LONG_OUT=2000   # < 2,049 with margin
mk() { uv run --no-project --with numpy --with pandas python experiments/make_workload.py "$@" --max-model-len $MML > /dev/null; }
W=data/workloads

# bagel_burst_v2: same structure as bagel_burst (shorts at 9/s for 300 s, a burst of longs at t = 30 s).
# 46 long x (6500 + 20 + 2000) = 392k = 1.50x POOL, now real: 6.5k prompt tokens of KV each plus 2,000 generated.
mk bimodal --out $W/bagel_burst_v2.jsonl --rate 9 --duration 300 --seed 0 \
   --prompt-lo 450 --prompt 1000 --short-lo 16 --short-hi 64 \
   --long $LONG_OUT --long-prompt 6500 --n-long 46 --long-at 30

# bagel_stag_v2_r{R}: Poisson arrivals at R req/s for STAG_DUR s, each request long with probability 0.15; a long has
# a prompt ~U[8000, 16000] and output 2000 (final KV ~14k), so a long holds ~13k tokens of KV for 2000 x ITL seconds.
# Rates 1.5 / 2.5 / 3 give a peak in-flight KV of ~0.7 / ~1.2 / ~1.5 x POOL (kv_report fluid estimate: 0.69-0.79 / 1.07-1.39 /
# 1.39-1.66 for ITL 15-20 ms), a decode demand of 0.46-1.1k tok/s (decode saturates near 2k tok/s) and a prefill demand of
# 3.4-7.8k tok/s. In analysis/admit_sim_bagel.py the short E2E mean of sum_fifo over peak_bf is ~8x for the burst, ~4x at
# 2.5 req/s (p99 ~9x) and ~30x at 3 req/s; at 1.5 req/s nothing queues (the no-pressure rung).
for R in $V2_RATES; do
  mk bimodal --out $W/bagel_stag_v2_r${R}.jsonl --rate $R --duration $STAG_DUR --seed 1 \
     --prompt-lo 450 --prompt 1000 --short-lo 16 --short-hi 64 \
     --long $LONG_OUT --long-prompt 16000 --long-prompt-lo 8000 --frac-long 0.15
done

# One peak_inflight key per ITL in meta.json: the fluid estimate assumes a fixed decode inter-token latency, and the
# measured effective ITL is 7 / 17.5 / 32 ms at decode batch 8 / 32 / 64, so 15-20 ms brackets the 20-40 requests in flight.
# Only the v2 files are annotated; kv_report.py rewrites the .meta.json of every file it is given.
for itl in $ITL_MS_LIST; do
  python3 experiments/mstar/kv_report.py --pool $POOL --overhead $OVH --max-seq-len $MAXLEN --itl-ms $itl \
     $W/bagel_burst_v2.jsonl $W/bagel_stag_v2_r*.jsonl
done
