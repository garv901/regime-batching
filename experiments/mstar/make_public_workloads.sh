#!/bin/bash
# Regenerate the public-trace workloads in data/workloads/: pub_mooncake_conv_r{R}, pub_azure_conv_r{R}, and for each the
# over-declared variants _mtx2 and _mt2048. Needs the raw traces: bash experiments/mstar/fetch_public_traces.sh
#   bash experiments/mstar/make_public_workloads.sh
#   POOL=<tokens> MC_RATES="0.25 0.5 0.75 1" AZ_RATES="1 1.25 2 3" SPAN=600 W=<dir> bash experiments/mstar/make_public_workloads.sh
# Windows (README "Public traces" has the numbers): every output file spans SPAN = 600 s, so the window of the ORIGINAL
# trace is SPAN * r seconds long, and the windows of one source are nested around one centre C (start = C - SPAN * r / 2):
# the windows of Mooncake are [1425, 1575) (r = 0.25), [1350, 1650), [1275, 1725), [1200, 1800) (C = 1500 s), those of Azure
# conv [2250, 2850), [2175, 2925) (r = 1.25), [1950, 3150), [1650, 3450) (C = 2550 s). Each centre has the best mean score of
# `public_trace_report.py --pick-window` (the distance of the window statistics to the whole-trace ones).
# r = 0.25 for Mooncake is the no-pressure control (the three rates of the brief, 0.5 / 0.75 / 1, all press on the pool).
# Fluid peak: kv_report.py at ITL 15 and 20 ms; a .meta.json keeps the last (20 ms), as in make_bagel_workloads_v3.sh.
# Variants: declare_max_tokens.py with a different --id-offset each, so that on one server a variant shares no prompt tokens
# with its base or with the other variant. A variant has the arrivals, lengths and (shifted) prompts of its base, so it
# gets its base's .meta.json (kv_report's "kv" block is the same: it counts generated tokens), and public_trace_report.py
# --write-meta then sets its path and adds a "declared" block (declared KV = input + 20 + max_tokens, and its ratio).
set -euo pipefail
cd "$(dirname "$0")/../.."
POOL=${POOL:-262144}; OVH=20; MAXLEN=32768
MC_RATES=${MC_RATES-"0.25 0.5 0.75 1"}
AZ_RATES=${AZ_RATES-"1 1.25 2 3"}
MC_CENTRE=${MC_CENTRE:-1500}
AZ_CENTRE=${AZ_CENTRE:-2550}
SPAN=${SPAN:-600}
ITL_MS_LIST=${ITL_MS_LIST:-"15 20"}
W=${W:-data/workloads}
mkdir -p "$W"

make_one() {  # make_one <source> <name prefix> <centre> <r>
  local src=$1 pre=$2 centre=$3 r=$4 start
  start=$(awk -v c="$centre" -v r="$r" -v s="$SPAN" 'BEGIN { printf "%g", c - s * r / 2 }')
  python3 -I experiments/mstar/convert_public_trace.py "$src" --start "$start" --span "$SPAN" --rate "$r" \
      --out "$W/${pre}_r${r}.jsonl"
}

BASES=()
for r in $MC_RATES; do make_one mooncake pub_mooncake_conv "$MC_CENTRE" "$r"; BASES+=("$W/pub_mooncake_conv_r$r.jsonl"); done
for r in $AZ_RATES; do make_one azure_conv pub_azure_conv "$AZ_CENTRE" "$r"; BASES+=("$W/pub_azure_conv_r$r.jsonl"); done

for itl in $ITL_MS_LIST; do
  python3 -I experiments/mstar/kv_report.py --pool "$POOL" --overhead $OVH --max-seq-len $MAXLEN --itl-ms "$itl" "${BASES[@]}"
done

for base in "${BASES[@]}"; do
  n=${base%.jsonl}
  python3 -I experiments/mstar/declare_max_tokens.py "$base" "${n}_mtx2.jsonl" --mode x2 --id-offset 1000000000
  python3 -I experiments/mstar/declare_max_tokens.py "$base" "${n}_mt2048.jsonl" --mode fixed --id-offset 2000000000
  cp "$n.meta.json" "${n}_mtx2.meta.json"
  cp "$n.meta.json" "${n}_mt2048.meta.json"
done
python3 -I experiments/mstar/public_trace_report.py --write-meta "${BASES[@]}"
