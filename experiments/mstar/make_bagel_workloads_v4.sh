#!/bin/bash
# Derive the v4 BAGEL workloads in data/workloads/: for each Phase 1c base trace two copies whose rows carry a declared
# max_tokens above the generated output_length (experiments/mstar/declare_max_tokens.py). The v1-v3 files are only read.
#   bash experiments/mstar/make_bagel_workloads_v4.sh
#   BASES="bagel_burst_v2" POOL=<tokens> ITL_MS_LIST="15 20" W=<dir> bash experiments/mstar/make_bagel_workloads_v4.sh
# Why v4: v1-v3 are ORACLE traces (the client sends max_tokens = output_length), so a summed reservation (prompt +
# max_tokens) and a future-peak fit test know every request's exact length. A production client over-declares max_tokens, or
# omits it (the server then assumes 2,048 for BAGEL). The variants keep the arrivals, prompts and generated outputs of the base
# and change only what the server is told:
#   <base>_mtx2     max_tokens = min(2048, 2 x output_length)   a predictor with a 2x margin (shorts 16-64 -> 32-128)
#   <base>_mt2048   max_tokens = 2048 for every request          no information about the output length
# The request still stops after output_length tokens (emulated EOS), so the KV actually used equals the base's; only the
# reservation grows. --id-offset shifts every hash_id (x2: +1e9, 2048: +2e9), so a variant replayed on the server that served
# its base (or another variant) builds different prompt tokens and cannot hit the prefix cache the other run left behind.
# Each variant's .meta.json is the base's with `path` and a `declared` block (base, mode, cap, id offset); kv_report.py then
# rebuilds its `kv` block, which carries the declared totals and the declared fluid peaks next to the generated ones.
set -euo pipefail
cd "$(dirname "$0")/../.."
POOL=${POOL:-262144}; OVH=20; MAXLEN=32768; CAP=2048
BASES=${BASES:-"bagel_burst_v2 bagel_stag_v2_r3 bagel_burst_v3 bagel_stag_v3_r5.5"}
ITL_MS_LIST=${ITL_MS_LIST:-"15 20"}
W=${W:-data/workloads}

variant() {  # variant <base> <suffix> <mode> <id offset>
  local base=$1 suffix=$2 mode=$3 off=$4
  python3 experiments/mstar/declare_max_tokens.py "$W/$base.jsonl" "$W/${base}_$suffix.jsonl" --mode "$mode" --cap $CAP --id-offset "$off"
  python3 - "$W/$base.meta.json" "$W/${base}_$suffix.meta.json" "$W/${base}_$suffix.jsonl" "$base" "$mode" "$CAP" "$off" <<'EOF'
import json, sys
src, dst, path, base, mode, cap, off = sys.argv[1:]
meta = json.load(open(src))
meta["path"] = path
meta["declared"] = dict(base=base, mode=mode, cap=int(cap), id_offset=int(off))
json.dump(meta, open(dst, "w"), indent=1)
EOF
}

FILES=()
for base in $BASES; do
  variant "$base" mtx2   x2    1000000000
  variant "$base" mt2048 fixed 2000000000
  FILES+=("$W/${base}_mtx2.jsonl" "$W/${base}_mt2048.jsonl")
done

# Fluid estimate per ITL in ITL_MS_LIST (every request admitted on arrival, fixed decode ITL, no queueing): the peak in-flight
# KV with the generated lengths, with the declared lengths, and of the summed reservation. kv_report.py rebuilds the "kv" block
# of a .meta.json on every call, so the file keeps the LAST ITL of the list only (as the v2/v3 files do); each call prints its own.
for itl in $ITL_MS_LIST; do
  python3 experiments/mstar/kv_report.py --pool $POOL --overhead $OVH --max-seq-len $MAXLEN --max-declared $CAP --itl-ms $itl "${FILES[@]}"
done
