#!/bin/bash
# Submit REPS independent admission_matrix.sbatch jobs (one job = one server = one rep) for each ARM, interleaved:
# rep 1 of every arm, then rep 2 of every arm, ..., so a busy or slow node does not bias one arm.
#   REPS=3 WORKLOADS="bagel_burst_v2 bagel_stag_v2_r2.5 bagel_burst_v2" ./experiments/mstar/submit_matrix.sh ARM...
#   REPS=3 WORKLOADS="..." ./experiments/mstar/submit_matrix.sh -f arms.txt
# An ARM is ONE quoted word: "label config [VAR=val ...]" (fields split on whitespace, so an env value has no spaces):
#   "peak_bf next MSTAR_KV_ADMISSION_FIT=peak MSTAR_KV_ADMISSION_ORDER=backfill"      # arm "peak_bf" on config next
#   "ac ac"                                                                           # label ac, no extra env
# label  names the arm: results land in <results>/<job>_<label>/, and the analysis pools every job with that label as reps
#        (so give two arms the same label only on purpose). config = a CONFIG case of admission_matrix.sbatch.
# An arm file (-f, repeatable) has one arm per line in the same syntax; blank lines and # comments are skipped.
# Env: REPS (default 1), WORKLOADS (required), WARMUP (default bagel_burst_v2; WARMUP= turns it off, as in the sbatch),
#      SBATCH_OPTS (extra sbatch flags, e.g. "-p team1"), MANIFEST (default below), SBATCH (default sbatch).
# Options: -n  dry run (print the sbatch calls, submit nothing, write no manifest).
# Each job is submitted with CONFIG/LABEL/EXTRA_ENV/WORKLOADS/WARMUP in its environment (--export=ALL carries values with
# spaces unmangled), and one TSV line per job is appended to the manifest after sbatch accepts it:
#   job_id  label  config  extra_env  workloads  warmup  submit_time  rep
set -euo pipefail
cd "$(dirname "$0")/../.."
SBATCH=${SBATCH:-sbatch}
MANIFEST=${MANIFEST:-/data/garv901-55613a/results/mstar/manifest.tsv}
REPS=${REPS:-1}
WARMUP=${WARMUP-bagel_burst_v2}   # same default as admission_matrix.sbatch, so the manifest records what the job will use
SCRIPT=experiments/mstar/admission_matrix.sbatch
die() { echo "submit_matrix: $*" >&2; exit 2; }

DRY=0; labels=(); configs=(); extras=()
mapfile -t KNOWN < <(sed -nE 's/^ +([a-z0-9_]+)\) +MSTAR_DIR=.*/\1/p' "$SCRIPT")   # the CONFIG cases of the sbatch
add_arm() {  # "label config [VAR=val ...]"
    local -a w; read -r -a w <<< "$1"
    [ "${#w[@]}" -ge 2 ] || die "arm needs 'label config [VAR=val ...]': '$1'"
    local label=${w[0]} cfg=${w[1]} kv
    [[ $label =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || die "label '$label': use letters, digits, _ . - (it is a dir name)"
    [[ $label =~ _rq[0-9]+$ ]] && die "label '$label': a trailing _rq<N> is what a requeued job's dir gets; the analysis strips it"
    case " ${KNOWN[*]} " in *" $cfg "*) ;; *) die "config '$cfg' (arm '$label') is not a CONFIG case of $SCRIPT: ${KNOWN[*]}" ;; esac
    for kv in "${w[@]:2}"; do [[ $kv =~ ^[A-Za-z_][A-Za-z0-9_]*=[^[:space:]]*$ ]] || die "arm '$label': '$kv' is not VAR=val"; done
    local l
    for l in "${labels[@]+"${labels[@]}"}"; do [ "$l" != "$label" ] || die "label '$label' appears twice"; done
    labels+=("$label"); configs+=("$cfg"); extras+=("${w[*]:2}")
}
while [ $# -gt 0 ]; do
    case $1 in
        -n) DRY=1 ;;
        -f) shift; [ -r "${1:-}" ] || die "-f needs a readable arm file"
            while IFS= read -r line || [ -n "$line" ]; do
                line=${line%%#*}; [[ $line =~ ^[[:space:]]*$ ]] || add_arm "$line"
            done < "$1" ;;
        -*) die "unknown option $1" ;;
        *) add_arm "$1" ;;
    esac
    shift
done
[ "${#labels[@]}" -gt 0 ] || die "no arms; usage in the header of $0"
[[ $REPS =~ ^[1-9][0-9]*$ ]] || die "REPS='$REPS' is not a positive integer"
: "${WORKLOADS:?set WORKLOADS=\"wl1 wl2 ...\" (names under data/workloads/, without .jsonl)}"
for wl in $WORKLOADS ${WARMUP:+$WARMUP}; do [ -f "data/workloads/$wl.jsonl" ] || die "data/workloads/$wl.jsonl does not exist"; done

[ "$DRY" = 1 ] || { mkdir -p "$(dirname "$MANIFEST")"; [ -s "$MANIFEST" ] || printf '#job_id\tlabel\tconfig\textra_env\tworkloads\twarmup\tsubmit_time\trep\n' >> "$MANIFEST"; }
ids=()
for rep in $(seq "$REPS"); do
    for i in "${!labels[@]}"; do
        label=${labels[$i]} cfg=${configs[$i]} extra=${extras[$i]}
        if [ "$DRY" = 1 ]; then
            echo "[dry] rep $rep: CONFIG=$cfg LABEL=$label EXTRA_ENV='$extra' WORKLOADS='$WORKLOADS' WARMUP='$WARMUP' $SBATCH --parsable --export=ALL ${SBATCH_OPTS:-} $SCRIPT"
            continue
        fi
        # OUT would send every job to one dir; the sbatch derives it from the job id and the label
        jid=$(env -u OUT CONFIG="$cfg" LABEL="$label" EXTRA_ENV="$extra" WORKLOADS="$WORKLOADS" WARMUP="$WARMUP" \
              "$SBATCH" --parsable --export=ALL ${SBATCH_OPTS:-} "$SCRIPT") || { echo "submit_matrix: sbatch failed for $label rep $rep; submitted so far: ${ids[*]:-none}" >&2; exit 1; }
        jid=${jid%%;*}; ids+=("$jid")
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$jid" "$label" "$cfg" "$extra" "$WORKLOADS" "$WARMUP" "$(date -u +%FT%TZ)" "$rep" >> "$MANIFEST"
        echo "$jid  rep $rep  $label  ($cfg${extra:+ $extra})"
    done
done
[ "$DRY" = 1 ] || echo "submitted ${#ids[@]} job(s): ${ids[*]}  (manifest: $MANIFEST)"
