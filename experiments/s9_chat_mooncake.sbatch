#!/bin/bash
#SBATCH -J s9_chat_mc
#SBATCH -p main
#SBATCH --gres=gpu:1
#SBATCH --time=01:20:00
#SBATCH -o logs/%x_%j.out
# Stage 2 (chat baseline), multi-turn chat: Mooncake conversation trace (first 307 requests; 512-token KV blocks shared
# across turns -> prefix-cache reuse; 23-28% of requests reach >= 16k context) replayed on Llama-3.1-8B at 0.43 and
# 0.86 req/s; fixes at the higher load; default repeated at the end.
cd "${SLURM_SUBMIT_DIR:-.}"
PORT=$((20000 + SLURM_JOB_ID % 10000))
OUT=results/runs/s9_chat_mc_$SLURM_JOB_ID
MODEL=meta-llama/Llama-3.1-8B-Instruct MAX_MODEL_LEN=131072
EXTRA_ENV="" CLIENT_ARGS="--timed-trace-chunk-hash-size 512" CLIENT_TIMEOUT=1200
source experiments/lib_serve.sh
unset VLLM_API_KEY OPENAI_API_KEY
nvidia-smi --query-gpu=index,name,memory.used --format=csv; uptime
# smoke: 12 requests must complete, and at least one must start with a prefix-cache hit (cached context on its first step)
run_case smoke data/workloads/chat_mc10.jsonl 12
python - "$OUT/smoke" <<'PY' || { echo SMOKE_FAILED; exit 1; }
import glob, json, sys
b = json.load(open(f"{sys.argv[1]}/bench.json"))
first = {}
for f in glob.glob(f"{sys.argv[1]}/telemetry/steps_*.jsonl"):
    for l in open(f):
        r = json.loads(l)
        if r.get("kind") == "step":
            for rid, q, kv, *_ in r.get("reqs") or []:
                first.setdefault(rid, kv - q)
hits = sum(1 for k, v in first.items() if v > 0 and not k.startswith("_warmup"))
print("smoke completed", b.get("completed"), "requests with a cached prefix on their first step:", hits, "of", len(first))
raise SystemExit(0 if b.get("completed") == 12 and hits >= 1 else 1)
PY
EXTRA_ENV="" run_case mc05_default data/workloads/chat_mc05.jsonl 307
EXTRA_ENV="" run_case mc10_default data/workloads/chat_mc10.jsonl 307
EXTRA_ENV="" run_case mc10_cap2048 data/workloads/chat_mc10.jsonl 307 -cc '{"max_cudagraph_capture_size":2048}'
EXTRA_ENV="REGIME_ATTN=1 REGIME_ATTN_POLICY=x_split_tile" run_case mc10_c3 data/workloads/chat_mc10.jsonl 307
EXTRA_ENV="" run_case mc10_mnbt512 data/workloads/chat_mc10.jsonl 307 --max-num-batched-tokens 512
EXTRA_ENV="" run_case mc10_default_rep data/workloads/chat_mc10.jsonl 307
uptime; echo "done $(date)"
