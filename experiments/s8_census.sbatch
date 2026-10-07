#!/bin/bash
#SBATCH -J s8_census
#SBATCH -p main
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH -o logs/%x_%j.out
# Stage 1b whole-step regime census: in-process vLLM (Llama-3.1-8B) driven step by step, default vs capture 2048.
cd /mnt/storage/garv901/regime-batching
source .venv/bin/activate
export HF_HUB_OFFLINE=1 PYTHONHASHSEED=0 VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 VLLM_ENABLE_V1_MULTIPROCESSING=0
unset VLLM_API_KEY OPENAI_API_KEY
OUT=results/runs/s8_census
nvidia-smi --query-gpu=index,name,memory.used --format=csv; uptime
# smoke: two scenarios, 3 probes, 1 repeat; abort unless probe steps reach the telemetry
rm -rf $OUT/smoke
REGIME_TELEMETRY_DIR=$OUT/smoke/telemetry timeout 900 python experiments/regime_census.py --config default \
    --out $OUT/smoke --scenarios none,1x128k --max-probes 3 --reps 1 || { echo SMOKE_FAILED; exit 1; }
python - <<'PY' || exit 1
import glob, json
n = sum(1 for f in glob.glob("results/runs/s8_census/smoke/telemetry/steps_*.jsonl") for l in open(f)
        if '"probe|' in l)
print("smoke probe steps in telemetry:", n)
raise SystemExit(0 if n >= 4 else 1)
PY
for cfg in default cap2048; do
  rm -rf $OUT/$cfg
  REGIME_TELEMETRY_DIR=$OUT/$cfg/telemetry python experiments/regime_census.py --config $cfg --out $OUT/$cfg \
      || echo "CONFIG_FAILED $cfg"
  uptime
done
echo "done $(date)"
