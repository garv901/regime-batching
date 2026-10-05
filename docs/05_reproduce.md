# Reproduce the experiments

## 1. Install

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install vllm==0.30.0          # pulls torch 2.13 (CUDA 13.0)
uv pip install -e .                   # registers the regime_batching vLLM plugin
uv pip install pandas pyarrow matplotlib tabulate pytest
```

Optional, for FlashInfer microbenchmarks only: `flashinfer-cubin` from the FlashInfer wheel index (the XQA decode
kernels on Hopper need it).

Download the models with the Hugging Face CLI: `Qwen/Qwen3-8B`, `Qwen/Qwen3-4B`,
`meta-llama/Llama-3.1-8B-Instruct` (gated: request access first). The job scripts set `HF_HUB_OFFLINE=1`.

The CPU tests use vLLM's own test helpers. Clone vLLM v0.30.0 into `third_party/vllm` to run them.

## 2. Run one configuration

`experiments/lib_serve.sh` starts a vLLM server on `127.0.0.1`, replays a workload with `vllm bench serve`,
saves the results and stops the server.

```bash
source experiments/lib_serve.sh
export MODEL=Qwen/Qwen3-8B OUT=results/runs/my_run PORT=21000

# vLLM default
EXTRA_ENV="" run_case B0 data/workloads/xheavy_lc24k.jsonl 2400

# step shaping
EXTRA_ENV="REGIME_SCHED_ADMIT=fcfs REGIME_SCHED_SHAPE=cg_cap" \
  run_case shape data/workloads/xheavy_lc24k.jsonl 2400 \
  --scheduler-cls regime_batching.scheduler.RegimeScheduler

# oracle KV admission on the burst workload
EXTRA_ENV="REGIME_SCHED_ADMIT=peak REGIME_SCHED_SHAPE=off" \
  run_case peak data/workloads/lburst80_8k.jsonl 4800 \
  --scheduler-cls regime_batching.scheduler.RegimeScheduler
```

`EXTRA_ENV` applies to the server only. Arguments after the request count go to `vllm serve`. Each case writes
`<OUT>/<case>/bench.json`, `server.log`, `client.log` and `telemetry/steps_<pid>.jsonl` (plus `sched_<pid>.jsonl`
for the scheduler).

Note: `lib_serve.sh` sets `ROOT` to the repository root (the parent of `experiments/`). Set `ROOT` before you
source it to use another location. If the server environment sets `VLLM_API_KEY`, the client must send the same
key; `run_case` passes it as `OPENAI_API_KEY`.

## 3. Job scripts used for the published results

The scripts are Slurm batch files for our cluster (partition `main`, 1 GPU). Submit them from the repository
root, and change the partition for your cluster. Each script runs all its cases in one job, so every comparison
shares a GPU.

The repository stores each `bench.json` as `bench.json.gz` (about 8 MB in place of 22 MB). The analysis scripts
read either file.

| Script | Content | Results |
|---|---|---|
| `experiments/s1c_xheavy.sbatch` | C3 correctness; X-heavy B0 and C3 (Qwen3-8B 24k and 16k, Qwen3-4B 24k) | `results/runs/s1c_1015/` |
| `experiments/s3a_static.sbatch` | X-heavy Qwen3-8B: B0, cap2048, mnbt512, shape, full | `results/runs/s3a_1017/` |
| `experiments/s4a_repeat.sbatch` | Correctness noise floor; X-heavy repeats 2 and 3 | `results/runs/s4a_1034/` |
| `experiments/s4b_burst.sbatch` | Burst Qwen3-8B: B0, peak, full | `results/runs/s4b_1033/` |
| `experiments/s4c_burst_static.sbatch` | Burst Qwen3-8B: `--max-num-seqs` 32, 64, 128, 256 | `results/runs/s4c_1039/` |
| `experiments/s4d_burst_abl.sbatch` | Burst Qwen3-8B: sjf ablation, B0 and peak repeat | `results/runs/s4d_1043/` |
| `experiments/s5a_llama.sbatch` | Llama-3.1-8B: X-heavy matrix and burst | `results/runs/s5a_llama_1040/` |
| `experiments/s5b_qwen4b.sbatch` | Qwen3-4B: X-heavy matrix and burst; Qwen3-8B reactive shaping | `results/runs/s5b_qwen4b_1041/` |

Each X-heavy case takes 3-5 minutes and each burst case 7-10 minutes, including server start.

## 4. Analyze

```bash
# per-class latency table, speedups against a reference case
python analysis/bench_compare.py results/runs/s4b_1033/{B0,peak,full} --ref B0 --md out.md

# mean +- std over repeats; each repeat is normalized to its own B0
cd analysis && python repeats.py --cases B0 mnbt512 C3 shape full \
  --rep 'r1=../results/runs/s3a_1017/{case}' \
  --rep 'r2=../results/runs/s4a_1034/{case}_r2' \
  --rep 'r3=../results/runs/s4a_1034/{case}_r3'

# regime ribbons, zoom view, regime bars, CDFs, Perfetto traces for every case of a job
python analysis/telemetry.py run results/runs/s4b_1033
```

## 5. Workloads

The traces of all published runs are in `data/workloads/`, with a `.meta.json` summary for each. To make new
traces, run `python experiments/make_workload.py --help`. The main options are the family (`bimodal`, `mix`,
`replay`, `waves`), arrival rate and duration, prompt range (`--prompt-lo`, `--prompt`), short output range
(`--short-lo`, `--short-hi`), long output (`--long`), long prompt (`--long-prompt`), number of long requests
(`--n-long`) and their arrival (`--long-first`, `--long-at`).

## 6. Other checks

```bash
# scheduler and planner tests (CPU only)
HF_HUB_OFFLINE=1 python -m pytest -q tests/test_oracle_scheduler.py

# C3 numerical check: stock, then C3, then compare greedy outputs
python experiments/c3_correctness.py --out base.json
REGIME_ATTN=1 python experiments/c3_correctness.py --out regime.json
python experiments/c3_correctness.py --compare base.json regime.json

# GEMM sweep
python microbench/gemm_sweep.py --out results/microbench/gemm_sweep.parquet
```
