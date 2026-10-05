# Setup, models, workloads and baselines

## Hardware and software

| Item | Value |
|---|---|
| GPU | 1 x NVIDIA H100-80GB SXM (132 SMs) |
| Host | Shared Slurm node. Load average approximately 650 during the runs (see the limitations page). |
| vLLM | 0.30.0, V1 engine, async scheduling, FlashAttention 3 backend, `FULL_AND_PIECEWISE` CUDA graphs, capture sizes up to 512, `max_num_batched_tokens` 8,192 |
| PyTorch | 2.13 (CUDA 13.0) |
| FlashInfer | flashinfer-python 0.6.18.post1 + flashinfer-cubin (microbenchmarks only) |
| Server flags | `vllm serve <model> --max-model-len 40960 --seed 0` plus the flags of each configuration |
| Client | `vllm bench serve --dataset-name timed_trace` (replays the arrival times and lengths of a trace) |

## Models

| Model | Layers | Query / KV heads | Head dim | Role | Why |
|---|---|---|---|---|---|
| Qwen/Qwen3-8B | 36 | 32 / 8 | 128 | Primary | Decode reads 15.1 GB of weights per step (6.6 ms floor). A 16-32k context gives a large straggler cost in X steps. |
| meta-llama/Llama-3.1-8B-Instruct | 32 | 32 / 8 | 128 | Replication | Different model family with the same attention geometry |
| Qwen/Qwen3-4B | 36 | 32 / 8 | 128 | Amplifier | Same attention as Qwen3-8B with about half the weight bytes, so attention is a larger part of each step |

We selected the models with a calibrated cost model (`analysis/pen_and_paper.py`, summary in
[../reports/00_pen_and_paper.md](../reports/00_pen_and_paper.md) section 6).

## Workloads

All workloads are synthetic traces in `data/workloads/`. Each line holds an arrival time, an input length and an
output length. Prompts are random tokens with no shared prefixes. The generator is `experiments/make_workload.py`.

| Workload | File | Requests | Arrivals | Short requests | Long requests | What it tests |
|---|---|---|---|---|---|---|
| X-heavy | `xheavy_lc24k.jsonl` | 2,400 over approximately 99 s | 24 per second | prompt 450-1,000 tokens, output 16-64 | 4 at t = 0: prompt 24,000, output 4,000 | Long-context decodes in steps with prefill chunks (X regime) |
| X-heavy 16k | `xheavy_lc16k.jsonl` | same | same | same | 4 at t = 0: prompt 16,000, output 4,000 | Same, smaller straggler |
| Memory-pressure burst | `lburst80_8k.jsonl` | 4,800 over approximately 297 s | 16 per second | prompt 450-1,000 tokens, output 16-64 | 80 at t = 30 s: prompt 512, output 8,000 | KV demand above the cache. The 80 long requests need 681k slots at the end; Qwen3-8B has 451k. |
| User waves (negative control) | waves family | 32 requests every 2 s | waves | 31 with outputs of about 30 tokens | 1 with output 4,000 (or 16,000) | The motivating example |

"Long" in all result tables means an output of 1,024 tokens or more.

## Baselines

| Name | Configuration | Why we include it |
|---|---|---|
| B0 | vLLM default | Reference for all speedups |
| cap2048 | `-cc '{"max_cudagraph_capture_size":2048}'` | Moves the 512-token boundary to 2,048 for CUDA graphs and KV splits |
| mnbt512 | `--max-num-batched-tokens 512` | Keeps every step at or below 512 tokens |
| mns32 ... mns256 | `--max-num-seqs 32 / 64 / 128 / 256` | Limits concurrency, the static control against KV overcommit |
| FlashInfer, Triton | other attention backends | Kernel alternatives (microbenchmarks and smoke runs) |

## Our configurations

| Name | Server environment and flags |
|---|---|
| C3 | `REGIME_ATTN=1` |
| shape | `--scheduler-cls regime_batching.scheduler.RegimeScheduler` with `REGIME_SCHED_ADMIT=fcfs REGIME_SCHED_SHAPE=cg_cap` |
| shape (reactive) | as `shape`, plus `REGIME_SHAPE_MIN_REMAIN=0` (no oracle input) |
| peak | `--scheduler-cls ...RegimeScheduler` with `REGIME_SCHED_ADMIT=peak REGIME_SCHED_SHAPE=off` |
| sjf | `--scheduler-cls ...RegimeScheduler` with `REGIME_SCHED_ADMIT=sjf REGIME_SCHED_SHAPE=off` |
| full | `REGIME_ATTN=1` + `--scheduler-cls ...RegimeScheduler` with `REGIME_SCHED_ADMIT=peak REGIME_SCHED_SHAPE=cg_cap` |

## Method rules

- Every comparison pairs runs from the same Slurm job on the same GPU. Speedups use the default run of the same job.
- Qwen3-8B X-heavy results have 3 runs (mean +- standard deviation). The burst has 2 runs for B0 and peak. Other
  results are single runs.
- Telemetry records every step in baseline and proposed runs alike.
