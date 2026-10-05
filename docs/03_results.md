# Results

All latencies are in milliseconds unless the column says seconds. "x" columns are speedups against the vLLM
default (B0) run of the same Slurm job; higher is better. Short = output below 1,024 tokens; long = output of 1,024
tokens or more. Source: `bench.json` of each case, summarized by `analysis/bench_compare.py` and
`analysis/repeats.py`.

## 1. Memory-pressure burst: oracle KV admission

Workload `lburst80_8k`: 16 short requests per second for 300 s, plus 80 long requests (8,000 output tokens) at
t = 30 s.

### 1.1 Qwen3-8B (jobs 1033, 1039, 1043)

| Configuration | Short TTFT mean | Short TTFT p99 | Short E2E mean | Short E2E p99 | Short TPOT | Long TPOT | Long E2E (s) | Output tok/s | Preemptions |
|---|---|---|---|---|---|---|---|---|---|
| B0 (vLLM default) | 26,691.5 | 99,228.7 | 29,657.9 | 105,107.9 | 76.8 | 29.3 | 235.0 | 2,487 | 52 |
| mns32 | 147,888.0 | 217,827.7 | 148,485.8 | 218,404.1 | 15.4 | 14.1 | 200.5 | 2,061 | 0 |
| mns64 | 78,216.6 | 160,318.8 | 79,194.9 | 161,364.9 | 25.3 | 20.9 | 199.9 | 2,362 | 18 |
| mns128 | 28,808.6 | 99,216.2 | 30,486.5 | 101,542.0 | 43.5 | 28.8 | 230.7 | 2,518 | 56 |
| mns256 (best static) | 26,855.4 | 99,367.5 | 29,565.8 | 104,829.0 | 70.0 | 29.0 | 232.3 | 2,507 | 53 |
| sjf (oracle order, no memory plan) | 17,128.6 | 118,361.0 | 20,799.3 | 129,858.1 | 86.8 | 30.0 | 240.2 | 2,495 | 306 |
| **peak (oracle KV plan)** | **197.0** | **3,603.4** | **1,029.5** | **5,999.7** | **21.7** | **20.5** | 240.1 | 2,351 | **0** |
| full (peak + C3 + shape) | 225.7 | 4,026.1 | 1,022.1 | 5,957.7 | 20.6 | 20.1 | 234.7 | 2,389 | 0 |

Preemptions are counted from the step telemetry: a request that starts a step below the context length it had
already reached was preempted. For RegimeScheduler runs, vLLM's own preemption list agrees (sjf 307, peak and
full 0). Tokens lost to preemption and computed again: B0 219k, mns64 123k, mns128 221k, mns256 222k, sjf 339k.

Prefix caching was on in all runs (the vLLM default). A preempted request can find its own blocks in the prefix
cache when it is admitted again. The tokens that it skips at the restart show how much the cache returned:

| Run | Preemptions | KV tokens held when preempted | Returned by the prefix cache | Computed again |
|---|---|---|---|---|
| B0 (job 1033) | 52 | 224,842 | 6,000 (3%) | 218,842 |
| B0 repeat (job 1043) | 49 | 223,913 | 5,328 (2%) | 218,585 |
| mns128 | 56 | 232,472 | 11,392 (5%) | 221,080 |
| mns256 | 53 | 230,779 | 9,232 (4%) | 221,547 |
| sjf | 306 | 403,478 | 64,336 (16%) | 339,142 |
| Llama-3.1-8B B0 | 44 | 196,991 | 6,368 (3%) | 190,623 |
| Qwen3-4B B0 | 47 | 201,145 | 13,776 (7%) | 187,369 |

The cache returns little because vLLM puts the freed blocks at the end of the free queue
(`vllm/v1/core/block_pool.py:770-798`) and takes new blocks from the front (`block_pool.py:678`). Preemption occurs
only when the cache is full. The free queue is then almost empty, so the long requests that continue to grow take
the freed blocks at once.

sjf ran in job 1043. Against the B0 of that job (29,111.2 ms), sjf gives 1.40x short E2E and 1.53x short TTFT. Its
short TPOT and p99 values are worse than B0, because it preempts six times as often.

Repeat (job 1043): B0 short E2E 29,111.2 ms, peak 986.7 ms (29.5x); short TTFT 26,259.9 ms against 177.5 ms
(148x); output 2,512 against 2,377 tok/s; preemptions 49 against 0.

Difference, peak minus B0 (job 1033):

| Metric | B0 | peak | Difference | Speedup |
|---|---|---|---|---|
| Short TTFT mean | 26,691.5 | 197.0 | -26,494.5 | 135x |
| Short TTFT p50 | 2,193.4 | 84.7 | -2,108.7 | 25.9x |
| Short E2E mean | 29,657.9 | 1,029.5 | -28,628.4 | 28.8x |
| Short E2E p50 | 3,586.5 | 829.0 | -2,757.5 | 4.3x |
| Short E2E p99 | 105,107.9 | 5,999.7 | -99,108.2 | 17.5x |
| Short TPOT mean | 76.77 | 21.65 | -55.12 | 3.55x |
| Long TPOT mean | 29.3 | 20.5 | -8.8 | 1.43x |
| Long TTFT mean | 434.4 | 76,015.0 | +75,580.6 | slower: 34 long requests held, 178.5 s on average |
| Long E2E mean | 235,032.4 | 240,107.5 | +5,075.1 | 2% slower |
| Output throughput | 2,487 tok/s | 2,351 tok/s | -136 tok/s | 5.5% lower |

Short E2E mean by arrival time (job 1033):

| Arrival window | B0 | peak | Difference |
|---|---|---|---|
| 0-30 s (before the burst) | 409.9 | 446.3 | +36.4 |
| 30-130 s (KV cache fills) | 1,073.9 | 851.7 | -222.2 |
| 130-230 s (KV cache full) | 71,745.9 | 1,528.2 | -70,217.7 |
| 230-300 s (catch-up) | 21,029.3 | 798.2 | -20,231.1 |

Step-level evidence from telemetry (job 1033):

| Metric | B0 | peak |
|---|---|---|
| Steps longer than 100 ms | 319 | 20 |
| Prefill tokens computed | 3,685,352 | 3,476,488 (209k fewer: no recompute) |
| Prefill-heavy step time, mean / p99 | 57.1 / 220.9 ms | 34.1 / 79.0 ms |
| Decode-only step time, mean | 16.81 ms | 14.81 ms |
| Scheduler calls that held at least one request | - | 7,992 of 19,059 |

### 1.2 Other models (single runs, jobs 1040 and 1041)

| Model | Configuration | Short TTFT mean | Short E2E mean | Long TPOT | Long E2E (s) | Output tok/s | Preemptions | Short E2E x | Short E2E p99 x |
|---|---|---|---|---|---|---|---|---|---|
| Llama-3.1-8B | B0 | 16,814.3 | 19,809.5 | 26.87 | 215.3 | 2,732 | 44 | 1.00 | 1.00 |
| Llama-3.1-8B | peak | 183.1 | 988.7 | 19.85 | 221.4 | 2,529 | 0 | **20.0** | 14.7 |
| Qwen3-4B | B0 | 13,801.9 | 15,657.5 | 24.15 | 193.3 | 2,784 | 47 | 1.00 | 1.00 |
| Qwen3-4B | peak | 141.8 | 807.9 | 17.57 | 194.5 | 2,786 | 0 | **19.4** | 19.3 |

## 2. X-heavy long-context workload: step shaping and C3

Workload `xheavy_lc24k`: 4 decodes with 24k-token contexts, plus 24 new requests per second with 450-1,000-token
prompts. The name means "many X steps". An X step has more than 512 tokens, holds a prefill chunk of 417 tokens or
more and also holds a long-context decode. In such a step, FA3 does not split the long decode, so one thread block
per KV head reads 24k tokens alone. In B0, X steps use 56% of GPU time on this workload.

### 2.1 Qwen3-8B, 3 repeats (jobs 1015/1017, 1034)

Absolute values are means over the runs. Speedups are mean +- standard deviation of the per-run ratio against the
B0 run of the same job.

| Configuration | Runs | Short TTFT | Short E2E mean | Short E2E p99 | Short TPOT | Long TPOT | Short E2E x | Short E2E p99 x | Long TPOT x | TTFT x |
|---|---|---|---|---|---|---|---|---|---|---|
| B0 | 4 | 185.5 | 1,056.0 | 5,231.7 | 22.6 | 22.0 | 1.00 | 1.00 | 1.00 | 1.00 |
| cap2048 | 1 | 171.6 | 900.0 | 5,078.9 | 18.9 | 19.0 | 1.14 | 0.99 | 1.15 | 1.03 |
| mnbt512 | 3 | 389.2 | 977.1 | 4,630.7 | 15.0 | 16.9 | 1.08 +- 0.01 | 1.13 +- 0.04 | 1.31 +- 0.01 | **0.47** |
| C3 | 3 | 169.9 | 902.4 | 5,169.7 | 19.0 | 19.2 | 1.18 +- 0.03 | 1.03 +- 0.01 | 1.15 +- 0.02 | 1.11 |
| **shape** | 3 | 193.0 | **839.0** | 4,637.4 | 16.7 | 17.5 | **1.26 +- 0.03** | 1.13 +- 0.02 | 1.26 +- 0.02 | 0.96 |
| full | 3 | 191.9 | 831.9 | 4,711.1 | 16.5 | 17.4 | 1.27 +- 0.03 | 1.11 +- 0.04 | 1.26 +- 0.01 | 0.96 |
| shape, reactive (no oracle) | 1 | 194.6 | 828.8 | 4,734.1 | 16.4 | 17.3 | 1.27 | 1.11 | 1.27 | 0.95 |

The reactive run (job 1041) has no B0 in the same job. Its speedups use the mean of the four B0 runs.

Difference against B0 (means): shape cuts short E2E by 217 ms and long TPOT by 4.5 ms per token (about 18 s over
a 4,000-token output). mnbt512 adds 204 ms to short TTFT.

GPU time (job 1017, telemetry): in B0, X steps use 56% of GPU time at 36 ms per X step, and 61.6% of GPU time runs
as "eager, 1 split". With shape, X steps fall to 7% and eager steps to 20.4% (mostly the long prompts' own
prefill, which shaping does not limit). In a 3-second window, B0 completes 132 steps and shape completes 177.

### 2.2 Other models (single runs, jobs 1040 and 1041)

| Model | Configuration | Short TTFT | Short E2E mean | Long TPOT | Short E2E x | Short E2E p99 x | Long TPOT x | TTFT x |
|---|---|---|---|---|---|---|---|---|
| Llama-3.1-8B | B0 | 163.3 | 913.2 | 19.92 | 1.00 | 1.00 | 1.00 | 1.00 |
| Llama-3.1-8B | cap2048 | 147.3 | 786.4 | 17.42 | 1.16 | 1.03 | 1.14 | 1.11 |
| Llama-3.1-8B | mnbt512 | 283.8 | 805.5 | 15.30 | 1.13 | 1.11 | 1.30 | 0.58 |
| Llama-3.1-8B | **shape** | 163.9 | **732.5** | 15.91 | **1.25** | 1.17 | 1.25 | 1.00 |
| Llama-3.1-8B | full | 160.9 | 730.7 | 15.82 | 1.25 | 1.14 | 1.26 | 1.01 |
| Qwen3-4B | B0 | 85.3 | 519.1 | 14.20 | 1.00 | 1.00 | 1.00 | 1.00 |
| Qwen3-4B | **cap2048** | 76.3 | **443.4** | 12.45 | **1.17** | 0.98 | 1.14 | 1.12 |
| Qwen3-4B | mnbt512 | 153.9 | 503.3 | 11.80 | 1.03 | 0.74 | 1.20 | 0.55 |
| Qwen3-4B | shape | 88.1 | 457.2 | 12.05 | 1.14 | 1.07 | 1.18 | 0.97 |
| Qwen3-4B | full | 87.6 | 452.2 | 11.93 | 1.15 | 1.06 | 1.19 | 0.97 |

On Qwen3-4B, cap2048 is better than shape on mean short E2E. Shape is still better on p99 and long TPOT.

## 3. Kernel-level measurements

### 3.1 Attention microbenchmarks (`results/microbench/ATTN_SWEEP.md`)

Best kernel for each batch shape against the vLLM default FA3 path:

| Batch type | Geomean gain | Largest gain |
|---|---|---|
| Decode, uniform lengths | 1.04x | 1.30x (only at very small batch and short context) |
| Decode, skewed lengths | 1.03x | 1.21x (FlashInfer native, 15 x 2k + 1 x 4.5k) |
| Steady mixed | 1.03x | |
| Single request | 1.01x | |
| **Mixed step above 512 tokens (X)** | **1.40x** | **4.81x** (512-token chunk + 15 short decodes + one 32k decode: 407 us against 85 us per layer) |

FlashInfer XQA, the default FlashInfer decode kernel on Hopper, is 1.66x slower than FA3 on skewed batches
(geomean), and up to 4.5x slower.

### 3.2 GEMM sweep (`results/microbench/gemm_sweep.parquet`)

Qwen3-8B linear layers with cuBLAS: total GEMM time is flat up to M = 128 tokens (1.06x of M = 1), then 1.28x at
256, 2.30x at 512 and 4.55x at 1,024. Triton is faster than cuBLAS only at M of 64 or fewer (o_proj up to 1.8x,
partly a warm-L2 effect), which is at most about 0.35 ms of a 6.6 ms decode step.

### 3.3 In-engine calibration (Qwen3-8B)

| Quantity | Value |
|---|---|
| Decode step floor (weight reads) | 6.6 ms (2.29 TB/s effective) |
| KV read cost in decode | 0.35 ms per GB (2.85 TB/s effective) |
| 512-token prefill step | 15 ms |
| One FA3 thread block streaming KV | 58 GB/s |
| X-step cost per 1,000 tokens of the longest decode context | 0.35 ms measured, 0.318 ms predicted |
| CPU launch time of an eager step on this node | about 16.5 ms (about equal to its GPU time) |

## 4. Negative controls

| Workload | Result |
|---|---|
| User waves (31 x 30 tokens + 1 x 4,000 tokens every 2 s), Qwen3-8B | GPU time: Ds 50%, Dl 26%, Dw 12%, P 12%, X 0.6%. C3 changes nothing. Estimated headroom below 3%. |
| xstress 16k (8 req/s, 512-token prompts, 1/64 long) | X is 5% of GPU time. C3: +0.6% throughput, E2E within 1.2%. |
| X-heavy with peak admission (no memory pressure) | full 1.27x against shape 1.26x: the KV plan has no effect when demand fits in the cache. |

## 5. Correctness

`experiments/c3_correctness.py`: 101 greedy requests, each step an X step (one long-context decode plus a new
prompt of 512 tokens or more).

| Comparison | Identical greedy outputs | Max logprob difference |
|---|---|---|
| Stock vs stock, two jobs, same code and config | 97 / 101 | 0.161 |
| C3 vs stock, same job (1034) | 101 / 101 | 0.067 |
| C3 vs stock, same job (1015) | 96 / 101 | 0.15 |

C3 stays inside the run-to-run noise of stock vLLM. A KV split changes the order of floating-point additions, so
results are not bit-for-bit equal. The scheduler (C4) does not change any computation.

## 6. Overheads

| Item | Cost |
|---|---|
| C3 decision per step | about 23 us (CPU, no GPU sync) |
| C4 `schedule()` wrapper | 76 us median per call (measured before the planner was vectorized) |
| KV planner construction | 29-47 us for 16-256 running requests (after vectorization) |
