# Regime-aware batching in vLLM with oracle output lengths

This repository holds a measurement study and a prototype. It asks one question:

> If the vLLM scheduler knows the exact output length of each request in advance, what can it do better?

We give vLLM this "oracle" knowledge through `max_tokens` with `ignore_eos`. Each request then generates exactly
`max_tokens` tokens, so the engine knows each output length when the request arrives. How to predict lengths in
production is a separate problem. This study measures the upper bound: the gain when the length is known.

All runs use vLLM 0.30.0 on one NVIDIA H100-80GB, with Qwen3-8B, Llama-3.1-8B-Instruct and Qwen3-4B.

## The two main workloads

### X steps and the X-heavy workload

Our telemetry gives each engine step a *regime* label (see [docs/01_onboarding.md](docs/01_onboarding.md)). The
regime **X** costs the most time. A step is an X step when all three conditions are true:

1. **The step has more than 512 tokens.** 512 is the largest CUDA-graph capture size, so the step runs without a
   CUDA graph. The CPU launches every kernel (about 16.5 ms per step on our node).
2. **The step holds a prefill chunk of 417 tokens or more.** Above 512 tokens, vLLM lets the FlashAttention-3
   heuristic select the number of KV splits. The heuristic counts query tiles: 8 KV heads x ceil(chunk / 32).
   A 416-token chunk gives 8 x 13 = 104 tiles, below 0.8 x 132 SMs = 105.6, so the heuristic splits. A 417-token
   chunk gives 8 x 14 = 112 tiles. The heuristic then decides that the GPU is full and selects 1 split for every
   sequence in the step. The limit is 417 tokens for all three models, because all three have 8 KV heads and 4
   query heads per KV head.
3. **The step also holds a decode with a long context**, for example 24,000 tokens.

In an X step, one thread block per KV head reads the long context alone: 8 of the 132 SMs work, and the other
SMs wait for them. One attention layer takes 407 us in place of 85 us (4.8x), and the step takes about 36 ms.
[reports/diagrams/flaw2_512_step_boundary/](reports/diagrams/flaw2_512_step_boundary/) shows this in four
diagrams.

The **X-heavy** workload (`data/workloads/xheavy_lc24k.jsonl`) makes many X steps on purpose:

| Part | Requests | Prompt | Output |
|---|---|---|---|
| Long-context requests | 4, all at t = 0 | 24,000 tokens | 4,000 tokens |
| Short requests | 24 per second for about 99 s (2,400 in total) | 450-1,000 tokens | 16-64 tokens |

The four long requests decode during the full run. vLLM puts each new short prompt into a step together with
these decodes, so most of these steps become X steps. With the vLLM default, X steps use 56% of the GPU time.
`xheavy_lc16k.jsonl` is the same workload with 16,000-token contexts.

The workload is synthetic, and it isolates one mechanism. Real traffic has the same pattern when long chats or
long documents decode while new prompts arrive, but X steps are then less frequent. For example, in our
`xstress` workload (1 long request in 64), X steps use only 5% of the GPU time.

### Memory-pressure burst

`lburst80_8k.jsonl`: 16 short requests per second for 300 s. At t = 30 s, 80 long requests arrive, each with
8,000 output tokens. At the end of their outputs, the long requests need 681,000 KV slots. Qwen3-8B has 451,000.

## Results in one table

Speedups compare against the vLLM default configuration in the same job. "Best static" is the best vLLM flag
configuration that we found for the same workload.

| Idea | Workload | vLLM default | Best static vLLM | This work | Needs the oracle? |
|---|---|---|---|---|---|
| **Admission with a plan of future KV memory** | Memory-pressure burst, Qwen3-8B | short-request E2E 29,658 ms | 29,566 ms (`--max-num-seqs 256`) | **1,029 ms (28.8x)** | **Yes** |
| | same, Llama-3.1-8B | 19,809 ms | not measured | **989 ms (20.0x)** | Yes |
| | same, Qwen3-4B | 15,657 ms | not measured | **808 ms (19.4x)** | Yes |
| **Step shaping at the 512-token boundary** | Long-context X-heavy, Qwen3-8B (3 repeats) | short-request E2E 1,056 ms | 900 ms, 1.14x (capture size 2048) | **839 ms (1.26x)** | No |
| | same, Llama-3.1-8B | 913 ms | 786 ms, 1.16x | **732 ms (1.25x)** | No |
| | same, Qwen3-4B | 519 ms | **443 ms, 1.17x** | 457 ms (1.14x) | No |
| Per-step attention split override (C3) | X-heavy, Qwen3-8B (3 repeats) | 1,056 ms | 900 ms | 902 ms (1.18x) | No |
| Switch decode kernels (FA3, FlashInfer, Triton) or GEMM kernels | microbenchmarks | - | - | 5% or less of a step | No |

E2E = mean end-to-end latency of the short requests. Full tables in milliseconds: [docs/03_results.md](docs/03_results.md).

## Three conclusions

1. **The oracle pays off in KV admission.** vLLM admits a request if its prompt fits now. It does not plan for
   output growth, and it corrects the overcommit later with preemption. A planner that knows each request's
   future KV peak removed all preemptions and cut short-request latency 20-30x on three models. No static flag
   and no length-based request order came close.
2. **vLLM has a 512-token step boundary.** Above the CUDA-graph capture size, a step loses its CUDA graph *and*
   the FlashAttention-3 KV split. A long-context decode in such a step becomes a straggler. A scheduler that keeps
   these steps at 512 tokens gains 1.25x on the 8B models. This fix does not need the oracle.
3. **Regime-specific kernels are a small lever on these models.** Outside the 512-token boundary, the vLLM default
   FA3 path is close to the hardware limit. Decode on an 8B model is limited by weight reads. An output-length
   oracle does not change this. The split decision needs only the current context length, which the engine
   already knows at each step. The oracle helps decisions that cannot be undone, such as admission
   ([why](docs/04_vllm_takeaways.md#36-an-output-length-oracle-does-not-improve-the-split-decision)).

Details: [docs/04_vllm_takeaways.md](docs/04_vllm_takeaways.md).

## Documentation

| Document | Content |
|---|---|
| [docs/01_onboarding.md](docs/01_onboarding.md) | Terms, the regime taxonomy, the components and how they attach to vLLM, how to read the figures |
| [docs/02_setup_models_workloads.md](docs/02_setup_models_workloads.md) | Hardware, software versions, models, workloads, baselines |
| [docs/03_results.md](docs/03_results.md) | All results in milliseconds, ablations, correctness |
| [docs/04_vllm_takeaways.md](docs/04_vllm_takeaways.md) | What the study shows about vLLM scheduling, KV management and kernel use |
| [docs/05_reproduce.md](docs/05_reproduce.md) | Install, run, and analyze |
| [docs/06_limitations_and_future_work.md](docs/06_limitations_and_future_work.md) | Limits of the evidence and next steps |
| [docs/07_mstar_gaps.md](docs/07_mstar_gaps.md) | Scheduling and kernel gaps in mstar, as hypotheses to test |
| [reports/00_pen_and_paper.md](reports/00_pen_and_paper.md) | Full analysis log: cost model, simulator, every measurement |
| [reports/related_work.md](reports/related_work.md) | Related work and how this study differs |
| [reports/diagrams/](reports/diagrams/) | Diagrams of the mechanisms |

## Repository layout

```
regime_batching/        vLLM plugin (no vLLM fork)
  plugin.py             entry point; env switches
  telemetry.py          per-step markers: regime, CUDA-graph mode, attention split, step time
  features.py           regime classifier and a replica of the vLLM/FA3 split decision
  scheduler.py          RegimeScheduler: oracle KV admission, step shaping, SJF ablation
  oracle.py             future-peak KV planner
  attn/backend.py       C3: per-step FA3 split override
analysis/               result tables, figures, Perfetto traces, calibrated simulator
experiments/            workload generator, Slurm job scripts, correctness test
microbench/             attention and GEMM sweeps
data/workloads/         the exact request traces used in every run
results/                benchmark outputs per job
reports/                analysis log, figures, diagrams, related work
tests/                  CPU tests for the planner and scheduler
```
