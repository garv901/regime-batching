# Onboarding: what this study does and how it works

Read this page first. It defines the terms, shows how the parts fit together, and tells you how to read the figures.

## 1. The question

Continuous batching puts very different requests into one forward pass. One example: 31 requests that generate
about 30 tokens run together with one request that generates 4,000 tokens. vLLM uses one scheduler policy and one
kernel configuration for all of them.

We ask: if the scheduler knows each output length in advance (the "oracle"), can it batch and dispatch better?

We test three uses of the oracle:
1. **Admission:** plan KV memory over the full life of each request.
2. **Batch composition:** keep each step in a fast regime.
3. **Kernel selection:** use the attention configuration that suits each step.

## 2. Terms

| Term | Meaning |
|---|---|
| Oracle | The exact output length of each request, known at arrival. We set `max_tokens` to the true length and `ignore_eos` to true. |
| Step | One forward pass of the model over the current batch. |
| T | The number of new tokens in a step: 1 per decode request plus the prefill-chunk tokens. |
| KV cache | GPU memory for attention keys and values. One context token uses one slot. Qwen3-8B on one H100 has 451,456 slots. |
| Admission | The scheduler decision to start a waiting request. |
| Preemption | vLLM removes a running request from the KV cache. Later it computes the request's tokens again. |
| CUDA graph | A captured sequence of GPU kernels. Replay avoids the CPU cost of each kernel launch. vLLM captures graphs for steps of up to 512 tokens by default. |
| KV split | FlashAttention cuts one long context into parts, and different SMs read the parts at the same time. A small combine step merges the results. Without a split, one thread block reads the whole context. |
| TTFT, TPOT, E2E | Time to first token; time per output token after the first; end-to-end request latency. |

## 3. Regime taxonomy

`regime_batching/features.py` labels every step with one regime. The baseline runs and our runs use the same
labels, so you can compare them step by step.

| Regime | Definition | What limits the step |
|---|---|---|
| P | Prefill-heavy mixed step | GEMM compute |
| M | Mixed prefill and decode | GEMM and attention |
| **X** | Mixed step in which FA3 does not split a long-context decode | One thread block per KV head reads the long context (straggler) |
| Dw | Decode, small total KV | Weight reads |
| Dk | Decode, large total KV | KV reads |
| Ds | Decode, a few long contexts dominate the KV | Skew between requests |
| Dl | Decode, few requests, long context | Low GPU occupancy |
| Dc | Decode, very large batch | Near the GEMM compute ridge |

The X regime is the important one. It occurs when a step has more than 512 tokens, holds a prefill chunk of 417
tokens or more (Qwen3-8B geometry), and holds a long-context decode. The diagrams in
[../reports/diagrams/flaw2_512_step_boundary/](../reports/diagrams/flaw2_512_step_boundary/) explain why.

## 4. Components

The prototype is a vLLM plugin. It does not modify vLLM.

| Component | File | What it does | How to turn it on |
|---|---|---|---|
| C1 Telemetry | `regime_batching/telemetry.py` | Writes one record per step: features, regime, CUDA-graph mode, attention split, GPU time. Adds NVTX ranges. | `REGIME_TELEMETRY_DIR=<dir>` |
| C2 Cost model | `regime_batching/features.py` | Classifies regimes. Replicates the vLLM/FA3 split decision. Predicts the straggler cost. | used by C1 and C3 |
| C3 Attention split | `regime_batching/attn/backend.py` | On a predicted X step, raises the FA3 split limit. Every other step is unchanged. | `REGIME_ATTN=1` |
| C4 Scheduler | `regime_batching/scheduler.py`, `regime_batching/oracle.py` | Oracle KV admission, step shaping, SJF ablation | `--scheduler-cls regime_batching.scheduler.RegimeScheduler` |

The plugin loads through the `vllm.general_plugins` entry point in `pyproject.toml`.

### C4 scheduler modes

| Env variable | Values | Effect |
|---|---|---|
| `REGIME_SCHED_ADMIT` | `peak` (default) | Admit a waiting request only if the future KV peak of all admitted requests, plus this one, fits in the cache. Then try the next waiting request (backfill). |
| | `fcfs` | Stock vLLM admission. |
| | `sjf` | Ablation: order the waiting queue by oracle remaining tokens. No memory plan. |
| `REGIME_SCHED_SHAPE` | `off` (default), `cg_cap` | `cg_cap` limits a step to the CUDA-graph capture size while a decode with 4,096 or more context tokens and 256 or more remaining tokens runs. It lifts the limit if more than 16,384 prefill tokens wait. |
| `REGIME_SCHED_WATERMARK` | `0.02` | Fraction of KV slots kept free in the plan. |
| `REGIME_SHAPE_MIN_KV`, `REGIME_SHAPE_MIN_REMAIN`, `REGIME_SHAPE_MAX_BACKLOG`, `REGIME_SHAPE_CAP` | `4096`, `256`, `16384`, capture size | Step-shaping thresholds. `REGIME_SHAPE_MIN_REMAIN=0` removes the oracle from step shaping. |

### How the KV planner works

Each admitted request holds `u` KV slots now and will grow by one slot per step for `r` more steps (the oracle
gives `r`). When it finishes, it frees all its slots. `oracle.PeakPlanner` calculates the highest total of
`u + k` over all future steps `k`, for the admitted set plus one candidate. If that peak is more than the cache
minus the watermark, the scheduler holds the candidate in the queue. Short requests have a small peak, so they
usually fit and start before a held long request. A planner call takes 29-47 us on the CPU for 16-256 running
requests.

## 5. How to read the figures

`python analysis/telemetry.py run <results/runs/job_dir>` writes figures to `reports/figures/<job>/`:

| File | What it shows |
|---|---|
| `compare/compare_ribbons.png` | One row per configuration. The top track gives the regime of each step. The `exec` track gives what ran: full or piecewise CUDA graph, or eager, and the attention split. The `sched` track gives step-cap and admission-hold periods. |
| `compare/compare_zoom.png` | A 3-second window with the most X steps in the baseline. One rectangle per step, so you can compare step by step. |
| `compare/compare_regime_bars.png` | GPU time per regime for each configuration. |
| `compare/compare_latency_cdfs.png` | TTFT, TPOT and E2E distributions. |
| `compare/compare_table.md` | The same data as tables, including GPU-time share for each execution type. |
| `<case>/trace.json.gz` | A Perfetto trace with regime, execution type, batch size, total KV and step time per step. Open it at ui.perfetto.dev. |

Good first figures:
- `reports/figures/s3a_1017/compare/compare_zoom.png`: the X regime and step shaping.
- `reports/figures/s4b_1033/compare/compare_ribbons.png`: the memory-pressure burst. The default stalls for
  about 90 seconds; the oracle planner does not.
