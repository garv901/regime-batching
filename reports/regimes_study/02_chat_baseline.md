# Stage 2. Chat baseline: do the costly regimes occur in ordinary serving?

Status: running. vLLM 0.30.0, FA3, one H100-80GB, Llama-3.1-8B-Instruct.

## Question

In chat traffic (not agents), which kinds of step does vLLM form, how much GPU time does each take, and do the
two lossy kinds from Stage 1b occur? Those are X steps (a prefill next to a long decode in a step over 512 tokens)
and A+L steps (a long-prefix append next to a long decode). Which of H1-H5 survive the kill criteria, and which
fixes pay?

Two chat traces bracket the question:

| Trace | What it is | Requests | Context | Prefix reuse |
|---|---|---|---|---|
| Azure 2024 conversation | single-turn chat (Microsoft production, 1 week) | 3,010 (first 2 min of the trace, time-scaled) | input p50 913, never >= 16k (max input + output 9,122 in the first 2M rows) | none in the trace |
| Mooncake conversation | multi-turn chat (Kimi production); history resent each turn | 307 (first 106 s, time-scaled) | input p50 8,770; 23-28% of requests reach >= 16k | 512-token KV blocks shared across turns; 14% of input tokens hit in this window with an unlimited cache (37% over the whole trace) |

## Prediction (written before the run)

Load: by the Stage 1b cost of an 8,192-token step (211 ms fresh, +7.5 ms per 1k cached tokens), prefill alone fills
56% of the window for Azure at 12.6 req/s and 48% for Mooncake at 0.86 req/s ("high" load). "Low" load is half
the rate, same requests.

Azure (no context reaches 16k, so X, A and A+L cannot occur):
1. Prefill-bearing steps (P, M) take >= 70% of GPU time at high load and >= 50% at low load.
2. H1 killed (no X steps). H2 killed: launch-bound steps above 512 tokens < 2% of GPU time (Stage 1b: eager
   excess a median 0.3 ms at this node's load).
3. Capture 2048 and step shaping change mean TTFT and mean TPOT by < 5%.
4. H3 not killed: prefill-bearing steps hold >= 20% of decode-weighted wall time. A 512-token step budget
   (`mnbt512`) improves ITL p99 by >= 1.5x and changes mean TTFT by <= 10%, because a 512-token step processes
   prefill at about the same tokens per second as an 8,192-token step (Stage 1b: 13.1 ms vs 211 ms at prefix 0).
5. H5 killed or near it: trace-weighted FA3 prefill efficiency (tile model) >= 85%.

Mooncake:
1. Long-prefix steps (A, A+L) take >= 40% of GPU time at high load.
2. X steps with a measured or modelled loss (X_cost) exist but are short-prefix chunks next to a long decode:
   2-15% of GPU time. H1 not killed.
3. C3 (split bound in eager X steps) improves mean TPOT more than capture 2048 does, because most prefill steps
   here are 8,192-token chunks and capture 2048 cannot reach them. Capture 2048 changes mean TPOT by < 3%.
4. H2 killed (as for Azure).
5. H3 not killed: prefill-bearing steps hold >= 50% of decode-weighted wall time. `mnbt512` improves ITL p99 by
   >= 2x (an 8,192-token chunk on a 32k-120k prefix takes 0.45-1.1 s, during which every decode waits) and changes
   mean TTFT by <= 15%.
6. H5: trace-weighted FA3 prefill efficiency 80-90%, below the kill threshold of 90%. The 8,192-token chunks fill
   whole waves; the loss sits in each prompt's last chunk.
7. Served prefix-hit share within 3 points of the unlimited-cache estimate (14%), since the window's working set
   (4.4M tokens over 307 requests, reuse within a few turns) mostly fits in 452k KV tokens.

## Method

| Step | Tool | Output |
|---|---|---|
| Workloads: `make_workload.py replay` (timestamps scaled to the target rate; Mooncake keeps its hash ids, replayed with 512-token blocks) | `data/workloads/chat_{az7,az14,mc05,mc10}.jsonl` | |
| Azure: default at 6.3 and 12.6 req/s; at 12.6: capture 2048, `mnbt512`, step shaping (RegimeScheduler `cg_cap`); default repeated last | `experiments/s9_chat_azure.sbatch` | `results/runs/s9_chat_az_<job>/` |
| Mooncake: default at 0.43 and 0.86 req/s; at 0.86: capture 2048, C3 (`REGIME_ATTN=1`, tile-model policy), `mnbt512`; default repeated last | `experiments/s9_chat_mooncake.sbatch` | `results/runs/s9_chat_mc_<job>/` |
| Per-step labels (tile model + A+L), occupancy, H1-H5 metrics, matched per-request TTFT/TPOT | analysis script (to be written before the data is read) | |
| ServingStudio: one dense simulation of the Azure workload | `servingstudio-simulate` | |

Kill and support criteria: unchanged from the plan (README).

## Results

Pending.
