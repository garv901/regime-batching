# Stage 2. Chat baseline: do the costly regimes occur in ordinary serving?

Status: done 2026-10-07 (jobs 1459, 1460; 1458 stopped on a bug in its own smoke check). vLLM 0.30.0, FA3, one
H100-80GB, Llama-3.1-8B-Instruct.

## Question

In chat traffic (not agents), which kinds of step does vLLM form, how much GPU time does each take, and do the
two lossy kinds from Stage 1b occur? Those are X steps (a prefill next to a long decode in a step over 512 tokens)
and A+L steps (a long-prefix append next to a long decode). Which of H1-H5 survive the kill criteria, and which
fixes pay?

Two chat traces bracket the question:

| Trace | What it is | Requests | Context | Prefix reuse |
|---|---|---|---|---|
| Azure 2024 conversation | single-turn chat (Microsoft production, 1 week) | 3,010 (first 2 min of the trace, time-scaled) | input p50 913, never >= 16k (max input + output 9,122 in the first 2M rows) | none in the trace |
| Mooncake conversation | multi-turn chat (Kimi production); history resent each turn | 307 (first 106 s, time-scaled) | input p50 8,770; 23-28% of requests reach >= 16k | 512-token KV blocks shared across turns; 14% of input tokens hit in this window with an unlimited cache (37% over the whole trace). **Correction after the run:** the same rule on the same 307 requests gives 9.3%; the 14% was an arithmetic error. 37% is right |

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
| Per-step labels (tile model + A+L), occupancy, H1-H5 metrics, matched per-request TTFT/TPOT | `analysis/stage_regimes.py` (committed in 75aa71b before the data was read) | `figures/stage2_tables.md`, `results/regimes_study/stage2_summary.json` |
| ServingStudio: one dense simulation of the Azure workload (first 2,000 requests, the API limit). The Mooncake trace cannot be simulated: the API accepts a cached prefix only on a later round of a chained session | `servingstudio-simulate` | `results/regimes_study/servingstudio/stage2_*` |

Kill and support criteria: unchanged from the plan (README).

## Results

Every case completed all requests except one (`mc10_mnbt512`: 306 of 307; one client connection dropped mid-run,
the engine logged no error). Output lengths match the trace exactly in every case, so the comparisons are request
by request. Node load average during the runs: 86-174.

### Summary

- **The costly kinds of step from Stage 1b barely occur in chat.**
  - Azure never forms an X, A or A+L step (no context reaches 16k).
  - Mooncake forms X *shapes* in 42% of GPU time. They are X *by cost* in only 0.5%, because the prefill next to the
    long decode is an 8,192-token chunk on a long prefix, and the one-split decode finishes underneath it.
  - Long-prefix appends next to a long decode (A+L) take 22% of GPU time. Separate launches would recover only
    0.01% of GPU time: same reason.
  - Every kernel-side fix (split bound, capture 2048, separate launches, perfect packing) is worth at most 0.9% of
    GPU time in default serving.
- **The one large effect is the decode stall (H3).**
  - Steps that carry prefill hold 53% of decode-weighted time in both traces.
  - ITL p99 is 17.6x ITL p50 for Azure and 26x for Mooncake: 184 ms and 433 ms.
  - A 512-token step budget removes the stall (ITL p99 8.0x and 6.1x lower) but costs 54% and 120% mean TTFT.
- **Small steps create the loss that big steps hide.**
  - With a 512-token budget, Mooncake's A+L steps take 35% of GPU time.
  - Separate prefill and decode launches would recover 5.0% of GPU time there.
  - FA3 prefill efficiency drops from 96% to 74%.
- **Mooncake serves 3.6% of input tokens from the prefix cache, against 9.3% with an unlimited cache.**
  - The misses are evictions. The 16 requests that miss more than 4k expected tokens reuse blocks first seen 71-276 s
    earlier (at 0.86 req/s). By then 1.1M-3.5M new tokens have passed through a 452k-token cache.
  - The share is identical at both rates and in every config, because eviction depends on token volume, not time.

### Regimes observed (share of measured GPU time; share of steps in brackets)

Labels:
- **P:** prefill-heavy. **M:** mixed.
- **A:** an append on >= 16k cached tokens. **A+L:** A with a >= 16k decode in the same launch.
- **X:** a prefill next to a >= 16k decode, eager, FA3 at 1 split, costing >= 1.2x the best split (tile model).
- Decode-only steps: **Dl** low-occupancy long-context, **Ds** skewed context lengths, **Dk** KV-bound,
  **Dw** weight-bound.

| Case | X | A+L | A | P | Decode-only (Dl / Ds / Dk / Dw) |
|---|---|---|---|---|---|
| Azure 6.3 req/s | - | - | - | 24.6% (4.4%) | 54.9% / 19.5% / - / 0.9% |
| Azure 12.6 req/s | - | - | - | 51.0% (13.1%) | 5.8% / 40.6% / 0.5% / 2.2% |
| Azure 12.6, `mnbt512` | - | - | - | 70.3% (55.7%) | 10.9% / 18.5% / - / 0.2% |
| Mooncake 0.43 req/s | 0.1% (0.0%) | 15.7% (0.6%) | 4.7% (0.2%) | 20.3% (1.4%) | 57.0% / 1.2% / - / 1.1% |
| Mooncake 0.86 req/s | 0.5% (0.1%) | 21.8% (1.2%) | 3.8% (0.2%) | 24.5% (2.4%) | 34.0% / 14.7% / - / 0.7% |
| Mooncake 0.86, `mnbt512` | - | 35.0% (16.3%) | 4.6% (2.7%) | 37.2% (32.3%) | 21.5% / 1.2% / - / 0.5% |

M is 0.0-0.1% everywhere. Default and repeat agree within 0.7 points on every label. Full tables, all cases:
[`figures/stage2_tables.md`](figures/stage2_tables.md).

- **Decode-only steps:** they take half the GPU time at high load. In Stage 1b they sit at the weight or KV
  bandwidth floor, and their only lever is batch size.
- **Long decodes:** in Mooncake, 84% of GPU time is in steps with a decode of >= 16k context. Long decodes are almost
  always running, so the X and A+L shapes are common.
- **Prefill step sizes:** almost all prefill GPU time is in steps over 2,048 tokens, beyond the reach of capture 2048.
  Azure 41.2% of GPU time (8.4% at 513-2,048 tokens), Mooncake 50.4% (0.1%).

### H1-H5 metrics (share of measured GPU time unless noted)

| Metric | Azure 6.3 | Azure 12.6 | Mooncake 0.43 | Mooncake 0.86 | Mooncake 0.86 `mnbt512` |
|---|---|---|---|---|---|
| H1: X shape / X by cost / X_cost (also >= 1 ms excess) | 0 / 0 / 0 | 0 / 0 / 0 | 25.5% / 0.1% / 0.1% | 42.3% / 0.5% / 0.5% | 0 / 0 / 0 |
| H1: recoverable by best split bound / separate launches (modelled) | 0.14% / 0.24% | 0.05% / 0.05% | 0.01% / 0.07% | 0.06% / 0.30% | 0.08% / 5.85% |
| H2: eager steps / launch-bound steps above 512 tokens | 23.8% / 0.34% | 49.6% / 0.80% | 40.7% / 0.19% | 50.6% / 0.59% | 0 / 0 |
| H3: prefill-bearing share of decode-weighted time | 25.5% | 52.6% | 45.3% | 52.9% | 83.9% |
| H5: FA3 prefill efficiency (tile model) / recoverable by perfect packing | 76.7% / 0.63% | 87.4% / 0.78% | 97.4% / 0.41% | 95.8% / 0.86% | 74.2% / 10.61% |
| Attention in prefill-bearing steps (modelled) | 2.7% | 6.2% | 16.0% | 20.3% | 41.1% |

Mooncake `mnbt512`'s separate-launch recovery is almost all in A+L steps (4.98% of GPU time).

### Latency (ms; request-matched ratios against the default at the same rate; < 1 is faster)

| Case | TTFT mean / p99 | TPOT mean / p99 | ITL p50 / p99 | Mean TTFT | Mean TPOT | ITL p99 | Mean E2E |
|---|---|---|---|---|---|---|---|
| Azure 12.6 default | 156.9 / 474.3 | 21.37 / 97.70 | 10.49 / 184.23 | | | | |
| repeat | 156.9 / 475.4 | 21.37 / 100.96 | 10.54 / 180.78 | 1.000 | 1.000 | 0.981 | 1.000 |
| capture 2048 | 158.0 / 475.8 | 21.43 / 98.66 | 10.42 / 183.05 | 1.007 | 1.003 | 0.994 | 0.999 |
| step shaping (`cg_cap`) | 216.5 / 690.0 | 15.48 / 51.71 | 12.69 / 51.71 | 1.380 | 0.724 | 0.281 | 0.826 |
| `mnbt512` | 241.8 / 918.2 | 13.82 / 18.50 | 14.66 / 22.96 | 1.540 | 0.647 | 0.125 | 0.794 |
| Mooncake 0.86 default | 2,829.5 / 12,947.1 | 49.16 / 349.67 | 16.55 / 432.70 | | | | |
| repeat | 3,050.8 / 12,847.4 | 45.23 / 294.83 | 16.51 / 413.24 | 1.078 | 0.920 | 0.955 | 0.993 |
| capture 2048 | 3,039.8 / 12,271.9 | 44.74 / 330.61 | 16.50 / 413.78 | 1.074 | 0.910 | 0.956 | 0.995 |
| C3 (split bound in eager X steps) | 2,961.0 / 12,796.5 | 44.27 / 279.09 | 16.54 / 416.13 | 1.046 | 0.900 | 0.962 | 0.988 |
| `mnbt512` (306 requests) | 6,253.5 / 18,580.3 | 24.15 / 44.18 | 24.23 / 71.36 | 2.204 | 0.490 | 0.165 | 0.989 |

- **Repeat gate:** Azure passes (means within 0.1%).
- **Mooncake fails the gate on means:** the repeat moves mean TTFT by 7.8% and mean TPOT by 8.0%. Per-request
  medians stay within 0.8% (TTFT 0.997, TPOT 0.992). The tail dominates the means: TTFT p99 is 4.6x the mean.
- **So Mooncake differences in means below about 8% are noise.** Capture 2048 and C3 sit inside that band; their
  per-request median changes are at most 1.3% (TTFT 1.009 and 1.013, TPOT 0.999 and 0.994).
- **C3 forced a split on 6 steps** (0.3% of GPU time).

### H1-H5 verdicts (chat)

| Hyp. | Verdict | Evidence |
|---|---|---|
| H1 X straggler | **killed** | X_cost is 0 (Azure) and 0.5% of GPU time (Mooncake), against a 2% threshold. Capture 2048 and C3 change per-request medians by <= 1.3%; mean changes are inside the 8% repeat spread |
| H2 512 graph cliff | **killed** | Launch-bound steps above 512 tokens take <= 0.80% of GPU time. Up to half the GPU time runs eager without being launch-bound; capture 2048 changes Azure latency by <= 0.7% |
| H3 token budget is the wrong cost unit | **not killed; support clause not met** | Prefill-bearing steps hold 53% of decode-weighted time (threshold 20%), and ITL p99 is 17.6-26x ITL p50. `mnbt512` cuts ITL p99 8.0x / 6.1x but raises mean TTFT 54% / 120% (allowed: 10%) |
| H4 re-prefill bursts | **killed** | Chat has no compaction or cold restarts. Eviction re-prefills at most 5.7% of Mooncake's input tokens (9.3% expected hits minus 3.6% served), fewer of its attention FLOPs (early positions), against a 10% threshold |
| H5 FA3 wave quantization | **not killed; bounded** | Azure 87.4% efficiency (6.3 req/s: 76.7%) misses the 90% kill line, but perfect packing would recover only 0.6-0.8% of GPU time, because attention is 3-6% of GPU time in short-prompt steps. Mooncake 95.8-97.4%: killed |

### Predictions checked

| # | Prediction | Result | Held? |
|---|---|---|---|
| Az 1 | P, M >= 70% of GPU time at high load, >= 50% at low | 51.0%, 24.6% | no |
| Az 2 | H1 and H2 killed | X absent; launch-bound 0.80% | yes |
| Az 3 | Capture 2048 and step shaping change mean TTFT and TPOT by < 5% | capture 2048: +0.7%, +0.3%. Shaping: +38%, -28% | half |
| Az 4 | H3 >= 20%; `mnbt512` ITL p99 >= 1.5x better with mean TTFT <= +10% | 52.6%; 8.0x; +54% | no (TTFT) |
| Az 5 | H5 efficiency >= 85% | 87.4% (low load 76.7%) | yes at high load |
| Mc 1 | A, A+L >= 40% of GPU time | 25.6% | no |
| Mc 2 | X_cost 2-15% of GPU time; H1 not killed | 0.5%; killed | no |
| Mc 3 | C3 beats capture 2048 on mean TPOT; capture 2048 < 3% | 0.900 vs 0.910, both inside the 8% repeat band | not testable |
| Mc 4 | H2 killed | 0.59% | yes |
| Mc 5 | H3 >= 50%; `mnbt512` ITL p99 >= 2x better with mean TTFT <= +15% | 52.9%; 6.1x; +120% | no (TTFT) |
| Mc 6 | H5 efficiency 80-90% | 95.8% | no |
| Mc 7 | Served hits within 3 points of 14% | 3.6% (unlimited-cache estimate recomputed: 9.3%) | no |

What the misses teach:

- **Az 1:** my own load estimate said prefill fills 56% of the window, and 51% was measured. The 70% threshold did
  not follow from it. I under-counted decode-only steps, which pay a weight-streaming floor every step whatever their
  batch size.
- **Az 3 (shaping):** RegimeScheduler `cg_cap` keeps prefill steps at or below 512 tokens. It therefore acts as a
  token budget, an H3 lever, not an H2 one. It lands between the default and `mnbt512`: ITL p99 3.6x lower, TTFT +38%.
- **Az 4, Mc 5:** the rationale was "a 512-token step processes prefill as fast as an 8,192-token one", which is true
  for prefill alone (Stage 1b). In serving, though, every small step also carries the running decodes and the
  per-step weight floor. Azure `mnbt512` runs 33% more steps (18,477 vs 13,939) in the same GPU time (243 s), so
  prefill advances more slowly per step. Mean E2E still improves 21% (Azure) and 1% (Mooncake).
- **Mc 2, Mc 6:** Stage 1 was right about agents, and it applies here too. When the prefill chunk is 8,192 tokens on
  a long prefix, the straggler and the wave tail are hidden under it. Both losses appear only when chunks are small
  (`mnbt512`: A+L 35% of GPU time, efficiency 74%).

### ServingStudio (H200 simulator) vs vLLM (H100), first 2,000 Azure requests

| | TTFT mean / p50 / p90 / p99 | TPOT mean / p50 / p90 / p99 | E2E mean / p99 |
|---|---|---|---|
| ServingStudio (`llama3_dense_tp_barebone`, tp 1) | 124.7 / 95.6 / 262.5 / 428.9 | 15.14 / 10.64 / 25.15 / 113.44 | 1,458 / 8,878 |
| vLLM `az14_default`, same requests (ours) | 153.8 / 130.0 / 294.9 / 459.3 | 21.0 / 17.1 / 35.5 / 93.4 | 2,235 / 13,984 |

- **TPOT:** the simulator's mean TPOT is 0.72x ours, close to the H100/H200 bandwidth ratio (3.35 / 4.8 TB/s = 0.70).
  This comparison is our arithmetic on API numbers.
- **Prefill handling:** the simulator does not chunk prefill. Its steps carry up to 16,914 prefill tokens, against
  vLLM's 8,192.
- **Attention cost:** it prices prefill and decode attention as separate FlashInfer kernels, so it cannot show X,
  A+L or H5. Its kernel-time split for this workload is single GEMMs 74.3%, decode attention 16.8% and prefill
  attention 3.4%. That agrees with the tile model on the main point: in chat, attention in prefill-bearing steps is
  a small part of GPU time.

### Optimization areas (chat)

1. **Decode stalls behind prefill chunks (H3), scheduling.**
   - This is the only effect worth more than 1%. ITL p99 runs at 17-26x p50.
   - A static small budget trades TTFT for ITL, and badly.
   - The area to work on is a budget that prices a step by its cost (new tokens x context) and by the decode SLO,
     not by token count. Step shaping already shows a middle point.
2. **If steps are made small, launch A+L steps separately (kernel / launch).**
   - At a 512-token budget, separate launches would recover 5.0% of GPU time (modelled).
   - At the default budget they recover nothing.
   - A cost-based budget (area 1) will produce small chunks on long prefixes, so the two should be designed together.
3. **Prefix-cache retention (scheduling / memory).**
   - Mooncake loses 5.7 of 9.3 points of reusable input tokens to eviction.
   - This is outside the regime question, but it is a direct prefill saving; it belongs to Stage 5's session-aware
     KV retention.
4. **Not worth pursuing for chat:** split-bound fixes (C3), capture 2048 and wave-aware tiling. Each is worth at
   most 0.9% of GPU time.

### Caveats

- **Workloads:**
  - Azure contexts never reach 16k, so Azure tests only the short-context regimes.
  - Mooncake is one 106-s window (307 requests), time-scaled down 3.4x. Prompts are synthetic tokens with the
    trace's block reuse.
- **Noise:** Mooncake mean latencies vary 8% between identical runs, so only per-request medians and large effects
  are reliable.
- **Model:** the tile model prices attention, and Stage 1b found it under-predicts whole steps by 6-18%. Label shares
  use measured GPU time; recoverable times are modelled.
- **Shared node:** load averaged 86-174 during the runs. Launch-bound shares could be higher on a busier node
  (Stage 1's 16.8 ms launch floor was measured at load 650).
- **Hardware:** ServingStudio models an H200, not an H100.
