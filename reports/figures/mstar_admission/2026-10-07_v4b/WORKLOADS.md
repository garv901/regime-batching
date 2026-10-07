# v4b workloads: characteristics, purpose, and the behavior of each admission policy

This file describes the four workload families of the v4b study.
For each family, it gives the shape of the trace, the reason for the trace, and the behavior of each arm on it.
All numbers come from the 103 valid runs in `runs.tsv`, through `compare.md` and `summary.json`.

## Terms

- **The pool (P):** the BAGEL KV pool on one H100. It holds 2,048 pages of 128 tokens, so 262,144 tokens (14 GiB).
- **"x P":** a quantity of KV tokens divided by 262,144.
- **Final KV of a request:** prompt tokens plus output tokens. A request holds this much KV just before it stops.
- **Fluid KV peak:** the largest total KV of the active requests over time, in a model with no queue and no prefill time. Each request starts at its arrival and decodes at 20 ms per token (`kv_report.py --itl-ms 20`). A fluid peak above 1.0x P means that the server cannot hold all requests at the same time.
- **Total KV:** the sum of the final KV of all requests in the trace.
- **Declared KV:** prompt plus the declared `max_tokens`. The admission arms reserve this quantity.
- **Short, mid, long:** the request class. The synthetic traces set the class directly. The public traces use percentiles of the final KV: short is at or below p50, long is at or above p90, and mid is between.
- **Admission wait:** the time from the first admission check of a request to its reservation, from the admission log. For the strict FIFO arms (`ac`, `ac_park`), the clock starts only when the request comes to the head of the queue. Thus admission wait is not comparable between FIFO and backfill. Use TTFT to compare the arms.
- **Host ms per step:** the CPU time between two decode steps, without the GPU time. The study bins it by the queue depth.

## The arms in one paragraph each

- **`main`** (mstar `origin/main`, no admission control). It admits a request for its current span only. When a request needs a new page and no page is free, `main` holds the request for 50 ms and tries again. With `cpu_offload_pages=0`, no request can give its pages back before it stops. Thus, if all active requests need pages, nothing can move.
- **`ac`** (Gaurav's admission control, `FIT=sum ORDER=fifo`). It reserves prompt plus `max_tokens` for each request. It admits a request only if the sum of all reservations fits in the pool. Nothing passes the head of the queue.
- **`ac_park`** (`ac` plus pass D). The decisions are the same as `ac`. A request that cannot be admitted is "parked": the scheduler does not examine it again until the KV state changes. This removes the per-scan cost of a long queue.
- **`peak_bf_pc_park`** ("peak", our policy: `FIT=peak ORDER=backfill`, plan cache on, park on). It admits a request if the future peak of KV fits. A request that stops gives its pages to the requests that continue, so the future peak is lower than the sum. If the head does not fit, a later request that fits can pass it (backfill), but only if the head does not start later.

## Summary of the four families

| family | file | requests / span / rate | classes (prompt p50, output p50) | fluid KV peak | total KV | declared KV, `_mtx2` / `_mt2048` | source |
|---|---|---|---|---|---|---|---|
| burst | `bagel_burst_v2` | 2,700 / 305 s / 8.86 per s | 2,654 short (726, 39); 46 long (6,500, 2,000), all at t = 30 s | 1.53x P | 9.42x P | 1.04-1.05x / 3.16-3.21x | synthetic |
| stag | `bagel_stag_v3_r5.5` | 3,300 / 591 s / 5.59 per s, Poisson | 2,820 short (727, 39); 480 long (8,100, 1,170) | 0.84-1.02x P | 25.3x P | 1.06x / 1.92-1.93x | synthetic |
| Mooncake | `pub_mooncake_conv_r0.5` | 1,036 / 594 s / 1.74 per s | 518 short (2,216); 414 mid (12,704); 104 long (32,269); output p50 approximately 350 in each class | 1.77x P | 42.4x P | 1.03x / 1.16x | Kimi production trace |
| Azure | `pub_azure_conv_r1.25` | 4,105 / 599 s / 6.85 per s | 2,053 short (391, 99); 1,641 mid (1,096, 396); 411 long (2,964, 101) | 0.30x P | not used | 1.16x / 2.46-2.48x (variants dropped) | Azure production trace |

"Declared KV" is the declared KV of the variant divided by the real KV, summed over the requests.
The ranges come from the different runs of the variant.

## Variants of each family

Each family has three variants. The arrivals, the prompts, and the real output lengths are the same in all three.
Only the declared `max_tokens` changes.

| variant | declared `max_tokens` | what it models |
|---|---|---|
| base | the real output length | a perfect length oracle |
| `_mtx2` | min(2048, 2 x the real length) | a length predictor with a 2x margin |
| `_mt2048` | 2048 for each request | no length information (the client sends the model cap) |

The server stops each request at its real length (emulated EOS, `emulated_eos_at`), so the KV that the request uses does not change.
The hash ids of the prompt blocks get an offset per variant (+1e9 for `_mtx2`, +2e9 for `_mt2048`).
Thus one variant cannot get prefix cache hits from the blocks of a different variant.

## 1. burst (`bagel_burst_v2`)

### Shape

- **Shorts:** 2,654 requests at approximately 9 per second from 0 to 305 s. The prompt is uniform in [450, 1,000] tokens (p50 726). The output is uniform in [16, 64] tokens (p50 39).
- **Longs:** 46 requests that all arrive at t = 30 s. Each has a prompt of 6,500 tokens and an output of 2,000 tokens. Together, the longs need 392k tokens of final KV, which is 1.50x P.
- **No prefix reuse:** each prompt block has a unique hash id.
- **Fluid KV peak** 1.53x P, total KV 9.42x P. The trace needs 648 output tokens per second.

### Purpose

This trace is the stress case for admission. The longs cannot all fit in the pool at the same time.
A steady flow of short requests arrives behind them.
The trace shows if a policy can stop the stall of `main`, and what the shorts pay while the longs occupy the pool.
The longs all have the same length, so they all stop at approximately the same time. This makes the summed fit and the peak fit give different answers.

### Production relevance

We did not measure a production burst. The shape is that of a tenant that sends many long-context requests at once.
Examples are a batch job or an agent fan-out. Interactive traffic continues at the same time.
The Mooncake trace (section 3) has a fluid peak of 1.77x P, so real traffic can also exceed this pool.

### Behavior of each arm (base variant)

| metric | `main` (n = 1) | `ac` | `ac_park` | peak |
|---|---|---|---|---|
| completed / timed out | 1,157 / 1,543 | 2,700 / 0 | 2,700 / 0 | 2,700 / 0 |
| short TTFT mean (s) | 411 | 16.2 | 7.4 | **0.20** |
| short E2E mean (s) [min-max] | 439 | 20.2 [18.9-22.2] | 9.8 [8.7-11.4] | **0.85** [0.76-1.02] |
| long TTFT mean (s) | - | 25.8 | 19.3 | 24.4 |
| long E2E mean (s) [min-max] | - | 103.9 [101.5-107.7] | 75.6 [71.5-80.3] | 76.8 [72.0-85.1] |
| long E2E max (s) | - | 170.8 | 123.7 | 101.7 |
| admission wait, short / long (s) | - | 0.03 / 1.45 (from the head) | 0.02 / 1.02 (from the head) | 0.05 / **20.9** |
| max stall (s) | 659 | 61.9 | 43.8 | 5.2 |
| holds (admit failures) | 108,919 | 0 | 0 | 0.7 |
| host ms per step at 400+ in queue | - | 29.1 | 11.1 | fewer than 30 steps at this depth |
| GPU busy / decode batch | 10% / 13.3 | 71% / 10.0 | 80% / 8.0 | 83.5% / 7.5 |
| output tok/s | 54 | 648 | 648 | 648 |

- **`main` stalls in each run**. The longs and the shorts fill the pool, and each active request then needs one more page. No request can give pages back, so `main` holds and retries 108,919 times. The GPU is busy for only 10% of the run. After 600 s, the server aborts the requests that wait, and 1,543 of 2,700 requests fail.
- **`ac` blocks the shorts behind the longs**. The sum of the long reservations is 1.5x P, so only part of the longs fit. The first long that does not fit becomes the head, and each short behind it waits. The queue grows above 400 requests. At that depth, each decode step costs 29.1 ms of host time, so the decode of the admitted requests also slows.
- **`ac_park` makes the same decisions, but faster**. The queue is as long as in `ac`. The scheduler does not ask parked requests again. Thus, at 400+ in queue, the host time per step goes from 29.1 ms to 11.1 ms. Short E2E is 0.49x of `ac`, and long E2E is 0.73x.
- **peak defers the longs and lets the shorts pass**. The longs that do not fit wait for 20.9 s on average. The shorts wait for 0.05 s. The queue stays short, so the host time per step stays low. Short E2E is 0.04x of `ac`.
- **The longs do not pay for this in peak**. Long TTFT is the same as `ac` (24.4 s against 25.8 s). Long E2E is 0.74x of `ac`, and the worst long finishes in 101.7 s against 170.8 s. The probable cause is the faster decode: the queue is short, so the host time per step is low.

### `_mtx2` (declared KV 1.04-1.05x)

- The result is the same as the base variant.
- Short E2E: `ac` 15.6 s, `ac_park` 11.3 s, peak **0.91 s** (0.06x of `ac`, ranges apart).
- Long E2E: `ac` 92.6 s, `ac_park` 79.7 s, peak 77.8 s.
- The prompt is most of the KV, so a 2x output margin adds only 4-5% to the declared KV.
- `main` stalls (1,472 of 2,700 requests fail).

### `_mt2048` (declared KV 3.16-3.21x)

- Each short declares 2,048 output tokens but uses approximately 39. Each short then looks like a long to the planner.
- peak cannot find the shorts any more: admission wait is 21.3 s for the shorts and 24.2 s for the longs.
- Short E2E: `ac` 38.9 [15.9-84.4], `ac_park` 17.7 [10.2-26.2], peak 24.2 [14.9-38.1]. All ranges overlap.
- Long E2E: 118.1, 92.8, 118.2 s, all within noise. The decode batch of peak (14.0) is near that of `ac` (14.6).
- Each arm changes by 2.6-5.3x between runs. This noise does not follow the outsider load: the worst runs of `ac_park` and peak ran at 2.3%.
- `main` stalls (1,687 of 2,700 requests fail).

## 2. stag (`bagel_stag_v3_r5.5`)

### Shape

- **Arrivals:** Poisson at 5.5 per second over 591 s (3,300 requests). Each request is long with a probability of 0.15.
- **Shorts:** 2,820 requests with the same distribution as the burst shorts (prompt p50 727, output p50 39).
- **Longs:** 480 requests. The prompt is uniform in [4,000, 12,000] tokens (p50 8,100). The output is uniform in [256, 2,000] tokens (p50 1,170). The largest final KV is 13,876 tokens.
- **No prefix reuse**.
- **Fluid KV peak** 0.84-1.02x P, total KV 25.3x P. The trace needs 1,102 output tokens per second.

### Purpose

This trace is the control. The pool is close to full at the peak, but no burst occurs.
The longs have different lengths, so they stop at different times. In the simulator, peak with backfill gave no short E2E advantage here.
If peak was much better on this trace, the harness and not the policy can be the cause.

### Production relevance

The shape is a steady mix of chat and long-document requests, for example retrieval with long contexts at 15% of the traffic.
The real traces in this study have a similar long tail of prompts: Mooncake has a long class with prompts of approximately 32k tokens.

### Behavior of each arm (base variant)

| metric | `main` (n = 1) | `ac` | `ac_park` | peak |
|---|---|---|---|---|
| short E2E mean (s) [min-max] | 0.98 | 1.22 [0.97-1.45] | 1.41 [1.07-1.99] | 1.00 [0.88-1.24] |
| long E2E mean (s) [min-max] | 22.4 | 20.8 [19.8-21.6] | 21.7 [19.9-24.5] | 22.0 [20.6-24.1] |
| admission wait, short / long (s) | - | 0.01 / 0.11 (from the head) | 0.01 / 0.11 (from the head) | 0.13 / 0.52 |
| holds | 3,415 | 0 | 0 | 0 |
| max stall (s) | 4.5 | 4.3 | 4.2 | 4.1 |
| GPU busy / decode batch | 86.9% / 17.4 | 88.2% / 17.0 | 87.6% / 17.5 | 87.7% / 17.5 |

- **All arms are within noise**. Each arm keeps up with the trace (1,102 tok/s).
- **`main` holds requests 3,415 times but does not stall**. The pool fills for short periods only, and a request that stops soon frees pages.
- **The control holds**. peak gives no gain here, as the simulator predicted.

### `_mtx2` (declared KV 1.06x)

| metric | `main` (n = 1) | `ac` | `ac_park` | peak |
|---|---|---|---|---|
| short E2E mean (s) [min-max] | 131.5, stall | 6.22 [1.47-15.62] | 1.61 [1.27-2.23] | **0.84** [0.82-0.87] |
| long E2E mean (s) [min-max] | 86.1 | 27.7 [21.0-40.8] | 21.9 [20.5-24.6] | 20.4 [20.0-21.2] |
| output tok/s | 459 | 1,063 [986-1,102] | 1,102 | 1,102 |

- The 6% of extra declared KV pushes the summed reservation above the pool at the peak. Then `ac` blocks at the head. One `ac` run has a short E2E of 15.6 s, and that run falls behind the trace (986 tok/s).
- peak does not see the extra 6% as a problem, because the future peak is lower than the sum. Short E2E is 0.13x of `ac` (ranges apart).
- `main` stalls in its one run (264 requests time out). Section "main stalls" in `SUMMARY.md` gives the limits of this result.

### `_mt2048` (declared KV 1.92-1.93x)

- Short E2E: `ac` 1.93, `ac_park` 3.17, peak 1.45 s. All ranges overlap. Long E2E: 22.3, 24.7, 24.3 s.
- peak waits more before it admits (0.48 s short, 1.08 s long), but the trace has enough slack to absorb it.
- `main` stalls in its one run (1,659 requests time out).

## 3. Mooncake (`pub_mooncake_conv_r0.5`)

### Shape

- **Source:** the Kimi (Moonshot AI) conversation trace of the Mooncake paper (FAST 2025). We use the window [1,350, 1,650) s of the original trace, at half the original rate (`r0.5`). The span is 594 s.
- **Requests:** 1,036 at 1.74 per second.
- **Classes by final KV:**
  - short (518): prompt p50 2,216 tokens.
  - mid (414): prompt p50 12,704 tokens.
  - long (104): prompt p50 32,269 tokens. The converter clamps the final KV to `max_seq_len` (32,768).
- **Outputs:** p50 approximately 350 tokens in each class, max 2,000. The prompt is most of the KV.
- **Prefix reuse:** 21.6% of the prompt blocks repeat an earlier block, so the prefix cache gets hits.
- **Fluid KV peak** 1.77x P, total KV 42.4x P. The trace needs 582 output tokens per second.
- **Data note:** 9 requests with a real output of 1 token return 2 tokens, in each arm (flag OVERRUN). The effect on KV is negligible.

### Purpose

This trace is real production traffic with long contexts.
It shows if the result on the synthetic traces is also true for a real mix of prompt lengths and prefix reuse.

### Production relevance

This is production traffic from a chat service with long contexts (Kimi). We changed only the rate and the window.

### Behavior of each arm (base variant)

| metric | `main` (n = 2) | `ac` | `ac_park` | peak |
|---|---|---|---|---|
| short TTFT mean (s) | 2.86 | 2.91 | 3.06 | **1.80** |
| short E2E mean (s) [min-max] | 12.55 [11.29-13.82] | 10.20 [9.90-10.70] | 10.55 [10.32-10.79] | **9.25** [9.02-9.38] |
| mid E2E mean (s) [min-max] | 16.50 [13.87-19.13] | 11.66 [11.32-12.15] | 12.02 [11.78-12.24] | 11.62 [11.50-11.74] |
| long E2E mean (s) [min-max] | 30.3 [23.3-37.4] | 14.6 [14.2-15.3] | 15.0 [14.8-15.3] | 15.1 [14.7-15.3] |
| long TTFT mean (s) | 18.96 | 5.56 | 5.74 | 5.91 |
| admission wait, short / long (s) | - | 0.05 / 0.50 (from the head) | 0.05 / 0.51 (from the head) | 0.27 / 2.97 |
| holds | 15,841 | 0 | 0 | 2.3 |
| GPU busy / decode batch | 85.9% / 14.4 | 94.5% / 10.9 | 94.8% / 11.2 | 94.9% / 11.0 |

- **`main` completes, but the longs take 2x longer**. `main` admits more requests than the pool can grow (decode batch 14.4 against 11). The active requests then wait for pages while they hold KV (15,841 holds). The GPU is busy for 86% of the run, against 94.5%. Long TTFT is 19.0 s against 5.6 s, and long E2E is 30.3 s against 14.6 s.
- **The three admission arms halve the long E2E of `main`**. A reservation makes sure that an admitted request never waits for a page.
- **peak is 9% faster than `ac` on the shorts (ranges apart)**. Short TTFT is 1.80 s against 2.91 s. peak defers the largest requests (long admission wait 2.97 s against 0.50 s), but the long E2E stays within noise.
- **park gives no gain here**. The queue is short (1.74 requests per second), so the scan cost is small.

### `_mtx2` (declared KV 1.03x) and `_mt2048` (declared KV 1.16x)

- Each variant adds little declared KV, because the prompts are long and the outputs are short.
- Short E2E: peak 9.37 s (`_mtx2`) and 9.35 s (`_mt2048`), against `ac` 10.29 s and 10.83 s. peak is 0.91x and 0.86x of `ac`, ranges apart.
- **The only measured cost of peak:** on `_mt2048`, long E2E is 15.9 [15.6-16.4] s. This is 1.07x of `ac` (14.9, within noise) and 1.11x of `ac_park` (14.3 [14.0-15.0], ranges apart). The long admission wait of peak goes up to 4.06 s.
- `main` stalls in 1 of 2 `_mt2048` runs (522 requests time out in that run).

## 4. Azure (`pub_azure_conv_r1.25`)

### Shape

- **Source:** the Azure LLM inference trace 2023, conversation service (CC-BY), from the Splitwise paper (ISCA 2024). We use the window [2,175, 2,925) s of the original trace, at 1.25x the original rate (`r1.25`). The span is 599 s.
- **Requests:** 4,105 at 6.85 per second.
- **Classes by final KV:**
  - short (2,053): prompt p50 391, output p50 99.
  - mid (1,641): prompt p50 1,096, output p50 396.
  - long (411): prompt p50 2,964, output p50 101.
- **Fluid KV peak 0.30x P**. The trace never fills the pool.
- **Load:** 1,408 output tokens per second, the highest of the four families.

### Purpose

This trace is real production traffic with short contexts and a high request rate.
It does not test KV admission. It measures what the admission code costs when the pool is never full.

### Production relevance

This is production traffic from a chat service (Azure, 2023). We changed only the rate and the window.
Many chat services have this shape: short prompts, short outputs, and many requests per second.

### Behavior of each arm (base variant)

| metric | `main` (n = 1) | `ac` | `ac_park` | peak |
|---|---|---|---|---|
| short E2E mean (s) [min-max] | 3.83 | 6.45 [5.29-7.07] | 5.57 [4.17-6.30] | 4.39 [3.60-5.03] |
| mid E2E mean (s) [min-max] | 10.96 | 19.30 [15.63-21.52] | 16.93 [11.70-19.63] | 12.62 [9.74-15.58] |
| long E2E mean (s) [min-max] | 4.0 | 6.7 [5.4-7.4] | 5.7 [4.3-6.5] | 4.5 [3.8-5.1] |
| admission wait, short / long (s) | - | 0.01 / 0.05 (from the head) | 0.00 / 0.03 (from the head) | 0.00 / 0.00 |
| host ms per step (all depths) | 6.11 | 10.05 | 9.25 | 7.11 |
| GPU busy / decode batch | 63.9% / 28.2 | 56.4% / 36.0 | 57.8% / 34.5 | 61.8% / 30.6 |
| output tok/s | 1,407 | 1,351 | 1,366 | 1,396 |

- **No arm waits for KV**. The admission wait is 0.05 s or less in each arm.
- **The host CPU limits the server, not the GPU**. The host time per decode step is 6-10 ms. On the other workloads, it is 0.5-2 ms. The GPU is busy for only 56-64% of the run.
- **Each admission arm adds host time, and that host time becomes latency**. The server has 2 CPUs and they are saturated. `ac` adds 3.9 ms per step to `main`, `ac_park` adds 3.1 ms, and peak adds 1.0 ms.
- **The slower steps make the decode batch larger**. Each request stays active for longer, so more requests are active at the same time (36 in `ac` against 28 in `main`).
- **peak is 0.68x of `ac` on short E2E (ranges apart)**, but it is still approximately 15% slower than `main`. peak runs with the plan cache on, and `ac` and `ac_park` run with it off. This is a probable cause of the gap, but this matrix does not isolate it.
- **The over-declared variants are dropped**. Each arm had only 1 valid run of `_mtx2` and `_mt2048`, so we do not report them.

## What the four families show together

- **The peak gain needs a length declaration that is close to the real length**.
  - On burst, peak is 0.04x of `ac` (base) and 0.06x (`_mtx2`). On stag `_mtx2`, it is 0.13x.
  - On Mooncake, it is 0.86-0.91x.
  - With `_mt2048`, the gain disappears on burst and stag.
- **The prompt is most of the KV in all four families**. Thus a 2x margin on the output adds only 3-6% of declared KV on the KV-bound traces.
- **`main` stalls when the pool fills with active requests that all need pages**. Each admission arm completes all requests in each valid run.
- **When the pool is never full (Azure), the cost of admission is host time**. Of the three admission arms, peak with the plan cache costs the least.
