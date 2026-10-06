# mstar: scheduling and kernel gaps (hypotheses)

This page applies the lessons of the vLLM study to mstar. It lists gaps in the mstar scheduler and in its kernel
use, and one hypothesis for each gap. **We ran no experiments on mstar.** Each item is a hypothesis to test later.

- **Source.** A read-only audit of mstar `main` at commit `3fed73c1` (2026-10-05). Paths are relative to the mstar
  repository root.
- **Verification.** We read each `path:line` citation again in the source. Items marked *(audit)* come from the
  audit report, and we did not read them again. Items marked *(unverified)* are inferences from the code.

## 1. Lessons from the vLLM study

| ID | Lesson | Evidence in this repository |
|---|---|---|
| L1 | An output-length oracle helps decisions that are expensive to undo, such as KV admission. | 19-29x short-request E2E on the burst ([03_results.md](03_results.md)) |
| L2 | The oracle does not help decisions that can change at every step: KV-split count and step size. | C3 reactive 1.18x; step shaping 1.27x without the oracle, 1.26x with it ([04, section 3.6](04_vllm_takeaways.md#36-an-output-length-oracle-does-not-improve-the-split-decision)) |
| L3 | The largest cost is a step boundary where the CUDA graph and the KV split change together. | X steps use 56% of GPU time; step shaping gives 1.26x |
| L4 | Preemption with recompute is expensive, and the prefix cache does not save the work. | 52 preemptions, 219k tokens recomputed; the prefix cache returns 2-7% |
| L5 | On a dense 8B model, a change of decode or GEMM kernel changes a step by 5% or less. | Microbenchmarks ([04, section 3.4](04_vllm_takeaways.md#34-outside-the-boundary-the-default-kernels-are-good)) |
| L6 | Placement on an instance is expensive to undo, so the oracle probably helps it. | Not tested ([06, item 6](06_limitations_and_future_work.md)) |
| L7 | Per-step telemetry with a regime label finds the real cost before any fix. | The X regime was found this way |

## 2. What carries over to mstar

| vLLM finding | In mstar? | Evidence |
|---|---|---|
| X step: a prefill chunk and a long-context decode in one eager step with 1 KV split | **No.** A batch holds one (node, walk), so prefill and decode do not share a step. The decode batch is capped at the largest captured graph. | `mstar/worker/micro_scheduler.py:380-387`, `mstar/engine/engine.py:1151-1167` |
| Admission plans for the present span only, so memory is overcommitted | **Yes.** The KV manager allocates the current span plus the stored length. It does not read `max_tokens`. | `mstar/engine/resources/kv/manager.py:405-415` |
| Head-of-line blocking | **Yes, at batch level.** If one request in a batch does not fit, the whole batch goes back to its queues. | `mstar/worker/worker.py:1465-1470` |
| A kernel switch gains little | Probably true for dense models. MoE and long contexts can differ. | L5 |
| The oracle helps irreversible decisions | **Yes:** admission, data-parallel routing and stage placement. | L1, L6 |

mstar has one advantage for this work. In many mstar workloads, the length is known at arrival, so the oracle is
more realistic than in chat:

- A diffusion model runs a fixed number of denoise steps (Cosmos3 `max_inference_steps`, BAGEL `num_timesteps`)
  *(audit)*.
- `max_tokens` sets the upper bound of each autoregressive loop.
- For TTS, the output length follows the text length.

## 3. Scheduling gaps

Each item gives the gap, the hypothesis and a test.

### S1. Memory pressure can stop all progress (highest risk)

- **Gap.**
  - Admission reserves only the current span (`kv/manager.py:405-415`).
  - On an allocation failure, the worker pushes the whole batch back to its queues (`worker.py:1465-1470`).
  - It then offloads one victim to CPU memory. This works only if the engine can evict (`worker.py:902-904`).
    CPU offload is off by default: `cpu_offload_pages = 0` (`mstar/engine/resources/kv/config.py:33-34`).
  - With no victim, the worker holds every request of the batch for a backoff period and tries again
    (`worker.py:1480-1486`; `HOLD_BACKOFF_SECONDS = 0.05` in `micro_scheduler.py`). There is no recompute path.
- **Hypothesis** *(unverified)*. With the defaults and a full page pool, a batch that holds one request that needs
  a new page fails each time. The other requests of that batch stall with it. If no request can finish, no page
  becomes free, and progress can stop. An admission planner that reserves prompt + `max_tokens` (or the known
  step count) removes this case. In vLLM, the same planner gave 19-29x (L1).
- **Test.** Set a small `max_num_pages`, keep CPU offload off and send a burst of long-output requests, like
  `lburst80_8k`. Count holds, offloads and completed requests over time.

### S2. A batch is never made smaller to fit

- **Gap.** Admission is all or nothing for each batch (`mstar/engine/resources/runner.py:226-244`). Pages that a
  refused admit already gave to a stream stay with that stream: `page_indices` is a high-water mark
  (`kv/manager.py:443-446`).
- **Hypothesis.** Admit the largest subset of the batch that fits, and fill the free seats with small requests.
  This decreases tail latency under memory pressure.
- **Test.** The S1 setup. Compare E2E p99 and the number of holds.

### S3. The eviction victim is selected only by recency

- **Gap.** `_select_eviction_victim` picks the request with the oldest `_last_active` time. It does not look at
  the number of pages or the remaining length (`worker.py:933-946`).
- **Hypothesis.** Pick the request that frees the most pages for the longest time: pages x remaining tokens.
  `max_tokens` gives the remaining length. This decreases the number of offload and reload cycles.
- **Test.** The S1 setup with CPU offload on. Count offloads, reloads and bytes moved.

### S4. Round-robin counts steps, not time, and there is no chunked prefill

- **Gap.** The micro-scheduler selects the (node, walk) with the oldest last step (`micro_scheduler.py:154-170`).
  A 2,048-token prefill step and a decode step count the same. Chunked prefill is a TODO
  (`engine.py:1263-1268`). Every decode on that worker waits for the full prefill step.
- **Hypothesis.** The tail of inter-token latency increases with prompt length. Cut prefills into pieces of the
  largest captured token bucket, or give prefill a time budget for each decode step. Do not cut smaller than the
  bucket: in our vLLM runs, `--max-num-batched-tokens 512` made TTFT about 2x worse.
- **Design warning.** If mstar later mixes prefill and decode in one step, it can create an X-step equivalent.
  Keep the KV split available for long-context decodes in mixed steps (L3).
- **Test.** Long prompts arrive while many decodes run. Measure ITL p99 against prompt length.

### S5. Stages share one GPU with no cost or deadline awareness

- **Gap.** The stages of one model (for example Thinker, Talker and Code2Wav, or ViT and LLM) are scheduled by the
  same round-robin. Each stage has its own KV pool, and nothing plans memory across stages *(audit)*.
- **Hypothesis.** For streaming audio, select the walk whose oldest request has the least time to its deadline.
  This improves first-audio latency and the real-time factor under mixed load. This needs no oracle.
- **Test.** Mixed text and audio requests on Qwen3-Omni. Measure first-audio latency and the real-time factor.

### S6. Data-parallel routing is random

- **Gap.** The conductor picks a replica with `np.random.randint` (`mstar/conductor/conductor.py:510-519`).
- **Hypothesis.** Routing cannot be undone without a KV move, so foresight helps (L6). Route by outstanding work
  (prompt + remaining `max_tokens`) or by the projected KV peak of each replica. With skewed lengths, this
  decreases tail latency. It is cheap to build.
- **Test.** Two or more replicas, bimodal output lengths. Compare random routing, least outstanding requests and
  least outstanding tokens.

### S7. Diffusion batches mix resolutions and step counts

- **Gap.** Cosmos3 batches up to 8 generations (`max_gen_batch_size = 8`) *(audit)*. Requests in one batch can
  have different resolutions and different remaining steps.
- **Hypothesis.** Group requests by resolution to avoid padding. Then run the request with the fewest remaining
  denoise steps first. The step count is known at arrival, so this oracle is exact.
- **Test.** Mixed resolutions and step counts. Measure mean and p99 latency per request.

### S8. A finished request can keep its state until the client reads the outputs

- **Gap.** The worker removes a request (and its engine state) when the conductor sends `REMOVE_REQUEST`. The
  conductor sends it only after the API server has read all outputs (`worker.py:516-524`). *(unverified)* KV
  pages may stay allocated during that time.
- **Hypothesis.** Slow clients hold KV pages under memory pressure. Free the KV pages when the request completes,
  and keep only the outputs until the read.
- **Test.** First, confirm when KV pages are freed. Then add slow clients to the S1 setup.

### S9. There is no prefix caching (low priority)

- **Gap.** The page allocator is a FIFO queue of free pages with no hash table (`kv/cache.py:11-53`). The prefix
  cache in the dense model code is per request *(audit)*.
- **Hypothesis.** Shared system prompts and multi-turn chat would gain. First, measure how often prompts share a
  prefix. Do not expect the prefix cache to help under memory pressure (L4).

## 4. Kernel gaps

### K1. The decode batch is capped at 64 by default

- **Gap.** The default capture sizes are `[1, 2, 4, 8, 16, 32, 64]` (`mstar/engine/cuda_graph_runner.py:24`).
  The scheduler does not exceed the largest capture (`engine.py:1151-1167`). With more than 64 running decodes,
  one token round needs several decode steps, and each step reads all the weights again.
- **Hypothesis.** Dense decode is limited by weight reads (6.6 ms per step on an 8B model in our study). Capture
  sizes of 128 and 256 raise decode throughput at high concurrency, almost linearly, until KV reads dominate.
- **Test.** A configuration change. Measure tokens per second for concurrency 32 to 256.

### K2. Large prefills fall out of the CUDA graph

- **Gap.** Above the largest token bucket, a prefill step runs eagerly. The audit reports 2,048 tokens for BAGEL
  and the Qwen3-Omni text stage, and 1,024 for Orpheus *(audit)*. A step with no lease runs eagerly
  (`engine.py:1195-1234`). Pre-planning of packed prefill steps is not supported yet (`engine.py:1263-1268`).
- **Hypothesis.** On our node, an eager step needed about 16.5 ms of CPU launch time (L3). On a loaded host,
  several graphed bucket-size steps are faster than one eager step. Combine this with S4.
- **Test.** Prompt lengths from 512 to 8k. Compare one eager step with bucket-size steps.

### K3. mstar does not set the FlashInfer split options

- **Gap.** `FlashInferDecodeWrapper` uses `BatchDecodeWithPagedKVCacheWrapper` with `backend="auto"`
  (`mstar/engine/resources/attn/wrappers.py:178-208`). `plan()` gets no `fixed_split_size` and no
  `disable_split_kv` *(audit)*. In FlashInfer `include/flashinfer/attention/scheduler.cuh`, the split choice
  differs between CUDA-graph mode (split forced, tiles sized for the worst case) and eager mode (tiles sized from
  the average) *(audit; the FlashInfer version that mstar deploys is not confirmed)*.
- **Hypothesis.** Find the decode kernel that `"auto"` selects on H100. In our microbenchmark, FlashInfer XQA was
  1.7x slower than FA3 on skewed batches (geomean). With 32k-token contexts, attention is a large part of the step,
  so this can matter.
- **Test.** Microbenchmark skewed decode batches (one long context, many short) through the mstar wrapper, in
  graph and eager mode, as in `microbench/`.

### K4. MoE has no tuned kernel configurations and no expert parallelism

- **Gap.** The Triton grouped GEMM for MoE has two tile regimes and no tuned tables *(audit)*.
- **Hypothesis.** For Kimi-K2-class models, tables tuned per token count, plus expert parallelism, gain more than
  any attention change. The MoE GEMM shape changes most with batch size. This comes from our pen-and-paper
  analysis, not from a measurement.
- **Test.** Sweep the token count per expert and compare the default tiles with a tuned table.

### K5. There is no per-step GPU time and no regime label (do this first)

- **Gap.** The timers measure CPU time. The logs do not record batch composition or KV use *(audit)*.
- **Why it matters.** In our study, per-step regime telemetry found that X steps used 56% of GPU time (L7).
  Without it, we would have worked on decode kernels, which are worth 5% or less (L5).
- **Proposal.** Write one row per step with:
  - node, walk, batch size, token count;
  - longest and mean context;
  - graph or eager, bucket, and the split and plan settings;
  - GPU time from CUDA events;
  - free pages, holds and offloads.

## 5. Which items need the oracle

| Item | Needs an output-length oracle? | Why |
|---|---|---|
| S1 admission planner | Yes (`max_tokens` or the step count as an upper bound) | Admission is expensive to undo (L1) |
| S3 eviction victim | Yes (remaining length) | The victim choice uses future page use |
| S6 routing | Yes (outstanding work) | Placement is expensive to undo (L6) |
| S7 diffusion order | Yes, and exact (the step count is known) | Shortest remaining work first |
| S2, S4, S5, S8, S9 | No | Reactive or per-step decisions (L2) |
| K1-K5 | No | Kernel choices depend on the present step only (L2) |

## 6. Suggested order

1. K5 telemetry. Every other test needs it.
2. S1 stall test. It is a robustness risk, and the test is cheap.
3. K1 capture sizes above 64. It is a configuration change.
4. S4 and K2: chunked prefill at the bucket size, against round-robin, for ITL.
5. S6 routing by outstanding work.
6. S5 deadline-aware stage selection for streaming audio.
7. K3 FlashInfer on skewed decode batches.
8. K4 MoE tuning, as part of the Kimi integration.

## 7. Open questions before the tests

- Does `engine.evictable()` return false when `cpu_offload_pages = 0`? If yes, S1 has no offload path by default.
- Can S1 stop all progress, or do smaller batches of the same walk still run?
- When are KV pages freed: at completion, or at `REMOVE_REQUEST` (S8)?
- Which FlashInfer version and decode kernel does mstar use on H100 (K3)?
- Do production model configs override the default capture sizes (K1) and token buckets (K2)?
