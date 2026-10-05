# What this study shows about vLLM today

This page explains three practices in vLLM 0.30.0 that the measurements expose: scheduling, KV-cache management
and kernel use. Each item gives the mechanism, the source location, the measured effect and what helps.

Source paths refer to vLLM v0.30.0.

## 1. KV-cache management

### 1.1 Admission checks present memory, not future memory

- **Mechanism.** The scheduler admits a waiting request if the KV cache has space for its prompt now
  (`vllm/v1/core/sched/scheduler.py`, `allocate_slots` in the waiting loop). It does not reserve space for output
  tokens. A request with 8,000 output tokens grows by 8,000 slots after admission.
- **Effect.** In the burst workload, vLLM admitted all 80 long requests at t = 30 s. They needed 681k slots at the
  end; the cache holds 451k.

### 1.2 Preemption corrects the overcommit with repeated work

- **Mechanism.** When a running request needs a new block and none is free, vLLM preempts the most recently
  admitted request (`scheduler.py:753`, `self.running[-1]`). It frees the request's blocks and puts it at the front
  of the waiting queue (`scheduler.py:1517`). Later, vLLM computes the request's tokens again.
- **Effect.** 52 preemptions, about 219k tokens computed a second time, and 319 steps longer than 100 ms. Every
  running decode waits during those steps: short TPOT 76.8 ms against 21.7 ms with the KV plan.

### 1.3 The waiting queue blocks at its head

- **Mechanism.** If the first waiting request does not fit, the scheduler stops admission for that step
  (`scheduler.py:1180`, `break`). A preempted long request at the head blocks every short request behind it.
- **Effect.** Short requests that arrived between 130 s and 230 s waited 66.6 s on average for their first token.

### 1.4 Static flags cannot fix this

`--max-num-seqs` trades one class against the other. With 32 or 64 seats, the long requests keep the seats and
short requests wait 79-148 s. With 128 or 256 seats, the overcommit stays. Ordering the queue by oracle length
(shortest first) gives only 1.40x, because the long requests that are already admitted still fill the cache.

### 1.5 What helps

A planner that knows each request's remaining length can calculate the future KV peak of the admitted set. It
admits a request only if the peak fits, and it lets small requests start before a held large request. Result:
0 preemptions and 20-30x lower short-request latency on three models, for 0-7% less throughput. This needs output
lengths. With real (unknown) lengths, the same planner can work from a predicted length or a reserved upper bound;
this study does not measure that.

## 2. Scheduling

### 2.1 The token budget does not see kernel regimes

- **Mechanism.** The scheduler fills each step up to `max_num_batched_tokens` (8,192 by default) with chunked
  prefill. It does not know that the attention backend and CUDA graphs change behavior at 512 tokens.
- **Effect.** Each new 450-1,000-token prompt goes into the same step as the running long-context decodes. That
  step crosses 512 tokens and becomes an X step.

### 2.2 What helps

Step shaping: limit a step to the CUDA-graph capture size while a long-context decode runs, and lift the limit when
the prefill backlog grows. Result: 1.26x short E2E and 1.26x long TPOT on Qwen3-8B, 1.25x on Llama-3.1-8B, with
TTFT almost unchanged. The rule needs only the current context length of each request. A version without oracle
input gave the same result.

## 3. Kernel use

### 3.1 The KV-split decision is tied to CUDA-graph eligibility

- **Mechanism.** `vllm/v1/attention/backends/flash_attn.py:805-815` lets FA3 split each context into a maximum of
  32 parts only if the step has 512 tokens or fewer (the CUDA-graph range). Above that, it passes
  `max_num_splits = 0`, which gives the decision to the FA3 heuristic. The source comment says splits need extra
  buffers, so vLLM allows them only for CUDA graphs.

### 3.2 The FA3 split heuristic counts tiles, not bytes

- **Mechanism.** The heuristic counts query tiles. With variable-length batches it treats the step as one sequence
  with the longest query: `tiles = h_kv x ceil(max_q x g / 128)`. If `tiles >= 0.8 x 132 SMs`, it selects 1 split.
  For Qwen3-8B (h_kv = 8, g = 4), any prefill chunk of 417 tokens or more triggers this. The heuristic never looks
  at the context length of decode rows.
- **Effect.** One thread block per KV head reads a long context alone at about 58 GB/s, and the step waits for it.
  Measured: 407 us against 85 us per attention layer (4.8x) for a step with one 32k-token decode. In the X-heavy
  workload, these steps used 56% of GPU time.

### 3.3 The same boundary removes CUDA graphs

- **Mechanism.** vLLM captures CUDA graphs for steps of up to 512 tokens. A larger step runs eagerly: the CPU
  launches every kernel.
- **Effect.** On our shared node, an eager step needs about 16.5 ms of CPU launch time, about equal to its GPU
  time. A kernel-only fix (C3) restores the split but not the graph, so it gains 1.18x where step shaping gains
  1.26x.

### 3.4 Outside the boundary, the default kernels are good

- Decode with FA3 is within 1-4% (geomean) of the best kernel that we tested for each batch shape.
- Decode on an 8B model is limited by weight reads (6.6 ms per step). Attention is a small part of the step, so a
  faster attention kernel changes the step by 5% or less.
- FlashInfer XQA, the default FlashInfer decode kernel on Hopper, is 1.7x slower than FA3 on skewed batches.
- Per-size GEMM selection gains 5% or less of a step.

### 3.5 What helps

| Fix | Type | Gain on X-heavy Qwen3-8B |
|---|---|---|
| `-cc '{"max_cudagraph_capture_size":2048}'` | Static flag | 1.14x (also 1.16x Llama, 1.17x Qwen3-4B) |
| Split decision based on the decode context length, not on graph eligibility (C3) | Kernel dispatch | 1.18x |
| Step shaping at the capture size | Scheduler | 1.26x |

The static flag is the simplest choice for users. For vLLM itself, two changes follow from these results: keep the
KV split available above the capture size for steps with long-context decodes, and let the scheduler know the
capture size.

### 3.6 An output-length oracle does not improve the split decision

- **Mechanism.** FA3 already chooses the number of KV splits for each sequence. In vllm-project/flash-attention
  (commit `506341a143fcabd4bb79052a7605ada727d6b3f5`), `hopper/flash_prepare_scheduler.cu:161` sets the split count
  of a sequence from the KV block count of that sequence. A step-level upper bound (`num_splits_static`) caps it.
  With an upper bound above 1, long-context decodes get many splits and short ones get 1. With an upper bound of 1,
  every sequence gets 1 split. So the question "enable the split for a request that will become long" becomes
  "keep the upper bound above 1 while a long-context decode is in the step".
- **Why the oracle does not help.** The split decision depends only on the context length at the current step.
  The engine knows this length without an oracle: it is the prompt length plus the tokens generated so far (vLLM
  keeps it in `seq_lens`). A request that will become long is still short now, and a split of a short context adds
  a combine step and gives no gain. The split count is also metadata that vLLM rebuilds on the CPU at every step.
  A change costs nothing, so foresight has no setup cost to plan around.
- **Evidence.** C3 (`regime_batching/attn/backend.py`) makes the decision reactively. It reads only CPU-side
  metadata of the current step (`seq_lens_cpu_upper_bound`, `query_start_loc_cpu`) and raises the upper bound on
  predicted X steps. It has a minimum-gain gate of 8 us per layer, and the decision costs about 23 us per step.
  Result on X-heavy Qwen3-8B: 1.18x +- 0.03 short-request E2E (3 repeats). An oracle version would make the same
  choice at every step. C3 still loses to step shaping (1.26x +- 0.03), because a correct split does not restore the
  CUDA graph of steps above 512 tokens. Step size also needs no oracle: step shaping with the oracle check gives
  1.26x, and a reactive version gives 1.27x (single run, compared with the mean of four B0 runs).
- **Rule.** The oracle helps decisions that are expensive to undo. It does not help decisions that can change at
  every step.

| Decision | Can it change at the next step? | Does the output-length oracle help? |
|---|---|---|
| Admission (KV memory) | No: undoing it needs preemption and recompute | Yes: 19-29x short-request E2E |
| Placement on an instance or lane | No: moving KV is expensive | Probably; not tested |
| Kernel KV-split count | Yes, at no cost | No (C3 reactive: 1.18x) |
| Step size (step shaping) | Yes | No (1.27x without, 1.26x with) |

Placement is the one case where "know in advance that a request will become long" applies. A request with a short
prompt and a long output looks short at arrival. See [future work item 6](06_limitations_and_future_work.md).

## 4. Summary

| Area | Practice today | Consequence | Fix that worked | Needs output lengths? |
|---|---|---|---|---|
| KV management | Admit on present memory; preempt and recompute later | Preemption storms, head-of-line blocking, 20-30x higher short latency under memory pressure | Future-peak KV plan with backfill | Yes |
| Scheduling | Fill steps to a fixed token budget | Steps cross the 512-token boundary next to long-context decodes | Step shaping | No |
| Kernels | Split only in the CUDA-graph range; FA3 heuristic counts tiles | Long-context decode straggler, eager launch cost | Larger capture size, split override, step shaping | No |
