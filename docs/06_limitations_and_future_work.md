# Limitations and future work

## Limitations of the evidence

1. **Oracle lengths.** Every result assumes exact output lengths. Real requests stop at an end-of-sequence token
   at an unknown time. These results are an upper bound for length-aware scheduling.
2. **Synthetic workloads.** We designed each workload to show one effect. The KV-plan gain exists only when the
   total KV demand is more than the cache. Without memory pressure, the plan has no effect (X-heavy: full 1.27x,
   shape alone 1.26x).
3. **Shared node.** The host had a load average of about 650. This makes eager steps more expensive than on a
   dedicated server, which increases the 512-token boundary cost and decreases the gain of kernel-only fixes.
4. **Number of runs.** Qwen3-8B X-heavy has 3 runs and the Qwen3-8B burst has 2 runs for B0 and peak. The
   Llama-3.1-8B and Qwen3-4B results are single runs.
5. **Dropped requests.** In the burst, B0 and the static configurations lost 12-16 requests to API-server
   disconnects under the backlog (peak lost 0-1). The tables exclude dropped requests, which makes the baselines
   look slightly better.
6. **Best-static search.** For the burst, we searched `--max-num-seqs` on Qwen3-8B only.
7. **One GPU, dense models.** All results are on one H100 with dense 8B and 4B models.
8. **Scheduler overhead.** The C4 wrapper measured 76 us per call before the planner was vectorized. We did not
   measure it again inside a running server.
9. **No shared prefixes.** Prefix caching was on, but the prompts are random tokens, so no two requests share a
   prefix. The study gives no result for shared system prompts, multi-turn chat or retrieval prompts. The KV
   planner counts the full prompt of each request (`regime_batching/oracle.py:66`). With shared prefixes, it
   counts a shared block once for each request that uses it, so it reserves too much memory.

## Future work, in order of priority

1. **Oracle tolerance.** Run the KV planner with noisy lengths (+-10%, 25%, 50%), with reservation at an upper
   quantile, and with a policy for requests that run past their prediction. This connects the upper bound to real
   length predictors.
2. **Recover the 0-7% throughput cost.** The planner knows when KV memory will become free. It can admit long
   requests into future gaps, or allow a small planned overcommit with CPU offload in place of recompute.
3. **Realistic memory pressure.** Production traces (Azure, BurstGPT, reasoning-length distributions), longer
   contexts, a 70B model on several GPUs, and goodput under latency targets.
4. **Cost-model step sizing.** Select each step size from the captured CUDA-graph sizes with the calibrated cost
   model. This addresses the Qwen3-4B case, where a larger capture size beats a 512-token limit.
5. **Upstream the 512-token boundary.** Report the split/graph coupling to vLLM. Two options: keep the KV split
   available above the capture size for steps with long-context decodes, or base the FA3 split decision on the
   decode context lengths.
6. **Across instances.** Route requests by their future KV peak and compare with length-grouped multi-instance
   systems such as CascadeInfer. The oracle matters for placement, because a request with a short prompt and a
   long output looks short at arrival. Route it to a long lane or instance at arrival, so that short-only
   instances never see X steps. We did not test this.
7. **Kernels.** Revisit regime-specific kernels for MoE models and very long contexts, where attention and GEMM
   shapes change more between steps.
8. **Prefix caching.** Make the KV planner count each shared block once, while at least one request uses it. The
   oracle gives the finish time of each of these requests, so the planner knows when the shared block becomes
   free. Test it on the burst with a shared 2,000-token system prompt and on a multi-turn workload. Compare B0, B0
   with `--no-enable-prefix-caching`, the current planner and the prefix-aware planner. Measure the cache hit
   rate, the recomputed tokens, short-request E2E and throughput. Note: the output-length oracle does not help the
   eviction choice. The best eviction needs the time when a future prompt uses a block again, which output lengths
   do not give.
