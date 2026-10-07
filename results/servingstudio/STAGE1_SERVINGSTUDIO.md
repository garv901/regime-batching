# Stage 0 / Stage 1: ServingStudio probe, kernel cross-check, timing predict

API `https://servingstudio.cs.washington.edu/api/public/v1`, `sim_commit` b87a574f0ae4d4b4fd10ce0eb54bc1d11c9c17be.
All raw responses are saved under `results/servingstudio/` (see "Files"). Numbers quoted from the API are marked as such; "(my arithmetic)" marks anything I computed.

Caveats up front:
- The GPU is **H200 (141 GB class), not H100**. Every kernel time, GEMM time and ratio is H200.
- Attention is modeled as **separate kernels** (`flashinfer_attn_prefill` for one request of prefix+append, `flashinfer_attn_decode` for a uniform-length batch). This is a regime-separated, best-backend counterfactual, not a fused/mixed-batch attention kernel as in vLLM's single ragged call.
- The decode kernel is measured with a **uniform batch** (kv_len = floor(total_tokens/batch_size)); skewed real batches can differ. The prefill kernel is measured for **one request, contiguous K/V**, not a paged cache.
- Kernel times are CUPTI kernel time with L2 flushed (cold cache). No CPU scheduling or launch gaps are in predictions.

## Part A. Stage 0 probe (simulate)

### Workload and run
- Converter: `experiments/servingstudio/tracelab_to_ss.py` (stdlib; `--in --out --max-rounds --max-model-len --seed`). Input: TraceLab `convert.py --provider claude --seed 0 --arrival-rate 0.05 --max-sessions 40` (2538 rounds, saved to /tmp, not kept). Output `workload_ss.csv`: **3 sessions, 50 rounds**, every round prefix+input+output <= 131072. Sessions are seeded-shuffled, truncated at first overflowing round, taken until 50 rounds.
- Format semantics learned from 400 errors (API text): (a) round 0 of a session must have `prefix_len` 0 ("a session's first round has no previous context to copy from"); my converter folds a nonzero round-0 prefix into `input_len` (one session, s0: 28527 prefix + 2026 input became 30553 input). (b) request_id must be `session_<sid>_round_<6 digits>`. (c) arrival_time_ms: I used the session arrival (20 s apart, 0.05 sessions/s) for every round of that session; later rounds chain on previous finish plus `tool_wait_after_ms`, as the skill says.
- Upload `de8896c2f61e40fd8fcec9f957f01671`: 50 requests, `prompt_tokens` 55424.6 mean (the API's number; it includes prefix), `output_tokens` 663.02, rate 0.05.
- Simulation `d89abfd7f8204baea143c316398e46e1`, preset `Llama-3.1-8B/llama3_dense_tp_barebone`, params `{"tp_size":1,"replicas":1}`, 1 GPU (H200), routing label "requests from your workload; the model routes no experts". Result (API): 50/50 finished, cause DrainComplete, sim_ms 735644.5; TTFT ms mean 439.0 / p50 169.5 / p90 1067.2 / p99 4715.8 / max 7679.0; TPOT ms mean 6.85 / p50 6.70 / p99 8.90 (n=49); e2e ms p50 1288.5 / max 95921.4; total_tok_s 521.3, prefill_tok_s 476.3, decode_tok_s 45.06.
- Files: `sim_record.json`, `an_descriptor.json`, `an/*.json` (every subject used).

### Analyzer surface
`GET analyzer/runs/{run_id}/descriptor` lists subjects (all `ready`): concurrency, request-state, slo-general, throughput, utilization, batch, kv-occupancy, kernel-input-distribution, kernel-time-share, scoped-optimality, optimality (+batch_locked variant), workload-conservation. It also lists "details" (worker-operation-index, worker-cost-tree, iteration-optimality-kernel-ladder, iteration-optimality-waterfall) but **every URL form I tried for those (`details/<name>/payload`, `subjects/<name>/payload`, `%2F`-encoded) returned 404**, so per-iteration operation detail was not reachable through the public route. Working URLs are `.../runs/{id}/subjects/<subject>/{report|payload}` (`subjects/summary/report` works; `summary/report` does not). `traces` is `{}`.

### Q1. Is prefix_len served as cached KV?  Yes (as a prefix-cache hit, not recomputed).
Evidence:
1. `subjects/workload-conservation/report` (all checks OK): `prefill_tokens` actual 350362 = expected 350362. My arithmetic: sum of `input_len` in my CSV is 350362 and sum of `prefix_len` is 2420868, so the prefill kernels processed only the new tokens. `prefix_token_balance` 2771230 = 350362 + 2420868 (hit + computed = fresh + declared). `num_reprocessed_prefill_episodes` 0. The report's definitions name `request_slo.prefix_cache_hit_tokens` ("admission-time resident-prefix observation") and `declared_prefix_tokens`.
2. `topology` payload: worker `prefix_cache_mode: "opportunistic"`, `prefix_cache_policy: "lru"`. `kv-occupancy` has a retained-prefix series (`peak_retained_prefix_mean_tokens` 123421.9 of `capacity_tokens` 610351).
3. Controlled TTFT probe (3 tiny uploads, same preset; `ttft_probe/`). The main run does not expose per-request TTFT (only distribution summaries), so I used single-purpose workloads:

| workload | API TTFT (ms) |
|---|---|
| B: one cold request, input 126040, prefix 0 | 8929.5 |
| A round 0: cold input 125000 (API `ttft max` of a 2-request run) | 8807.7 |
| A round 1: prefix 125016 (cached), input 1024 | 124.9 (my arithmetic: 2 x mean 4466.3 - max 8807.7; the API returns only mean/p50/max for n=2) |
| C: one cold request, input 1024 | 23.0 |

A's `prefill_tokens` (subject `summary`) is 126025 = 125000 + 1024 + 1 decode token, i.e. the 125016 cached prefix was not recomputed. So a 126k-context round with 1k append costs ~125 ms TTFT here versus ~8.9 s if the same context were prefilled cold. The cached round is slower than a cold 1k prompt (23 ms) because attention reads the 125k prefix.
Caveat: this holds while the prefix stays resident (LRU, ~610k-token KV budget, `attn_gpu_memory_gb` 80). Eviction/reprocess is modeled (`num_reprocessed_prefill_episodes`), none happened in this run.

### Q2. Is per-iteration batch composition available?  Only partially, sampled, no per-step KV lengths.
- `subjects/batch/payload`: per pool/worker scatter of `time_ms`, `batch_tokens`, `prefill_tokens`, `decode_request_count` per iteration. It is a sample: stats use "regular iter_id-stride sample" (stride 50 elsewhere; `n` 4536 of `num_calls` 31756), and the scatter is downsampled to 2268 points. Report: batch_tokens mean 29.9, p50 1, max 115017; decode_request_count max 2.
- `subjects/kernel-input-distribution/payload`: for the decode attention position, sampled (batch_size, total_tokens) points (e.g. (1, 111224)). So decode batch size and summed KV length per step are available for sampled steps, but not the per-request KV list. The prefill position (`unified.attn_block.attn.prefill`) was **omitted**: "never executed in the sampled iterations".
- No endpoint returned (prefix_len, append_len) of each prefill chunk or per-request kv_lens per step. `kv-occupancy` is a binned KV-token time series. The unreachable `worker-operation-index` / `worker-cost-tree` details might hold it; I could not fetch them.
- The decode attention backend picked in the sim was `fa2` (635/635 sampled slots; candidates fa2, fa3).

### What `barebone` does (from topology, batch, and conservation subjects)
- Worker params: `attn_gpu_memory_gb` 80, `gpu_time_multiplier` 1.0, `pending_order` "session-start", prefix cache opportunistic/LRU; KV capacity 610351 tokens per shard.
- Steps can mix prefill and decode: the sampled scatter contains one mixed point (prefill_tokens 653, decode_request_count 1, batch_tokens 654).
- No visible chunked-prefill cap: the largest iteration is `batch_tokens` max 115017 with `prefill_tokens` max 115017, i.e. a whole 115k-token prompt in one iteration (decodes stall behind it; main-run TTFT max 7679 ms). I did not find a `max_num_batched_tokens` field; the unchunked behaviour is inferred from that max, not stated by the API.
- Iteration-wise logging: `num_iterations` 31756, `num_layers` 0 in conservation meta.

## Part B. Stage 1 kernel cross-check

Kind `flashinfer_attn_prefill`, GPU `NVIDIA H200`, num_qo_heads 32, num_kv_heads 8, head_dim 128, q/kv/o bf16 (coverage has fa2 and fa3 with 1080 bf16 rows each for all head configs; for this config 135 rows per backend). Rows `profiler_run_at` 2026-05-24 (CUDA 12.8, driver 570.211.01). `trt` records no rows; `cudnn` has no H200 rows.
**Method (API text):** CUPTI kernel time, L2 flushed; "fa2 first times the default plan and fixed KV split sizes of 1,024 to 16,384 tokens below kv_len, then measures the fastest." So fa2 is the best of several split-KV configurations; fa3 gets one configuration.
**Grid limit:** the measured grid has prefix_len + append_len <= ~65536 (max prefix_len 65472 at append 128). **There are no rows at 128k prefix**; the 128k column is therefore absent from the table (the timing predictor extrapolates there, Part C). The grid is sparse and differs per append (e.g. append 128: prefix 0..65472; append 8192: prefixes 0, 4096, 12288, ..., 61440), so cells are the nearest measured pair, marked `*` when not exact.

Prefill fa2 vs fa3 (time in ms, TFLOPS from the API's `tflops` column; ratio = fa2_ms/fa3_ms, my arithmetic; `*` = nearest measured cell):

| target P | target A | measured (prefix, append) | fa2 ms | fa3 ms | fa2 TFLOPS | fa3 TFLOPS | fa2/fa3 time | faster |
|---|---|---|---|---|---|---|---|---|
| 0 | 128 | (0, 128) | 0.0083 | 0.0072 | 16 | 19 | 1.15 | fa3 |
| 8192 | 128 | (8128, 128) * | 0.0668 | 0.1017 | 257 | 169 | 0.66 | fa2 |
| 32768 | 128 | (32704, 128) * | 0.2073 | 0.4292 | 332 | 160 | 0.48 | fa2 |
| 65536 | 128 | (65472, 128) * | 0.3775 | 0.7826 | 364 | 176 | 0.48 | fa2 |
| 0 | 256 | (0, 256) | 0.0333 | 0.0092 | 16 | 58 | 3.61 | fa3 |
| 8192 | 256 | (8064, 256) * | 0.1249 | 0.1040 | 275 | 330 | 1.20 | fa3 |
| 32768 | 256 | (32640, 256) * | 0.3870 | 0.4331 | 355 | 317 | 0.89 | fa2 |
| 65536 | 256 | (65408, 256) * | 0.7277 | 0.8921 | 378 | 308 | 0.82 | fa2 |
| 0 | 512 | (0, 512) | 0.0641 | 0.0132 | 34 | 163 | 4.87 | fa3 |
| 8192 | 512 | (7936, 512) * | 0.2347 | 0.1195 | 293 | 575 | 1.96 | fa3 |
| 32768 | 512 | (32512, 512) * | 0.7552 | 0.4264 | 364 | 645 | 1.77 | fa3 |
| 65536 | 512 | (65280, 512) * | 1.4501 | 0.8669 | 379 | 634 | 1.67 | fa3 |
| 0 | 1024 | (0, 1024) | 0.0505 | 0.0301 | 170 | 285 | 1.68 | fa3 |
| 8192 | 1024 | (7680, 1024) * | 0.3804 | 0.2271 | 361 | 605 | 1.67 | fa3 |
| 32768 | 1024 | (32256, 1024) * | 1.3975 | 0.8346 | 393 | 659 | 1.67 | fa3 |
| 65536 | 1024 | (65024, 1024) * | 2.9129 | 1.6733 | 377 | 657 | 1.74 | fa3 |
| 0 | 2048 | (0, 2048) | 0.1448 | 0.0776 | 237 | 443 | 1.87 | fa3 |
| 8192 | 2048 | (7168, 2048) * | 0.7649 | 0.4674 | 359 | 588 | 1.64 | fa3 |
| 32768 | 2048 | (31744, 2048) * | 2.9054 | 1.5621 | 378 | 704 | 1.86 | fa3 |
| 65536 | 2048 | (64512, 2048) * | 5.8148 | 3.3435 | 378 | 658 | 1.74 | fa3 |
| 0 | 8192 | (0, 8192) | 1.4977 | 0.8391 | 367 | 655 | 1.78 | fa3 |
| 8192 | 8192 | (4096, 8192) * | 2.8616 | 1.5785 | 384 | 697 | 1.81 | fa3 |
| 32768 | 8192 | (28672, 8192) * | 11.7274 | 6.4352 | 375 | 683 | 1.82 | fa3 |
| 65536 | 8192 | (61440, 8192) * | 22.6431 | 13.7692 | 388 | 639 | 1.64 | fa3 |

Full 135-cell join: `kernels/prefill_all_cells.csv`; table CSV: `kernels/prefill_table.csv`. Target append 417-530 maps to the measured 512 column (no 417-530 value other than 512 exists in the grid).

**Where fa2's split-KV beats fa3.** Across all 135 measured cells, fa2 is faster in only 7 (my count from the rows):

| prefix | append | fa2 ms | fa3 ms | fa2/fa3 |
|---|---|---|---|---|
| 4032 | 128 | 0.0459 | 0.0570 | 0.81 |
| 8128 | 128 | 0.0668 | 0.1017 | 0.66 |
| 16320 | 128 | 0.1145 | 0.1996 | 0.57 |
| 32704 | 128 | 0.2073 | 0.4292 | 0.48 |
| 65472 | 128 | 0.3775 | 0.7826 | 0.48 |
| 32640 | 256 | 0.3870 | 0.4331 | 0.89 |
| 65408 | 256 | 0.7277 | 0.8921 | 0.82 |

So fa2 wins only for **long prefix with a tiny append (128 to 256 tokens)**: that regime is memory-bound with too few query blocks for fa3 to fill the SMs, and fa2's KV split recovers parallelism (fa2 reaches 332-378 TFLOPS there vs fa3's 160-176 at append 128). At append >= 512 fa3 wins everywhere measured, by about 1.6-2.0x at long prefix (fa3 ~575-700 TFLOPS vs fa2 ~360-390; fa2 sits at ~360-390 TFLOPS whenever it is compute-bound). At prefix 0 and small append fa3 also wins (0.0072 vs 0.0083 ms at 128). At append 512 and above, our ~530-token append case, fa3 is about 1.6-2.0x faster than fa2 at 8k-65k prefix (my reading of the table above). No cell exists for 128k prefix, so whether fa2's split-KV win extends to 128k prefix with 256-530 append is untested in the Kernel Library (Part C's predictor picked fa2 for P=126976, A=256 by extrapolation).

### Decode (`flashinfer_attn_decode`, same heads, bf16, H200)
Method (API text): paged cache, 16-token pages, FlashInfer with tensor cores; GB/s counts all allocated pages; fa3 and fa2_cudagraph run once before timing; fa2_cudagraph uses the CUDA-graph plan that splits the KV and adds a merge kernel (kernels launched directly, no captured graph replay). Uniform batch, kv_len = floor(total_tokens/batch_size). Grid: batch_size 1..256, total_tokens 32..4194304. Rows below are batch 1,2,4,8 (full CSV `kernels/decode_table.csv`).

| batch | total_tokens | fa3 ms | fa2 ms | fa2_cudagraph ms | fa3 GB/s | fa2 GB/s | fa2_cg GB/s | best |
|---|---|---|---|---|---|---|---|---|
| 1 | 8192 | 0.1140 | 0.0224 | 0.0205 | 295 | 1497 | 1640 | fa2_cudagraph |
| 1 | 32768 | 0.4332 | 0.0502 | 0.0434 | 310 | 2675 | 3094 | fa2_cudagraph |
| 1 | 131072 | 1.7068 | 0.1416 | 0.1339 | 315 | 3792 | 4008 | fa2_cudagraph |
| 1 | 262144 | 3.4061 | 0.2598 | 0.2538 | 315 | 4132 | 4231 | fa2_cudagraph |
| 1 | 524288 | 6.7881 | 0.4981 | 0.4914 | 316 | 4311 | 4370 | fa2_cudagraph |
| 1 | 1048576 | 13.5778 | 0.9732 | 0.9707 | 316 | 4413 | 4425 | fa2_cudagraph |
| 2 | 8192 | 0.0602 | 0.0223 | 0.0202 | 558 | 1508 | 1661 | fa2_cudagraph |
| 2 | 32768 | 0.2223 | 0.0494 | 0.0435 | 604 | 2719 | 3083 | fa2_cudagraph |
| 2 | 131072 | 0.8651 | 0.1398 | 0.1322 | 621 | 3841 | 4061 | fa2_cudagraph |
| 2 | 262144 | 1.7082 | 0.2585 | 0.2534 | 629 | 4154 | 4238 | fa2_cudagraph |
| 2 | 524288 | 3.4133 | 0.4966 | 0.4934 | 629 | 4324 | 4352 | fa2_cudagraph |
| 2 | 1048576 | 6.8065 | 0.9735 | 0.9741 | 631 | 4412 | 4409 | fa2 |
| 4 | 8192 | 0.0357 | 0.0221 | 0.0205 | 941 | 1521 | 1644 | fa2_cudagraph |
| 4 | 32768 | 0.1213 | 0.0496 | 0.0433 | 1107 | 2706 | 3100 | fa2_cudagraph |
| 4 | 131072 | 0.4460 | 0.1402 | 0.1331 | 1204 | 3830 | 4034 | fa2_cudagraph |
| 4 | 262144 | 0.8746 | 0.2582 | 0.2516 | 1228 | 4158 | 4267 | fa2_cudagraph |
| 4 | 524288 | 1.7208 | 0.4972 | 0.4938 | 1248 | 4319 | 4349 | fa2_cudagraph |
| 4 | 1048576 | 3.4477 | 0.9770 | 0.9699 | 1246 | 4396 | 4429 | fa2_cudagraph |
| 8 | 8192 | 0.0381 | 0.0223 | 0.0205 | 884 | 1512 | 1645 | fa2_cudagraph |
| 8 | 32768 | 0.1254 | 0.0498 | 0.0435 | 1072 | 2698 | 3086 | fa2_cudagraph |
| 8 | 131072 | 0.4504 | 0.1400 | 0.1331 | 1192 | 3836 | 4034 | fa2_cudagraph |
| 8 | 262144 | 0.8807 | 0.2577 | 0.2535 | 1219 | 4168 | 4237 | fa2_cudagraph |
| 8 | 524288 | 1.7303 | 0.4977 | 0.4945 | 1241 | 4315 | 4343 | fa2_cudagraph |
| 8 | 1048576 | 3.5717 | 0.9728 | 0.9815 | 1203 | 4415 | 4376 | fa2 |

Reading (my arithmetic on API rows): fa3 decode at batch 1 is ~315 GB/s regardless of length (1.71 ms at 131072 tokens vs fa2 0.142 ms, a 12x gap), because with one request fa3 apparently does not split the KV (inferred from the throughput, not stated by the API); fa2 and fa2_cudagraph reach ~3.8-4.4 TB/s at 131k-1M tokens. fa3 catches up only with larger batches (1.2 TB/s at batch 4-8) but still trails fa2 (best of fa2/fa2_cudagraph is chosen in all 24 rows shown). A single 126976-token decode costs 0.14 ms/layer in fa2 (rows: batch 1, 131072 tokens), about 4.4 ms over 32 layers (my arithmetic), matching Part C's 4.410 ms.

## Part C. Stage 1 timing predict

Preset `Llama-3.1-8B/llama3_dense_tp`, params `{"tp_size":1}`, H200, 1 GPU, selector `iter`, 1 group, `max_model_len` 131072, `missing` empty, `analyze: true`. `prediction_id` p_b61dc7f8078e43bc9a2ba7d2a6a04692 (Analyzer keeps it 6 h), `sim_commit` b87a574f0ae4d4b4fd10ce0eb54bc1d11c9c17be. 37 cases in one POST (`predict_request.json` -> `prediction.json`).

**Deviation from the brief:** prefix 126976 + append 8192 = 135168 exceeds max_model_len 131072 (API 400: "case 14: prefill context 126976 + 8192 = 135168 exceeds max_model_len 131072"). For P=126976 I used append 4096 instead of 8192 (126976+4096 = 131072) in both the prefill-only and prefill+decode groups.

Columns: total, attention-prefill (slot 4 leaf `total_ms`), attention-decode (slot 5), GEMM (sum of the five `single_gemm` leaves: qkv_proj, o_proj, up_gate_proj, down_proj, lm_head), other (my arithmetic: total minus the three; embedding, norms, kv_cache_append, activation). Leaf `total_ms` already includes the 32-layer scaling. Slot backend = `slot_backend` of slot 4 / 5; GEMM leaves chose `torch`/`torch_linear`. "extrapolated" lists slot indices from `coverage` (slot 4 = prefill attention; slots 0,1,7,9,11 = embedding, input_norm, post_norm, activation, final_norm, which lie below the measured m grid in decode-only batches).

| case | total ms | attn prefill | attn decode | GEMM | other | prefill be | decode be | extrapolated slots |
|---|---|---|---|---|---|---|---|---|
| pf[8192,256] | 10.051 | 3.382 | 0.000 | 6.099 | 0.570 | fa3 | - | none |
| pf[8192,530] | 16.190 | 4.399 | 0.000 | 10.874 | 0.916 | fa3 | - | none |
| pf[8192,1024] | 29.715 | 7.737 | 0.000 | 20.500 | 1.479 | fa3 | - | none |
| pf[8192,2048] | 60.202 | 16.757 | 0.000 | 40.794 | 2.651 | fa3 | - | none |
| pf[8192,8192] | 249.440 | 75.188 | 0.000 | 164.565 | 9.688 | fa3 | - | none |
| pf[32768,256] | 19.095 | 12.426 | 0.000 | 6.099 | 0.570 | fa2 | - | none |
| pf[32768,530] | 27.264 | 15.473 | 0.000 | 10.874 | 0.916 | fa3 | - | none |
| pf[32768,1024] | 49.106 | 27.128 | 0.000 | 20.500 | 1.479 | fa3 | - | none |
| pf[32768,2048] | 95.214 | 51.769 | 0.000 | 40.794 | 2.651 | fa3 | - | none |
| pf[32768,8192] | 409.515 | 235.262 | 0.000 | 164.565 | 9.688 | fa3 | - | none |
| pf[126976,256] | 50.439 | 43.770 | 0.000 | 6.099 | 0.570 | fa2 | - | extrapolated:[4] |
| pf[126976,530] | 71.860 | 60.070 | 0.000 | 10.874 | 0.916 | fa3 | - | extrapolated:[4] |
| pf[126976,1024] | 126.268 | 104.289 | 0.000 | 20.500 | 1.479 | fa3 | - | extrapolated:[4] |
| pf[126976,2048] | 259.099 | 215.654 | 0.000 | 40.794 | 2.651 | fa3 | - | extrapolated:[4] |
| pf[126976,4096] | 506.609 | 420.008 | 0.000 | 81.580 | 5.021 | fa3 | - | extrapolated:[4] |
| pf[32768,530]+1x126976dec | 31.698 | 15.473 | 4.410 | 10.897 | 0.917 | fa3 | fa2 | none |
| pf[32768,530]+3x126976dec | 39.068 | 15.473 | 11.727 | 10.948 | 0.920 | fa3 | fa2 | none |
| pf[32768,1024]+1x126976dec | 53.540 | 27.128 | 4.410 | 20.523 | 1.480 | fa3 | fa2 | none |
| pf[32768,1024]+3x126976dec | 60.910 | 27.128 | 11.727 | 20.572 | 1.483 | fa3 | fa2 | none |
| pf[32768,8192]+1x126976dec | 413.946 | 235.262 | 4.410 | 164.585 | 9.689 | fa3 | fa2 | none |
| pf[32768,8192]+3x126976dec | 421.310 | 235.262 | 11.727 | 164.630 | 9.691 | fa3 | fa2 | none |
| pf[126976,530]+1x126976dec | 76.294 | 60.070 | 4.410 | 10.897 | 0.917 | fa3 | fa2 | extrapolated:[4] |
| pf[126976,530]+3x126976dec | 83.664 | 60.070 | 11.727 | 10.948 | 0.920 | fa3 | fa2 | extrapolated:[4] |
| pf[126976,1024]+1x126976dec | 130.702 | 104.289 | 4.410 | 20.523 | 1.480 | fa3 | fa2 | extrapolated:[4] |
| pf[126976,1024]+3x126976dec | 138.072 | 104.289 | 11.727 | 20.572 | 1.483 | fa3 | fa2 | extrapolated:[4] |
| pf[126976,4096]+1x126976dec | 511.050 | 420.008 | 4.410 | 81.610 | 5.022 | fa3 | fa2 | extrapolated:[4] |
| pf[126976,4096]+3x126976dec | 518.435 | 420.008 | 11.727 | 81.675 | 5.025 | fa3 | fa2 | extrapolated:[4] |
| dec1x126976 | 8.688 | 0.000 | 4.410 | 3.942 | 0.336 | - | fa2 | extrapolated:[0, 1, 7, 9, 11] |
| dec3x126976 | 16.002 | 0.000 | 11.727 | 3.938 | 0.337 | - | fa2 | extrapolated:[0, 1, 7, 9, 11] |
| dec8x126976 | 34.457 | 0.000 | 30.180 | 3.934 | 0.342 | - | fa2 | extrapolated:[0, 1, 7, 9, 11] |
| dec1x32768 | 5.884 | 0.000 | 1.606 | 3.942 | 0.336 | - | fa2 | extrapolated:[0, 1, 7, 9, 11] |
| dec3x32768 | 7.794 | 0.000 | 3.519 | 3.938 | 0.337 | - | fa2 | extrapolated:[0, 1, 7, 9, 11] |
| dec8x32768 | 12.522 | 0.000 | 8.246 | 3.934 | 0.342 | - | fa2 | extrapolated:[0, 1, 7, 9, 11] |
| cold[0,8192] | 201.103 | 26.851 | 0.000 | 164.565 | 9.688 | fa3 | - | none |
| cold[8192,8192] | 249.440 | 75.188 | 0.000 | 164.565 | 9.688 | fa3 | - | none |
| cold[0,8192]+3x126976dec | 212.898 | 26.851 | 11.727 | 164.630 | 9.691 | fa3 | fa2 | none |
| cold[8192,8192]+3x126976dec | 261.235 | 75.188 | 11.727 | 164.630 | 9.691 | fa3 | fa2 | none |

All the prefill attention numbers at P=126976 are extrapolated beyond the prefill kernel grid (max prefix 65472), so treat them as less certain than the P=8192 and P=32768 ones. `slot_backend` of the prefill slot: fa2 only for append 256 at P=32768 and P=126976 (the same small-append regime as Part B); fa3 for every other prefill case. Decode attention chose fa2 in every case.

### Mixed vs separated vs decode-only (my arithmetic on API totals)
Separated = prefill-only total + decode-only total, each a full iteration (each pays the dense GEMM/weights once). Ratio columns are mixed/(sum) and mixed/decode-only.

| mixed case | mixed ms | prefill-only ms | decode-only ms | mixed - (pf+dec) ms | mixed/(pf+dec) | mixed/decode-only |
|---|---|---|---|---|---|---|
| pf[32768,530]+1x126976dec | 31.698 | 27.264 | 8.688 | -4.254 | 0.882 | 3.65 |
| pf[32768,530]+3x126976dec | 39.068 | 27.264 | 16.002 | -4.197 | 0.903 | 2.44 |
| pf[32768,1024]+1x126976dec | 53.540 | 49.106 | 8.688 | -4.254 | 0.926 | 6.16 |
| pf[32768,1024]+3x126976dec | 60.910 | 49.106 | 16.002 | -4.198 | 0.936 | 3.81 |
| pf[32768,8192]+1x126976dec | 413.946 | 409.515 | 8.688 | -4.257 | 0.990 | 47.64 |
| pf[32768,8192]+3x126976dec | 421.310 | 409.515 | 16.002 | -4.207 | 0.990 | 26.33 |
| pf[126976,530]+1x126976dec | 76.294 | 71.860 | 8.688 | -4.254 | 0.947 | 8.78 |
| pf[126976,530]+3x126976dec | 83.664 | 71.860 | 16.002 | -4.197 | 0.952 | 5.23 |
| pf[126976,1024]+1x126976dec | 130.702 | 126.268 | 8.688 | -4.254 | 0.968 | 15.04 |
| pf[126976,1024]+3x126976dec | 138.072 | 126.268 | 16.002 | -4.198 | 0.970 | 8.63 |
| pf[126976,4096]+1x126976dec | 511.050 | 506.609 | 8.688 | -4.247 | 0.992 | 58.82 |
| pf[126976,4096]+3x126976dec | 518.435 | 506.609 | 16.002 | -4.176 | 0.992 | 32.40 |
| cold[0,8192]+3x126976dec | 212.898 | 201.103 | 16.002 | -4.207 | 0.981 | 13.30 |
| cold[8192,8192]+3x126976dec | 261.235 | 249.440 | 16.002 | -4.207 | 0.984 | 16.33 |

Reading:
- A mixed step costs about prefill-only plus the decode attention: the difference mixed - prefill-only is 4.41 ms for 1 decode at 126976 and 11.73 ms for 3 (the decode attention leaf) plus <= 0.06 ms of extra GEMM/other work. The batched GEMMs barely change with 1-3 extra rows.
- Mixed is 4.2-4.3 ms below separated for every case (my arithmetic: -4.176 to -4.257 ms), which is the decode-only step's GEMM+other (3.94 + 0.34 ms of weight-bound work) paid once instead of twice. In relative terms that is 1% (8192-token appends) to 12% (32768+530) of the separated total.
- Versus the decode-only step with the same decodes, a mixed step takes 2.4x (32768+530, 3 decodes) up to 58.8x (126976+4096, 1 decode). Decodes sharing a step with a 1k-token in-session append at 126976 wait 130.7 ms (1 decode) or 138.1 ms (3 decodes) versus 8.7 or 16.0 ms alone: 15.0x and 8.6x.
- Decode-only: 1x126976 = 8.688 ms (decode attention 4.410), 3x = 16.002 (11.727), 8x = 34.457 (30.180); 3x32768 = 7.794 ms. Decode attention scales ~linearly with batch (3.9 ms/req of 126976-length KV), so decode-only time is dominated by KV reads when the context is ~127k.
- Cold [0,8192] mixed with 3 decodes (212.9 ms) vs cold [8192,8192] (261.2 ms): the 8192 cached prefix adds 48.3 ms of attention to the chunk (75.19 vs 26.85 ms, my arithmetic).
- At the prefill-only side the 1k append on a 126976 prefix is 126.3 ms, of which attention is 104.3 ms (83%) and GEMM 20.5 ms; at append 8192 on a 32768 prefix GEMM is 40% (164.6 of 409.5 ms).

### Optimality waterfall (`cases/{case}/subjects/optimality-waterfall/payload`)
Buckets are in GPU-seconds (1 GPU); I show ms and share of total (my arithmetic). `meta`: gpu H200-SXM-141GB spec, peaks_source "sidecar", `necessary_work_available` true, **`necessary_work_mode` "replicated_large_batch"** with `necessary_work_replication_factor` 10000, `batch_size_locked` false, `caveats` [] (empty). The mode name means the necessary work is estimated from a replicated large batch rather than labeled exactly for this iteration; read `hardware_necessary` and `batching` as model-based estimates. Buckets `fusion` and `excess_over_necessary` also appear (not in the skill's list).

| bucket (ms, % of total) | case 24: 126976+1024 with 3 decodes | case 35: cold [0,8192] with 3 decodes | case 28: decode-only 3x126976 |
|---|---|---|---|
| total | 138.072 | 212.898 | 16.002 |
| hardware_necessary | 83.823 (60.7%) | 133.525 (62.7%) | 10.402 (65.0%) |
| hardware_gap | 32.650 (23.6%) | 53.254 (25.0%) | 0.675 (4.2%) |
| batching | 10.522 (7.6%) | 8.927 (4.2%) | 4.876 (30.5%) |
| fusion | 10.228 (7.4%) | 10.424 (4.9%) | 0.045 (0.3%) |
| excess_over_necessary | 0.848 (0.6%) | 6.768 (3.2%) | 0.003 (0.0%) |
| communication, imbalance, idle | 0 | 0 (imbalance 0) | 0 |

(Case 24 imbalance 4.5e-9 s, zero for practical purposes.) The decode-only step is 30.5% "batching" loss: three rows leave the weight-bound GEMMs under-batched. In the mixed steps that loss falls to 4-8%, which is the batching benefit of mixing.

## Files (all under /mnt/storage/garv901/regime-batching)
- Scripts: `experiments/servingstudio/tracelab_to_ss.py`, `kernel_tables.py`, `predict_cases.py`, `predict_tables.py`.
- Part A: `results/servingstudio/workload_ss.csv`, `upload_response.json`, `sim_post.json`, `sim_record.json`, `an_descriptor.json`, `an/*.json`, `ttft_probe/*`, `presets.json`, `workloads_desc.json`, `openapi.json`.
- Part B: `results/servingstudio/kernels/` (`catalog.json`, `*.meta.json`, `prefill_rows.json`, `decode_rows.json`, `prefill_table.{csv,md}`, `prefill_all_cells.csv`, `prefill_fa2_wins.json`, `decode_table.{csv,md}`, `tree_llama_tp1.json`).
- Part C: `results/servingstudio/predict_request.json`, `predict_case_names.json`, `prediction.json`, `predict_table.{csv,md}`, `predict_mixed_table.md`, `waterfall/case{24,35,28}.json`.
