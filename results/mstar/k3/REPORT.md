# K3: does FlashInfer's decode split-KV choice leave a straggler in mstar?

Hypothesis K3: mstar sets no split-KV options and `backend="auto"` may pick a slow decode kernel for skewed batches (one long-context row plus many short rows).

## Verdict

- `auto` is not a slow kernel. On sm90 with bf16 it resolves to `fa2` (FlashInfer 0.7.0.post1, `decode.py:1937-1953`; confirmed by `resolved_backend` on every row and by the kernel names below). fa3 is chosen automatically only for FP8 KV. mstar's decode wrapper runs the FA2 batch-prefill kernel with one query row per request plus a `PersistentVariableLengthMergeStatesKernel` merge.
- A straggler exists, but only in one corner: bs=64 with one 32k row among 63 short rows. FlashInfer's default planner spends its grid budget on the short rows and gives the long row about 3 CTAs of 11008 tokens each. At bs <= 32 the planner balances well (chunks of 512 to 1920 tokens) and no knob helps by more than 1.02x on a skewed batch.
- Gain at bs=64, 1x32k + 63x512 (bagel, per attention layer): graph 156 us -> 72 us (`fixed_split_size=16`, 2.2x) and eager 183 us -> 70 us (`fixed_split_size=8`, 2.6x). That is 2.3 ms (8.0%) of a ~29 ms BAGEL step in graph mode and 3.2 ms (10.8%) in eager mode. With 63x2048 shorts: 1.5x (graph), 1.7x (eager), 6.4% and 8.8% of the step.
- The per-layer gain clears the 10% bar on 2 of 17 compositions (both bs=64, one 32k row), in both modes and both geometries. So the opt-in knob was wired (see "What was wired"). A static knob is a blunt tool, though: it loses on other batches. I do not recommend enabling it globally. See "Is K3 worth wiring".

## What `auto` selects

| item | finding |
|---|---|
| resolved backend (bf16, sm90) | `fa2` |
| decode kernel | `flashinfer::BatchPrefillWithPagedKVCacheKernel<true, KernelTraits<MaskMode 0, CTA_TILE_Q=16, ... bf16>>` then `PersistentVariableLengthMergeStatesKernel<8,16,8,4,bf16,bf16,int>` |
| `fa3` | `PrefillWithKVCacheKernel<SparseCollectiveMainloop ...>` on sm90; 1.1x to 9.2x slower than auto here (about equal at bs=1, 8x slower with a 32k row; FlashInfer's SM90 path has no split-KV) |
| `fa2_notc` (use_tensor_cores=False) | raises `Unsupported group_size: 7` for BAGEL (28/4/128); runs for Omni (32/4/128, group 8) |
| `trtllm-gen`, `cute-dsl`, `prims-ts`, `cutile` | all raise on this node: Blackwell-only (`Unsupported architecture`), page_size must be <= 64, needs HND layout, no `tileiras` compiler |

Source-level facts (file:line citations, the plan() kwarg list, the planner arithmetic, what graph mode does to the split decision) are in `perf_testing/k3_decode_split/NOTES.md` in the worktree.

## The straggler, by planner arithmetic

The default planner picks the smallest KV chunk such that the number of (row, chunk) tiles is at most `2 * num_sm / num_kv_heads = 66`. Each short row needs at least one tile, so at bs=64 the 63 short rows take 63 tiles and the 32k row is split into 3. Measured: chunk 11008 tokens, 66 tiles, 729 GB/s effective (versus 1500 to 2700 GB/s on balanced batches). At bs=32 the same long row gets 32 chunks of 1024 tokens (63 tiles) and runs at 1761 GB/s.

## Measured: us per attention layer (GPU time of one `run()`, KV from HBM, 128 MB L2 flush subtracted)

Bagel (28/4/128, 28 layers), graph mode:

| composition | bs | auto | nosplit | fix8 | fix16 | fix32 | fix64 | fa3 |
|---|---|---|---|---|---|---|---|---|
| 1x32768+7x512 | 8 | 49 | 433 | 49 | 54 | 73 | 125 | 434 |
| 1x32768+31x512 | 32 | 57 | 410 | 57 | 62 | 82 | 133 | 496 |
| 2x32768+30x512 | 32 | 80 | 447 | raises | 79 | 91 | 136 | 513 |
| **1x32768+63x512** | 64 | **156** | 422 | raises | **72** | 95 | 146 | 571 |
| **1x32768+63x2048** | 64 | **208** | 484 | raises | **142** | 144 | 180 | 888 |
| uniform 64x2048 | 64 | 112 | 115 | raises | 115 | 115 | 115 | 460 |
| uniform 8x32768 | 8 | 198 | 448 | raises | raises | 197 | 206 | 868 |

Bagel, eager mode:

| composition | bs | auto | nosplit | fix8 | fix16 | fix32 | fix64 | fa3 |
|---|---|---|---|---|---|---|---|---|
| 1x32768+7x512 | 8 | 48 | 366 | 48 | 50 | 66 | 108 | 434 |
| 1x32768+31x512 | 32 | 57 | 367 | 57 | 60 | 75 | 115 | 493 |
| 2x32768+30x512 | 32 | 79 | 358 | 80 | 79 | 90 | 118 | 506 |
| **1x32768+63x512** | 64 | **183** | 373 | **70** | 71 | 86 | 146 | 568 |
| **1x32768+63x2048** | 64 | **231** | 424 | **139** | 140 | 140 | 194 | 886 |
| uniform 64x2048 | 64 | 108 | 108 | 117 | 108 | 108 | 108 | 456 |
| uniform 8x32768 | 8 | 197 | 365 | 204 | 200 | 198 | 196 | 868 |

Omni Thinker geometry (32/4/128, 48 layers): every fa2 cell is within 3 us of Bagel's, except fix32 and fix64 at graph 1x32768+63x512 (8 and 19 us) and graph nosplit at 1x16k+31x512 (14 us). fa3 is up to 115 us slower on Omni. fix8 also works in graph mode at 1x32768+63x512 (71 us), because group 8 gives a larger padded grid (95 instead of 91). Full tables are in `summary.md`.

Estimated share of a decode step (BAGEL ~29 ms at bs=64; the attention cost of the whole skewed step, auto, is 4.4 ms graph and 5.1 ms eager, 28 layers x per-layer us):

| composition | mode | knob | per-layer gain | saved per step | share of 29 ms |
|---|---|---|---|---|---|
| 1x32768+63x512 | graph | fix16 | 156 -> 72 us | 2.3 ms | 8.0% |
| 1x32768+63x512 | graph | fix32 (never raised) | 156 -> 95 us | 1.7 ms | 5.9% |
| 1x32768+63x512 | eager | fix8 | 183 -> 70 us | 3.2 ms | 10.8% |
| 1x32768+63x2048 | graph | fix16 | 208 -> 142 us | 1.9 ms | 6.4% |
| 1x32768+63x2048 | eager | fix8 | 231 -> 139 us | 2.6 ms | 8.8% |
| every other bs<=32 skewed composition | both | any | <= 0.5 us | < 0.02 ms | < 0.1% |

Omni at the same compositions saves 4.0 ms (graph) and 5.4 ms (eager) per step on 48 layers; its step time was not in the brief, so no percentage.
The 29 ms step time is the brief's number for a typical bs=64 step; a step with a 32k row is probably longer, which would make the share smaller. Treat the share as an upper-end estimate.

Other findings:
- `disable_split_kv` is up to 9x slower than the default when one long row dominates (one CTA walks the whole row: 360 to 480 us for a 32k row, versus 48 to 57 us), 1.9x slower on 8x32k, and neutral when every row is long enough to fill the grid anyway (32x8k, 64x2k). It is hazardous in graph mode (see below).
- Static knobs cost something elsewhere. Versus auto in graph mode: fix16 is +21 us at 1x2048, +11 us at 8x2048, +5 us at 1x32k+7..31x512, +10 us at 1x16k+31x512; fix32 is +25 us at 1x32k+31x512, +24 us at 1x32k+15x512. On uniform 512-token batches both knobs are between 3 us worse and 4 us better.
- For uniform bs=32 x 512 tokens, `nosplit` and fix >= 4 beat auto by 15 to 18% (31 -> 26 us). Not a skewed batch, and it is 0.8% of a step, so it does not count for K3.
- `fa2_notc` on Omni (the only geometry where it runs): graph 1x32k+63x512 is 110 us versus auto 155 (1.4x better), but it is 1.3x to 1.8x worse on 1x32k+63x2048 (264 vs 207), uniform 64x2048 (204 vs 112) and 8x32k (341 vs 198). Not a candidate.

## Hazards found (they bound how a knob can be used)

- `fixed_split_size` in graph mode raises `PrefillSplitQOKVIndptr` (`FLASHINFER_CHECK(new_batch_size <= padded_batch_size)`) when the plan needs more (row, chunk) tiles than the padded grid. In `plan()`, that would be an exception in the serving loop. Bagel graph failures: fix8 on 6 of 17 compositions, fix16 on uniform 8x32k and 32x8k, fix2/fix4 on most. fix32 and fix64 never raised in this sweep. Eager mode never raised.
- `disable_split_kv` in graph mode: 1 illegal-memory-access (bagel, 1x16k+31x512; a retry in a fresh process passed, 230 us) and 3 hangs out of 17 runs on Omni (1x512, 1x32k+7x512, 8x32k; each killed by the watchdog after 180 s). Likely cause: in this branch the graph-padded blocks have no `block_valid_mask` and read uninitialised schedule entries. The root cause was not confirmed (it did not reproduce under `CUDA_LAUNCH_BLOCKING=1`). Never use `disable_split_kv` with graphs.
- One more hang occurred in cancelled job 13074 at `bagel|u32x8192|fa2|fix64|graph`; the same row ran fine (207 us) in job 13083. So a non-`nosplit` config hung once and not reproducibly; cause unknown.

## What was wired (step 5; defaults unchanged)

`AttentionConfig.flashinfer_decode_plan_kwargs: dict = {}` is forwarded to `FlashInferDecodeWrapper.plan()` (graph buckets with bs == num_tokens and the eager decode path). `AttentionSpec.apply_yaml_overrides(flashinfer_decode_plan_kwargs=...)` sets it from yaml. Cross-attention uses only the prefill wrapper, so `CrossAttentionConfig` does not get the knob. CPU test: `test/modular/test_flashinfer_decode_plan_kwargs.py` (7 tests; passes with the wiring, fails without it). Commit `0ccb27950` in the worktree.

## Is K3 worth wiring?

Mostly no, as a static setting. Reasons:
- The straggler needs bs ~64 with a single very long row, and 63 short rows. If BAGEL traffic never has that mix at bs=64, the gain is zero. At bs <= 32 the heuristic is already within 2% of the best knob.
- `fixed_split_size` is a global number (KV pages per CTA). Each value that fixes the straggler costs 5 to 25 us per layer (about 0.1 to 0.7 ms per step) on other skewed batches and +21 us on 1x2048, and the values that help most (8, 16) raise in graph mode on some compositions. A raise in `plan()` takes the step down.
- If it is used at all: `{"fixed_split_size": 32}` in graph buckets was the only value that never raised and still gave 1.6x on the straggler (156 -> 95 us, 5.9% of a step); for eager decode above the graph cap, `{"fixed_split_size": 8}` gave 2.6x with no raise.
- The right fix is adaptive and per step: `plan()` runs on the host every step, so mstar could pass `fixed_split_size` only when `bs >= ~48` and one row's pages are much larger than the rest, sized so `n_short + ceil(long_pages / k) <= padded grid`. That is a policy change in shared code and was not built. Cost of not building it: the 6 to 11% step loss on the straggler batches stays.
- fa3, `disable_split_kv` and `use_tensor_cores=False` are not candidates.

## Method and caveats

- GPU: H100 80GB HBM3 (sm90, node1), torch 2.12.1+cu132, FlashInfer 0.7.0.post1 (the brief said 0.6.18.post1; the installed one was used), page size 128, bf16, seeded random page permutation over one shared paged KV tensor per geometry.
- Timing: per iteration a 128 MB memset flushes L2, then `torch.cuda._sleep` (>= 300 us), then event, call, event; median of 200 iterations after 20 warmup. Flush cost is measured separately (`flush_us`) and `call_sub_us` is the cross-check (total minus flush); `call_us` is the headline. In eager mode `call_sub_us` includes CPU launch gaps and is unreliable. Every row's output is checked against an fp32 torch reference (max error is below 0.05 for every ok row; the worst is about 0.002).
- Graph mode here is a CUDA graph of one `run()`; the planned data is uploaded by `plan()` before replay. mstar's real graph buckets replay a whole layer-stack graph; the attention kernel is the same.
- 17 compositions per geometry, bs 1 to 64, ctx 512 to 32768. The skewed set is a designed worst case, not a trace of BAGEL traffic.
- Rows: 652 ok, 101 error rows (mostly fa2_notc on bagel, the Blackwell backends, the raises above). Jobs: 13059 (cancelled by me after 38 s, ninja missing from PATH), 13071 (ended on the IMA below before the supervisor existed; partial rows archived), 13073 (debug), 13074 (cancelled by me after 14 min, a worker hung with no watchdog; its rows are kept), 13083 (COMPLETED in 13 min, the final stage; resumed from 13074's rows).
- Not verified: the root cause of the nosplit graph IMA and hangs and of the one fix64 hang; the 32/4/128 Omni Thinker numbers use my geometry (qo 32, kv 4, hd 128, 48 layers), not a load of the real model; no end-to-end server measurement of the share of a step (the percentages are per-layer microbenchmark gains times layer count over the brief's step times).

Reproduce: `sbatch perf_testing/k3_decode_split/run.sbatch` from the worktree root (resume-safe), then `python perf_testing/k3_decode_split/summarize.py results/mstar/k3/bench.jsonl --out results/mstar/k3/summary.md`.
Raw data: `bench.jsonl`, `smoke.jsonl`; all tables: `summary.md`; logs: `slurm_k3_decode_split_*.out`.
