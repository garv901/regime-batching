# ATTN_SWEEP: attention kernel config space on H100 (Qwen3-8B geometry)

Job 1008 (sbatch, partition main, 1 GPU, ~10 min wall for 234 specs x up to 14 configs = 3114 timed rows, 0 errors).
Code: microbench/attn_sweep.py (sweep), microbench/summarize_attn_sweep.py (tables). Raw: attn_sweep.{jsonl,parquet,csv},
attn_sweep_best_vs_default.csv. Full per-family tables are appended below (generated).

## Method and what each config really executes
- Real vLLM 0.30 backend impls and metadata builders, KV cache in vLLM's layout, random bf16 K/V, scattered (random-permutation) block tables, page size 16, decodes ordered first (as vLLM's reorder does).
- Timing: CUDA-graph replay, 128 MB memset before each call to flush the 50 MB L2 (memset cost, 42 us, subtracted), so KV streams from HBM like consecutive layers. `time_us_warm` (no flush) only for family 5. Eager time (includes Python/launch) for families 2, 3, 5.
- fa3_auto: builder default with cudagraph off: max_num_splits=0, AOT scheduler metadata from FA3 heuristic. fa3_cg / fa3_sN: metadata built by vLLM's builder, then AOT scheduler_metadata recomputed with num_splits=32 / N and max_num_splits set (consistent). For batches <= 512 tokens fa3_cg scheduler metadata was verified bit-identical to what vLLM's real full-cudagraph builder produces (all such rows `vllm_cg_builder_md_match=True`). For >512 tokens real vLLM would use auto; fa3_cg there is a hypothetical config.
- IMPORTANT: FA3 `num_splits=N` is a static upper bound; the prepare kernel then picks per-sequence dynamic splits (<= N, proportional to KV blocks, and 1 for everyone if batch has too many CTAs already). The `notes` column records the dynamic split vector actually in the metadata (max / for longest request / number of split requests). fa3_s1 truly disables splitting.
- fa3_auto in mixed steps: notes show `heuristic_static_bound=1;no_dyn_split_vec` for the prefill+decode batches whose static bound is 1 (as predicted): no sequence is split, so the long decode is a single-CTA-per-kv-head straggler.
- fi_xqa: notes confirm `decode_path=xqa` (FlashInferTrtllmAPIDecode, kernel XQA, use_xqa=True) and `prefill_path=FIPrefill` (native FI prefill, SM90 has no trtllm prefill). fi_native: `use_trtllm_attention=False` -> `decode_path=FIDecode` (BatchDecodeWithPagedKVCacheWrapper, tensor cores). nosplit / fs256 / fs1024 set builder.disable_split_kv / decode_fixed_split_size (units: pages; 256 pages = 4096 tokens), all accepted by flashinfer (no skips). FI decode wrappers were planned without FI's own cudagraph wrappers (builder.enable_cuda_graph is off), the run() call is graph-captured fine.
- fa3_phase_split: two FA3 calls (decode rows with max_num_splits=32 AOT, prefill rows with auto) into one output buffer, timed together.
- Correctness: max abs diff vs fa3_auto <= 8e-3 for every config/spec (0 rows flagged at 2e-2).
- Harness patched (third_party/vllm/benchmarks/attention_benchmarks/runner.py): model now Qwen/Qwen3-8B (env ATTN_BENCH_MODEL), and --num-splits override now recomputes scheduler_metadata. attn_sweep.py reuses its helpers but builds its own config/metadata.

## Headline numbers
- Single-CTA streaming bandwidth (fa3_s1, B=1, 8 CTAs one per kv head, cold L2): 50 / 56 / 58 / 59 GB/s per CTA at L=4k/16k/32k/64k (asymptote ~59 GB/s per kv head; L2-warm identical, so it is a per-CTA pipeline limit, not HBM). A 32k sequence unsplit takes 291 us vs 62 us auto-split (4.7x); 64k: 567 vs 108 us (5.3x). With splitting, one sequence reaches 2.5 TB/s at 64k.
- vLLM default vs oracle best (geomean / max speedup of best over default): F1 homogeneous decode 1.04 / 1.30 (the gains are all in L<=2k and are small in absolute terms, 1-3 us at B=1; for B>=16, L>=2k fa3_s1 ~ fa3_cg within 1%); F2 skewed decode 1.03 / 1.21; F3 mixed steps 1.40 / 4.81; F4 steady long-heavy decode 1.03 / 1.10; F5 1.01.
- The big, structural loss is mixed steps with a long-KV decode: fa3_auto (default for >512 tokens) gives 407-455 us for a 32k-KV decode row regardless of prefill size, vs 85-200 us with splitting: geomean auto/best = 1.03 (no long decode), 1.11 (4.5k), 1.78 (16k), 2.55 (32k), up to 4.8x (q512 chunk + 15 short + one 32k decode: 407 us vs 85 us). fa3_phase_split recovers it (geomean within 1.04-1.05 of oracle best for L_long>=16k; auto/phase_split = 1.69x at 16k, 2.45x at 32k).
- But phase_split / forced splits cost ~10-20% (geomean 1.22x slower than best) when there is no long decode (extra launch + split-combine overhead, 0.84 auto/phase_split): so it must be gated on skew (long decode KV >= ~8k).
- FlashInfer XQA (the FI-backend default for decode): ties FA3 on homogeneous decode (geomean fi_xqa/fa3_cg = 0.99; within 1-5% of best, 1.00 for small B / short L, 4-5% slower at B>=8, L>=8k), but is poor on skewed batches: 1.66x slower than fa3_cg geomean over family 2 (1.97x for L_long>=16k, up to 4.5x) because it does not balance one long sequence across SMs; in mixed steps fi_xqa is 1.3-1.4x slower than fa3_auto and 3.3x slower than best at 32k long decode (XQA straggles the same way). fi_native (BatchDecode TC with FI's split-KV planner) is the best FlashInfer variant on skewed batches (1.36x slower than fa3_cg in F2 overall but wins ~1/4 of specs; in mixed steps with L_long>=16k it is 6-18% faster than fa3_auto but 1.4-2x slower than split FA3).
- Triton attention is 1.18x (F1) to 3.3x (F2) slower than fa3_cg geomean, only wins at tiny B with L=512 (<~10us, launch-level).

## Crossovers (details in tables)
- Homogeneous decode: B=1-8: XQA/FI native slightly beat FA3 at L<=2k (<=15%, few us); L>=16k at B=2-8 FA3 split variants win by 1-5%. B>=16, L>=2k: unsplit (fa3_s1) ties auto/cg (HBM-bound, 99-100%), i.e. splitting is irrelevant once B*8 CTAs >= ~132. Splitting matters when B*8 << 132 and L is long (see family 5: unsplit is 2.5-5.3x slower).
- Skewed decode (F2): the oracle best varies among fa3_cg/auto/s4/s16/fi_native/fs256 within 0-21% (win counts spread, differences mostly small); consistent finding: default fa3_cg (<=512 tokens, 32-split bound) is within 3% geomean of best, worst 1.21x (FI native wins by 15-21% when the batch has one 4.5k decode among 15-31 2k decodes).
- Mixed (F3): auto wins or ties when no long decode (and when L_long=4.5k: phase_split ~ tied, cg-forced 4-16% better than auto); once L_long >= 16k forced splits / phase_split dominate (1.7x-4.8x). Prefill chunk size matters mostly through the static bound: chunks of q>=512 tokens (all of 512/1024/2048, incl. ctx 4k) trigger the no-split behaviour (notes: heuristic_static_bound=1); q=256 and q=416 chunks still get dynamic splits from fa3_auto (e.g. max 5 splits for the longest row at q=416).

## Caveats
- Single run per point (10 graph replays, median); no clock locking, shared node but exclusive GPU. Differences below ~2-3% should be treated as noise (F1 "winners" at B>=16 are noise).
- F1 B=1 specs also appear as family 5 (separate data, different random seeds). F5 spec names are prefixed `single_`.
- FI native decode planned with fast_plan_decode but FI cudagraph-specific wrappers (fixed split for graph capture) are not exercised; vLLM full-cudagraph mode for FI native would use different (cuda-graph) wrappers.
- Mixed rows: FI decode/prefill run as two calls (decode first), FA3 runs one varlen call.
- Memory-layout: contiguity of KV is randomized by page; real allocator fragmentation may differ.


---
# Generated tables


Generated by microbench/summarize_attn_sweep.py from results/microbench/attn_sweep.{parquet,csv}. Timings are CUDA-graph replays with a 128 MB L2 flush before each call (flush cost subtracted), i.e. KV is streamed from HBM.

Rows: 3114 (3114 timed), specs: 234.

## (a) Single long sequence / single-CTA streaming bandwidth (family 5, B=1)

Time (us, cold L2):

| L | fa3_s1 | fa3_s2 | fa3_s4 | fa3_s8 | fa3_s16 | fa3_auto | fa3_cg | fi_xqa | fi_native | fi_native_nosplit | triton |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 4096.0 | 42.0 | 26.8 | 19.6 | 17.0 | 17.0 | 16.9 | 17.0 | 16.6 | 17.8 | 61.9 | 27.5 |
| 16384.0 | 148.8 | 80.7 | 49.8 | 39.4 | 38.5 | 38.5 | 38.5 | 38.4 | 39.5 | 236.3 | 94.3 |
| 32768.0 | 291.0 | 152.2 | 86.7 | 63.9 | 62.4 | 62.3 | 62.4 | 62.3 | 63.9 | 446.6 | 174.4 |
| 65536.0 | 567.4 | 294.4 | 161.4 | 110.6 | 107.7 | 107.6 | 107.5 | 108.2 | 106.7 | 884.7 | 334.6 |

Bandwidth:

| L | fa3_s1: per-CTA GB/s (bytes per kv head / time) | fa3_auto: total GB/s | fa3_cg: total GB/s | fi_xqa: total GB/s | fa3_s1 warm-L2: per-CTA GB/s | fa3_s1 / fa3_auto slowdown |
|---|---|---|---|---|---|---|
| 4096.0 | 50.0 | 989.9 | 988.3 | 1009.5 | 51.4 | 2.5 |
| 16384.0 | 56.4 | 1744.8 | 1744.4 | 1749.3 | 57.3 | 3.9 |
| 32768.0 | 57.6 | 2152.7 | 2150.1 | 2152.9 | 58.2 | 4.7 |
| 65536.0 | 59.1 | 2495.8 | 2498.1 | 2481.7 | 59.4 | 5.3 |

## (b) Best config vs vLLM default, per family

Default FA = fa3_cg if num_actual_tokens <= 512 else fa3_auto; FlashInfer default = fi_xqa. speedup = default_time / best_time over ALL configs (oracle); `fa_best_vs_default` restricts best to FA3 configs; `xqa/best` = fi_xqa time / best time.

| family | n_specs | geomean_speedup | median_speedup | max_speedup | frac_gt_1p05 | geomean_fa_best_vs_default | geomean_xqa_over_best | best_cfg_wins |
|---|---|---|---|---|---|---|---|---|
| F1_homog | 60 | 1.04 | 1.01 | 1.30 | 0.25 | 1.01 | 1.02 | {'fa3_s1': 23, 'fi_xqa': 18, 'triton': 5, 'fa3_auto': 3} |
| F2_skew | 90 | 1.03 | 1.00 | 1.21 | 0.19 | 1.01 | 1.70 | {'fa3_auto': 18, 'fa3_cg': 13, 'fi_native_fs256': 11, 'fa3_s16': 11} |
| F3_mixed | 72 | 1.40 | 1.08 | 4.81 | 0.54 | 1.39 | 2.03 | {'fa3_phase_split': 23, 'fa3_s1': 11, 'fa3_auto': 10, 'fa3_cg': 9} |
| F4_steady | 8 | 1.03 | 1.01 | 1.10 | 0.12 | 1.01 | 1.07 | {'fa3_s1': 5, 'fi_native_fs256': 2, 'fi_native': 1} |
| F5_single | 4 | 1.01 | 1.01 | 1.03 | 0.00 | 1.00 | 1.01 | {'fi_xqa': 3, 'fi_native_fs256': 1} |

### F1_homog

Top 15 of 60 specs by speedup:

| spec | tokens | default_cfg | default_us | best_cfg | best_us | speedup_vs_default | xqa_us | xqa_speedup_vs_best |
|---|---|---|---|---|---|---|---|---|
| 1q1s512 | 1 | fa3_cg | 10.88 | triton | 8.39 | 1.30 | 9.33 | 1.11 |
| 2q1s512 | 2 | fa3_cg | 11.54 | triton | 9.39 | 1.23 | 9.88 | 1.05 |
| 4q1s512 | 4 | fa3_cg | 13.07 | triton | 11.14 | 1.17 | 11.45 | 1.03 |
| 1q1s1024 | 1 | fa3_cg | 11.85 | fi_xqa | 10.20 | 1.16 | 10.20 | 1.00 |
| 16q1s512 | 16 | fa3_cg | 23.22 | fi_xqa | 20.63 | 1.13 | 20.63 | 1.00 |
| 2q1s1024 | 2 | fa3_cg | 13.01 | fi_xqa | 11.63 | 1.12 | 11.63 | 1.00 |
| 64q1s512 | 64 | fa3_cg | 64.56 | triton | 58.03 | 1.11 | 61.43 | 1.06 |
| 4q1s1024 | 4 | fa3_cg | 16.58 | fi_xqa | 15.11 | 1.10 | 15.11 | 1.00 |
| 1q1s2048 | 1 | fa3_cg | 13.38 | fi_xqa | 12.20 | 1.10 | 12.20 | 1.00 |
| 8q1s512 | 8 | fa3_cg | 16.05 | fa3_auto | 14.65 | 1.10 | 14.97 | 1.02 |
| 16q1s1024 | 16 | fa3_cg | 37.25 | fi_xqa | 34.87 | 1.07 | 34.87 | 1.00 |
| 4q1s2048 | 4 | fa3_cg | 23.72 | fi_xqa | 22.32 | 1.06 | 22.32 | 1.00 |
| 32q1s512 | 32 | fa3_cg | 38.70 | fi_xqa | 36.46 | 1.06 | 36.46 | 1.00 |
| 64q1s1024 | 64 | fa3_cg | 107.44 | triton | 101.73 | 1.06 | 105.66 | 1.04 |
| 8q1s1024 | 8 | fa3_cg | 23.66 | fi_xqa | 22.43 | 1.05 | 22.43 | 1.00 |

### F2_skew

Top 15 of 90 specs by speedup:

| spec | tokens | default_cfg | default_us | best_cfg | best_us | speedup_vs_default | xqa_us | xqa_speedup_vs_best |
|---|---|---|---|---|---|---|---|---|
| 15q1s2048_1q1s4512 | 16 | fa3_cg | 81.50 | fi_native | 67.61 | 1.21 | 82.02 | 1.21 |
| 127q1s2048_1q1s16384 | 128 | fa3_cg | 482.85 | fi_native_fs256 | 413.70 | 1.17 | 512.19 | 1.24 |
| 31q1s2048_1q1s4512 | 32 | fa3_cg | 126.17 | fi_native | 109.61 | 1.15 | 130.92 | 1.19 |
| 7q1s2048_1q1s32768 | 8 | fa3_cg | 94.95 | fa3_s8 | 82.66 | 1.15 | 205.81 | 2.49 |
| 15q1s2048_2q1s4512 | 17 | fa3_cg | 82.90 | fi_native | 72.54 | 1.14 | 101.46 | 1.40 |
| 7q1s542_2q1s4512 | 9 | fa3_cg | 36.93 | fa3_s4 | 32.88 | 1.12 | 61.50 | 1.87 |
| 7q1s2048_1q1s4512 | 8 | fa3_cg | 49.77 | fi_native | 44.31 | 1.12 | 52.93 | 1.19 |
| 127q1s2048_1q1s32768 | 128 | fa3_cg | 486.94 | fi_native_fs256 | 434.22 | 1.12 | 676.02 | 1.56 |
| 127q1s2048_2q1s16384 | 129 | fa3_cg | 485.83 | fi_native_fs256 | 434.17 | 1.12 | 516.13 | 1.19 |
| 7q1s2048_2q1s16384 | 9 | fa3_cg | 91.85 | fa3_s4 | 82.41 | 1.11 | 205.59 | 2.49 |
| 7q1s542_1q1s16384 | 8 | fa3_cg | 49.06 | fa3_s8 | 44.98 | 1.09 | 107.44 | 2.39 |
| 15q1s2048_1q1s16384 | 16 | fa3_cg | 88.90 | fi_native | 81.80 | 1.09 | 198.66 | 2.43 |
| 7q1s2048_2q1s4512 | 9 | fa3_cg | 54.25 | fi_native | 50.35 | 1.08 | 67.54 | 1.34 |
| 15q1s542_1q1s4512 | 16 | fa3_cg | 37.24 | fi_native | 34.70 | 1.07 | 60.47 | 1.74 |
| 7q1s542_4q1s4512 | 11 | fa3_cg | 50.17 | fa3_s2 | 47.15 | 1.06 | 63.67 | 1.35 |

### F3_mixed

Top 15 of 72 specs by speedup:

| spec | tokens | default_cfg | default_us | best_cfg | best_us | speedup_vs_default | xqa_us | xqa_speedup_vs_best |
|---|---|---|---|---|---|---|---|---|
| 1q512s512_15q1s542_1q1s32768 | 528 | fa3_auto | 407.18 | fa3_cg | 84.59 | 4.81 | 366.68 | 4.33 |
| 1q512s512_31q1s542_1q1s32768 | 544 | fa3_auto | 424.27 | fa3_s16 | 105.28 | 4.03 | 394.07 | 3.74 |
| 1q1024s1024_15q1s542_1q1s32768 | 1040 | fa3_auto | 406.93 | fa3_s8 | 101.43 | 4.01 | 388.23 | 3.83 |
| 1q1024s1024_31q1s542_1q1s32768 | 1056 | fa3_auto | 424.05 | fa3_phase_split | 122.65 | 3.46 | 414.74 | 3.38 |
| 1q512s512_15q1s542_1q1s16384 | 528 | fa3_auto | 212.51 | fa3_s16 | 61.69 | 3.44 | 204.58 | 3.32 |
| 1q512s512_63q1s542_1q1s32768 | 576 | fa3_auto | 453.53 | fa3_phase_split | 140.50 | 3.23 | 426.68 | 3.04 |
| 1q1024s1024_63q1s542_1q1s32768 | 1088 | fa3_auto | 453.86 | fa3_phase_split | 152.56 | 2.97 | 445.23 | 2.92 |
| 1q512s4608_15q1s542_1q1s32768 | 528 | fa3_auto | 410.04 | fi_native | 143.66 | 2.85 | 418.97 | 2.92 |
| 1q1024s1024_15q1s542_1q1s16384 | 1040 | fa3_auto | 213.85 | fa3_cg | 78.51 | 2.72 | 225.97 | 2.88 |
| 1q512s512_31q1s542_1q1s16384 | 544 | fa3_auto | 230.23 | fa3_cg | 85.17 | 2.70 | 229.21 | 2.69 |
| 1q512s4608_31q1s542_1q1s32768 | 544 | fa3_auto | 426.55 | fa3_phase_split | 159.65 | 2.67 | 445.60 | 2.79 |
| 1q2048s2048_15q1s542_1q1s32768 | 2064 | fa3_auto | 409.83 | fa3_phase_split | 155.20 | 2.64 | 442.22 | 2.85 |
| 1q2048s2048_31q1s542_1q1s32768 | 2080 | fa3_auto | 428.36 | fa3_phase_split | 168.04 | 2.55 | 466.30 | 2.77 |
| 1q512s4608_63q1s542_1q1s32768 | 576 | fa3_auto | 455.40 | fa3_phase_split | 190.07 | 2.40 | 477.54 | 2.51 |
| 1q2048s2048_63q1s542_1q1s32768 | 2112 | fa3_auto | 455.43 | fa3_phase_split | 198.91 | 2.29 | 501.43 | 2.52 |

### F4_steady

| spec | tokens | default_cfg | default_us | best_cfg | best_us | speedup_vs_default | xqa_us | xqa_speedup_vs_best |
|---|---|---|---|---|---|---|---|---|
| rand_B16_oL16k | 16 | fa3_cg | 208.63 | fi_native | 189.47 | 1.10 | 230.85 | 1.22 |
| rand_B16_oL4k | 16 | fa3_cg | 86.50 | fa3_s1 | 82.62 | 1.05 | 86.63 | 1.05 |
| rand_B32_oL4k | 32 | fa3_cg | 125.05 | fa3_s1 | 123.29 | 1.01 | 129.76 | 1.05 |
| rand_B32_oL16k | 32 | fa3_cg | 430.11 | fi_native_fs256 | 424.59 | 1.01 | 451.42 | 1.06 |
| rand_B64_oL16k | 64 | fa3_cg | 744.15 | fi_native_fs256 | 736.25 | 1.01 | 779.79 | 1.06 |
| rand_B64_oL4k | 64 | fa3_cg | 231.60 | fa3_s1 | 229.41 | 1.01 | 240.33 | 1.05 |
| rand_B128_oL4k | 128 | fa3_cg | 458.69 | fa3_s1 | 455.34 | 1.01 | 474.12 | 1.04 |
| rand_B128_oL16k | 128 | fa3_cg | 1513.94 | fa3_s1 | 1511.54 | 1.00 | 1575.64 | 1.04 |

### F5_single

| spec | tokens | default_cfg | default_us | best_cfg | best_us | speedup_vs_default | xqa_us | xqa_speedup_vs_best |
|---|---|---|---|---|---|---|---|---|
| single_1q1s65536 | 1 | fa3_cg | 107.46 | fi_native_fs256 | 104.68 | 1.03 | 108.17 | 1.03 |
| single_1q1s4096 | 1 | fa3_cg | 16.98 | fi_xqa | 16.62 | 1.02 | 16.62 | 1.00 |
| single_1q1s16384 | 1 | fa3_cg | 38.47 | fi_xqa | 38.36 | 1.00 | 38.36 | 1.00 |
| single_1q1s32768 | 1 | fa3_cg | 62.42 | fi_xqa | 62.34 | 1.00 | 62.34 | 1.00 |

## (c) Crossovers

Homogeneous decode, best config per (B rows x L cols):

| B | 512 | 1024 | 2048 | 4096 | 8192 | 16384 | 32768 |
|---|---|---|---|---|---|---|---|
| 1 | triton | fi_xqa | fi_xqa | fi_xqa | fi_xqa | fi_xqa | fi_xqa |
| 2 | triton | fi_xqa | fi_xqa | fi_xqa | fi_xqa | fa3_auto | fa3_auto |
| 4 | triton | fi_xqa | fi_xqa | fi_xqa | fa3_cg | fa3_cg | fa3_s4 |
| 8 | fa3_auto | fi_xqa | fi_xqa | fa3_cg | fa3_s2 | fa3_s2 | fa3_s2 |
| 16 | fi_xqa | fi_xqa | fa3_s1 | fa3_s1 | fa3_s1 | fa3_s1 | fa3_s1 |
| 32 | fi_xqa | fi_native_fs1024 | fa3_s1 | fa3_s1 | fa3_s1 | fa3_s1 | fa3_s1 |
| 64 | triton | triton | fa3_s1 | fa3_s1 | fa3_s1 | fa3_s1 | fa3_s1 |
| 128 | fi_native_fs256 | fi_native_fs256 | fa3_s1 | fa3_s1 | fa3_s1 | fa3_s1 |  |
| 256 | fi_native_fs1024 | fa3_s1 | fa3_s1 | fa3_s1 | fa3_s1 |  |  |

Homogeneous decode, default/best speedup:

| B | 512 | 1024 | 2048 | 4096 | 8192 | 16384 | 32768 |
|---|---|---|---|---|---|---|---|
| 1.00 | 1.30 | 1.16 | 1.10 | 1.02 | 1.02 | 1.00 | 1.00 |
| 2.00 | 1.23 | 1.12 | 1.04 | 1.04 | 1.01 | 1.00 | 1.00 |
| 4.00 | 1.17 | 1.10 | 1.06 | 1.02 | 1.00 | 1.00 | 1.00 |
| 8.00 | 1.10 | 1.05 | 1.03 | 1.00 | 1.00 | 1.00 | 1.00 |
| 16.00 | 1.13 | 1.07 | 1.03 | 1.02 | 1.01 | 1.00 | 1.00 |
| 32.00 | 1.06 | 1.04 | 1.02 | 1.01 | 1.00 | 1.00 | 1.00 |
| 64.00 | 1.11 | 1.06 | 1.01 | 1.01 | 1.00 | 1.00 | 1.00 |
| 128.00 | 1.04 | 1.02 | 1.01 | 1.00 | 1.00 | 1.00 |  |
| 256.00 | 1.04 | 1.02 | 1.01 | 1.00 | 1.00 |  |  |

Homogeneous decode, fi_xqa/best:

| B | 512 | 1024 | 2048 | 4096 | 8192 | 16384 | 32768 |
|---|---|---|---|---|---|---|---|
| 1.00 | 1.11 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| 2.00 | 1.05 | 1.00 | 1.00 | 1.00 | 1.00 | 1.01 | 1.04 |
| 4.00 | 1.03 | 1.00 | 1.00 | 1.00 | 1.00 | 1.03 | 1.05 |
| 8.00 | 1.02 | 1.00 | 1.00 | 1.01 | 1.03 | 1.05 | 1.05 |
| 16.00 | 1.00 | 1.00 | 1.01 | 1.03 | 1.04 | 1.04 | 1.05 |
| 32.00 | 1.00 | 1.01 | 1.04 | 1.04 | 1.04 | 1.04 | 1.05 |
| 64.00 | 1.06 | 1.04 | 1.03 | 1.03 | 1.04 | 1.04 | 1.04 |
| 128.00 | 1.01 | 1.02 | 1.03 | 1.04 | 1.04 | 1.04 |  |
| 256.00 | 1.02 | 1.02 | 1.03 | 1.03 | 1.04 |  |  |

Mixed steps (family 3): fa3_auto (vLLM default for >512 tokens) vs fa3_cg-style forced splits vs fa3_phase_split:

| spec | tokens | auto_us | cg_us | phase_split_us | auto/cg | auto/phase_split | best_cfg | best_us |
|---|---|---|---|---|---|---|---|---|
| 1q1024s1024_15q1s542 | 1039 | 44.94 | 53.15 | 57.92 | 0.85 | 0.78 | fa3_s1 | 44.85 |
| 1q1024s1024_15q1s542_1q1s16384 | 1040 | 213.85 | 78.51 | 86.76 | 2.72 | 2.46 | fa3_cg | 78.51 |
| 1q1024s1024_15q1s542_1q1s32768 | 1040 | 406.93 | 101.68 | 108.99 | 4.00 | 3.73 | fa3_s8 | 101.43 |
| 1q1024s1024_15q1s542_1q1s4512 | 1040 | 72.44 | 62.18 | 71.67 | 1.16 | 1.01 | fa3_cg | 62.18 |
| 1q1024s1024_31q1s542 | 1055 | 60.98 | 69.40 | 75.09 | 0.88 | 0.81 | fa3_s1 | 60.92 |
| 1q1024s1024_31q1s542_1q1s16384 | 1056 | 230.92 | 104.88 | 103.22 | 2.20 | 2.24 | fa3_phase_split | 103.22 |
| 1q1024s1024_31q1s542_1q1s32768 | 1056 | 424.05 | 125.51 | 122.65 | 3.38 | 3.46 | fa3_phase_split | 122.65 |
| 1q1024s1024_31q1s542_1q1s4512 | 1056 | 90.17 | 77.51 | 91.36 | 1.16 | 0.99 | fa3_s16 | 77.31 |
| 1q1024s1024_63q1s542 | 1087 | 89.50 | 99.88 | 102.79 | 0.90 | 0.87 | fa3_auto | 89.50 |
| 1q1024s1024_63q1s542_1q1s16384 | 1088 | 259.55 | 144.69 | 135.71 | 1.79 | 1.91 | fa3_phase_split | 135.71 |
| 1q1024s1024_63q1s542_1q1s32768 | 1088 | 453.86 | 169.49 | 152.56 | 2.68 | 2.97 | fa3_phase_split | 152.56 |
| 1q1024s1024_63q1s542_1q1s4512 | 1088 | 119.89 | 131.13 | 119.54 | 0.91 | 1.00 | fa3_phase_split | 119.54 |
| 1q2048s2048_15q1s542 | 2063 | 90.70 | 106.93 | 104.17 | 0.85 | 0.87 | fa3_auto | 90.70 |
| 1q2048s2048_15q1s542_1q1s16384 | 2064 | 218.62 | 146.23 | 133.23 | 1.50 | 1.64 | fa3_phase_split | 133.23 |
| 1q2048s2048_15q1s542_1q1s32768 | 2064 | 409.83 | 182.32 | 155.20 | 2.25 | 2.64 | fa3_phase_split | 155.20 |
| 1q2048s2048_15q1s542_1q1s4512 | 2064 | 95.17 | 111.37 | 118.10 | 0.85 | 0.81 | fa3_auto | 95.17 |
| 1q2048s2048_31q1s542 | 2079 | 106.98 | 123.18 | 120.84 | 0.87 | 0.89 | fa3_auto | 106.98 |
| 1q2048s2048_31q1s542_1q1s16384 | 2080 | 234.81 | 154.96 | 148.51 | 1.52 | 1.58 | fa3_phase_split | 148.51 |
| 1q2048s2048_31q1s542_1q1s32768 | 2080 | 428.36 | 199.90 | 168.04 | 2.14 | 2.55 | fa3_phase_split | 168.04 |
| 1q2048s2048_31q1s542_1q1s4512 | 2080 | 111.58 | 130.31 | 137.37 | 0.86 | 0.81 | fa3_s1 | 111.53 |
| 1q2048s2048_63q1s542 | 2111 | 135.43 | 155.23 | 147.94 | 0.87 | 0.92 | fa3_auto | 135.43 |
| 1q2048s2048_63q1s542_1q1s16384 | 2112 | 263.62 | 187.85 | 181.41 | 1.40 | 1.45 | fa3_phase_split | 181.41 |
| 1q2048s2048_63q1s542_1q1s32768 | 2112 | 455.43 | 224.43 | 198.91 | 2.03 | 2.29 | fa3_phase_split | 198.91 |
| 1q2048s2048_63q1s542_1q1s4512 | 2112 | 139.81 | 162.32 | 166.33 | 0.86 | 0.84 | fa3_auto | 139.81 |
| 1q256s256_15q1s542 | 271 | 31.68 | 31.79 | 41.21 | 1.00 | 0.77 | fa3_s1 | 29.22 |
| 1q256s256_15q1s542_1q1s16384 | 272 | 130.75 | 59.11 | 71.03 | 2.21 | 1.84 | fa3_s16 | 59.00 |
| 1q256s256_15q1s542_1q1s32768 | 272 | 230.38 | 80.68 | 93.12 | 2.86 | 2.47 | fa3_s16 | 80.64 |
| 1q256s256_15q1s542_1q1s4512 | 272 | 57.56 | 44.76 | 55.43 | 1.29 | 1.04 | fa3_s4 | 44.67 |
| 1q256s256_31q1s542 | 287 | 47.95 | 47.89 | 58.77 | 1.00 | 0.82 | fa3_s1 | 44.65 |
| 1q256s256_31q1s542_1q1s16384 | 288 | 138.72 | 79.30 | 87.19 | 1.75 | 1.59 | fa3_s8 | 79.00 |
| 1q256s256_31q1s542_1q1s32768 | 288 | 238.46 | 99.07 | 106.72 | 2.41 | 2.23 | fa3_s16 | 99.05 |
| 1q256s256_31q1s542_1q1s4512 | 288 | 65.49 | 65.28 | 75.62 | 1.00 | 0.87 | fa3_cg | 65.28 |
| 1q256s256_63q1s542 | 319 | 78.18 | 78.12 | 86.53 | 1.00 | 0.90 | triton | 68.59 |
| 1q256s256_63q1s542_1q1s16384 | 320 | 170.35 | 120.36 | 119.63 | 1.42 | 1.42 | fa3_phase_split | 119.63 |
| 1q256s256_63q1s542_1q1s32768 | 320 | 269.82 | 140.05 | 137.06 | 1.93 | 1.97 | fa3_phase_split | 137.06 |
| 1q256s256_63q1s542_1q1s4512 | 320 | 96.60 | 96.72 | 103.92 | 1.00 | 0.93 | fa3_s4 | 96.45 |
| 1q416s416_15q1s542 | 431 | 34.91 | 34.96 | 44.31 | 1.00 | 0.79 | fa3_s1 | 31.44 |
| 1q416s416_15q1s542_1q1s16384 | 432 | 73.91 | 60.88 | 73.57 | 1.21 | 1.00 | fa3_cg | 60.88 |
| 1q416s416_15q1s542_1q1s32768 | 432 | 115.16 | 83.47 | 96.09 | 1.38 | 1.20 | fa3_s16 | 83.36 |
| 1q416s416_15q1s542_1q1s4512 | 432 | 45.78 | 45.70 | 58.48 | 1.00 | 0.78 | fa3_s8 | 45.59 |
| 1q416s416_31q1s542 | 447 | 51.59 | 51.62 | 61.63 | 1.00 | 0.84 | fa3_s1 | 47.34 |
| 1q416s416_31q1s542_1q1s16384 | 448 | 90.83 | 84.21 | 89.67 | 1.08 | 1.01 | fa3_cg | 84.21 |
| 1q416s416_31q1s542_1q1s32768 | 448 | 131.47 | 100.67 | 109.65 | 1.31 | 1.20 | fa3_cg | 100.67 |
| 1q416s416_31q1s542_1q1s4512 | 448 | 66.37 | 66.61 | 78.25 | 1.00 | 0.85 | fa3_auto | 66.37 |
| 1q416s416_63q1s542 | 479 | 81.55 | 81.53 | 89.44 | 1.00 | 0.91 | fa3_s1 | 76.01 |
| 1q416s416_63q1s542_1q1s16384 | 480 | 130.85 | 130.66 | 122.69 | 1.00 | 1.07 | fa3_phase_split | 122.69 |
| 1q416s416_63q1s542_1q1s32768 | 480 | 162.97 | 141.09 | 139.62 | 1.16 | 1.17 | fa3_phase_split | 139.62 |
| 1q416s416_63q1s542_1q1s4512 | 480 | 97.95 | 97.90 | 106.79 | 1.00 | 0.92 | fa3_cg | 97.90 |
| 1q512s4608_15q1s542 | 527 | 83.81 | 91.61 | 95.19 | 0.91 | 0.88 | fa3_auto | 83.81 |
| 1q512s4608_15q1s542_1q1s16384 | 528 | 219.18 | 149.02 | 124.62 | 1.47 | 1.76 | fi_native | 121.29 |
| 1q512s4608_15q1s542_1q1s32768 | 528 | 410.04 | 152.56 | 146.23 | 2.69 | 2.80 | fi_native | 143.66 |
| 1q512s4608_15q1s542_1q1s4512 | 528 | 130.35 | 138.06 | 109.26 | 0.94 | 1.19 | fi_native | 104.87 |
| 1q512s4608_31q1s542 | 543 | 100.34 | 108.39 | 112.01 | 0.93 | 0.90 | fa3_s1 | 100.19 |
| 1q512s4608_31q1s542_1q1s16384 | 544 | 234.98 | 166.20 | 140.51 | 1.41 | 1.67 | fa3_phase_split | 140.51 |
| 1q512s4608_31q1s542_1q1s32768 | 544 | 426.55 | 167.96 | 159.65 | 2.54 | 2.67 | fa3_phase_split | 159.65 |
| 1q512s4608_31q1s542_1q1s4512 | 544 | 147.01 | 155.27 | 129.16 | 0.95 | 1.14 | fa3_phase_split | 129.16 |
| 1q512s4608_63q1s542 | 575 | 129.22 | 138.81 | 139.35 | 0.93 | 0.93 | fa3_auto | 129.22 |
| 1q512s4608_63q1s542_1q1s16384 | 576 | 263.36 | 192.39 | 172.89 | 1.37 | 1.52 | fa3_phase_split | 172.89 |
| 1q512s4608_63q1s542_1q1s32768 | 576 | 455.40 | 198.62 | 190.07 | 2.29 | 2.40 | fa3_phase_split | 190.07 |
| 1q512s4608_63q1s542_1q1s4512 | 576 | 174.89 | 184.82 | 157.40 | 0.95 | 1.11 | fa3_phase_split | 157.40 |
| 1q512s512_15q1s542 | 527 | 33.22 | 37.44 | 45.36 | 0.89 | 0.73 | fa3_s1 | 33.09 |
| 1q512s512_15q1s542_1q1s16384 | 528 | 212.51 | 61.83 | 74.51 | 3.44 | 2.85 | fa3_s16 | 61.69 |
| 1q512s512_15q1s542_1q1s32768 | 528 | 407.18 | 84.59 | 96.48 | 4.81 | 4.22 | fa3_cg | 84.59 |
| 1q512s512_15q1s542_1q1s4512 | 528 | 71.96 | 50.41 | 59.11 | 1.43 | 1.22 | fa3_s16 | 50.39 |
| 1q512s512_31q1s542 | 543 | 49.14 | 54.00 | 62.43 | 0.91 | 0.79 | fa3_s1 | 49.09 |
| 1q512s512_31q1s542_1q1s16384 | 544 | 230.23 | 85.17 | 90.92 | 2.70 | 2.53 | fa3_cg | 85.17 |
| 1q512s512_31q1s542_1q1s32768 | 544 | 424.27 | 105.42 | 110.26 | 4.02 | 3.85 | fa3_s16 | 105.28 |
| 1q512s512_31q1s542_1q1s4512 | 544 | 89.99 | 67.20 | 78.89 | 1.34 | 1.14 | fa3_s8 | 67.03 |
| 1q512s512_63q1s542 | 575 | 77.74 | 84.53 | 90.51 | 0.92 | 0.86 | fa3_auto | 77.74 |
| 1q512s512_63q1s542_1q1s16384 | 576 | 259.46 | 131.57 | 123.38 | 1.97 | 2.10 | fa3_phase_split | 123.38 |
| 1q512s512_63q1s542_1q1s32768 | 576 | 453.53 | 150.78 | 140.50 | 3.01 | 3.23 | fa3_phase_split | 140.50 |
| 1q512s512_63q1s542_1q1s4512 | 576 | 119.49 | 98.75 | 107.40 | 1.21 | 1.11 | fa3_s4 | 98.71 |

Overall count of best-config wins (all specs):

| best_cfg | wins |
|---|---|
| fa3_s1 | 48 |
| fa3_auto | 31 |
| fa3_cg | 25 |
| fi_xqa | 23 |
| fa3_phase_split | 23 |
| fa3_s16 | 19 |
| fi_native_fs256 | 16 |
| fi_native | 15 |
| fa3_s4 | 14 |
| fa3_s8 | 8 |
| triton | 6 |
| fa3_s2 | 4 |
| fi_native_fs1024 | 2 |

## (d) Anomalies / failures

Rows with errors / graph failures / maxdiff>2e-2: 0

Max abs diff vs fa3_auto per config:

| config | max maxdiff vs fa3_auto |
|---|---|
| fa3_auto | 0.0000 |
| fa3_cg | 0.0020 |
| fa3_phase_split | 0.0005 |
| fa3_s1 | 0.0010 |
| fa3_s16 | 0.0020 |
| fa3_s2 | 0.0020 |
| fa3_s4 | 0.0020 |
| fa3_s8 | 0.0020 |
| fi_native | 0.0039 |
| fi_native_fs1024 | 0.0039 |
| fi_native_fs256 | 0.0039 |
| fi_native_nosplit | 0.0039 |
| fi_xqa | 0.0039 |
| triton | 0.0078 |

fa3_cg metadata identical to vLLM's full-cudagraph builder output (<=512 tokens): 97 match, 89 mismatch.
