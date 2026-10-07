# Stage 1. Foundations: which regimes exist in a vLLM step, and what does each cost?

Status 2026-10-07. vLLM 0.30.0, FA3 (vllm-flash-attention @506341a), one H100-80GB, Llama-3.1-8B geometry
(32 query heads, 8 KV heads, head dim 128; Qwen3-8B has the same attention geometry). Times are per attention
layer unless marked; multiply by 32 layers for a Llama step. ServingStudio numbers are H200 and are marked.

## Summary

1. **The X step is real at the kernel level, but only while the prefill in the step is cheap.** With a prefix of
   8k tokens or less, a 513-2,048-token chunk next to one 128k decode runs 2.3-7.7x slower than the best split.
   At a 32k prefix the loss falls to 1.2-1.9x, and at 128k it disappears (at most 1.05x): the decode tiles
   run under the long prefill.
2. **FA3 treats vLLM's `num_splits` as a cap, not a choice.** FA3 always applies its own per-sequence split
   rule, and with an expensive prefill in the launch that rule gives every sequence 1 split. No split setting,
   and so no C3-style override, can fix a step with a long-prefix or large prefill.
3. **H5 (wave quantization) is confirmed, and it is large.** An append that needs one tile more than a full wave
   pays for a whole extra wave. At prefixes of 8k-128k, 513-544-token appends run at 55-62% efficiency, 640 at
   67-72%, and 1,057-1,088 at 76-81%. One token more, 512 -> 513, costs 1.71-1.79x; 1,056 -> 1,057 costs
   1.28-1.31x. No split bound recovers it.
4. **Mixing decodes into a long-prefix step changes the wave count.** Against separate prefill and decode
   launches, the mixed launch costs 0.88x (the decodes fill an idle second wave) to 1.17-1.41x (they push the
   prefill into one more wave, or a cold 8k chunk waits for a 1-split 128k decode).
5. **The corrected cost model (FA3 tile scheduler) predicts all of this.** With constants fit before the F6 run
   it is within 7.0% mean error on F6 (5% at prefixes of 8k and more), and within 1.5% on 40 X steps recorded
   in earlier engine runs. Both Stage 1 gates pass. The old additive model overstates X at long prefixes.
6. **Earlier results stand for short prefixes.** Re-priced with the new model, the X-heavy runs still spend
   48-54% of GPU time in X steps (55.6% and 50.5% under the old labels).

## Question

What did the original study find, which parts survive a corrected cost model, and how do the regimes extend to
long cached prefixes, which every coding-agent round has?

## Prediction (written before the F6 run)

1. The earlier results stand for short prefixes.
2. The additive straggler model (`features.x_ratio`, `step_cost`, C3 `_decide`) overstates X when the prefill in
   the step is expensive, because FA3 runs the decode tiles and the prefill tiles at the same time.
3. H5 exists at prefixes of 32k and more: an append of 513-544 tokens on a 128k prefix runs at about 52% SM
   efficiency.

The model's predictions for every F6 shape were written to `results/microbench/attn_f6_pred_pre.jsonl`
(2026-10-07 01:56 UTC, sha256 `8ff9d6ea...d034`) before job 1444 produced data (03:14 UTC).

Outcome: 1 and 2 hold. 3 holds in direction and is slightly milder than predicted: 58-62% measured against
49-52% predicted, and it already appears at an 8k prefix.

## Method and tools

| Step | Tool | Output |
|---|---|---|
| Restate the original findings | `docs/03_results.md`, `reports/fa2_vs_fa3_engine.md`, `results/microbench/FA_OCCUPANCY.md` | section 1 |
| Read the FA3 split and scheduler code | `third_party/vllm-flash-attention/hopper/{flash_api.cpp,flash_prepare_scheduler.cu,tile_size.h}` | section 2 |
| Replace the additive model with a tile-scheduler model | `regime_batching/features.py` (`fa3_tile_time`, `attn_step_model`); fit and check: `microbench/validate_tile_model.py` | section 3 |
| F6 long-prefix microbench and replay of 40 recorded X steps (GPU, job 1444, 9 min) | `microbench/attn_sweep.py`, `microbench/summarize_attn_f6.py`, `analysis/stage1_f6_tables.py` | sections 3, 5 |
| Regime atlas | F6 + `analysis/regime_atlas.py` | section 4 |
| Re-price recorded steps of earlier runs | `analysis/relabel_telemetry.py` | section 6 |
| ServingStudio kernel library and timing predict (H200) | `experiments/servingstudio/*.py` | section 7 |
| Stage 0 setup checks | TraceLab v0.0.2 + DuckDB (checksum OK), `session_runner` probe (job 1442), ServingStudio probe | section 8 |

## Results

### 1. What the original study found (baseline for every later stage)

| Finding | Evidence | Status after Stage 1 |
|---|---|---|
| X steps (T > 512, a prefill chunk >= 417 tokens, a long decode) cost the most kernel time: 56% of GPU time on X-heavy, 36 ms per step | job 1017 telemetry | stands for short prefixes (section 6); vanishes at long prefixes (section 5) |
| One X attention layer: 407 us against 85 us (4.8x) | microbench F3 | reproduced: 1.55 ms against 0.20 ms with a 128k decode (F6) |
| Step shaping at 512 tokens: 1.25-1.26x short E2E on the 8B models | jobs 1015/1017/1034, 1040 | not re-tested |
| C3 (per-step split override): 1.18x short E2E alone, nothing on top of shaping | same | works only while the prefill is cheap (section 2) |
| Decode kernel choice (FA3, FlashInfer, Triton): <= 5% of a step | microbench F1/F2/F4 | stands |
| FA2 never beats FA3 in vLLM | job 1390, FA_OCCUPANCY | stands; ServingStudio finds one narrow FlashInfer-fa2 win that vLLM already covers (section 7) |
| 31 x 30 + 1 x 4,000 tokens (the original example) is a negative control: X 0.6% of GPU time | job 1033 | stands |
| Oracle KV admission: 20-30x short E2E under memory pressure | jobs 1033/1039/1040/1041/1043 | not re-tested |

### 2. FA3 mechanics that decide the kernel cost of a step

Read from the source vLLM 0.30 builds:

1. **Static split bound.** For T <= 512 under full CUDA graphs vLLM passes `max_num_splits = 32`. Above 512 it
   passes 0, and FA3's heuristic treats the whole varlen batch as one sequence: 1 split when the prefill chunk
   has >= 0.8 x 132 (m-block x KV-head) tiles, which is 417 tokens for 4 query heads per KV head.
2. **The bound is a cap.** Whenever the bound is above 1, FA3 runs its own per-sequence rule
   (`flash_api.cpp:676`, `flash_prepare_scheduler.cu:159`): splits = ceil(n_blocks / blocks_per_sm), with
   blocks_per_sm = 1.1 x (sum over sequences of m_blocks x n_blocks) x 8 / 132. A large prefill makes
   blocks_per_sm large, so every sequence, including a 128k decode, gets 1 split even with a bound of 32. F6
   confirms it: at a 128k prefix no bound from 2 to 32 beats the default by more than 1.05x (section 5).
3. **Order.** vLLM puts decodes first. FA3's persistent varlen scheduler does not sort sequences
   (`varlen_sort_batches = false`); within a causal sequence it starts with the longest m-block. All tiles of a
   launch share the 132 SMs, so a long decode and a prefill run at the same time.
4. **Tile count.** A launch with a prefill uses kBlockM = 128 rows with 4 query heads packed per KV head, so a
   q-token chunk is 8 x ceil(q / 32) tiles. 512 tokens give 128 tiles (one wave); 513 give 136 (two waves for
   1.03 waves of work). 1,056 give exactly 264 (two waves); 1,057 give 272 (three).

### 3. Corrected cost model

`features.fa3_tile_time` lists the tiles of one launch in launch order and gives each to the SM that frees first.
A decode tile costs KV bytes / per-CTA bandwidth; a prefill tile costs FLOPs / per-CTA throughput. On top come
fixed per-tile and per-launch costs, a combine cost when any sequence splits, and an HBM floor for decode bytes.
Decode-only launches (kBlockM = 64) and mixed launches (kBlockM = 128) have separate constants.
`attn_step_model` prices a step four ways:

| Quantity | Meaning |
|---|---|
| t_default | the split vLLM picks |
| t_best | the best split bound in {1, 2, 4, 8, 16, 32} for the same launch (what C3 can reach) |
| t_sep | prefill and decodes as two launches, each at its best bound (the regime-separated counterfactual) |
| t_ideal | the default launch's work packed perfectly on 132 SMs (bound for re-tiling fixes) |

The new X label is t_default / t_best >= 1.2 on a structural X shape. The old `x_ratio` and `regime` fields are
unchanged, so earlier telemetry stays comparable. The live telemetry path only gains cheap fields
(`structural_x`, `pre_prefix`, `pre_tiles`, `pre_wave_eff`). `step_cost(..., attn_model="tile")` and
`REGIME_ATTN_POLICY=x_split_tile` (C3) use the tile model.

Accuracy (mean / max absolute error; `python3 microbench/validate_tile_model.py <files>`):

| Data | Rows (all split bounds) | vLLM default split | All split bounds |
|---|---|---|---|
| F3 mixed, used for the fit (job 1008) | 432 | 5.3% / 31.7% | 4.7% / 31.9% |
| F1, F2, F4, F5 decode-only, used for the fit | 972 | 4-9% / 37% | 4-11% / 36% |
| **F6, predicted before the run** | 2,184 | **7.0% / 56%** | 7.2% / 56% |
| F6 at prefixes of 8k, 32k, 128k | 1,620 | 5.0-5.4% / 17% | 6.2-7.2% / 20% |
| **40 recorded X steps, replayed** | 240 | **1.5% / 3.4%** | 6.9% / 20% |

The largest F6 errors are short chunks at prefix 0 (20-50 us, over-predicted by up to 56%, per-tile overhead).
8k chunks on long prefixes are under-predicted by about 13%. A refit on F1-F6 does not reduce the errors (6.9%
on F6), so the constants stay as fit before the run. For the 40 replayed X steps, the measured gain of the best
split is a median of 1.78x (model 1.89x), and model and measurement agree on the X label for all 40.

**Gates: within 10% on F3 (5.3%) and within 15% on F6 (7.0%). Both pass.**

The old additive model is close on F3 (mean 6.3% after a refit), because there the prefill is short. It adds the
straggler to the prefill, so at a 128k prefix it predicts that C3 saves 15% of the step for a 1k-token append
with three 128k decodes (`step_cost`, policies baseline vs proposed). The tile model and F6 both show no saving.

### 4. Regime atlas

Measured on H100 (F6 and earlier microbenchmarks), per attention layer. "Loss" = default / best reachable
alternative. Model-only shapes: `figures/atlas_model.md`; all F6 cells: `figures/f6_tables.md`,
`results/microbench/ATTN_F6.md`.

| Regime | Example step | FA3 path vLLM takes | Default | Best alternative | Loss | What reaches the alternative |
|---|---|---|---|---|---|---|
| Dw / Dk | decode-only, <= 512 tokens | CUDA graph, bound 32 | at the HBM limit | other decode kernels | <= 1.05x | nothing needed |
| Ds | 31 x 542 + 1 x 4.6k decodes | CUDA graph, bound 32 | 57.3 us | 57.2 us | 1.00x | nothing needed |
| X, short prefix | 513-2,048 chunk at prefix 0-8k + 1 x 128k decode | eager, 1 split | 1.55-1.58 ms | 0.20-0.69 ms (split bound) | 2.3-7.7x | split bound (C3) or keep T <= 512 (shaping) |
| X, 32k prefix | 513-1,024 chunk + 1 x 128k | eager, 1 split | 1.56-1.61 ms | 0.85-1.30 ms | 1.2-1.9x | split bound |
| X, cold big chunk (H4) | 8k chunk at prefix 0 + 1 x 128k | eager, 1 split | 1.63 ms | 1.15 ms (separate launches) | 1.41x | separate launches only (split bound: 1.02x) |
| A, wave-aligned | 512 / 1,056 / 2,112 append on 8k-128k | 1 split | 98-100% efficient | | 1.0x | |
| A, misaligned (H5) | 513-544 append on 8k-128k | 1 split | 55-62% efficient | aligned size | 1.6-1.8x per unit of work | wave-aligned chunk size or smaller tiles; no split bound |
| A + long decodes | 1,024 append on 128k + 3 x 128k | 1 split | 4.97 ms | 4.26 ms (separate launches) | 1.17x | separate launches; no split bound (1.00x) |
| A + long decodes, decodes in wave slack | 513 append on 128k + 3 x 128k | 1 split | 3.20 ms | 3.20 ms | 1.0x (separate: 0.88x) | the mixed launch is already the best |
| P / M | chunk + short decodes, no long decode | eager, 1 split | | | <= 1.05x | |

### 5. Long prefixes (F6, job 1444)

**X fades as the prefix grows** (default / best split bound, one 128k decode; full table in `figures/f6_tables.md`):

| Chunk | prefix 0 | 8k | 32k | 128k |
|---|---|---|---|---|
| 513 | 7.71x | 5.83x | 1.85x | 1.00x |
| 1,024 | 7.30x | 3.98x | 1.24x | 1.05x |
| 2,048 | 6.06x | 2.29x | 1.00x | 1.00x |
| 8,192 | 1.02x | 1.00x | 1.03x | 1.00x |

With three 128k decodes the short-prefix loss is 2.4-2.9x for 513-1,024-token chunks (the three decodes share 24
CTAs, so the best split gains less). Coding-agent rounds append about 1k tokens to 120-141k cached tokens (TraceLab medians), which is
the right-hand column: an in-session agent round is not an X step even when its shape is.

**H5: efficiency of a prefill alone** (per unit of causal attention work, relative to the best wave-aligned size
at the same prefix; model in brackets):

| Append | Tiles | prefix 8k | 32k | 128k |
|---|---|---|---|---|
| 417 | 112 | 83% (76%) | 85% (78%) | 85% (79%) |
| 512 | 128 | 98% (93%) | 98% (96%) | 100% (97%) |
| **513** | 136 | **55% (50%)** | **57% (49%)** | **58% (49%)** |
| 544 | 136 | 58% (53%) | 61% (52%) | 62% (52%) |
| 640 | 160 | 67% (61%) | 69% (61%) | 72% (61%) |
| 768 | 192 | 79% (72%) | 81% (73%) | 82% (73%) |
| 1,024 | 256 | 96% (94%) | 98% (96%) | 98% (97%) |
| 1,056 | 264 | 99% (97%) | 100% (99%) | 99% (100%) |
| **1,057** | 272 | **77% (71%)** | **76% (68%)** | **78% (67%)** |
| 1,088 | 272 | 79% (73%) | 80% (70%) | 81% (69%) |
| 1,584 | 400 | 85% (81%) | 85% (77%) | 85% (75%) |
| 2,112 | 528 | 100% | 100% | 100% |
| 2,113 | 536 | 89% (86%) | 89% (82%) | 89% (80%) |
| 8,192 | 2,048 | 100% (105%) | 97% (100%) | 97% (98%) |

At a 128k prefix, 512 tokens take 1.81 ms and 513 take 3.10 ms; 1,056 take 3.76 ms and 1,057 take 4.80 ms.
The loss shrinks with chunk size (one extra wave out of more), and is gone at 8k. The best split bound beats the
default by 5% or more in only one cell (1,536 at 32k, 1.07x).

**Mixed vs separate launches at a 128k prefix:** 1,024 + 3 x 128k: 4.97 ms mixed vs 4.26 ms separate (1.17x);
1,056 + 1 x 128k: 4.98 vs 3.96 ms (1.26x); 417 + 3 x 128k: 3.17 vs 2.26 ms (1.40x); 513 + 3 x 128k: 3.20 vs
3.65 ms (0.88x, the decodes fill the idle wave). Which way it goes depends on the wave slack the prefill leaves,
and the tile model predicts it (4,903 / 4,898 / 3,266 / 3,272 us predicted for these four).

### 6. Earlier runs re-priced with the tile model

`python3 analysis/relabel_telemetry.py` on the recorded steps (`results/regimes_study/relabel_prior_runs.md`):

| Run | X, old label | X, new label | Structural X | Modelled attention recoverable, share of GPU time: split bound / separate launches / perfect packing |
|---|---|---|---|---|
| Qwen3-8B X-heavy, B0 (job 1017) | 55.6% | 53.7% | 56.9% | 9.6% / 7.4% / 9.9% |
| Qwen3-8B X-heavy, shape | 7.3% | 4.3% | 7.0% | 0.6% / 0.5% / 0.9% |
| Llama-3.1-8B X-heavy, B0 (job 1040) | 50.5% | 48.2% | 50.8% | 8.0% / 6.2% / 8.3% |
| Llama-3.1-8B X-heavy, shape | 9.1% | 5.0% | 7.5% | 0.7% / 0.5% / 0.9% |
| Qwen3-8B memory burst, B0 (job 1033) | 1.4% | 0.0% | 0.0% | 0.0% / 0.0% / 0.1% |

The recoverable share counts attention kernel time only. The measured shaping gain (1.25x) is larger because
shaping also removes eager launches (H2), which this model does not price.

### 7. ServingStudio cross-check (H200, separate prefill and decode kernels)

- **Long-prefix prefill attention agrees with the tile model.** For a 1,024-token append on 126,976 cached
  tokens, ServingStudio's timing predict gives 104.3 ms of prefill attention (32 layers). The tile model gives
  101.9 ms on H100 (2% apart; FA3 is compute-bound here and both GPUs have 132 SMs). At 2,048 and 4,096 tokens:
  215.7 vs 204.2 ms and 420.0 vs 410.0 ms. ServingStudio extrapolates past its 65k-prefix kernel grid here.
- **ServingStudio misses H5.** For a 530-token append on 126,976 tokens it predicts 60.1 ms; the tile model
  predicts 101.3 ms, and F6 measures 3.11 ms per layer for 544 tokens on 131,072 (99.5 ms over 32 layers).
  ServingStudio's nearest measured cell is a 512-token append (one wave), so its extrapolation cannot show the
  wave edge.
- **Few-tile long-prefix appends.** In the kernel library, FlashInfer fa3 prefill reaches 160-176 TFLOPS at a
  128-token append on 32k-65k prefixes, 308-330 at 256 and 575-660 at 512. Throughput scales with the tile
  count (32, 64, 128 tiles on 132 SMs), the same mechanism as H5. FlashInfer fa2 with a KV split wins exactly
  there (7 of 135 cells: append 128-256, prefix >= 4k, 0.48-0.89x the time) and nowhere at append >= 512. vLLM
  already covers this case when the step stays at <= 512 tokens: the bound is then 32, and FA3's rule splits a
  256-token append on 128k into 2.
- **ServingStudio prices a mixed step as prefill plus decode attention** (138.1 ms for a 1,024-token append on
  126,976 tokens with 3 decodes of 126,976, against 126.3 ms without). It cannot show the wave interaction of
  section 5; it is the regime-separated counterfactual.
- **Its prefix cache serves `prefix_len` as cached KV**: a 1,024-token round on 125k cached tokens has a TTFT of
  about 125 ms, against 8.9 s for the same context prefilled cold.

### 8. Stage 0 setup checks (all pass)

- TraceLab v0.0.2 cloned, `syfi_coding_trace.duckdb` checksum matches.
- `session_runner` built (rustup stable; the system cargo 1.75 is too old). Live probe, job 1442
  (`experiments/s7_probe_session.sbatch`, Llama-3.1-8B, `--max-model-len 262144`, 48 small sessions,
  303 rounds): 303 of 303 rounds finished; server prefix hit rate 81.47% against 81.66% planned. vLLM's KV
  cache holds 443,184 tokens (3.4 sessions at 128k). Step telemetry records the KV length including the cached
  prefix, and requests join to the runner log by id (`cmpl-<session>_round_<idx>-...`).
- ServingStudio dense preset `Llama-3.1-8B/llama3_dense_tp_barebone` simulates TraceLab sessions (3 sessions,
  50 rounds). Its Analyzer gives only sampled per-step batch composition, not per-request lengths.

## Regimes observed

Stage 1 measures kernels and recorded synthetic runs, not production traffic. Step and GPU-time shares for chat
and agent traffic come in Stages 2 and 3. What Stage 1 adds is the map those shares are read against: X is a
short-prefix regime; long-prefix steps are A, and A has its own losses (H5, wave interaction with decodes).

## H1-H5 verdicts

Kernel-level verdicts only. The production verdicts need the GPU-time shares from Stages 2 and 3.

| Hyp. | Stage 1 verdict | Evidence |
|---|---|---|
| H1 X straggler | **Confirmed at the kernel level, prefix-dependent.** 2.3-7.7x at prefixes <= 8k, 1.2-1.9x at 32k, none at 128k. | F6, replay of 40 recorded X steps, re-pricing (section 6) |
| H2 512 graph cliff | Not tested at the kernel level (it is a CPU launch cost). | Stage 2/3 telemetry |
| H3 wrong budget unit | Shape-level only: decodes that share a step with a 1k-token append on 127k wait 8.6-15x as long as in a decode-only step (ServingStudio). | Stage 3 measures stalls |
| H4 re-prefill bursts | **Confirmed at the kernel level.** A short-prefix chunk next to 128k decodes is an X step (2.7-7.7x at prefix 0); an 8k cold chunk hides the split loss but still loses 1.15-1.41x against separate launches. | F6 |
| H5 wave quantization | **Confirmed at the kernel level:** 55-62% efficiency at 513-544 tokens, below 80% for 513-640 and 1,057 tokens (768 and 1,088: 79-82%), at prefixes of 8k and more. | F6 |

**Deviation from the fixed criteria (H5).** The support test was "efficiency < 80%, and a forced split gains >=
1.15x". Section 2 shows that no split bound can split a long-prefix prefill, and F6 confirms it (at most 1.07x),
so the second clause cannot be met by any configuration. The loss is recoverable, but by a different chunk size:
an aligned chunk is 1.28-1.79x cheaper per unit of work than the next misaligned one. I propose to replace the
clause with "a wave-aligned chunk size gains >= 1.15x per unit of work". This is for your decision; until then
H5 is reported against both versions.

## Optimization areas

Ranked here by kernel-level size; Stages 2-3 re-rank them by how much GPU time each regime takes in traffic.

| Area | Kind | Fixes | Size at kernel level | Notes |
|---|---|---|---|---|
| Keep steps that hold a long decode at <= 512 tokens (step shaping) | scheduling | H1 at short prefixes, H2 | 1.25x E2E measured earlier | known; no oracle needed |
| Wave-aligned chunk sizes: end chunks at or just below 512, 1,056, 1,568, 2,112, ... tokens, where 8 x ceil(q/32) tiles just fill whole waves of 132 | scheduling | H5 | up to 1.8x per unit of attention work | TraceLab's median adjusted append (~1k tokens) sits next to the 1,056 edge; Stage 3 measures the size distribution |
| Separate decode and prefill attention launches when it pays (per-step choice with the tile model) | scheduling + launch | H4 cold chunks, A + long decodes | 1.15-1.41x attention per step; can lose (0.88x) when decodes fill wave slack | needs a per-step decision, which `attn_step_model` provides |
| Split decodes independently of the prefill in a mixed launch | kernel (FA3 split rule) | H1 at 32k, H4 | up to 1.4x | FA3's rule sizes splits from the whole launch |
| Smaller m-tiles or split-KV prefill for few-tile, long-prefix appends | kernel | H5 | up to perfect packing (t_ideal) | FlashInfer fa2's KV split shows the effect at 128-256-token appends |

## Caveats

- F-family microbenchmarks time one attention layer in isolation (CUDA graph replay, median, L2 flush
  subtracted). In the engine a step adds GEMMs, launch gaps and, above 512 tokens, eager CPU launch.
- The tile model ignores L2 reuse and bandwidth contention between prefill and decode tiles beyond an HBM floor;
  its per-tile constants absorb both. It over-predicts very short chunks and under-predicts 8k chunks on long
  prefixes by about 13%.
- ServingStudio is an H200 simulator with separate kernels and uniform decode batches; its numbers past a 65k
  prefix are extrapolated.
- `reports/fa2_vs_fa3_engine.md` and `results/microbench/FA_OCCUPANCY.md`, cited in section 1, are from the
  previous session and are not committed yet.
