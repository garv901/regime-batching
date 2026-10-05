# Pen-and-paper analysis: where regime-aware batching + kernel dispatch wins (and where it does not)

Status 2026-10-03. Setting: vLLM 0.30.0 (FA3 backend, async scheduling, FULL_AND_PIECEWISE CUDA graphs, capture
max 512, max_num_batched_tokens 8192), one H100-80GB, oracle output lengths (`ignore_eos` + per-request
`max_tokens` through `vllm bench serve --dataset-name timed_trace`). Calibrated simulator:
`analysis/pen_and_paper.py`; in-engine telemetry: `regime_batching/telemetry.py`; real runs: `results/runs/`.

## TL;DR
1. **Highest-gain, low-hanging lever: the T = 512 regime boundary.** A vLLM step with more than 512 tokens
   loses (a) the full-CUDA-graph split bound (FA3 then picks 1 split whenever a prefill chunk is >= 417 tokens,
   so every long-KV decode row becomes a one-CTA-per-KV-head straggler, "X regime") and (b) CUDA graphs
   altogether (eager; launch-bound on our shared node). Workloads with long-KV decodes plus a steady stream of
   >= 417-token prompts spend most GPU time there: **56% of GPU time in X steps** (36 ms/step) on the X-heavy
   workload (4 x 24k-context decodes + 24 req/s of 450-1000-token prompts, Qwen3-8B).
2. **Regime-aware step shaping with oracle lookahead (C4 `cg_cap`) is the best single mechanism on X-heavy
   workloads** (real H100, Qwen3-8B, X-heavy 24k, 3 repeats, each normalized to its own same-job default run;
   `results/runs/s4a_1034/repeats.md`):

   | config | short E2E | short E2E p99 | long TPOT | short TPOT | TTFT |
   |---|---|---|---|---|---|
   | static capture size 2048 (1 run) | 1.14x | 0.99x | 1.15x | 1.17x | 1.03x |
   | static `--max-num-batched-tokens 512` | 1.08 +- 0.01x | 1.13 +- 0.04x | 1.31 +- 0.01x | 1.51x | **0.47x** |
   | C3 kernel-only (per-step FA3 split override) | 1.18 +- 0.03x | 1.03 +- 0.01x | 1.15 +- 0.02x | 1.20 +- 0.03x | 1.11 +- 0.02x |
   | **shape** (C4 cg_cap) | **1.26 +- 0.03x** | 1.13 +- 0.02x | 1.26 +- 0.02x | 1.36 +- 0.04x | 0.96 +- 0.01x |
   | full (shape + C3 + peak admission) | 1.27 +- 0.03x | 1.11 +- 0.04x | 1.26 +- 0.01x | 1.37 +- 0.02x | 0.96 +- 0.05x |

   Shaping beats every static config on mean short E2E without the 2x TTFT cost of the 512-token budget.
   C3 adds nothing on top of shaping (shaping removes most X steps); alone it gives 1.18x short E2E and 1.15x
   long TPOT (Qwen3-4B, 1 run: 1.15x short E2E, 1.14x long TPOT).
3. **Where the oracle is essential: KV admission under memory pressure.** Calibrated simulation: when vLLM's
   greedy admission over-commits KV (tens to thousands of recompute preemptions), an oracle peak-KV planner with
   backfill improves short-request E2E by 10-1000x (median 7x across unstable scenarios). Reactive policies
   cannot do this: they do not know which admitted request will grow to 8-32k tokens. **Real H100 burst**
   (Qwen3-8B, 16 req/s shorts + 80 x 8000-token outputs at t = 30 s, job 1033) vs vLLM default: short TTFT
   26.7 s -> 0.20 s (135x), short E2E 29.7 s -> 1.03 s (28.8x; p99 105 s -> 6.0 s, 17.5x), long TPOT 1.43x,
   0 preemptions vs 52 (219k recomputed tokens), at -5% output throughput; repeat (job 1043): 29.5x short E2E, 148x
   TTFT. Replicated on Llama-3.1-8B (20.0x) and Qwen3-4B (19.4x). Oracle SJF ordering without KV planning: 1.40x (306 preemptions). **No static config comes close:**
   capping concurrency (`--max-num-seqs` 32/64/128/256, job 1039) either starves shorts behind the longs'
   seats (short E2E 148 s / 79 s at 32/64) or changes nothing (30.5 s / 29.6 s at 128/256).
4. **Where there is little to gain:** the user's literal example (waves of 31 x 30-token + 1 x 4000-token
   requests) on an 8B model is weight-bound decode (Ds 50%, Dl 26%, Dw 12% of GPU time; X 0.6%). The long
   request's attention is already balanced by vLLM's full-CUDA-graph split path; the best alternative decode
   kernel saves <= ~3% of a step. Measured: C3 changes nothing there (as predicted).
5. **Novelty**: no prior work found that reports the T = 512 split/graph cliff, or that co-designs the
   scheduler and the per-step attention kernel inside one engine using output-length foresight
   (`reports/related_work.md`; closest: CascadeInfer (multi-instance), PersistentKV / PackInfer (decode-only
   kernels), Past-Future scheduler (estimated, kernel-agnostic peak admission), vLLM PR #58013 (FA2 call split)).

## 1. Cost model and calibration
Per step: `t = max(W/BW_W, 2*P*T/F) + L*(attn_dec + attn_pre + straggler excess) + overhead`.

| constant | value | source |
|---|---|---|
| BW_W (weight streaming) | 2.29 TB/s | in-engine fit, decode-only steps: 6.60 ms + 0.35 ms/GB KV (xstress_16k B0, 27k steps, rmse 0.30 ms) |
| BW_KV (balanced decode attention) | 2.85 TB/s | same fit |
| F (dense step throughput) | 480 TF/s | in-engine: 512-token prefill step 15 ms |
| BW_CTA (single CTA streaming) | 58 GB/s | microbench fa3_s1, B=1 (50/56/58/59 GB/s at 4k/16k/32k/64k) |
| beta_X | 0.8 | X attention ~ t_straggler + 0.8 t_balanced (microbench) |
| forced-split penalty on non-X mixed steps | 1.08x attention (max 1.18x) | microbench F3 |

Validation: the analytical X tax is `L_layers * maxL * 2*2*hd / BW_CTA` = **0.318 ms per 1k tokens** of the
longest decode (Qwen3-8B); the in-engine regression of X-step time gives **0.350 ms per 1k** (902 X steps).
X-predictor vs microbench: log-correlation 0.85.

## 2. Regime taxonomy and measured time shares
P prefill-heavy; M mixed; **X** mixed with an unsplit long-KV decode; Dw weight-bound decode; Dk KV-bound decode;
Ds skewed decode; Dl low-occupancy long-context decode; Dc large-batch decode.

| workload (Qwen3-8B, default vLLM) | GPU-time share by regime | note |
|---|---|---|
| smoke bimodal 4k (240 req) | Ds 46, Dw 31, Dl 18, X 4.9 | X steps T~522, 14.6 ms, all eager + nosplit |
| user waves 4k (31x30 + 1x4000 per 2 s) | Ds 50, Dl 26, Dw 12, P 12, X 0.6 | P bursts 148 ms (ITL p99 45 ms) |
| xstress 16k (8 req/s, 512 prompts, 1/64 long) | Dl 82, Ds 8, X 5 | X 17.3 ms vs Dl 11.1 ms |
| **X-heavy 24k** (4 x 24k ctx + 24 req/s 450-1000) | **X 56**, Ds 27, P 8 | X 36 ms/step |

## 3. The X regime: mechanism
- vLLM's FA builder passes `max_num_splits = 32` only when full CUDA graphs are on and T <= max capture size
  (512). Above that it passes 0 and FA3's heuristic treats the varlen batch as batch size 1: if
  `h_kv * ceil(max_q*g/kBlockM) >= 0.8 * 132` (i.e. any prefill chunk >= 417 tokens for Qwen3-8B geometry), it
  picks 1 split. No sequence is split, so each long decode row is streamed by one CTA per KV head at ~58 GB/s.
- Microbench (F3 mixed): default vs best 1.40x geomean, up to **4.81x** in attention (one 512-token chunk + 15
  short decodes + one 32k decode: 407 us vs 85 us per layer). Chunks of 256/416 tokens still split.
- Second cliff at the same boundary: steps above the capture size run without any CUDA graph. On this node
  (load average ~650) eager X steps are CPU-launch-bound: host time 16.5 ms ~ GPU-event time 16.8 ms, while
  graphed decode steps need 1.5 ms of host time. This caps what a kernel-only fix can recover (C3 cut X-step GPU
  time 16.8 -> 15.3 ms but wall stayed ~17.5 ms on xstress) and is why step shaping wins: it fixes both cliffs.
- Relative tax per X step at T~520: maxL 16k -> ~34% (8B), ~64% (4B); 32k -> ~68% / ~128%. For large chunks
  (T = 8192, GEMM ~235 ms) the tax is <= 4%: X hurts most when chunks are just above the boundary.

## 4. Kernel headroom outside X (microbench, `results/microbench/ATTN_SWEEP.md`)
Default vs per-shape best: homogeneous decode 1.04 geomean (1.30 max, only at tiny B/L), skewed decode 1.03
(1.21 max, FlashInfer native on 15x2k + 1x4.5k), steady 1.03, single 1.01. FlashInfer XQA is 1.66x slower than
FA3 on skewed batches. These are attention-only; with weights at 6.6 ms/step a 15% attention gain is <= ~5% of a
decode step for an 8B model. Conclusion: per-step decode kernel switching alone is not where the gains are.

## 5. Gain search (calibrated simulator v3, 108 scenarios x 9 variants)
Grid: Qwen3-8B/4B x prompts {512, ~1930 lognormal, ~4096 lognormal} x long outputs {16k, 32k, reasoning CDF} x
long fraction {1/256, 1/64} x {4, 8, 16} req/s. Variants: B0 default, B1s forced splits everywhere, B512 static
512 budget, K kernel-only (C3), Bx chunk cap 416, Bc oracle step shaping, S oracle peak-KV admission, KS, KSc.
Files: `results/pen_and_paper/sim_v3_focus.jsonl`, `sim_v3b_focus.jsonl`, `sim_v3_merged.csv`.

| lever | short-E2E speedup (sim) | real H100 so far | oracle needed? |
|---|---|---|---|
| Oracle peak-KV admission + backfill (S) | vs best static: p90 91x, max 998x where vLLM preempts; ~1.0x otherwise | burst vs default: **28.8x** short E2E, 135x TTFT (section 5b) | **yes** (future peak) |
| Step shaping at the CG boundary (Bc / KSc) | vs B0: median 1.20x (KSc), p90 1.29x (Bc); vs best static ~1.0 median, max 1.3x (B0-stable) | **1.28x** E2E, 1.28x long TPOT | lookahead (remaining tokens); mostly reactive in this workload |
| X kernel dispatch (K / C3) | vs B0 median 1.02x, p90 1.07x, max 1.14x (stable) | 1.16-1.19x (8B), 1.15x (4B) | no (current KV) |
| Decode kernel switching (FI/XQA/Triton) | <= ~5% of a step | - | no |
| GEMM per-M | Triton beats cuBLAS only at M <= 64 on o/down (<= ~5% of a decode step, partly warm-L2 artifact) | not built | no |

Sim caveats: it does not model the eager/launch cliff (so it under-predicts shaping and over-predicts the static
512 budget, which the real run shows loses 2x TTFT), and its 512-budget variant beats the regime-aware variants
in some low-rate cases (`analysis/sim_v3_rank.py` prints the worst cases; the peak planner is too conservative in
one 4B case: 0.26x).

### 5a. GEMM per-M sweep (`results/microbench/gemm_sweep.parquet`, job 1016)
Qwen3-8B linear layers (qkv, o, gate_up, down; cuBLAS): summed GEMM time is flat up to M = 128 (1.06x of M = 1),
then 1.28x at 256, 2.30x at 512, 4.55x at 1024 -- the weight-to-compute ridge sits at M ~ 256-384. cuBLAS reaches
0.73-0.90 of the bandwidth roofline at M <= 64 and ~0.67-0.70 of peak compute at M >= 1024. Triton autotune
beats cuBLAS only at M <= 64: o_proj up to 1.8x (12.7 -> 7 us; o_proj's 32 MB weight fits in L2, so part of
this is a warm-cache artifact) and down_proj ~1.1x; at M >= 96 cuBLAS wins (Triton 0.5-0.9x). Upper bound
~0.35 ms of a 6.6 ms decode step (~5%), not regime-coupled -> GEMM dispatch not built. The flat region is what
makes step shaping cheap: a 512-token step costs 2.3x a decode step's GEMMs, while a 513-token step additionally
loses CUDA graphs.

### 5b. Memory-pressure burst (real H100, job 1033; figures `reports/figures/s4b_1033/`)
`lburst80_8k`: 16 req/s shorts (prompts 450-1000, outputs 16-64) for 300 s, plus 80 requests (512-token
prompt, 8000-token output) arriving at t = 30 s; final KV of the longs ~681k tokens vs 451k cache.

| case | short TTFT mean / p99 | short E2E mean / p99 | short TPOT | long TPOT | long E2E | out tok/s | preemptions |
|---|---|---|---|---|---|---|---|
| B0 (vLLM default) | 26.7 s / 99.2 s | 29.7 s / 105 s | 76.8 ms | 29.3 ms | 235 s | 2487 | 52 (219k tokens recomputed) |
| peak (oracle admission only) | 0.20 s / 3.6 s | 1.03 s / 6.0 s | 21.7 ms | 20.5 ms | 240 s | 2351 | 0 |
| full (peak + C3 + cg_cap) | 0.23 s / 4.0 s | 1.02 s / 5.96 s | 20.6 ms | 20.1 ms | 235 s | 2389 | 0 |
| static `--max-num-seqs 32` | 148 s / 218 s | 148 s / 218 s | 15.4 ms | 14.1 ms | 201 s | 2061 | 0 |
| static `--max-num-seqs 64` | 78.2 s / 160 s | 79.2 s / 161 s | 25.3 ms | 20.9 ms | 200 s | 2362 | 18 |
| static `--max-num-seqs 128` | 28.8 s / 99.2 s | 30.5 s / 102 s | 43.5 ms | 28.8 ms | 231 s | 2518 | 56 |
| static `--max-num-seqs 256` | 26.9 s / 99.4 s | 29.6 s / 105 s | 70.0 ms | 29.0 ms | 232 s | 2507 | 53 |

Best static on short E2E is `--max-num-seqs 256` (~= default): the oracle planner is **28.7x** better on short
E2E mean, 17.5x on p99, 136x on TTFT, and 1.41x on long TPOT. Static seat caps trade the classes against each
other: fewer seats let the admitted longs decode faster (long TPOT 14 ms at 32 seats), but the longs hold the
seats and the shorts queue for minutes. Only knowing each request's future KV peak avoids both. Caveat: the
static runs drop 12-16 requests to API-server disconnects under the backlog (peak: 0-1); dropped requests are
excluded from their latency statistics, which flatters the static configs.

What happens (`compare/compare_ribbons.png`): B0 admits every long at t = 30 s, the KV cache fills as they grow,
and from ~137 s to ~230 s no new request can be admitted -- the timeline is pure KV-bound decode (Dk) while
shorts queue, then ~65 s of preemption recompute plus backlog prefill (P steps, 57 ms mean, 35% of GPU time).
The oracle planner admits longs only while the peak of the admitted set's future KV fits, and backfills
shorts into the remaining headroom: 7,992 of 19,059 scheduler calls held back at least one request (up to 128 at once).
Shorts never wait for KV, longs still finish at the same time (235-240 s), and throughput drops ~5% because
fewer longs decode concurrently early on. Shaping/C3 add little here (X is 1.4% of GPU time).

### 5c. Ablations and cross-model replication (jobs 1040, 1041, 1043)
**Does shaping need the oracle?** No, not on this workload. Reactive shaping (cap whenever a >= 4k-KV decode
runs, ignoring remaining tokens; `REGIME_SHAPE_MIN_REMAIN=0`) gives 1.27x short E2E, 1.27x long TPOT on Qwen3-8B
X-heavy 24k, inside the oracle variant's 1.22-1.33x range. The long requests have thousands of tokens left for
almost their whole life, so the remaining-length check rarely changes a decision. Shaping is a *regime-aware*
win, not an *oracle* win.

**Does the burst win need peak-KV planning, or just short-first ordering?** Peak-KV planning. Oracle
shortest-remaining-first ordering without KV planning (`REGIME_SCHED_ADMIT=sjf`, what SSJF/LTR-style
schedulers do given perfect lengths): short E2E 20.8 s (1.40x vs the same-job default) vs 1.03 s (28.8x) for the peak
planner; its ITL p99 gets worse (1.4 s vs 0.44 s) because preemptions continue. Ordering cannot stop the
already-admitted longs from growing into the cache; only planning their future peak can.

**Replication** (each vs its own same-job default; single runs):

| model | X-heavy: shape short E2E / p99 / long TPOT | best static on X-heavy (short E2E) | burst: peak short E2E / TTFT / long TPOT / throughput |
|---|---|---|---|
| Qwen3-8B (3 repeats) | 1.26 +- 0.03x / 1.13x / 1.26x | cap2048 1.14x | 28.8x / 135x / 1.43x / -5% |
| Llama-3.1-8B | 1.25x / 1.17x / 1.25x | cap2048 1.16x | 20.0x / 92x / 1.35x / -7% |
| Qwen3-4B | 1.14x / 1.07x / 1.18x | **cap2048 1.17x** (p99 0.98x) | 19.4x / 97x / 1.37x |

Qwen3-4B is the exception on X-heavy: raising the capture size beats shaping on mean short E2E (shaping still
wins p99 and long TPOT). With half the GEMM work per token, a 512-token step is cheap and the extra steps that
shaping creates cost relatively more than the X tax it removes; a cost-model-driven cap (choose between 512 and a
larger graphed size per step) is the natural fix.

## 6. Model choice
| model | role | why |
|---|---|---|
| **Qwen3-8B** (36L, 32q/8kv, hd128) | primary | weight floor 6.6 ms; 16-32k KV gives 34-68% X tax at T~520 |
| **Qwen3-4B** (same attention geometry, half the GEMM bytes) | amplifier | same X tax, ~2x relative effect; measured C3 1.15x |
| Llama-3.1-8B (32L, 32/8/128) | replication | same family of geometry; cached |
| Llama-3.2-3B-Instruct | optional | now cached (access granted) |

## 7. Favourable scenarios (to demonstrate the oracle idea)
- **X-heavy long context / long output**: a few requests with 16-32k KV (long prompts or long outputs) alive for
  most of the run + a steady 16-32 req/s stream of 450-1000-token prompts with short outputs
  (`data/workloads/xheavy_lc{16,24}k.jsonl`). Demonstrates C3 + C4 shaping.
- **Memory-pressure burst**: steady shorts + a burst of long-output requests whose final KV exceeds the cache
  (`data/workloads/lburst80_8k.jsonl`: 80 x 8000-token outputs at t=30 s, ~1.5x the 451k-token cache).
  Demonstrates the oracle planner (no preemption, backfilled shorts) - the part a reactive system cannot do.
- Not favourable (report as negative control): user-wave example on 8B, xstress at 8 req/s.

## 8. Differentiation (details and links: `reports/related_work.md`)
| work | what it does | difference |
|---|---|---|
| CascadeInfer (arXiv 2512.19179) | multi-instance length-range grouping because attention is length-mix sensitive | we stay in one engine on one GPU and make mixing cheap (kernel + step shaping) instead of separating it; oracle lookahead |
| PersistentKV (2606.26666), PackInfer (2602.06072) | per-step decode kernel policy / length-packing kernels | decode-only, no hybrid steps, no CUDA-graph boundary, no scheduler co-design |
| POD-Attention (ASPLOS'25), LeanAttention | fused prefill+decode / stream-K decode kernels | new kernels; we dispatch existing kernels and reshape steps; orthogonal |
| vLLM PR #58013 (FA2), forks | split mixed varlen call so decodes get split-KV | FA2/Metal only, static; we found the FA3 + CG-boundary mechanism and gate per step with a cost model |
| vLLM #56975 | steps with prefill leave CUDA graphs (raise capture size) | static knob, we include it as a baseline (capture 2048: 1.14x vs our 1.28x) |
| Sarathi-Serve | chunked prefill, token budget | token-count budget; ours keys the budget on the kernel/graph regime and on oracle remaining work, lifted under backlog |
| S3, LTR, SSJF, Past-Future (2507.10150), SageSched | output-length prediction for admission/ordering | scheduling only, predicted lengths; ours couples admission with kernel regimes and evaluates the oracle upper bound |
| DistServe/Splitwise/TetriInfer | disaggregate prefill and decode | multi-GPU separation; we keep one engine |

## 9. Caveats
- Shared node (load avg ~650, other users on GPUs 1-2): eager steps are more CPU-bound than on a dedicated
  server, which inflates the graph cliff and deflates kernel-only gains. All headline comparisons are same-job,
  same-GPU pairs; repeats pending.
- C3 numerics (X-step stress test `experiments/c3_correctness.py`: one long-context decode + a new 512+-token
  prompt every step, 101 greedy requests). **C3 stays inside stock vLLM's own run-to-run noise:**

  | comparison | identical greedy outputs | max logprob diff (identical seqs) |
  |---|---|---|
  | stock vs stock, two separate jobs (1015 vs 1034, same code/config) | 97/101 | 0.161 |
  | C3 vs stock, same job (1034) | 101/101 | 0.067 |
  | C3 vs stock, same job (1015) | 96/101 | 0.15 |
  | stock FULL_AND_PIECEWISE vs stock PIECEWISE, same job | 101/101 | 0.0 (all steps eager in both modes) |

  Split-KV reorders the softmax accumulation, so C3 is not bitwise equal to stock; neither is stock to itself
  across runs (vLLM #51902 reports batch-composition-dependent FA3 numerics).
- C4 wrapper overhead 76 us median per schedule() call (target 50 us); async scheduling hides it off the GPU
  critical path, but it is not free on a CPU-starved node.
- Simulator is a guide only; every claim above that matters is or will be confirmed on the GPU.

## 10. Verdict (2026-10-03)
- **Supported:** given oracle output lengths, *planning admission against each request's future KV peak* is a
  large, robust win under memory pressure (20-30x short E2E, 90-150x TTFT, 1.35-1.43x long TPOT, 3 models,
  repeat within 3%), at a 5-7% throughput cost. Neither static knobs (`--max-num-seqs`) nor oracle-length
  ordering (SJF, 1.40x) reproduce it. This is the part that needs the oracle.
- **Supported, but not oracle-dependent:** regime-aware step shaping at the CUDA-graph / split boundary (1.25x
  short E2E on two 8B models, above every static config). A reactive rule does equally well here.
- **Small:** per-step kernel dispatch. C3 gives 1.15-1.18x alone on X-heavy workloads, nothing on top of shaping;
  decode-kernel switching outside X and GEMM dispatch are <= ~5%.
- **Not supported:** the motivating 31 x 30 + 1 x 4000 example on an 8B model (weight-bound, < 3% headroom).

