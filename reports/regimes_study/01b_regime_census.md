# Stage 1b. Regime census: every kind of step, its whole-step cost, and the best reachable alternative

Status: running (in-engine and kernel-level jobs resubmitted after the 120k change). vLLM 0.30.0, FA3, one H100-80GB, Llama-3.1-8B-Instruct.

## Question

Stage 1 measured only attention, and went deep on two regimes (X and long-prefix appends). For every kind of step
vLLM forms (decode-only of any size and context mix, prefill alone, mixed, long-prefix), where does the step's
time go (GEMMs and other non-attention kernels, attention, CPU launch), how far is it from the best reachable cost
for the same work, and what reaches it (kernel configuration or batch formation)?

## Prediction (written before the run)

Quantitative: `analysis/census_predict.py` predicts every census step and every F7 attention cell. Whole step =
tile-model attention (32 layers) + non-attention cost by graph mode, fit on the earlier Llama-3.1-8B telemetry
(job 1040; 6,000 steps; mean error 2.9% full-graph decode, 4.1% piecewise, 4.5% eager):

| Graph mode | When | Predicted step time |
|---|---|---|
| FULL | decode-only, T <= capture size | attention + max(4.21 ms, 0.020 ms x T) + 1.92 ms |
| PIECEWISE | prefill in the step, T <= capture size | attention + max(4.21 ms, 0.020 ms x T) + 2.37 ms |
| NONE (eager) | T > capture size | max(16.8 ms, attention + max(4.21 ms, 0.020 ms x T) + 2.48 ms) |

Written to `results/regimes_study/census_pred_pre.jsonl` before either job produced census data: first at
06:59 UTC (sha256 `2df85b74...afa`, commit 842a089), then regenerated at 08:12 UTC (sha256 `afa72c14...7048`)
after the grid moved from 128k to 120k contexts (see Method). The fit is unchanged and all 2,061 predictions
for shapes in both grids are identical; only the 120k shapes are new. The 16.8 ms eager floor is the CPU launch time of one step; it was
measured when the node's load average was about 650 and is the least certain constant.

Qualitative:
1. Decode-only steps sit at the weight or KV bandwidth floor; the attention kernel moves them by <= 5%. Their only
   lever is batch size (cost per token).
2. Steps above the capture size pay the eager launch cost, which capture 2048 removes up to 2,048 tokens.
3. X steps lose to both the split choice and the eager cliff; capture 2048 fixes both, because FA3 then gets split
   bound 32.
4. Long-prefix appends keep the H5 loss under every engine config. A small remainder (<= 64 tokens on 120k) in a
   step of <= 512 tokens costs <= 0.25 ms of attention per layer.

## Method

| Step | Tool | Output |
|---|---|---|
| Census grid: 9 decode sets x prefill probes (q 16-8,192, step size crossing 512, cached prefix 0-120k) + 5 two-chunk steps; 385 shapes | `experiments/census_shapes.py` | |
| Whole step in-engine: in-process vLLM driven step by step, each probe 3 times; default vs `max_cudagraph_capture_size=2048` | `experiments/regime_census.py`, job 1448 | `results/regimes_study/census_steps.jsonl` |
| Attention for the same shapes + Stage 1 gaps under FA3 split bounds, separate launches, FlashInfer, Triton (420 shapes) | `microbench/attn_sweep.py` F7, job 1449 | `results/microbench/attn_f7.jsonl` |
| Pricing, roofline, grouping | `analysis/regime_census.py` | `figures/census_tables.md` |

Decode sets: none; 1 x 120k; 3 x 120k; 4 x 24k; 31 x 1k + 1 x 120k; 32 x 1k; 8 x 8k; 128 x 2k; 256 x 1k.

**Why 120k, not 128k.** The first attempt (job 1448) crashed in its smoke test with a device-side assert on the
first step whose tokens sat past position 131,071. vLLM builds Llama-3.1's rotary-embedding table with
`max_position_embeddings` = 131,072 entries (`rotary_embedding/base.py`, `_compute_cos_sin_cache`), and
`VLLM_ALLOW_LONG_MAX_MODEL_LEN` does not enlarge it. So Llama-3.1-8B cannot serve contexts above 131,072 tokens
in vLLM, and the longest census context is 120k plus an 8k chunk. (This also limits Stage 3: TraceLab sessions
past 128k need a cap or another model.) The kernel job was cancelled after 103 shapes so that both jobs measure
the same grid; its 128k rows are kept in `results/microbench/attn_f7_128k_partial.jsonl` (960 rows, 0 errors).

## Results

Pending.
