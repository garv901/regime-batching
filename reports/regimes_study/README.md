# Batching regimes in production serving

Question: does production serving form batches that land in regimes where vLLM's kernels run badly, and which
scheduling or kernel changes would recover the loss? The earlier study (repository README) found the X step and
the 512-token step boundary on synthetic workloads. This study checks them against chat traffic and real
coding-agent sessions (TraceLab), and extends the regime set to long cached prefixes.

Setting: vLLM 0.30.0, FA3, one H100-80GB, Llama-3.1-8B-Instruct; ServingStudio (H200 simulator) for the
regime-separated counterfactual and for MoE. Stage 3 uses Llama-3.1-Nemotron-8B-UltraLong-1M-Instruct (see
[the Stage 3 model](#stage-3-model)). Plan and fixed kill/support criteria:
[plan](#kill--support-criteria).

## Stages

Each report uses the same template: question, prediction (written before the run), method, results, regimes
observed, H1-H5 verdicts, optimization areas, caveats.

| Stage | Report | Status |
|---|---|---|
| 1 | [Foundations: regimes in a vLLM step and what each costs](01_foundations.md) | done, reviewed |
| 1b | [Regime census: every kind of step, whole-step cost, best reachable alternative](01b_regime_census.md) | done |
| 2 | [Chat baseline: Azure and Mooncake conversation traces](02_chat_baseline.md) | running (jobs 1459, 1460) |
| 3 | Coding agents (TraceLab replay on H100, UltraLong-1M model) | model checked (job 1461); not started |
| 4 | MoE (ServingStudio simulation) | not started |
| 5 | Synthesis: regimes x scenarios -> optimization areas | not started |

### Stage 3 model

Coding-agent sessions reach 120k-260k tokens of context. vLLM's RoPE table for Llama-3.1-8B ends at 131,072
positions (positions past it assert; Stage 1b job 1448), so Stage 3 uses `nvidia/Llama-3.1-Nemotron-8B-UltraLong-1M-Instruct`:
the Llama-3.1-8B architecture (32 layers, 32 query / 8 KV heads, head dim 128, hidden 4,096, MLP 14,336) continued-
pretrained with a llama3 RoPE scaling factor of 128 (1,073,152 positions). Differences: vocabulary 129,024 (vs
128,256), weights stored in float32 (served with `--dtype bfloat16`), license cc-by-nc-4.0 (research use).
Check (job 1461, `experiments/s10_ultralong_probe.sbatch`, `analysis/model_compare_census.py`):
- 140,000- and 250,000-token prompts run (32 output tokens each); a passkey 119,970 tokens deep in a 200,000-token
  prompt is retrieved.
- The Stage 1b default-config census (393 shapes) runs at Llama-3.1-8B's step times: median ratio 1.001 (p10 0.986,
  p90 1.017), 98% of shapes within 5% (Llama against its own repeat: 81%). So the Stage 1/1b cost models carry over.
- KV cache: 441,264 tokens at max-model-len 262,144 and utilization 0.92 (Llama-3.1-8B: 452,816).

## Vocabulary

Regimes describe what a step *is*:

| Label | Step | Source |
|---|---|---|
| X | prefill + long decode, T > 512, FA3 at 1 split, and the default split costs >= 1.2x the best split bound (tile model) | `features.attn_step_model` |
| A | a prefill chunk appended to >= 16k cached tokens (an agent round) | new in this study |
| A+L | A with a decode >= 16k in the same attention launch; FA3 gives the decode 1 split even at split bound 32 | new in Stage 1b |
| P / M | other prefill-bearing steps, prefill-heavy / mixed | `features.classify` |
| Dc, Dl, Ds, Dk, Dw | decode-only: GEMM ridge, low-occupancy long, skewed, KV-bound, weight-bound | `features.classify` |

"Structural X" is the shape rule alone (T > 512, prefill present, FA3 1 split, a decode >= 16k); it does not say
the step is slow.

Hypotheses describe what can go wrong:

| ID | Pathology | Kind |
|---|---|---|
| H1 | X straggler: a 1-split long decode in a mixed step | kernel configuration |
| H2 | 512 graph cliff: prefill-bearing steps above 512 tokens run eager | batching / runtime |
| H3 | Wrong budget unit: step cost grows with append x prefix, but vLLM budgets tokens, so decodes stall | batch formation |
| H4 | Re-prefill bursts after cold start, compaction, preemption or eviction | batch formation |
| H5 | Long-prefix wave quantization in FA3's 1-split path | kernel configuration |

## Verdict matrix

| | Stage 1 (kernel level) | Stage 1b (whole step, designed grid; evidence, not verdicts) | Stage 2 chat | Stage 3 agents | Stage 4 MoE sim |
|---|---|---|---|---|---|
| H1 | confirmed, prefix-dependent: 2.3-7.7x at prefix <= 8k, 1.2-1.9x at 32k, none at 128k ([01](01_foundations.md#h1-h5-verdicts)) | whole step up to 3.36x (<= 2,048 tokens; capture 2048 fixes) and 1.78x (> 2,048); at 120k prefixes up to 1.48x, fixed only by separate launches (A+L) | | | n/a (MLA/DSA) |
| H2 | not tested at kernel level (CPU launch cost) | launch cost median 0.3 ms at load 86-173; the 512 boundary costs through the split bound (H1) and the wave edge (H5) | | | |
| H3 | shape-level evidence only (ServingStudio) | an 8,192-token step: 211 ms fresh, 1,110 ms on a 120k prefix; co-scheduled decodes wait it out | | | |
| H4 | confirmed: cold short-prefix chunks are X steps; 8k cold chunks lose 1.15-1.41x vs separate launches | 1.6-3.4x for 512-1,057-token chunks at prefix 0-8k next to 120k decodes | | | |
| H5 | confirmed: 55-62% efficiency at 513-544-token appends on 8k-128k; no split bound helps | revised kernel-level clause met: same tokens 1.14-1.53x faster as aligned chunk + remainder (8 of 9 shapes >= 1.15x); trace-weighted clause pending Stage 3 | | | n/a (MLA/DSA) |

## Kill / support criteria

Fixed before the runs (plan of 2026-10-07). One change, approved 2026-10-07: H5's second support clause could not be
met by any configuration (FA3 treats the split setting as a cap), so it was reworded; both versions are below and
the reason is in [01](01_foundations.md#h1-h5-verdicts).

| Hyp. | Kill (not a production issue) | Support |
|---|---|---|
| H1 | measured X_cost steps (>= 1.2x attention and >= 1 ms per step) are < 2% of GPU time, and C3/cap2048 change matched-round TTFT/TPOT by < 3% | >= 10% of GPU time, or >= 1.1x from a fix |
| H2 | launch-bound steps with T > 512 are < 2% of GPU time | >= 10% |
| H3 | prefill-bearing steps are < 10% of decode-weighted wall time | >= 20%, and `mnbt512` improves decode ITL p99 by >= 1.5x at <= 10% TTFT cost |
| H4 | forced cold/miss/compaction rounds are < 10% of prefill attention FLOPs | >= 30% |
| H5 | trace-weighted FA3 prefill efficiency >= 90% | original: < 80%, and a forced split gains >= 1.15x. **Revised (2026-10-07):** < 80%, and an alternative that processes the same tokens (aligned chunk now + remainder in a later step of <= 512 tokens, or other work filling the idle wave) is >= 1.15x faster in total attention time, measured |
