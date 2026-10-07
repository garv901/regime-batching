# Batching regimes in production serving

Question: does production serving form batches that land in regimes where vLLM's kernels run badly, and which
scheduling or kernel changes would recover the loss? The earlier study (repository README) found the X step and
the 512-token step boundary on synthetic workloads. This study checks them against chat traffic and real
coding-agent sessions (TraceLab), and extends the regime set to long cached prefixes.

Setting: vLLM 0.30.0, FA3, one H100-80GB, Llama-3.1-8B-Instruct; ServingStudio (H200 simulator) for the
regime-separated counterfactual and for MoE. Plan and fixed kill/support criteria:
[plan](#kill--support-criteria).

## Stages

Each report uses the same template: question, prediction (written before the run), method, results, regimes
observed, H1-H5 verdicts, optimization areas, caveats.

| Stage | Report | Status |
|---|---|---|
| 1 | [Foundations: regimes in a vLLM step and what each costs](01_foundations.md) | done, awaiting review |
| 2 | Chat baseline (Azure conversation lengths) | not started |
| 3 | Coding agents (TraceLab replay on H100) | not started |
| 4 | MoE (ServingStudio simulation) | not started |
| 5 | Synthesis: regimes x scenarios -> optimization areas | not started |

## Vocabulary

Regimes describe what a step *is*:

| Label | Step | Source |
|---|---|---|
| X | prefill + long decode, T > 512, FA3 at 1 split, and the default split costs >= 1.2x the best split bound (tile model) | `features.attn_step_model` |
| A | a prefill chunk appended to >= 16k cached tokens (an agent round) | new in this study |
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

| | Stage 1 (kernel level) | Stage 2 chat | Stage 3 agents | Stage 4 MoE sim |
|---|---|---|---|---|
| H1 | confirmed, prefix-dependent: 2.3-7.7x at prefix <= 8k, 1.2-1.9x at 32k, none at 128k ([01](01_foundations.md#h1-h5-verdicts)) | | | n/a (MLA/DSA) |
| H2 | not tested at kernel level (CPU launch cost) | | | |
| H3 | shape-level evidence only (ServingStudio) | | | |
| H4 | confirmed: cold short-prefix chunks are X steps; 8k cold chunks lose 1.15-1.41x vs separate launches | | | |
| H5 | confirmed: 55-62% efficiency at 513-544-token appends on 8k-128k; no split bound helps | | | n/a (MLA/DSA) |

## Kill / support criteria

Fixed before the runs (plan of 2026-10-07). Stage 1 records one deviation for H5 (see 01).

| Hyp. | Kill (not a production issue) | Support |
|---|---|---|
| H1 | measured X_cost steps (>= 1.2x attention and >= 1 ms per step) are < 2% of GPU time, and C3/cap2048 change matched-round TTFT/TPOT by < 3% | >= 10% of GPU time, or >= 1.1x from a fix |
| H2 | launch-bound steps with T > 512 are < 2% of GPU time | >= 10% |
| H3 | prefill-bearing steps are < 10% of decode-weighted wall time | >= 20%, and `mnbt512` improves decode ITL p99 by >= 1.5x at <= 10% TTFT cost |
| H4 | forced cold/miss/compaction rounds are < 10% of prefill attention FLOPs | >= 30% |
| H5 | trace-weighted FA3 prefill efficiency >= 90% | < 80%, and a forced split gains >= 1.15x |
