# mstar v4b: KV admission policies on synthetic and production traces, with oracle and over-declared `max_tokens`

## Result in five lines

- With a length declaration close to the real length, our policy (peak fit plus backfill) cuts short E2E to 0.04-0.13x of AC under KV pressure.
- On the Kimi production trace (Mooncake), our policy is 0.86-0.91x of AC on short E2E in all three variants. Long E2E stays the same.
- With no length information (`max_tokens` = 2048 for each request), the gain on the burst disappears. A 2x margin on the length keeps the full gain.
- mstar `main` stalls when the KV pool fills. Each admission arm completes each request in each valid run.
- On a trace that never fills the pool (Azure), admission costs host time. Our policy with the plan cache costs the least of the three admission arms.

The workloads and the behavior of each arm on each workload are in [`WORKLOADS.md`](WORKLOADS.md).
The review guide for the code and the documents is in [`REVIEW.md`](REVIEW.md).

## Methodology

### Model, and why we use it

- **Model:** BAGEL-7B-MoT (ByteDance-Seed), text-to-text only, with the mstar config `configs/bagel_single_gpu.yaml`.
- **The text path is a dense 7B decoder of the Qwen2 family**. It has 28 layers, a hidden size of 3,584, and 4 KV heads of dimension 128. In text-to-text, only the understanding experts of the MoT run.
- **Why BAGEL** (decision with the user on 2026-10-05):
  - mstar has no plain dense LLM model entry. BAGEL text-to-text is the nearest supported model to a dense LLM.
  - The vLLM study of this repo used a dense 8B model (Qwen3-8B). A dense 7B decoder keeps the results comparable.
  - The model fits on one H100, and `bagel_single_gpu.yaml` is a supported mstar config.
  - The KV pool is small and fixed (2,048 pages), because the server also loads the ViT and the VAE. Thus KV pressure occurs at moderate request rates, and it is the same in each run.
  - The text path uses the paged KV, the admission control (AC) code, and the prefix cache. These are the parts of mstar that this study changes.
- **Limit of this choice:** a pure LLM on the same GPU has a larger pool. A `qwen3_dense` model entry can remove this difference (Phase 5 of the plan).

### Server and hardware

| item | value |
|---|---|
| GPU | 1 H100 80 GB per server, on node1 (Slurm partition `team1`) |
| CPU and memory | 2 CPUs (one core, two hyperthreads) and 16 GB per server, as one `srun --exact` job step |
| transport | `--tensor-comm-protocol SHM` |
| KV pool | 2,048 pages x 128 tokens = 262,144 tokens (14 GiB), 57,344 bytes per token, bf16 |
| host offload, prefix cache | `cpu_offload_pages=0`, prefix cache on |
| output cap | 2,049 output tokens per request (`MAX_OUTPUT_TOKENS` = 2048 decode iterations) |
| decode batch cap | 64 rows (the largest CUDA graph capture size) |
| `max_seq_len` | 32,768 |

One Slurm job (pool 13218) held the 8 GPUs of node1 and ran 8 tasks at the same time.
Each task started a fresh server and replayed one discarded `bagel_burst_v2` warmup. Then it replayed its measured runs on that server.
A fresh server is slower on its first burst (Phase 1c), so the warmup removes that effect.

### Arms

| arm | mstar worktree @ commit | fit / order | `MSTAR_KV_PLAN_CACHE` | `MSTAR_SCHED_PARK` |
|---|---|---|---|---|
| `main` | `regime-harness` @ `945aabf6e` (origin/main + telemetry + trace client + emulated EOS) | none (no admission control) | | |
| `ac` | `bench-next` @ `bbd8b85a3` | sum / fifo (the AC of `origin/admission-control`, on our branch) | 0 | 0 |
| `ac_park` | `bench-next` | sum / fifo | 0 | **1** |
| `peak_bf_pc_park` (peak) | `bench-next` | **peak / backfill** (ours) | 1 | 1 |

- `MSTAR_KV_RELEASE_AT_COMPLETION=0` in each arm.
- `bench-next` is `admission-peak` (the branch for the PR) plus step telemetry, the trace client, and emulated EOS.
- On our branch, the default mode (sum / fifo) gives the decisions of AC. The v2 baselines (jobs 13043-13049) measured it against the AC branch: 15.7 s against 16.4 s short E2E on `bagel_burst_v2`.
- Arms file: `experiments/mstar/arms_v4.txt`. Queue script: `experiments/mstar/queue_v4b.sh`.
- Reps: 3 for each admission arm. By design, `main` has 1 rep on burst and stag, 2 on Mooncake, and 3 on Azure. Each stall of `main` costs more than 600 s.

### Workloads and variants

| family | file | requests / span | fluid KV peak | source |
|---|---|---|---|---|
| burst | `bagel_burst_v2` | 2,700 / 305 s | 1.53x pool | synthetic: 46 long requests at t = 30 s, shorts at 9 per s |
| stag | `bagel_stag_v3_r5.5` | 3,300 / 591 s | 0.84-1.02x pool | synthetic: Poisson 5.5 per s, 15% long; the control |
| Mooncake | `pub_mooncake_conv_r0.5` | 1,036 / 594 s | 1.77x pool | Kimi conversation trace (FAST 2025), long prompts |
| Azure | `pub_azure_conv_r1.25` | 4,105 / 599 s | 0.30x pool | Azure LLM inference trace 2023, conversation; no KV pressure |

Each family has three variants. Only the declared `max_tokens` changes. The server stops each request at its real length (emulated EOS), so the real KV is the same.

| variant | declared `max_tokens` | declared / real KV (prompt + output) |
|---|---|---|
| base | the real length (a perfect oracle) | 1.00 |
| `_mtx2` | min(2048, 2 x the real length) | burst 1.04-1.05, stag 1.06, Mooncake 1.03 |
| `_mt2048` | 2048 for each request (no length information) | burst 3.16-3.21, stag 1.92-1.93, Mooncake 1.16 |

The Azure variants are not in the tables. Section "Validity and pruning" gives the reason.

### Metrics

- **Short and long E2E:** the mean over the completed requests of the class, from the client `bench.json`. Then we take the mean over the reps.
- **"[min-max]":** the range of the rep means. **"Apart":** the ranges of two arms do not overlap. **"≈":** the ranges overlap.
- With 3 reps, "apart" is weak evidence. It is not a significance test.
- **Holds, GPU busy, decode batch, host ms per step:** from the step telemetry of the server, in the window of the run.
- **Admission wait:** from the admission log. Its clock starts at the first check for backfill and at the head of the queue for FIFO. `WORKLOADS.md` gives the details.

## Validity and pruning

node1 is also the login node of the cluster. Processes of other users ("outsiders") run outside Slurm, and some of them use our CPUs.
A Slurm cpuset does not keep other processes off our cores. Each server has only 2 CPUs and they are saturated, so outsider load slows the server directly.

- **Metric:** the busy share of the pool CPUs that no job step uses (CPUs 8-35 and 112-139). All load on these CPUs comes from outsiders.
- **Source:** `envmon.py` samples it every 2 s from 05:38 UTC. For earlier runs, `env_valid.py` uses the `sar` 10-min bins.
- **Rule for Azure:** mean outsider load at or below 3.5%, and p90 at or below 7% (envmon) or 14% (sar). Azure is the most host-bound trace. It degrades at 4-12% outsider load.
- **Rule for all other workloads:** mean at or below 15%, and p90 at or below 30% (envmon) or 60% (sar). stag degrades at approximately 24%. Mooncake and burst stay stable up to approximately 22%.
- **Gate:** `gate.py` held the queue while the 2-min outsider mean was above 6%. It released the queue when the 10-min mean was at or below 3.5% and the 2-min mean at or below 4%.
- **Replacements:** a replacement task ran the same workload with the same arm. `/data/garv901-55613a/results/mstar/manifest_v4b_replaced.tsv` lists the 73 replaced runs and the task that ran each again.
- **Main source of noise:** a process of another user (`pt_main_thread`) used approximately 14 cores for 8-20 s every 132 s from approximately 09:30 UTC.

`runs.tsv` gives the status of each measured run (186 runs):

| status | runs | meaning |
|---|---|---|
| kept | 103 | valid; in all tables |
| excluded: outsider load above limit | 43 | not valid by the rule above |
| dropped: Azure over-declared variant | 40 | `_mtx2` and `_mt2048` of Azure. Each arm had only 1 valid run of each, so we do not report them |

We also cancelled the last 8 Azure replacement tasks (13218144-13218151) when we closed the pool. Their partial result directories are in `/data/garv901-55613a/results/mstar/_cancelled_13218/`.

## Short-request E2E mean (s): mean [min-max] over valid reps (n)

| workload | `main` | `ac` | `ac_park` | peak |
|---|---|---|---|---|
| burst base | 438.6 (1), **stall** | 20.23 [18.91-22.21] (3) | 9.82 [8.72-11.38] (3) | **0.85** [0.76-1.02] (3) |
| burst `_mtx2` | 443.4 (1), **stall** | 15.57 [12.82-20.30] (3) | 11.28 [11.04-11.73] (3) | **0.91** [0.81-1.09] (3) |
| burst `_mt2048` | 428.0 (1), **stall** | 38.89 [15.87-84.42] (3) | 17.67 [10.17-26.18] (3) | 24.21 [14.85-38.10] (3) |
| stag base | 0.98 (1) | 1.22 [0.97-1.45] (3) | 1.41 [1.07-1.99] (3) | 1.00 [0.88-1.24] (3) |
| stag `_mtx2` | 131.5 (1), **stall** | 6.22 [1.47-15.62] (3) | 1.61 [1.27-2.23] (3) | **0.84** [0.82-0.87] (3) |
| stag `_mt2048` | 350.7 (1), **stall** | 1.93 [1.68-2.06] (3) | 3.17 [1.74-5.99] (3) | 1.45 [0.88-1.98] (3) |
| Mooncake base | 12.55 [11.29-13.82] (2) | 10.20 [9.90-10.70] (3) | 10.55 [10.32-10.79] (3) | **9.25** [9.02-9.38] (3) |
| Mooncake `_mtx2` | 12.08 [11.44-12.71] (2) | 10.29 [9.90-10.65] (3) | 10.77 [10.33-11.57] (3) | **9.37** [9.24-9.54] (3) |
| Mooncake `_mt2048` | 162.0 [12.3-311.7] (2), 1 stall | 10.83 [10.26-11.70] (3) | 10.32 [10.06-10.81] (3) | **9.35** [9.17-9.62] (3) |
| Azure base | **3.83** (1) | 6.45 [5.29-7.07] (3) | 5.57 [4.17-6.30] (3) | 4.39 [3.60-5.03] (3) |

"stall" means that `main` had no free page and no request to evict, and the server aborted the requests that waited after 600 s.
A `main` mean includes only the completed requests. In the stalled runs, 264-1,687 requests failed (burst 1,472-1,687 of 2,700).

## Long-request E2E mean (s) and output tok/s

| workload | `main` | `ac` | `ac_park` | peak | tok/s (`main` / `ac` / `ac_park` / peak) |
|---|---|---|---|---|---|
| burst base | - | 103.9 [101.5-107.7] | 75.6 [71.5-80.3] | 76.8 [72.0-85.1] | 54 / 648 / 648 / 648 |
| burst `_mtx2` | - | 92.6 [85.9-103.7] | 79.7 [78.8-81.0] | 77.8 [73.1-87.1] | 57 / 648 / 648 / 648 |
| burst `_mt2048` | - | 118.1 [91.2-169.4] | 92.8 [75.5-111.3] | 118.2 [100.7-144.2] | 47 / 615 / 648 / 648 |
| stag base | 22.4 | 20.8 [19.8-21.6] | 21.7 [19.9-24.5] | 22.0 [20.6-24.1] | 1,102 / 1,102 / 1,102 / 1,102 |
| stag `_mtx2` | 86.1 | 27.7 [21.0-40.8] | 21.9 [20.5-24.6] | 20.4 [20.0-21.2] | 459 / 1,063 / 1,102 / 1,102 |
| stag `_mt2048` | 281.0 | 22.3 [21.0-23.0] | 24.7 [21.3-30.4] | 24.3 [20.1-28.0] | 150 / 1,102 / 1,096 / 1,102 |
| Mooncake base | 30.3 [23.3-37.4] | 14.6 [14.2-15.3] | 15.0 [14.8-15.3] | 15.1 [14.7-15.3] | 579 / 582 / 582 / 582 |
| Mooncake `_mtx2` | 28.0 [23.5-32.5] | 14.6 [14.2-15.1] | 15.0 [14.6-15.8] | 15.1 [14.8-15.3] | 580 / 582 / 582 / 582 |
| Mooncake `_mt2048` | 125.3 [28.7-221.9] | 14.9 [14.3-15.9] | 14.3 [14.0-15.0] | 15.9 [15.6-16.4] | 359 / 582 / 581 / 581 |
| Azure base | 4.0 | 6.7 [5.4-7.4] | 5.7 [4.3-6.5] | 4.5 [3.8-5.1] | 1,407 / 1,351 / 1,366 / 1,396 |

- "-" means that no long request completed.
- The trace sets the output tok/s when an arm keeps up with it: 648 (burst), 1,102 (stag), 582 (Mooncake), 1,408 (Azure).
- The Mooncake and Azure "long" class is the top 10% by final KV. The Azure longs have long prompts but short outputs (p50 101 tokens), so their E2E is short.

## Findings

### 1. With a useful declaration, peak plus backfill is better than AC on each workload with KV pressure

This table gives the ratio of the short E2E means, peak / `ac`.

| workload | base | `_mtx2` | `_mt2048` |
|---|---|---|---|
| burst | **0.04** (apart) | **0.06** (apart) | 0.62 ≈ |
| stag | 0.82 ≈ | **0.13** (apart) | 0.75 ≈ |
| Mooncake | **0.91** (apart) | **0.91** (apart) | **0.86** (apart) |
| Azure | **0.68** (apart) | dropped | dropped |

- **burst:** short E2E goes from 20 s to 0.85 s. Long E2E is 0.74x of `ac` (apart), and the worst long takes 101.7 s against 170.8 s. tok/s is the same. This repeats Phase 1c (0.06x).
- **A 2x margin costs nothing**. On burst `_mtx2` and stag `_mtx2`, peak stays at 0.84-0.91 s. The margin adds only 4-6% to the declared KV, because the prompt is most of the KV.
- **stag `_mtx2` hurts `ac`, not peak**. `ac` goes from 1.22 s to 6.22 s, with one rep at 15.6 s. peak stays at 0.84 s.
- **Mooncake (production trace):** peak is 9-14% faster than `ac` on short E2E in all three variants (apart). All three admission arms halve the long E2E of `main` (30 s to 15 s).
- **The only measured cost:** on Mooncake `_mt2048`, the long E2E of peak is 1.07x of `ac` (≈) and 1.11x of `ac_park` (apart).

### 2. With no length information (`_mt2048`), the burst gain disappears

- **burst `_mt2048`:** peak 24.2 s, `ac` 38.9 s, `ac_park` 17.7 s. The ranges overlap, and each arm changes by 2.6-5.3x between reps.
- The short E2E of peak is 28x its base value. The declared KV is 3.2x the real KV, so the planner sees a pool that is 3x fuller than it is.
- With every request declared at 2048, peak cannot tell a short from a long. Its admission wait is 21.3 s for the shorts and 24.2 s for the longs.
- **stag `_mt2048`** (declared KV 1.9x): the three admission arms are within noise (1.45-3.17 s). **Mooncake `_mt2048`** (1.16x): no change.
- Thus the gain of peak plus backfill needs a length prediction that keeps the declared KV close to the real KV. A 2x margin is sufficient on these traces. A fixed cap of 2048 is not.

### 3. Park (pass D) helps AC when the queue is long

- `ac_park / ac` short E2E: burst 0.49x (apart), burst `_mtx2` 0.72x (apart), burst `_mt2048` 0.45x ≈, stag `_mtx2` 0.26x ≈.
- On burst, the host time per decode step at 400+ in queue goes from 29.1 ms to 11.1 ms.
- Long E2E on burst goes down too (0.73x), because the decode steps are faster.
- On the other workloads, park is within noise. This repeats Phase 1c.

### 4. `main` stalls when the KV pool fills

- **burst:** `main` stalls in each run of each variant. 1,472-1,687 of 2,700 requests fail, and the GPU is busy for 9-10% of the run.
- **stag and Mooncake:** `main` completes the base trace. It stalled on stag `_mtx2` and stag `_mt2048` (1 run each), and in 1 of 2 Mooncake `_mt2048` runs.
- The code of `main` does not read `max_tokens` for admission. Only the stop check reads it, and the stop occurs at the same token in each variant.
- Thus we cannot identify the variant as the cause. We read these stalls as an instability of `main` near a full pool, with n = 1-2.
- On Mooncake base, `main` completes but the long E2E is 2x that of the admission arms (30.3 s against 14.6 s). `main` held requests 15,841 times.
- `ac`, `ac_park` and peak did not stall in a valid run. They completed each request in each variant.

### 5. The host CPU limits Azure, not the KV pool

- The fluid KV peak is 0.30x the pool, and no arm waits for KV (admission wait 0.05 s or less).
- The host time per decode step is 6-10 ms, against 0.5-2 ms on the other workloads. The GPU is busy for 56-64% of the run.
- Each admission arm adds host time per step: `ac` +3.9 ms, `ac_park` +3.1 ms, peak +1.0 ms over `main` (6.1 ms). The server has 2 saturated CPUs, so this time becomes latency.
- peak is 0.68x of `ac` on short E2E (apart), but approximately 15% slower than `main` (n = 1 for `main`).
- peak has the plan cache on, and `ac` and `ac_park` have it off. This is a probable cause of the gap between them, but this matrix does not isolate it.

### 6. Deployment bugs and data notes

- **API-server result-delivery TTL (deployment bug, Track B items B1 and B5)**. With 4 or 8 CPUs per server, `ac` and `main` fail 139-171 requests with "HTTP 500: result delivery timed out", and then the server stalls.
  - With 2 CPUs, no request fails.
  - The suspected cause is the GIL: hundreds of requests that wait poll every 1 ms under `request_lock`, and the single data-worker thread falls behind. We did not prove this (no py-spy).
  - A failed request is not torn down, so its KV stays held.
  - All v4b runs use 2 CPUs, and a run with this error is not valid. Report this bug to the mstar owners.
- **1-token requests return 2 tokens**. 9 Mooncake requests with a real length of 1 return 2 tokens, in each arm and each variant (flag OVERRUN in `compare.md`). The effect on KV is negligible.
- **burst `_mt2048` is noisy in each arm, and the noise does not follow outsider load**. The worst rep of `ac_park` (26.2 s) and of peak (38.1 s) ran at 2.3% outsider load. Treat its ratios as ≈.

## Limits

- **Few reps**. Each admission arm has 3 reps. `main` has 1-3. A ratio marked "apart" is weak evidence.
- **One model, one GPU type, one node**. The pool is small because BAGEL loads the ViT and the VAE. A pure LLM has a larger pool, so the same traces give less KV pressure.
- **2 CPUs per server**. Each server is CPU-saturated, so host costs show as latency more than on a server with more CPUs. The TTL bug prevents more CPUs.
- **Emulated EOS**. The server stops each request at the trace length, and not at the EOS token of the model.
- **No Azure variants**. The over-declared Azure runs had only 1 valid run per arm.
- **The 64-row decode cap is not in this matrix**. Phase 1c showed that this cap, and not admission, limits `bagel_burst_v3` (90 longs).
- **Absolute values differ from Phase 1c**. For example, `ac` on burst is 15.9 s in Phase 1c and 20.2 s here (ranges apart). Compare ratios between the two reports, not absolute values.

## Files

- `WORKLOADS.md`: each workload in depth, and the behavior of each arm on it.
- `REVIEW.md`: the order in which to review the documents and the code.
- `compare.md`: each metric per workload, mean [min-max] over the valid reps. The excluded runs are listed at the top.
- `declared.md`: the ratio of each variant to the base, per arm (`analysis/mstar_declared.py`).
- `step_gains.md`: the ratio of each step (`analysis/mstar_step_gains.py`).
- `runs.tsv`: the status of each measured run (kept, excluded, dropped) and its outsider load.
- `summary.json`: the values of each rep.
- `*_short_e2e_cdf.png`, `*_long_e2e.png`, `*_completions.png`, `*_free_kv.png`: plots per workload.

Commands to make the report again (on node1, from the repo root):

```bash
OUT=reports/figures/mstar_admission/2026-10-07_v4b
R=/data/garv901-55613a/results/mstar
python3 experiments/mstar/env/env_valid.py --job 13218 --sar-day 2026-10-07 --tsv > /tmp/valid.tsv
EXC=$(awk -F'\t' '$5==0 {printf " --exclude %s_%s/%s", $1, $2, $3}' /tmp/valid.tsv)
EXC="$EXC --exclude */pub_azure_conv_r1.25_mtx2_r1 --exclude */pub_azure_conv_r1.25_mt2048_r1"
DIRS=$(ls -d $R/132180{56..99}_* $R/13218{100..143}_* 2>/dev/null | grep -v -E "_(smoke|diag)")
UV="uv run --no-project --with numpy --with pandas --with matplotlib python"
$UV analysis/mstar_admission.py $DIRS $EXC --out $OUT
$UV analysis/mstar_declared.py $OUT/summary.json --md $OUT/declared.md
$UV analysis/mstar_step_gains.py $OUT/summary.json ac/main ac_park/ac peak_bf_pc_park/ac peak_bf_pc_park/ac_park peak_bf_pc_park/main --md $OUT/step_gains.md
```

## SHARED CODE TOUCHED

- None in mstar in this step. The arms use `bench-next` @ `bbd8b85a3` and `regime-harness` @ `945aabf6e` as committed before.
- `945aabf6e` adds `emulated_eos_at` to `conductor.py`, `sampler/config.py`, `bagel_model.py`, `bagel/submodules.py` and `model/base.py`. It is off by default.
- Study repo: `experiments/mstar/env/` (new: outsider-load monitor, gate, validity check, replacement queue, summary). `experiments/mstar/README.md` (section "Outsider load on node1").

## PERF ASSUMPTIONS

- **2 CPUs per server**. Faster alternative: 4-8 CPUs. Not taken, because the API-server TTL bug fails requests at 4 or more CPUs. Cost: each server is CPU-saturated, so host overhead and outsider load show as latency. Azure is the most affected.
- **Validity filter instead of a clean host**. Faster alternative: a node that is not the login node. Not available: the guest nodes were full, and they do not mount `/data`. Cost: 73 replaced runs and hours of held queue time.
- **Python monitor that reads cgroup files every 2 s**. Faster alternative: eBPF or `perf sched`. Not necessary at this resolution. Cost: 33 s of CPU in 7 h for the monitor, and 80 s per speed probe (0.1-0.3% of one CPU each).
- **8 tasks in parallel on one node**. Faster than sequential runs. The cgroups isolate the CPUs, but the tasks share memory bandwidth. Rep 1 ran 8 at a time at low outsider load and agreed with Phase 1c, so we kept it.
- **The plan cache, park, and the peak planner** keep the costs measured in Phase 1c: reserve p50 150-200 µs at 64 reserved, with numpy call overhead in `from_arrays`, `peak_from` and `banker_safe`.
- **AC arms without the plan cache**. The `ac` and `ac_park` arms run with `MSTAR_KV_PLAN_CACHE=0`, as the AC branch has no cache. Faster alternative: an `ac` arm with the cache on. Not run in v4b. Cost: on Azure, part of the host-time gap between `ac` and peak can come from the cache, and not from the policy.
