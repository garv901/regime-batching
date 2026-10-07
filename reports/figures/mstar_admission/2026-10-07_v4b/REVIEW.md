# Review guide: mstar KV admission (v4b)

This guide gives the order in which to read the documents and the code of the KV admission work.
Part A is the documents. Part B is the mstar code for the PR. Part C is the mstar harness code, which is not for the PR. Part D is the code of this study repo.
Each item tells you what to look for.

## Part A: documents (this repo)

Read them in this order.

1. **`SUMMARY.md` (this directory)**. The result, the methodology (model, hardware, arms, workloads), the validity rule, the E2E tables, the findings and the limits. Start here.
2. **`WORKLOADS.md` (this directory)**. Each workload in depth, its purpose, its production relevance, and the behavior of each arm on it, with metrics.
3. **`reports/diagrams/mstar_kv_admission/`**. Four diagrams. Open the `.png` files or the `.html` sources. `README.md` in that directory explains each diagram.
   - `01_main_vs_ac`: why `main` stalls (the fail-hold-retry loop) and how the AC gate prevents it.
   - `02_admission_policy`: the summed fit against the peak fit, EASY backfill, and the guard on each page grant.
   - `03_host_time_passes`: pass C (plan cache), pass D (park), and pass E (KV release at completion).
   - `04_results`: short E2E by arm, the gain of each pass, and the 64-row decode cap on `bagel_burst_v3`. These numbers come from Phase 1c.
4. **`compare.md` (this directory)**. All metrics per workload, with the ratio rows and the per-rep ranges. Use it to examine a number in `SUMMARY.md`.
5. **`runs.tsv` (this directory)**. The status of each of the 186 measured runs: kept (103), excluded for outsider load (43), or dropped (40 Azure variant runs).
6. **`declared.md` and `step_gains.md` (this directory)**. The ratio of each variant to its base, and the ratio of each step (for example `ac_park/ac`).
7. **`experiments/mstar/README.md`**. The harness manual. Read these sections:
   - "Facts": the BAGEL KV pool and the limits, with the evidence for each value.
   - "Workloads v2", "Workloads v3" and "Workloads v4": how each synthetic trace and each variant was made.
   - "Admission matrix: repeats, warmup, guest nodes": the run protocol.
   - "Outsider load on node1": the monitor, the gate and the validity rule.
   - "Public traces": the source, the conversion and the window of Mooncake and Azure.
8. **`reports/figures/mstar_admission/2026-10-06_phase1c/SUMMARY.md`**. The earlier matrix that measured each pass (C, D, E) by itself. Compare its ratios with v4b, not its absolute values.

## Part B: mstar code for the PR

- **Worktree:** `/shared/home/garv901-55613a/mstar-worktrees/regime-batching/admission-peak`.
- **Branch:** `garv/admission-peak-backfill` @ `e9c06f6d9`. It has 34 commits on `origin/admission-control` (merge base `2efdc2061`).
- **Size:** 34 files, +7,546 / -65 lines. Most of the added lines are tests and benches.
- **Command for the diff:** `git diff origin/admission-control...garv/admission-peak-backfill`.

The `bench-next` worktree that the v4b arms ran is this branch plus telemetry hooks, the trace client and emulated EOS. The admission code is the same.

### B1. The planner (read first)

| file | what to look for |
|---|---|
| `mstar/engine/resources/kv/peak_plan.py` (new, 295 lines) | `PlanEntry`, `PeakPlanner`. `peak_from`, `exceeds` and `fits` give the future peak of the admitted set with one more request. `banker_safe` (line 68) is the safe-state test. `shadow` (EASY backfill) gives the earliest start of the head, and `shadow_sum` gives the same for the summed fit. |
| `mstar/engine/resources/kv/config.py` | New keys: `admission_fit` (`sum` or `peak`), `admission_order` (`fifo` or `backfill`), `backfill_protect` (`easy`), `backfill_window` (128). The env variables `MSTAR_KV_ADMISSION_FIT`, `MSTAR_KV_ADMISSION_ORDER` and `MSTAR_KV_BACKFILL_WINDOW` override the YAML. The defaults give the AC behavior. |

### B2. The admission decision in `kv/manager.py` (+891 lines)

Read these functions in this order. The line numbers are at `e9c06f6d9`.

| function | line | what to look for |
|---|---|---|
| `_gate` | 2235 | The entry point for each check. FIFO waits behind the head. Backfill looks at the first `backfill_window` requests in the queue. |
| `_try_reserve` | 2275 | The fit test: summed or peak, the safe-state test, and the EASY rule (a request behind the head must not delay the head). |
| `_wait_add` | 2352 | The queue and the backfill window. |
| `_reserve_new` | 2378 | The reservation for a new request. |
| `_room` | 2504 | The room that the queue can use. |
| `_plan_entry`, `_plan_state` | 2638, 2730 | How the plan is built from the admitted requests. With the plan cache, the rows come from `PlanTable`. |
| `_defer_grant` | 2860 | The guard on each page grant (`GrantDeferred`): a grant that makes the state unsafe waits. |
| `admission_keys` | 2567 | The key that pass D uses to know if the KV state changed. |
| `release_kv` | 1952 | Pass E: give the pages and the reservation of a finished request back before its removal. |

- `_PLAN_CACHE` (line 74) reads `MSTAR_KV_PLAN_CACHE`. **Default on**.
- With `MSTAR_KV_DEBUG_ASSERTS=1` (`_DEBUG_ASSERTS`, line 70), the code also checks the KV page records after each change.

### B3. Pass C: plan cache and evictable pages

| file | what to look for |
|---|---|
| `mstar/engine/resources/kv/plan_table.py` (new, 174 lines) | `PlanTable`: numpy rows per admitted request, set at each change and not built again at each decision. |
| `mstar/engine/resources/kv/prefix_index.py` | The evictable pages are kept current as the arena reports owner changes, so a decision does not walk the index. |
| `mstar/engine/resources/kv/admission_log.py` (new, 94 lines) | The JSONL admission log (`reserve`, `wait`, `refuse`, `grant_deferred`, `release`, `kv_release`). Off unless `MSTAR_STEP_TELEMETRY_DIR` is set. |

### B4. Pass D: park waiters

| file | what to look for |
|---|---|
| `mstar/engine/resources/step.py` (lines 185-220) | The sentinels `ADMIT_WAIT` and `ADMIT_WAIT_BEHIND`, and their `FULL_ADMIT_*` forms. |
| `mstar/engine/resources/runner.py` (around line 265) | The runner gives the gate sentinel only if each "not ready" answer was a gate wait. |
| `mstar/engine/engine.py` | `admission_keys` for a node, combined from the pools that decide. |
| `mstar/worker/micro_scheduler.py` | `_SCHED_PARK` (line 24, `MSTAR_SCHED_PARK`, **default off**), `_parked` (line 241), `_park_scan` (line 873), and the purge in `clear_rid` and `fail_rids` (lines 927, 938). A first check is never skipped. The TP follower path is never skipped. |

### B5. Pass E: KV release at completion

| file | what to look for |
|---|---|
| `mstar/utils/ipc_format.py` | The new `WorkerMessageType.RELEASE_KV`. It uses the `RemoveRequest` body. |
| `mstar/conductor/conductor.py` (lines 59, 1287) | `MSTAR_KV_RELEASE_AT_COMPLETION` (**default off**). The conductor sends `RELEASE_KV` when a request completes. |
| `mstar/worker/worker.py` (`_release_kv`, line 750) | The same TP-follower guard and in-flight deferral as the removal. |
| `mstar/worker/rid_table.py`, `mstar/worker/engine_manager.py`, `mstar/graph/runtime/base.py` | Small changes for the released state. |
| `docs/environment_variables.rst` | Documents `MSTAR_KV_RELEASE_AT_COMPLETION` and the debug check. |

### B6. Tests and benches

| file | what it proves |
|---|---|
| `test/modular/test_kv_peak_plan.py` | The planner against hand-made cases. |
| `test/modular/test_kv_admission_peak.py` | Peak fit, backfill order, the grant guard, the config and the event log. |
| `test/modular/test_kv_backfill_cost.py` | Backfill decisions and their cost. It has the template for the on/off log comparison. |
| `test/modular/test_kv_plan_cache.py`, `test_kv_evictable_cache.py` | The cached plan and the evictable pages give the same answers as a fresh count. |
| `test/modular/test_sched_park.py` | Check counts with park, and park against no park over random workloads. |
| `test/modular/test_kv_release_at_completion.py`, `test_worker_release_kv.py`, `test_conductor_release_kv.py`, `test_kv_release_plan_cache.py` | Pass E: pages free at release, removal is idempotent, deferral, TP follower, and no new admission of a released request. |
| `test/modular/bench_kv_admission_cost.py` | CPU cost of one decision, with a digest of the decisions and `--diff` between runs. |
| `test/modular/bench_sched_park.py` | CPU cost of one scan with and without park. |

### B7. Open items for the PR

- `docs/environment_variables.rst` does not document `MSTAR_KV_PLAN_CACHE`, `MSTAR_SCHED_PARK`, `MSTAR_KV_ADMISSION_FIT`, `MSTAR_KV_ADMISSION_ORDER` or `MSTAR_KV_BACKFILL_WINDOW`. Add them before the PR.
- Proposed defaults: plan cache on, park on after review, KV release at completion off.
- The API-server TTL bug (`SUMMARY.md`, Findings, item 6) is not in this branch. Report it to the mstar owners as a separate issue.

## Part C: mstar harness code (not for the PR)

| worktree @ commit | what it adds | files |
|---|---|---|
| `regime-harness` @ `945aabf6e` (on `origin/main`) | Step telemetry, the trace replay client, and emulated EOS. The `main` arm and the client of each arm use it. | `mstar/profile/step_telemetry.py`, `benchmark/trace_replay.py`, `benchmark/runner.py` (`--trace`, `--request-timeout`), `benchmark/request.py`, `emulated_eos_at` in `conductor.py`, `sampler/config.py`, `bagel_model.py`, `bagel/submodules.py`, `model/base.py`. Tests: `test_step_telemetry.py`, `test_trace_replay.py`, `test_emulated_eos.py`. |
| `bench-next` @ `bbd8b85a3` | The PR code (Part B) plus the same telemetry hooks, trace client and emulated EOS, and a `kv_release` telemetry event. The `ac`, `ac_park` and peak arms use it. | The same files as Part B and the row above. |

Review the emulated EOS change with care. It decides where each request stops, so it decides the real KV of each run.

## Part D: code of this study repo

### D1. Run protocol

| file | what to look for |
|---|---|
| `experiments/mstar/arms_v4.txt` | The four arms and their env toggles. |
| `experiments/mstar/queue_v4b.sh` | The v4b matrix: reps per arm, the order of the workloads, the warmup, the smoke tasks. |
| `experiments/mstar/gpu_pool.sbatch`, `pool_sbatch.sh`, `submit_matrix.sh` | The pool job that holds the GPUs, and the queue of tasks. Each task is one `srun --exact` step with 2 CPUs and 1 GPU. |
| `experiments/mstar/admission_matrix.sbatch` | One task: start the server, replay the warmup, replay each run, restart the server after a failed request. |
| `experiments/mstar/env/` | `envmon.py`, `gate.py`, `env_valid.py` (line 48 `evaluate`: the validity rule), `sar_poolidle.py`, `speedprobe.py`, `summ.py`, `requeue_env.py`. |

### D2. Workloads

| file | what to look for |
|---|---|
| `experiments/mstar/make_bagel_workloads_v2.sh`, `make_bagel_workloads_v3.sh` | The burst and stag traces. |
| `experiments/mstar/make_bagel_workloads_v4.sh`, `declare_max_tokens.py` | The `_mtx2` and `_mt2048` variants and the hash id offsets. |
| `experiments/mstar/fetch_public_traces.sh`, `convert_public_trace.py`, `make_public_workloads.sh` | The download, the conversion (classes, clamps, prefix blocks) and the window of the public traces. |
| `experiments/mstar/kv_report.py`, `public_trace_report.py` | The fluid KV peak and the other trace statistics. |
| `data/workloads/pub_mooncake_conv_r0.5*.jsonl`, `pub_azure_conv_r1.25.jsonl` | The converted public traces of this report, with their `.meta.json`. |

### D3. Analysis

| file | what to look for |
|---|---|
| `analysis/mstar_admission.py` | `max_stall` (line 435, the stall flag), `bench_metrics` (line 488), `host_time_metrics` (line 528, host ms per step by queue depth), `admission_metrics` (line 626, the admission wait at lines 659-661), `ranges_overlap` (line 904, the noise rule). |
| `analysis/mstar_declared.py` | The variant / base ratios. |
| `analysis/mstar_step_gains.py` | The ratio of each step. |
| `tests/test_mstar_admission.py` | The tests of the analysis. |
