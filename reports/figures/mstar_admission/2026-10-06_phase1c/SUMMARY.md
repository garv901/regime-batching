# mstar Phase 1c: plan cache (C), parked waiters (D), early KV release (E/S8), v3 workloads (B), repeats (A)

BAGEL-7B-MoT text-to-text, one H100 80GB per job (node1), 2,048-page KV pool, decode capture cap 64 rows.
Every arm runs on ONE worktree, `bench-next` @ `19ccf6d5d` (= `admission-peak` + telemetry + trace client), and differs
from the others only by env toggles; `main` = `regime-harness` @ `8e3a994e5` (origin/main + telemetry).
Each job is one fresh server: a discarded `bagel_burst_v2` warmup replay, then `bagel_burst_v2`, `bagel_stag_v2_r3`,
`bagel_burst_v3`, `bagel_stag_v3_r5.5`. 3 jobs per arm, interleaved (rep 1 of every arm, then rep 2, ...).
Jobs 13086-13106 (`experiments/mstar/arms_phase1c.txt`, manifest `/data/garv901-55613a/results/mstar/manifest.tsv`).

| arm | fit / order | `PLAN_CACHE` | `SCHED_PARK` | `RELEASE_AT_COMPLETION` |
|---|---|---|---|---|
| `ac` | sum / fifo (Gaurav's AC, on our branch) | 0 | 0 | 0 |
| `ac_park` | sum / fifo | 0 | **1** | 0 |
| `peak_bf` | peak / backfill | 0 | 0 | 0 |
| `peak_bf_pc` | peak / backfill | **1** | 0 | 0 |
| `peak_bf_pc_park` | peak / backfill | 1 | **1** | 0 |
| `peak_bf_all` | peak / backfill | 1 | 1 | **1** |
| `main` | origin/main | | | |

Files: `compare.md` (every metric, per workload, `mean [min–max]` over reps), `step_gains.md` (the ratio of each step),
`summary.json` (per rep), `*_short_e2e_cdf.png`, `*_long_e2e.png`, `*_completions.png`, `*_free_kv.png`.
A ratio is marked ≈ when the two arms' [min–max] ranges over the 3 reps overlap (rep spread alone could give it); with 3
reps, "apart" is weak evidence, not a significance test.

## Short-request E2E mean (s), mean [min–max] over reps

| arm | burst_v2 | stag_v2_r3 | burst_v3 | stag_v3_r5.5 |
|---|---|---|---|---|
| `main` | 422 [417–426] (1730–1757 timed out) | 312 [311–313] (505–534 timed out) | 414 [393–440] (1502–1884 timed out) | 146 [1–304] (0–1390 timed out) |
| `ac` | 15.90 [13.39–17.91] | 6.57 [5.91–6.93] | 12.13 [9.39–13.74] | 1.62 [1.27–1.84] |
| `ac_park` | 10.60 [9.95–11.57] | 4.89 [4.58–5.34] | 11.39 [9.45–13.11] | 1.56 [1.23–2.15] |
| `peak_bf` | 0.88 [0.85–0.92] | 0.83 [0.77–0.87] | 11.44 [10.68–11.85] | 1.01 [0.95–1.12] |
| `peak_bf_pc` | 0.86 [0.84–0.88] | 0.80 [0.76–0.82] | 10.60 [8.39–12.43] | 1.02 [0.90–1.22] |
| `peak_bf_pc_park` | 0.82 [0.77–0.85] | 0.81 [0.77–0.84] | 9.32 [8.99–9.52] | 0.95 [0.90–1.02] |
| `peak_bf_all` | 0.88 [0.78–0.96] | 0.83 [0.81–0.85] | 10.11 [8.40–11.65] | 1.16 [0.93–1.35] |

Long-request E2E mean (s) and output tok/s (mean over reps):

| arm | burst_v2 long E2E | stag_v2_r3 long E2E | burst_v3 long E2E | stag_v3_r5.5 long E2E | tok/s (all four) |
|---|---|---|---|---|---|
| `main` | - | 277.1 | - | 114.2 | 44 / 183 / 44 / 590 |
| `ac` | 93.4 | 37.3 | 79.8 | 22.7 | 648 / 1051 / 634 / 1101 |
| `ac_park` | 77.8 | 34.4 | 78.0 | 22.6 | 648 / 1051 / 634 / 1102 |
| `peak_bf` | 79.0 | 32.0 | 89.5 | 22.5 | 648 / 1051 / 634 / 1102 |
| `peak_bf_pc` | 78.0 | 31.9 | 84.5 | 22.9 | 648 / 1051 / 634 / 1101 |
| `peak_bf_pc_park` | 75.4 | 31.4 | 81.4 | 21.6 | 648 / 1051 / 634 / 1102 |
| `peak_bf_all` | 77.3 | 33.2 | 83.5 | 24.4 | 648 / 1051 / 634 / 1102 |

`-` = no long request completed (every long timed out). Output tok/s is the same in every non-`main` arm: the trace, not the policy, sets it.

## Verdict per step

| step | what it targets | result |
|---|---|---|
| **D park, on AC** (`ac_park / ac`) | host time per scheduler scan at deep queues | **short E2E 0.67x on burst_v2, 0.74x on stag_v2** (ranges apart), long E2E 0.83x / 0.92x; host ms per decode step at 400+ waiting 24.1 -> 13.0 (0.54x). v3: 0.94 ≈ / 0.96 ≈ (within noise). |
| **peak+backfill vs AC** (`peak_bf / ac`) | KV admission policy | short E2E **0.06x burst_v2, 0.13x stag_v2, 0.62x stag_v3**; **burst_v3 0.94 ≈** (see finding 2); tok/s equal everywhere. |
| **C plan cache** (`peak_bf_pc / peak_bf`) | planner time per decision | planner µs p50 0.76-0.89x (all logged decisions), **2.1x at >= 64 reserved** (738 -> 348 µs, rep 1); E2E within noise on every workload. |
| **D park, on peak** (`peak_bf_pc_park / peak_bf_pc`) | same as above | within noise on every workload (0.88-1.02); peak+backfill keeps the queue < 50 on v2, so there is little to skip. |
| **E early release** (`peak_bf_all / peak_bf_pc_park`) | KV held by finished requests | **KV free lag p99 0.27-0.40x** (e.g. burst_v2 494 -> 145 ms), p50 0.61-0.92x; E2E within noise; host ms per decode step up on stag_v2 (0.65 -> 0.82 ms, ranges apart). |
| everything vs `main` | | `main` stalls in 11 of its 12 runs (505-1,884 timed-out requests per stalled run; one stag_v3 rep ran clean); `peak_bf_all` short E2E is 0.00-0.01x of `main`'s, tok/s 1.9-14.8x. |

## Findings

1. **Parking waiters is the one new step with an end-to-end gain, and only where the queue is deep (AC on v2).**
   Under strict FIFO the queue reaches 530-605 requests on burst_v2 (440-450 with parking); re-asking each of them every scan cost 18-23 ms of
   host time per decode step (Step 0). With parking, a waiter is re-asked only when the KV state it was refused over
   moves: host ms per decode step at 200-400 waiting 17.8 -> 11.7, at 400+ 24.1 -> 13.0, and short E2E -33% (burst) /
   -26% (stag_v2). Peak+backfill admits shorts as they come, so its queue stays < 50 on v2 and parking changes nothing.
   On burst_v3 every arm has 300-390 waiting and 24-27 ms of host time per decode step at 200-400 waiting, with or
   without parking: those waiters are not what holds the step (finding 2).
2. **burst_v3 does not separate the peak fit from the summed one on the GPU** (peak_bf 11.4 [10.7–11.9] s vs ac
   12.1 [9.4–13.7] s short E2E), though the CPU simulator predicted 2.4-3.9x. Two reasons, both visible in telemetry:
   - peak_bf admits all 90 longs within ~3 s of the burst (long TTFT 2.6 vs 7.5 s under AC, 3 reps); shorts then wait
     for pages (admission wait 5.8 [5.4–6.1] s mean; rep 1: p99 27 s, 1,267 'peak' and 424 'easy' refusals).
   - decode is capped at 64 rows: from t = 30 s more than 64 requests run, so mstar alternates a 64-row decode batch
     with the rest and every running request advances every other step (rep 1: 64-row steps are 50% of decode steps
     at t = 30-80 s and 100% from t = 90 s, in both ac and peak_bf). burst_v2 has 46 longs (< 64); the simulator has no row cap.
   - So peak_bf's extra admitted concurrency buys no tokens (tok/s equal) and slows the longs: long E2E 1.12x AC.
   burst_v3 is bound by the decode capture cap (K1) as much as by KV: it is the natural workload for Phase 2, and it
   argues for capping admitted decode rows at the largest capture size in the planner.
3. **The plan cache does what it was built for, but planning is not on the critical path.** At >= 64 reserved a
   decision's planner time halves on the GPU (738 -> 348 µs p50; rep 1, whole server log), as on the CPU bench (420 -> 161 µs reserve p50,
   `--per-token`). Every GPU decision costs ~2x its CPU-bench time at any reserved count (a busy process on a loaded
   node). Full evaluations total ~12k x 0.4 ms ≈ 5 s per server over ~40 min of runs, too little to move E2E; the
   cache adds 30-50 µs of bookkeeping per scheduler pass (~0.15% of a 20-30 ms decode step).
4. **Early release removes the KV free-lag tail but the tail was not costing latency.** loop stop -> pages back:
   p50 15 -> 9 ms, p99 494 -> 145 ms on burst_v2; p99 0.27-0.40x on every workload. The lag held 0.2-0.3% of the pool
   time-averaged (Step E measurement), so E2E does not move. It adds one RELEASE_KV message per request; host ms per
   decode step rose on stag_v2 (0.65 [0.63–0.67] -> 0.82 [0.74–0.88]) and is noisy elsewhere (stag_v3: every arm spans
   1.8-3.2 ms). Keep it off by default.
5. **AC on our branch = Gaurav's AC** (v2 baselines 13046-13049: burst short E2E 15.7 vs 16.4 s, within noise), so
   `ac` and `ac_park` here measure his policy with and without parking.

## Recommendation

- **C plan cache: keep default ON.** Exact (2.7M identical decisions on CPU; no E2E change on GPU), 2x cheaper planning
  at >= 64 reserved; costs 30-50 µs per pass.
- **D park: propose default ON, after review.** -26-33% short E2E for the AC (sum/fifo) mode under deep queues, neutral
  for peak+backfill. It touches `micro_scheduler.py` (free-scan loops only) and `step.py` sentinels.
- **E early release: keep default OFF.** It works (tail lag 3x lower) but buys no latency here and costs a message per
  request.
- **Next:** Phase 2 (K1, decode capture sizes > 64) on burst_v3, and an admission cap at the decode capture size.

## Caveats

- `main` is never restarted after a stall: the sbatch restarts on timeout/"request failed" errors, but `main`'s
  timeouts are the server's own 600 s aborts ("Response payload is not completed"), which it does not match. Every
  `main` run (v2 matrix, baselines, this matrix) shares this; the stall clears once the aborted requests are removed.
- One `ac` rep (13100, burst_v2) is flagged STALLED: 61.9 s with no completion right after the long burst (strict
  FIFO waits on the long head); it has no timeout.
- Admission wait is not comparable fifo vs backfill: a fifo pool stamps the ask when the request reaches the head, so
  `ac`'s logged wait leaves out the wait behind the head. Only backfill arms' admission waits are quoted.
- `min free KV pages` reads 0 in every arm, even before the burst: the prefix index keeps finished requests' pages
  cached (evictable), so free pages are not headroom.
- 3 reps on one node; [min–max] overlap is a coarse noise rule.

## SHARED CODE TOUCHED (mstar `admission-peak` since `f77fa34bf`, all local: 28 files, +3,818 / -54, of which ~2.9k lines tests)

- `mstar/engine/resources/kv/manager.py` (+377): plan rows kept per request (C); `admission_keys()`, the gate's own wait
  sentinels, `queue_epoch` (D); `release_kv`, released rids refused everywhere (E); `PageArena.release` (in manager.py)
  frees pages before ticking `owner_changes`, which runs with park off too.
- `mstar/engine/resources/kv/plan_table.py` (new, C), `kv/peak_plan.py` (`PeakPlanner.from_arrays`, C),
  `kv/admission_log.py` (`kv_release`/plan fields).
- `mstar/engine/resources/step.py` (sentinels), `engine/resources/runner.py`, `engine/engine.py`,
  `worker/engine_manager.py` (`admission_keys`, `release_kv` plumbing).
- `mstar/worker/micro_scheduler.py` (+91): `_parked` in the two free-scan loops only (D).
- `mstar/worker/worker.py` (+68, `_release_kv` behind the removal's deferral), `mstar/conductor/conductor.py` (+18,
  RELEASE_KV send), `mstar/utils/ipc_format.py` (+3) (E); docstrings in `graph/runtime/base.py`, `worker/rid_table.py`;
  `docs/environment_variables.rst` (the three toggles).
- `bench-next` adds `mstar/profile/step_telemetry.py` `kv_release` (telemetry branch only).
- Study repo: `analysis/mstar_admission.py` (KV free lag, arm order), `analysis/mstar_step_gains.py` (new).

## PERF ASSUMPTIONS

- **Plan cache keeps the stale-`rounds` plan semantics.** Faster: an absolute-clock planner (shift-invariant under
  lockstep) would make rows reusable across steps. Not now: it changes decisions. Cost: rows of every running request
  are dirty after each decode step, so the cache helps within a step's scan, not across steps (2x at 64 reserved, ~10%
  at < 16).
- **Plan cache bookkeeping per pass:** +30-50 µs per scheduler pass (dirty-row marks per token). Faster: mark rows
  per batch, not per stream. Cost: ~0.15% of a decode step.
- **Remaining ~340 µs per full decision on the GPU** is numpy call overhead (`from_arrays`, `peak_from`, `banker_safe`,
  untouched shared code) at ~2x the CPU bench. Faster: a pure-Python fast path for < 16 rows, or a compiled planner.
  Not on the critical path (~5 s CPU per 40 min of runs).
- **Park re-asks the front tier whenever num_free moves** (every grant). Faster: key the window on `_room`'s key.
  Measured: on v2 peak arms the queue is < 50, so not needed now.
- **Early release sends one extra conductor -> worker message per request** instead of a worker-local release at loop
  stop (unsafe for multi-walk requests). Cost: possibly +0.2 ms host per decode step at 3 req/s (stag_v2), noisy.
- **One GPU job per rep (one server each), 15-40 min cold start, ~1 min with a warm compile cache.** Faster: several
  arms per server (impossible: the toggles are read at start).
- **Analysis: KV free lag is computed per request with pandas** over each run's events; a streaming pass in the
  telemetry reader would be faster. ~7 min for 19 jobs x 4 workloads, offline.

## Reproduce

```bash
cd regime-batching
REPS=3 WORKLOADS="bagel_burst_v2 bagel_stag_v2_r3 bagel_burst_v3 bagel_stag_v3_r5.5" SBATCH_OPTS="-p team1 --time=04:00:00" \
  ./experiments/mstar/submit_matrix.sh -f experiments/mstar/arms_phase1c.txt
uv run --no-project --with numpy --with pandas --with matplotlib python analysis/mstar_admission.py \
  /data/garv901-55613a/results/mstar/{13086..13106}_* --out reports/figures/mstar_admission/2026-10-06_phase1c
uv run --no-project --with numpy --with pandas --with matplotlib python analysis/mstar_step_gains.py \
  reports/figures/mstar_admission/2026-10-06_phase1c/summary.json ac_park/ac peak_bf/ac peak_bf_pc/peak_bf \
  peak_bf_pc_park/peak_bf_pc peak_bf_all/peak_bf_pc_park peak_bf_all/peak_bf peak_bf_all/ac ac/main peak_bf_all/main \
  --md reports/figures/mstar_admission/2026-10-06_phase1c/step_gains.md
```
