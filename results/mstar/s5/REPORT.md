# S5: deadline-aware micro-scheduling for colocated stages

Status: **unmeasured**. The GPU A/B is Slurm job 13064 (queued 2026-10-06, `-p team1,guest -w node1 --gres=gpu:1`).
When it lands, `results/mstar/s5/run1/analysis.md` is written by the job itself; to redo it by hand:

    python3 experiments/mstar/s5/analyze.py --run results/mstar/s5/run1 > results/mstar/s5/run1/analysis.md

Nothing below the "Result" heading has been measured yet.

## Hypothesis

With Thinker, Talker and Code2Wav colocated on rank 0 (`configs/qwen3omni_colocated.yaml`), `MicroScheduler` picks the
next (node, walk) by round-robin with no cost or deadline awareness. A long Thinker prefill can sit between two Talker
decode steps and stretch the gap an audio client hears. Picking the walk with the least slack should improve TTFA and
audio RTF under mixed load, at a small cost to text throughput.

## What was built (mstar branch `garv/s5-stage-deadline`, commit a0c259547)

Opt-in: `MSTAR_SCHED_POLICY=deadline` (default `round_robin`, byte-for-byte today's behaviour).

* `NodeSubmodule.walk_priority(walk)` and `walk_deadline_slack(walk, rid, info, now)`: model hints, default None.
* Qwen3-Omni declares Code2Wav 100 > Talker 80 > Thinker decode 60 > Thinker prefill 20 > encoders 10.
* Slack comes from an `AudioClock`: a stream that emitted its first chunk at t0 and `s` seconds of audio since runs dry
  at t0 + s (a listener plays at 1x). Code2Wav reports `t0 + emitted - now`; Talker reports `t0 + frames*80 ms - now`
  (frames it has produced but Code2Wav has not yet turned to audio count as buffered); a Thinker prefill of an audio
  request has slack 0 (the first chunk is due now).
* Pick order over the walks that have a ready request: (1) aged: the walk whose oldest ready request waited more than
  `MSTAR_SCHED_MAX_WAIT_MS` (default 200) goes first, oldest first; (2) least slack, walks with a deadline before
  those without; (3) higher priority; (4) least recently scheduled.
* The worker's speculation chain yields to another ready walk after one step. Under `deadline` it yields only to a walk
  at least as high-priority, or one past the wait guard; otherwise the chain would undo the ranking.
* `sched_pick` telemetry event (only with `MSTAR_STEP_TELEMETRY_DIR`): chosen walk, reason, slack, and the wait of every
  ready walk. It is emitted under `round_robin` too, so the two arms are compared by the same measure.

Tests: `test/modular/test_micro_scheduler_deadline.py`, 43 tests with multi-walk fakes. Full `test/modular`: 2323 passed,
193 skipped, 2 failed; the 2 failures (`test_graph_runtime_factory::test_rust_refuses_the_pyzmq_communicator`,
`test_kokoro_model::test_missing_language_backend_is_reported_once`) fail identically on a clean checkout of the base
commit (a missing `mstar_rust` module, a missing kokoro language backend).

## Experiment

One server per policy, all cases against it, `MSTAR_STEP_TELEMETRY_DIR` on, one throwaway warm-up run per server
(audio conc 8 + text conc 8, discarded), then 2 reps of each case:

| case | load |
|---|---|
| a4 | 24 text-to-speech requests, closed loop at concurrency 4 |
| a8 | 32 text-to-speech requests, concurrency 8 |
| b4 | a4 plus text-only requests with 1-2k-token prompts at concurrency 8, refilled for the whole audio window |
| b8 | a8 plus the same text load |

Arms: `round_robin`, `deadline` (aging guard 200 ms). An extra arm with another guard can be added to the same run dir:
`ARMS="deadline_w1000" RES=$PWD/results/mstar/s5/run1 sbatch --export=ALL experiments/mstar/s5/s5_ab.sbatch`.

Metrics: TTFA p50/p99, RTF (e2e / audio seconds), listener stall time, text TTFT and ITL p50/p99, text tok/s and audio
s/s inside the audio window, and per-walk longest wait (max, p99) and GPU share from the step telemetry.

## Limitations to read the result with

* Backlog-first serving is unchanged: a prefill burst above the batch cap (4) goes through the backlog, which is served
  before the ready scan, so priority does not apply to it. Only the aging guard bounds it.
* Yield gating uses priority and aging only, not slack.
* Under `round_robin`, the telemetry waits can understate by about 2 steps (the speculation-yield check does not stamp
  ready times in that mode).
* Only 2 reps per case; the rep-to-rep spread of TTFA p50 is printed in the analysis. Treat differences inside it as noise.
* 8-rep-scale confidence intervals are not computed. The first real GPU run is also the first time the new code paths
  run on a GPU.

## Result

Not measured yet (job 13064 pending).
