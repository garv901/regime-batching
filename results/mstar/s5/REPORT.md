# S5: deadline-aware micro-scheduling for colocated stages

## Verdict

**The hypothesis is not confirmed for TTFA.** On one H100 with Thinker, Talker, Code2Wav and both encoders colocated on rank 0
(`configs/qwen3omni_colocated.yaml`), picking the next (node, walk) by deadline slack instead of round-robin:

* as first specified (Code2Wav > Talker > Thinker decode > prefill, least slack first), **doubled TTFA p50 in every case
  (+107% to +155%)** and cut text throughput by 63%. Cause, from the `sched_pick` telemetry: Thinker decode reported no slack, so it
  ranked behind every walk that had one and ran only once per aging guard (200 ms; steps/s 35 -> 15, wait p99 206 ms). The Talker
  then ran alone at batch 1.7 instead of 3.7 (73% of GPU time against 54%), waiting for tokens.
* with one fix (`MSTAR_SCHED_PIPELINE_AWARE=1`: Thinker decode carries the deadline of the stream it feeds, and the speculation
  chain yields on urgency rather than priority) and a 50 ms aging guard (`deadline_pa_w50`), **TTFA p50 is level with round-robin**
  (-3%, +12%, +2%, +3% in a4, a8, b4, b8; the repeat of round-robin differs by -1% to +6%). Under mixed load (b4, b8) the audio
  chunk gap p50 is 18-22% shorter and RTF p50 9-12% lower (both well outside the round-robin repeat's -2%..+3%), at a **text cost of
  15-18% of tok/s and +44-59% text TTFT p50** (text ITL p99 within noise). With the default 200 ms guard (`deadline_pa`) TTFA p50
  is +14..+25% worse than round-robin and text tok/s is 25-31% lower.
* Nobody stalls. In every round-robin and pipeline-aware arm the listener-side stall is 0 s: the server makes audio 3-5x faster
  than real time at these loads (RTF 0.18-0.42), so a stream's runway never gets short. The only deadline that binds is the
  first chunk, where every stage of the new request is equally late, so there is little for slack ordering to improve.
* The p99 columns do not resolve effects below about 40%: n is 48-64 audio requests per cell, and round-robin against its own
  repeat moves RTF p99 by +41% (a4) and TTFA p99 by up to 11%.

So: do not enable `MSTAR_SCHED_POLICY=deadline` for Qwen3-Omni on this evidence. All of it stays opt-in; the default is the
round-robin it was.

## Results

Job 13085 appended three arms to the run of job 13084, so all five arms are in `run1/analysis.md` (full tables: samples, every audio
and text metric with deltas, rep-to-rep spread, per-walk longest wait and GPU share, `sched_pick` reasons). Headline numbers, pooled
over 2 reps per case (a4/a8: 48/64 audio requests, b4/b8: 48/64 audio plus 48-192 text requests):

| case | metric | round_robin | round_robin_b | deadline | deadline_pa | deadline_pa_w50 |
|---|---|---|---|---|---|---|
| a4 | TTFA p50 | 0.65 s | 0.65 s | 1.34 s | 0.74 s | 0.63 s |
| a4 | TTFA p99 | 1.17 s | 1.03 s | 3.03 s | 0.88 s | 0.93 s |
| a4 | RTF p50 | 0.18 | 0.18 | 0.26 | 0.19 | 0.18 |
| a4 | audio chunk gap p50 | 0.26 s | 0.25 s | 0.19 s | 0.23 s | 0.25 s |
| a8 | TTFA p50 | 0.76 s | 0.80 s | 1.93 s | 0.95 s | 0.85 s |
| a8 | TTFA p99 | 1.33 s | 1.33 s | 3.27 s | 1.31 s | 1.36 s |
| a8 | RTF p50 | 0.24 | 0.24 | 0.30 | 0.24 | 0.24 |
| a8 | audio chunk gap p50 | 0.40 s | 0.39 s | 0.24 s | 0.30 s | 0.36 s |
| b4 | TTFA p50 | 0.84 s | 0.83 s | 1.95 s | 1.01 s | 0.86 s |
| b4 | TTFA p99 | 1.76 s | 1.59 s | 3.45 s | 2.00 s | 1.74 s |
| b4 | RTF p50 | 0.36 | 0.35 | 0.29 | 0.29 | 0.33 |
| b4 | audio chunk gap p50 | 0.58 s | 0.59 s | 0.22 s | 0.28 s | 0.47 s |
| b4 | TTFT p50 | 0.40 s | 0.39 s | 0.67 s | 0.67 s | 0.57 s |
| b4 | ITL p99 | 86.3 ms | 83.2 ms | 249.6 ms | 236.4 ms | 93.3 ms |
| b4 | text tokens | 292.2 tok/s | 293.3 tok/s | 107.2 tok/s | 200.2 tok/s | 240.0 tok/s |
| b8 | TTFA p50 | 0.96 s | 0.95 s | 2.25 s | 1.13 s | 0.99 s |
| b8 | TTFA p99 | 1.87 s | 2.03 s | 3.40 s | 2.28 s | 2.32 s |
| b8 | RTF p50 | 0.41 | 0.42 | 0.34 | 0.33 | 0.36 |
| b8 | audio chunk gap p50 | 0.70 s | 0.68 s | 0.25 s | 0.43 s | 0.55 s |
| b8 | TTFT p50 | 0.32 s | 0.35 s | 0.68 s | 0.65 s | 0.50 s |
| b8 | ITL p99 | 116.2 ms | 110.0 ms | 256.3 ms | 244.5 ms | 96.1 ms |
| b8 | text tokens | 259.9 tok/s | 257.6 tok/s | 92.3 tok/s | 194.0 tok/s | 222.0 tok/s |

Arms. `round_robin`: today's behaviour. `round_robin_b`: the same on a second server, the noise floor. `deadline`: the policy as first
specified, 200 ms guard. `deadline_pa`: plus `MSTAR_SCHED_PIPELINE_AWARE=1`. `deadline_pa_w50`: the same with a 50 ms guard.
Cases: a4/a8 audio-only at concurrency 4/8; b4/b8 the same plus text-only requests with 1-2k-token prompts at concurrency 8.
Each server ran one throwaway warm-up (audio conc 8 + text conc 8), then 2 reps of each case.

## What was built

mstar worktree `/shared/home/garv901-55613a/mstar-worktrees/regime-batching/s5-stage-deadline`, branch `garv/s5-stage-deadline`:

* `NodeSubmodule.walk_priority(walk)` and `walk_deadline_slack(walk, rid, info, now)`: model hints, default None.
* `SchedulingType.DEADLINE`, chosen by `MSTAR_SCHED_POLICY=deadline` (default `round_robin`). Pick order over the walks with a ready
  request: (1) a walk whose oldest ready request waited past `MSTAR_SCHED_MAX_WAIT_MS` (default 200), oldest first; (2) least
  slack, walks with a deadline before those without; (3) higher priority; (4) least recently scheduled. The speculation chain
  yields only to a walk at least as high-priority, or past the guard.
* Qwen3-Omni: Code2Wav 100 > Talker 80 > Thinker decode 60 > Thinker prefill 20 > encoders 10. Slack from an `AudioClock`: a stream
  whose first chunk went out at t0 and that has sent `s` seconds of audio since runs dry at t0 + s; before its first chunk it is
  0 (due now).
* `MSTAR_SCHED_PIPELINE_AWARE=1` (opt-in): Thinker decode of an audio request reports that slack too; the chain yields to a walk whose
  request is within the guard of its deadline rather than by priority.
* `sched_pick` telemetry event (only with `MSTAR_STEP_TELEMETRY_DIR`): the chosen walk, reason, slack, and the wait of every ready
  walk. Emitted under round-robin too, so the arms are compared by one measure.
* `MSTAR_CODE2WAV_MAX_GRAPH_BS` (opt-in, default 32 = unchanged): drops the large Code2Wav CUDA-graph buckets. Needed to run the
  colocated config at all on an 80 GB GPU: with the bs=32 vocoder graph in place the Thinker's graphs failed to capture and the
  first request ran out of memory (job 13068). The experiment sets 8 in every arm.

Tests: `test/modular/test_micro_scheduler_deadline.py` (51 tests, with mutation checks on the pick, aging, yield and slack rules).
Full `test/modular`: 2331 passed, 193 skipped, 2 failed; the 2 failures (`test_graph_runtime_factory::test_rust_refuses_the_pyzmq_communicator`,
`test_kokoro_model::test_missing_language_backend_is_reported_once`) fail identically on a clean checkout of the base commit.

## Jobs

| job | outcome |
|---|---|
| 13064 | died in 4 min: the worktree `.venv` has no `flash_attn` (logs in `failed_13064_no_flash_attn/`) |
| 13068 | worker OOM on the first request: graph capture left no memory (logs in `failed_13068_graph_oom/`); cancelled |
| 13084 | round_robin and deadline arms, `run1/` |
| 13085 | deadline_pa, round_robin_b, deadline_pa_w50 arms, same `run1/` |

The server runs the worktree's code in `/data/garv901-55613a/venvs/mstar-cu129` (torch 2.12.1+cu129, flash-attn 2.8.3.post1) with
`PYTHONPATH` set to the worktree; `experiments/mstar/lib_mstar.sh` gained `MSTAR_VENV` (default `$MSTAR_DIR/.venv`, so other jobs are
unchanged) for that.

## Caveats

* **19 CUDA graphs failed to capture on every server** (all arms alike): Thinker decode bs=4 (both slots), Thinker prefill_text bs=4
  at every length and bs=1 at 2048 tokens, the large vision buckets. Those shapes run eagerly, so absolute latencies are
  pessimistic for a fully captured server; the arms are compared on identical footing. Fitting every node on one 80 GB GPU leaves
  too little room for them.
* Backlog-first serving is unchanged: a prefill burst above the batch cap goes through the backlog, which is served before the
  ready scan, so priority does not apply to it.
* The yield test uses priority and aging only for backlogged walks (slack only for ready requests), and only under
  `MSTAR_SCHED_PIPELINE_AWARE` for those.
* Under round-robin the telemetry waits can understate by about two steps (the speculation-yield check does not stamp ready times
  in that mode).
* 2 reps, one server per arm. The round-robin repeat is the only estimate of server-to-server noise.
* Only audio concurrency 4 and 8 were run; a case where streams actually stall (higher concurrency, or a slower GPU) was not.
  The Code2Wav graph cap and the memory ceiling limit how high this config can go.
* The host also ran other users' jobs (load average 28-40 during the runs), which the repeat arm absorbs only in part.

## Reproduce

    sbatch experiments/mstar/s5/s5_ab.sbatch                                       # round_robin, deadline
    ARMS="deadline_pa round_robin_b deadline_pa_w50" sbatch --export=ALL experiments/mstar/s5/s5_ab.sbatch
    python3 experiments/mstar/s5/analyze.py --run results/mstar/s5/run1 > results/mstar/s5/run1/analysis.md

`analysis_first_ab_round_robin_vs_deadline.md` is the analysis after job 13084 alone. Raw step telemetry (12 MB per server) is
not in git; it is under `run1/<arm>/server_*/telemetry` on the machine that ran it.
