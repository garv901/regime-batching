# mstar harness (BAGEL text-to-text)

GPU-side harness for benchmarking mstar (in-house serving engine) with BAGEL-7B-MoT, text-to-text only.
Baselines run from the pristine worktree `mstar-worktrees/bench-main` (`origin/main`, venv at `.venv`).

## Files

| File | Purpose |
|---|---|
| `lib_mstar.sh` | Port of `experiments/lib_serve.sh`: `start_server`, `wait_health`, `gpu_drain`, `stop_server`, plus `run_client` (mstar's `benchmark.runner`) and a 1 Hz GPU-memory sampler |
| `smoke_bagel.sbatch` | Server from bench-main, then 8 closed-loop client phases, then a warm restart for startup time |
| `summarize_smoke.py` | Prints the facts below from a smoke output dir |
| `kv_probe.py` | Prints the KV pool (pages, page size, bytes) from the model spec and the config YAML (main does not log it) |
| `make_prompt_file.py` | One-prompt-per-line text file with exact BAGEL token counts for `benchmark.runner --request-txt-file` |
| `make_bagel_workloads.sh`, `kv_report.py` | Generate the BAGEL workloads and add their KV pressure to the `.meta.json` files |
| `make_bagel_workloads_v2.sh` | The v2 workloads (below); writes only the v2 files and their `.meta.json` |
| `make_bagel_workloads_v3.sh` | The v3 workloads (below); writes only the v3 files and their `.meta.json` |
| `admission_matrix.sbatch`, `submit_matrix.sh` | One admission config per job (one server, every workload); `submit_matrix.sh` submits REPS jobs per arm, interleaved, and logs them in `manifest.tsv` (section "Admission matrix") |
| `probe_guest.sbatch` | CPU-only check of what a guest node (node2, node4, node5) sees; result in section "Admission matrix" |

## How to run

```bash
sbatch experiments/mstar/smoke_bagel.sbatch      # from the repo root; output in /data/garv901-55613a/logs/mstar/
python3 experiments/mstar/summarize_smoke.py     # newest smoke_* dir
```

In your own job script:

```bash
source /data/mstar/env-laude.sh; export HF_HOME=/data/garv901-55613a/hf
PORT=...; MSTAR_DIR=/shared/home/garv901-55613a/mstar-worktrees/<worktree>     # needs its own .venv
EXTRA_ENV="MSTAR_STEP_TELEMETRY_DIR=$D/telemetry MSTAR_KV_ADMISSION_FIT=1 MSTAR_KV_ADMISSION_ORDER=lpt"   # server only
source experiments/mstar/lib_mstar.sh
start_server "$D" [extra mstar serve args]       # waits for /health; sets SRV and SERVER_STARTUP_S
run_client "$D" --request-type text_to_text --profiling-type closed_loop --max-concurrency 8 \
    --ignore-eos --output-len-min 128 --output-len-max 128 --num-requests 16
stop_server $SRV
```

The server command is `mstar serve bagel --config configs/bagel_single_gpu.yaml --tensor-comm-protocol SHM --port $PORT`
(all flags exist on main; `lib_mstar.sh` adds `--host 127.0.0.1`, a per-job `--socket-path-prefix` and `--upload-dir`,
because the default `/tmp/mstar_<user>/` is shared by every job of this user on the node).
`EXTRA_ENV` is applied after the defaults (`HF_HUB_OFFLINE=1`, `MSTAR_STEP_TELEMETRY_DIR=$case/telemetry`), so it can override them.
`LOG_STATS=1` adds `--log-stats-file`; it turns on worker profiling, so it is off by default.

Client notes: `benchmark.runner` has `--ignore-eos` and `--output-len-min/--output-len-max` (sent as `max_tokens` and
`max_output_tokens`; the default for BAGEL is 128 and the server default is 2048). Greedy decoding is set by the client.
It sends one `POST /generate` per request, with Poisson arrivals at a fixed `--rate` or a closed loop; it does not replay
`timed_trace` JSONL files, so the workloads below need a replay client on top of `lib_mstar.sh`.
The server runs with `HF_HUB_OFFLINE=1`; the client does not, because transformers 4.57 `AutoTokenizer` calls the hub API
even for a cached repo (without it the client only loses the per-token ITL).

## Facts (BAGEL-7B-MoT, `configs/bagel_single_gpu.yaml`, 1x H100 80 GB, SHM, mstar `origin/main` 1ce2e92b2)

| Fact | Value | Evidence |
|---|---|---|
| KV resources | one, `kv` (labels `main`, `cfg_text`, `cfg_img` share it); text-to-text uses `main` only | `BagelModel.get_node_resources` |
| `page_size` | 128 tokens | `kv_probe.py` on the real spec + YAML (no `resources:` override in `bagel_single_gpu.yaml`) |
| `max_num_pages` | 2048, fixed (not fitted to free GPU memory) | same |
| KV pool | 262,144 tokens = 14.0 GiB | 2048 x 128 |
| Bytes per page / per token | 7,340,032 (7.0 MiB) / 57,344 (56 KiB) | `kv_bytes_per_token=57344 bytes_per_page=7340032` |
| KV geometry (assumption confirmed) | 28 layers, 4 KV heads, head_dim 128 (3584 / 28 heads), bf16: 28 x 2 x 4 x 128 x 2 B = 57,344 B per token | `llm_config.json` and `PagedKVConfig(num_layers=28, num_kv_heads=4, head_dim=128, ...)` |
| Host offload, prefix cache | `cpu_offload_pages=0`, `prefix_cache=True` | `PagedKVConfig` printed by `kv_probe.py` |
| `max_seq_len` | 32768 (YAML and `max_position_embeddings`) | `configs/bagel_single_gpu.yaml` |
| Output tokens per request | at most 2,049 (1 from prefill + 2,048 decode-loop iterations), whatever `max_tokens` asks | `MAX_OUTPUT_TOKENS = 2048` (`mstar/model/base.py:32`) is the loop's `max_iters` (`bagel_model.py:988`); every v1 long (asked 8,000) ended at 2,049 |
| CUDA graph capture (from source, not yet seen in a log) | decode: batch 1, 2, 4, 8, 16, 32, 64, each for guidance off and on. `prefill_text` (guidance off): 128, 256, 512, 1024, 2048 tokens x batch 1, 2, 4, with `torch.compile`. ViT capture off (`MSTAR_VIT_CUDA_GRAPH=0`). The scheduler splits a batch above the largest captured size, so decode runs at most 64 requests per step. | `cuda_graph_runner.py` (`DEFAULT_CAPTURE_BATCH_SIZES`), `LLMSubmodule.get_cuda_graph_configs` (bagel/submodules.py) |
| flash-attn | not needed for text-to-text. Only `BagelViT` imports it (falls back to torch SDPA with a warning). LLM attention is FlashInfer (default `AttnBackend.FLASHINFER`). Not installed in bench-main. | `vit_encoder.py:25`, `attn/config.py` |
| venv | torch 2.12.1+cu132, flashinfer 0.7.0.post1 (same as the other new bench worktrees; the older `graphapi-testing` venv is cu130 / 0.6.18) | `uv pip list` |
| GPU memory after weight load, after CUDA-graph capture (free memory left) | NOT YET MEASURED: needs a GPU job | `smoke_bagel.sbatch` samples `nvidia-smi` at 1 Hz; `summarize_smoke.py` maps it to the log markers |
| Startup time (cold and warm compile caches) | NOT YET MEASURED | `startup.txt` |
| Per-request latency, tokens/s, decode capacity | NOT YET MEASURED | `client_*.log`, `out_*/results.json` |

The first smoke job has not run: node1's eight GPUs were held by 8-GPU jobs of other users when this was written.
After it runs, fill the three NOT YET MEASURED rows from `summarize_smoke.py`. If the measured pool differs from 262,144,
rerun `POOL=<tokens> ITL_MS=<measured decode ITL> bash experiments/mstar/make_bagel_workloads.sh`.

## Workloads

All in `data/workloads/`, `timed_trace` format like `lburst80_8k.jsonl`, generated by `make_bagel_workloads.sh`.
`input_length` is the prompt text in tokens; BAGEL adds a 20-token template
(`<|im_start|>{system}<|im_end|><|im_start|>{prompt}<|im_end|><|im_start|>`, measured with the BAGEL tokenizer), so a
request's final KV is `input_length + 20 + output_length`. `max_seq_len` is 32768, enforced by `kv_report.py`.
Pool P = 262,144 tokens. "Long" means the long class of each file.

**The v1 files below ask for 8,000 output tokens on every long, but BAGEL stops a request at 2,049 (see the Facts table), so
the KV pressure in the last column was never reached: the v1 burst's longs held ~0.45 P, not 1.5 P, and the staggered runs had no
pressure. The server still reserved prompt + 8,000 per long, so v1 runs only test over-declared `max_tokens`. Use the v2 files
(next section) for KV-pressure results; `analysis/mstar_admission.py` flags the v1 runs `TRUNCATED`.**

| File | Requests | Arrivals | Short class | Long class | KV pressure (pool P) |
|---|---|---|---|---|---|
| `bagel_burst.jsonl` | 2,700 over 305 s | shorts at 9/s (16/s x 262k/451k), 46 long at t = 30 s | prompt 450-1,000, output 16-64 (2,654) | prompt 512, output 8,000 (46) | long final KV 392k = **1.50 P** (lburst80_8k: 681k vs 451k = 1.51x) |
| `bagel_stag_r0.5.jsonl` | 300 over ~600 s | Poisson 0.5/s | same (262) | same, each request long with p = 0.15 (38) | peak in-flight 0.31 P |
| `bagel_stag_r1.jsonl` | 600 over ~600 s | Poisson 1/s | same (521) | same (79) | peak in-flight 0.60 P |
| `bagel_stag_r2.jsonl` | 1,200 over ~600 s | Poisson 2/s | same (1,008) | same (192) | peak in-flight 1.12 P |
| `bagel_control.jsonl` | 144 over 58 s | Poisson 2.4/s | same (136) | prompt 512, output 1,024 (8) | total final KV of all requests 0.46 P (<= 0.5 P) |

Peak in-flight KV in the stag rows is a fluid estimate (`kv_report.py --itl-ms 20`): every request admitted on arrival and
decoded at a fixed 20 ms per token, no queueing. The 20 ms is an assumption until the smoke job gives the real ITL.
The rates 0.5 / 1 / 2 req/s are PROVISIONAL for the same reason. The reasoning: with 15% long requests of 8,000 tokens a request
needs 1,234 decode tokens on average, so 2 req/s asks for about 2,500 decode tokens/s (near the capacity of one H100 at
batch 64 if a step takes 25 ms), and its in-flight KV reaches about 1.1 P. Check this against `dec_c64` and `long_c16` of the
smoke job: low should leave decode below ~30% of the measured tokens/s, medium about 60%, overload above 100%.
Prompts are random tokens (no prefix sharing). Longest request: 512 + 20 + 8,000 = 8,532 tokens, far below `max_seq_len`.

## Workloads v2 (every requested output <= 2,048, so requested == generated)

Same short class (prompt 450-1,000, output 16-64). The longs carry their KV in the prompt instead of the output (output 2,000, below the
2,049 cap). Generated by `make_bagel_workloads_v2.sh` (uses `make_workload.py --long-prompt-lo`; the v1 files are untouched and
regenerate byte-identically). `bagel_control` is unchanged (its longs ask for 1,024).

| File | Requests | Arrivals | Long class | KV pressure (pool P) | Demand (average) |
|---|---|---|---|---|---|
| `bagel_burst_v2.jsonl` | 2,700 over 305 s | shorts at 9/s, 46 long at t = 30 s | prompt 6,500, output 2,000 | long final KV 392k = **1.50 P**; fluid peak 1.52-1.53 P | decode 650 tok/s; prefill 7.5k tok/s (the burst is 300k prompt tokens, ~10 s of prefill) |
| `bagel_stag_v2_r1.5.jsonl` | 900 over 601 s | Poisson 1.5/s, 15% long (123) | prompt U[8,000, 16,000], output 2,000 | fluid peak **0.69-0.79 P** | decode 460 tok/s; prefill 3.4k tok/s |
| `bagel_stag_v2_r2.5.jsonl` | 1,500 over 604 s | Poisson 2.5/s (244 long) | same | fluid peak **1.07-1.39 P** | decode 900 tok/s; prefill 6.4k tok/s |
| `bagel_stag_v2_r3.jsonl` | 1,800 over 603 s | Poisson 3/s (295 long) | same | fluid peak **1.39-1.66 P** | decode 1.08k tok/s; prefill 7.8k tok/s |

The fluid peak is `kv_report.py --itl-ms 15` to `--itl-ms 20` (each `.meta.json` keeps only the 20 ms value, the script prints both): every request admitted on arrival,
a fixed decode ITL, no queueing. The measured effective ITL is 7 / 9 / 17.5 / 32 ms at decode batch 8 / 16 / 32 / 64, so 15-20 ms fits
the 20-40 requests in flight. Decode saturates near 2k tok/s, so all four stay below ~55% of it; prefill is assumed to run at 20-40k tok/s.

Policy check on CPU (`python analysis/admit_sim_bagel.py data/workloads/bagel_*_v2*.jsonl`; BAGEL step cost from the smoke
numbers, prefill 30k tok/s assumed; the order of the policies matters, not the seconds). Short E2E mean (p99), seconds:

| Workload | sum_fifo | sum_bf | peak_fifo | peak_bf |
|---|---|---|---|---|
| `bagel_burst_v2` | 9.2 (57) | 4.3 (29) | 9.5 (59) | 1.1 (10) |
| `bagel_stag_v2_r1.5` | 0.38 (1.3) | 0.38 (1.3) | 0.38 (1.3) | 0.38 (1.3) |
| `bagel_stag_v2_r2.5` | 2.7 (21) | 0.82 (4.1) | 2.5 (21) | 0.74 (2.4) |
| `bagel_stag_v2_r3` | 44 (88) | 1.7 (7.9) | 43 (94) | 1.3 (6.9) |

The separation is strict FIFO against backfill (head-of-line blocking of the shorts), and peak against sum only inside backfill;
`sum_fifo` and `peak_fifo` are alike. `bagel_stag_v2_r1.5` is the no-pressure rung. Submit the v2 set with the `WORKLOADS=` line in
the header of `admission_matrix.sbatch`.

## Workloads v3 (long outputs ~U[256, 2,000], so the longs finish at different times)

In v1 and v2 every long of a file has the same output, so the longs of a trace start and finish in lockstep and a summed
reservation (prompt + `max_tokens` for the whole life) and a future-peak fit test see nearly the same set of requests. In v3 each
long's output is ~U[256, 2,000] (every request <= 2,000 < 2,049, so requested == generated), and the KV a running long still has to
grow by differs per request. Same short class as v1/v2 (prompt 450-1,000, output 16-64, `cls` "short"). Generated by
`make_bagel_workloads_v3.sh` with the new `make_workload.py --long-lo` (long outputs ~U[long-lo, long]; the draw comes after every
other draw and only when the flag is set, so the v1 and v2 files regenerate byte-identically: checked by running both scripts in a
copy of the tree under /tmp and comparing every file). The v3 files are new; nothing under `data/workloads/` was modified.

| File | Requests | Arrivals | Long class | KV pressure (pool P) |
|---|---|---|---|---|
| `bagel_burst_v3.jsonl` | 2,700 over 305 s | shorts at 9/s, 90 long at t = 30 s | prompt U[1,000, 4,000], output U[256, 2,000] | long final KV 315k = **1.20 P**; fluid peak 0.96-0.98 P |
| `bagel_stag_v3_r4.5.jsonl` | 2,700 over 593 s | Poisson 4.5/s, 15% long (425) | prompt U[4,000, 12,000], output U[256, 2,000] | fluid peak 0.78-0.91 P; sim `open` 1.31 P |
| `bagel_stag_v3_r5.5.jsonl` | 3,300 over 591 s | Poisson 5.5/s (480 long) | same | fluid peak 0.84-1.02 P; sim `open` 4.71 P |
| `bagel_stag_v3_r6.5.jsonl` | 3,900 over 591 s | Poisson 6.5/s (563 long) | same | fluid peak 0.84-1.03 P; sim `open` 8.21 P |

Fluid peak is `kv_report.py --itl-ms 15` to `--itl-ms 20`; a `.meta.json` keeps only the last call (20 ms; `kv_report.py` rebuilds the
`kv` block on every call, so the v2 files hold only `itl20ms` too, although the v2 script says one key per ITL), the 15 ms values are
printed by the script. The fluid model keeps the decode ITL fixed, so it stays near 1 P for the staggered files; the simulator's
`open` row (no limit, decode slows as the batch grows) is the demand the pool would have to carry. Decode demand is 0.9-1.0k tok/s
(burst 0.6k): the stag runs sit at ~50% of the ~2k tok/s decode capacity and are bound by the KV pool, not by decode.

Policy check on CPU (`python analysis/admit_sim_bagel.py data/workloads/bagel_*_v3*.jsonl`; same calibration as v2). Short E2E mean
(p99) in seconds, then long E2E max and output tokens/s:

| Workload | sum_fifo | sum_bf | peak_fifo | peak_bf | open |
|---|---|---|---|---|---|
| `bagel_burst_v3`, short E2E | 6.40 (38.7) | 3.81 (20.8) | 1.63 (9.9) | 1.62 (9.9) | 1.25 (11.0) |
| long E2E max / tok/s | 101.8 / 634 | 101.3 / 634 | 99.0 / 634 | 99.0 / 634 | 99.0 / 634 |
| `bagel_stag_v3_r4.5`, short E2E | 1.53 (9.8) | 0.97 (2.8) | 1.38 (9.8) | 0.99 (3.2) | 0.97 (3.0) |
| long E2E max / tok/s | 65.1 / 942 | 67.5 / 942 | 66.2 / 942 | 68.4 / 942 | 74.3 / 942 |
| `bagel_stag_v3_r5.5`, short E2E | 36.5 (67.9) | 2.13 (7.3) | 34.4 (64.2) | 6.76 (21.2) | 4.13 (10.5) |
| long E2E max / tok/s | 129.1 / 993 | 202.7 / 991 | 127.9 / 997 | 165.1 / 995 | 271.8 / 1020 |
| `bagel_stag_v3_r6.5`, short E2E | 64.7 (138.4) | 3.65 (18.7) | 62.6 (134.1) | 13.90 (56.7) | 7.68 (20.5) |
| long E2E max / tok/s | 190.3 / 978 | 320.9 / 980 | 189.2 / 982 | 212.4 / 983 | 417.8 / 1012 |

Nothing is unfinished or over 600 s in any cell; the traces end by 305 s (burst) and 610-747 s (stag) of simulated time, so each run
fits RUN_LIMIT (2400 s) with the 900 s client timeout to spare.

What this says (simulator only; the real engine has to confirm it):
- **The burst separates the future-peak policies from the summed ones**: short E2E mean 3.9x better for peak_fifo than sum_fifo
  and 2.4x for peak_bf over sum_bf (p99 3.9x and 2.1x), at equal throughput; the price is long E2E (mean 72 s against 65-67 s) and
  long TTFT is better. It needs growth-dominated longs: a long's growth (output) is ~31% of its final KV here.
- **The required staggered family (prompt U[4,000, 12,000]) does not.** A long's KV is ~88% prompt, so summing prompt + `max_tokens`
  over-books the pool by only ~12% and the peak test admits barely more (mean admitted 33.9 against 31.5 at 5.5/s). peak_fifo is 5%
  better than sum_fifo (36.5 to 34.4 s), throughput is within 0.5%, and **peak_bf is worse than sum_bf on short E2E** (6.8 against 2.1 s
  at 5.5/s, 13.9 against 3.7 s at 6.5/s) but better on the tail of the longs (long E2E max 165 against 203 s, 212 against 321 s).
  Hypothesis (not checked in the simulator): peak fills the pool to 1.00 now, so a short waits for pages that a summed reservation
  would still have left unallocated. With other prompt ranges (U[1,000, 4,000], U[2,000, 6,000]), 15-30%
  long and 6-12 req/s, peak_bf was never better than sum_bf on short E2E either (equal where nothing queues, e.g. 6 req/s at 15%),
  and the throughput of the peak policies was within -3% to +3% of the summed ones. These files are the
  control where peak is NOT expected to win (and could lose short E2E); use the burst for the case it should win.
- r4.5 is the low-pressure rung (everything within 1 s of `open`); r5.5 and r6.5 are where the pool binds (sum_fifo makes the shorts wait 36-65 s).

How the burst was picked (short E2E mean, seconds, simulator; seed 0; all with outputs U[256, 2,000]):

| Burst long prompt, n | sum_fifo | sum_bf | peak_fifo | peak_bf |
|---|---|---|---|---|
| U[1,000, 4,000], 70 (0.9 P) | 0.96 | 0.95 | 0.90 | 0.90 |
| **U[1,000, 4,000], 90 (1.20 P), chosen** | 6.40 | 3.81 | 1.63 | 1.62 |
| U[1,000, 4,000], 100 | 11.32 | 5.64 | 7.31 | 5.30 |
| U[500, 4,000], 90 | 4.79 | 3.29 | 1.19 | 1.19 |
| U[500, 4,000], 100 | 9.83 | 5.47 | 3.09 | 3.01 |
| U[2,000, 6,000], 90 | 9.67 | 3.86 | 8.75 | 6.76 |
| U[4,000, 12,000], 52 | 5.64 | 2.01 | 5.06 | 3.31 |
| U[4,000, 12,000], 80 | 19.01 | 5.16 | 18.94 | 8.97 |

Peak wins where the summed reservation saturates the pool but the KV actually in use does not yet: at n = 70 nothing queues, at n = 90
with prompts <= 4,000 the summed policies queue the shorts and the peak policies do not. At n = 100 or with larger prompts the
advantage shrinks or reverses (peak_bf then fills the pool to 1.00; not checked further).
Seed spread of the chosen burst (same parameters, seeds 1-3; seed 0 is the best of the four): sum_bf / peak_bf short E2E 6.21 / 4.74,
5.25 / 3.51, 4.66 / 2.60 (seed 0: 3.81 / 1.62), sum_fifo / peak_fifo 10.11 / 4.87, 9.49 / 3.57, 8.74 / 2.64 (seed 0: 6.40 / 1.63): the sign
holds, the size is 0.56-0.76x (bf) and 0.30-0.48x (fifo) over seeds 1-3. Stag r5.5, seeds 2 and 3: sum_bf / peak_bf 2.14 / 4.14 and
1.95 / 4.67; sum_fifo / peak_fifo 40.3 / 38.3 and 25.6 / 23.7: also the same sign as seed 1.

Submit: `WORKLOADS="bagel_burst_v3 bagel_stag_v3_r4.5 bagel_stag_v3_r5.5 bagel_stag_v3_r6.5 bagel_burst_v3"` (the burst twice is the
first-run check, as in v2; the default WARMUP replays `bagel_burst_v2`, set `WARMUP=bagel_burst_v3` to warm up on the v3 burst).
The analysis orders the v3 files after v2 (`WORKLOAD_ORDER`).

## Admission matrix: repeats, warmup, guest nodes

`admission_matrix.sbatch` runs one config per job (one server, every workload, a fresh server only if one run leaves it unhealthy).
`CONFIG` picks the worktree and admission env (`main ac sum_bf peak_fifo peak_bf ac_fast peak_bf_fast next`); `next` is the worktree
`bench-next` with no admission env of its own, so an arm on it differs only by `EXTRA_ENV` (+ `LABEL`).
- `WARMUP` (default `bagel_burst_v2`; empty = none): one throwaway replay on each fresh server into `$OUT/warmup_<wl>/`, before the
  measured workloads. The first burst on a fresh server is ~40% worse in short E2E in every arm. It costs ~6 min per server.
  The analysis skips every `warmup_*` directory.
- Partitions `-p team1,guest` (team1 = node1; guest = node1, 2, 4, 5, preemptable). A requeued job writes to a new
  `<job>_<label>_rq<N>` directory and the analysis pools it with its label (`--open-mode=append` keeps both attempts in the Slurm log).
- `submit_matrix.sh` submits REPS jobs per arm, interleaved (rep 1 of every arm, then rep 2, ...), e.g.
  `REPS=3 WORKLOADS="bagel_burst_v3 bagel_stag_v3_r5.5" ./experiments/mstar/submit_matrix.sh "peak_bf next MSTAR_KV_ADMISSION_FIT=peak MSTAR_KV_ADMISSION_ORDER=backfill" "sum_bf next MSTAR_KV_ADMISSION_FIT=sum MSTAR_KV_ADMISSION_ORDER=backfill"`
  (`-n` = dry run, `-f file` = arms from a file). Every job is appended to `/data/garv901-55613a/results/mstar/manifest.tsv`
  (job id, label, config, extra env, workloads, warmup, submit time, rep).
- **A guest node cannot run the matrix yet.** `probe_guest.sbatch` (jobs 13056-13058, CPU only) found: `/shared` is NFS4 from node1 on
  node2, node4 and node5 (the repo, worktrees and venvs are visible: `import mstar, torch, flashinfer` works, torch 2.12.1+cu132), but `/data`
  is a node-local XFS on each of them, and none has `/data/mstar/env-laude.sh` (the CUDA toolkit), the BAGEL snapshot under
  `/data/garv901-55613a/hf` (node4 has no `hf` dir at all; node2 and node5 have the dir but not the model) or
  `/data/garv901-55613a/results/mstar` (nothing to write results to). The sbatch therefore stops in its preflight (exit 3, at once) on
  such a node. To use a guest node, stage `/data/mstar`, the HF snapshot and a results dir there first, and copy each job's results
  back to node1 (guest `/data` is not visible from node1). Until then run with `SBATCH_OPTS="-p team1"` so no job is lost to a guest node.

`analysis/mstar_admission.py <results dirs...> --out DIR` pools the jobs of one label as reps (`mean [min-max]` and n for the key
metrics), marks a ratio row "within noise" when the min-max ranges of the two configs overlap (a range-overlap rule, not a
significance test; the rule is in the header of `compare.md`), takes `--exclude GLOB` (repeatable, on case names), and reports host
ms per decode step overall and by waiting depth [0, 50), [50, 200), [200, 400), [400, inf) in `compare.md` and `summary.json`.
