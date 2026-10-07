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
| `make_bagel_workloads_v4.sh`, `declare_max_tokens.py`, `arms_v4.txt` | The v4 workloads (below): copies of four Phase 1c traces with a declared `max_tokens` above the generated output, and the four arms to run them under |
| `admission_matrix.sbatch`, `submit_matrix.sh` | One admission config per job (one server, every workload); `submit_matrix.sh` submits REPS jobs per arm, interleaved, and logs them in `manifest.tsv` (section "Admission matrix") |
| `probe_guest.sbatch` | CPU-only check of what a guest node (node2, node4, node5) sees; result in section "Admission matrix" |
| `fetch_public_traces.sh`, `convert_public_trace.py`, `make_public_workloads.sh`, `public_trace_report.py` | Download (SHA-256 checked) and convert the Mooncake and Azure production traces into `data/workloads/pub_*` files, characterise them and run the admission simulator on them (section "Public traces") |

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

## Workloads v4 (declared max_tokens above the generated output)

v1-v3 are ORACLE traces: the replay client sends `max_tokens = output_length` (and ignores EOS), so a summed reservation
(prompt + `max_tokens`) and a future-peak fit see every request's exact length. A production client over-declares `max_tokens` or
omits it (the server then assumes 2,048 for BAGEL). v4 asks what that costs each arm. A trace row may carry a declared
`max_tokens` (>= `output_length`, <= 2,048); the client sends it as `max_tokens` / `max_output_tokens`, and the server stops the
request after `output_length` tokens (emulated EOS), so the KV a request really uses is the base's and only what admission
reserves grows. A row without the field behaves as before. `make_bagel_workloads_v4.sh` derives two variants of each Phase 1c
base with `declare_max_tokens.py` and writes them to `data/workloads/` (the v1-v3 files are only read; the script rebuilds the
same bytes every time):

- `<base>_mtx2`: `max_tokens = min(2048, 2 x output_length)` (`--id-offset 1000000000`): a predictor with a 2x margin.
- `<base>_mt2048`: `max_tokens = 2048` for every request (`--id-offset 2000000000`): no information about the output length, or a
  client that omits `max_tokens`.

The id offset shifts every `hash_id`, so a variant replayed on the server that served its base builds different prompt tokens and
cannot hit the prefix cache the base left behind. (The same offset is used for every base, so variants of different bases share ids
as the bases already do among themselves.) Arrivals, prompts and generated outputs are the base's, so a `_mtx2` / `_mt2048` run generates
exactly the base's tokens (the tok/s a policy can reach is the base's). `kv_report.py` annotates
each variant's `.meta.json` (a copy of the base's with a `declared` block) with the declared figures next to the generated ones: per
class and total `declared_final_kv_tokens` / `declared_ratio_to_pool` (sum of prompt + 20 + declared) and `decl_kv_ratio`
(declared / generated), `decl_ratio_mean` (mean declared / output per request), and three fluid peaks (`--itl-ms 15` to `20`, every
request admitted on arrival, no queueing): with the generated lengths (the base's number), with every request running for its
DECLARED length (`peak_declared_inflight_kv_ratio_*`: what a planner must be ready for if nothing stops early), and of the summed
reservation (`peak_reserved_inflight_kv_ratio_*`: each in-flight request holds prompt + 20 + declared until it finishes; this is
what a summed fit sees). Oracle files get none of these keys (their output is byte-identical to before).

| File | Requests | Mean declared / generated | Declared / generated KV: all / short / long | Total final KV / P: generated -> declared | Fluid peak / P: generated lengths | declared lengths | summed reservation |
|---|---|---|---|---|---|---|---|
| `bagel_burst_v2_mtx2` | 2,700 (46 long) | 1.98 | 1.04 / 1.05 / 1.01 | 9.42 -> 9.83 | 1.52-1.53 | 1.54-1.55 | 1.54-1.55 |
| `bagel_burst_v2_mt2048` | 2,700 (46 long) | 58.82 | 3.16 / 3.57 / 1.01 | 9.42 -> 29.76 | 1.52-1.53 | 3.13-3.78 | 1.62-1.67 |
| `bagel_stag_v2_r3_mtx2` | 1,800 (295 long) | 1.84 | 1.01 / 1.05 / 1.00 | 20.46 -> 20.74 | 1.39-1.66 | 1.40-1.66 | 1.47-1.75 |
| `bagel_stag_v2_r3_mt2048` | 1,800 (295 long) | 49.81 | 1.57 / 3.55 / 1.00 | 20.46 -> 32.04 | 1.39-1.66 | 1.91-2.35 | 1.49-1.78 |
| `bagel_burst_v3_mtx2` | 2,700 (90 long) | 1.99 | 1.07 / 1.05 / 1.18 | 9.01 -> 9.62 | 0.96-0.97 | 1.10 | 1.44-1.45 |
| `bagel_burst_v3_mt2048` | 2,700 (90 long) | 57.88 | 3.26 / 3.56 / 1.30 | 9.01 -> 29.36 | 0.96-0.97 | 3.15-3.79 | 1.66-1.68 |
| `bagel_stag_v3_r5.5_mtx2` | 3,300 (480 long) | 1.95 | 1.06 / 1.05 / 1.06 | 25.30 -> 26.80 | 0.84-1.02 | 1.24-1.48 | 0.96-1.17 |
| `bagel_stag_v3_r5.5_mt2048` | 3,300 (480 long) | 52.19 | 1.92 / 3.56 / 1.10 | 25.30 -> 48.56 | 0.84-1.02 | 2.52-3.08 | 1.01-1.24 |

The longs of v2 already ask for 2,000 of the 2,048 cap, so v2 variants over-declare the SHORTS (16-64 tokens -> 32-128, or 2,048)
and barely the longs (1.00-1.01x); the v3 longs (output U[256, 2,000]) are over-declared up to 2x (1.06-1.30x by KV).
The "total" column sums every request of the 305-600 s trace and says little about pressure; the fluid peaks do. The fluid peak of
the summed reservation is 1.02-1.21x the fluid peak of the KV in use (`burst_v3`: 1.49x for `_mtx2`, 1.72x for `_mt2048`), so a
summed fit books only a few percent to a fifth more than the KV in use on most files.

Policy check on CPU (`python analysis/admit_sim_bagel.py data/workloads/bagel_*_v2*.jsonl data/workloads/bagel_*_v3*.jsonl`;
same calibration as v2; unchanged for the oracle rows, which were checked to give bit-identical results before and after the
change). `admit_sim_bagel.py` now reserves and plans with the declared `max_tokens` of a row that has one (sum: prompt + declared;
peak: growth up to declared) while the request still finishes after `output_length` tokens and frees its KV then. Short E2E mean
(p99) in seconds; `sum_fifo` is Gaurav's AC, `sum_bf` (AC with backfill) is added to separate head-of-line blocking from
over-reservation:

| Workload | sum_fifo (AC) | sum_bf | peak_bf | open |
|---|---|---|---|---|
| `bagel_burst_v2` | 9.21 (56.6) | 4.29 (28.5) | 1.11 (10.3) | 1.43 (12.4) |
| `bagel_burst_v2_mtx2` | 9.17 (56.3) | 5.53 (35.9) | 0.94 (7.4) | 1.43 (12.4) |
| `bagel_burst_v2_mt2048` | 9.12 (50.8) | 5.96 (33.0) | 7.41 (38.6) | 1.43 (12.4) |
| `bagel_stag_v2_r3` | 43.67 (88.4) | 1.65 (7.9) | 1.34 (6.9) | 2.65 (7.0) |
| `bagel_stag_v2_r3_mtx2` | 43.78 (88.9) | 1.49 (7.4) | 1.22 (4.8) | 2.65 (7.0) |
| `bagel_stag_v2_r3_mt2048` | 44.36 (88.0) | 7.25 (24.8) | 2.48 (14.2) | 2.65 (7.0) |
| `bagel_burst_v3` | 6.40 (38.7) | 3.81 (20.8) | 1.62 (9.9) | 1.25 (11.0) |
| `bagel_burst_v3_mtx2` | 6.58 (39.9) | 3.10 (16.7) | 1.25 (9.3) | 1.25 (11.0) |
| `bagel_burst_v3_mt2048` | 7.77 (40.8) | 7.00 (36.0) | 6.42 (36.4) | 1.25 (11.0) |
| `bagel_stag_v3_r5.5` | 36.46 (67.9) | 2.13 (7.3) | 6.76 (21.2) | 4.13 (10.5) |
| `bagel_stag_v3_r5.5_mtx2` | 37.85 (69.5) | 2.13 (5.9) | 1.51 (3.6) | 4.13 (10.5) |
| `bagel_stag_v3_r5.5_mt2048` | 39.65 (72.5) | 22.32 (63.8) | 3.15 (10.8) | 4.13 (10.5) |

Long E2E mean (s) / output tok/s (nothing is unfinished or over 600 s in any cell):

| Workload | sum_fifo (AC) | sum_bf | peak_bf | open |
|---|---|---|---|---|
| `bagel_burst_v2` | 76.8 / 648 | 88.1 / 648 | 94.6 / 648 | 121.6 / 648 |
| `bagel_burst_v2_mtx2` | 76.8 / 648 | 85.1 / 648 | 95.6 / 648 | 121.6 / 648 |
| `bagel_burst_v2_mt2048` | 76.1 / 648 | 83.2 / 648 | 81.4 / 648 | 121.6 / 648 |
| `bagel_stag_v2_r3` | 81.8 / 933 | 85.8 / 933 | 86.9 / 932 | 121.1 / 976 |
| `bagel_stag_v2_r3_mtx2` | 81.8 / 931 | 85.9 / 934 | 86.9 / 933 | 121.1 / 976 |
| `bagel_stag_v2_r3_mt2048` | 81.7 / 931 | 85.0 / 935 | 86.2 / 933 | 121.1 / 976 |
| `bagel_burst_v3` | 64.9 / 634 | 67.2 / 634 | 72.0 / 634 | 72.4 / 634 |
| `bagel_burst_v3_mtx2` | 58.6 / 634 | 65.6 / 634 | 72.0 / 634 | 72.4 / 634 |
| `bagel_burst_v3_mt2048` | 53.8 / 634 | 54.4 / 634 | 57.2 / 634 | 72.4 / 634 |
| `bagel_stag_v3_r5.5` | 69.2 / 993 | 73.9 / 991 | 74.5 / 995 | 108.9 / 1020 |
| `bagel_stag_v3_r5.5_mtx2` | 68.9 / 991 | 74.5 / 988 | 76.8 / 994 | 108.9 / 1020 |
| `bagel_stag_v3_r5.5_mt2048` | 69.0 / 987 | 71.7 / 986 | 75.6 / 992 | 108.9 / 1020 |

What each variant is meant to show (simulator only; the real engine has to confirm it):
- **AC (`sum_fifo`) hardly cares about the declared bound on these workloads** (short E2E within 4% of the oracle row for `_mtx2`
  and within 9% for `_mt2048`, except `burst_v3_mt2048` +21%): strict FIFO is already bound by the long at the head of the queue,
  and the peak of the summed reservation is only 2-21% above the peak KV in use (49-72% on `burst_v3`, the file where AC moves most). So
  `ac/main` should stay where it is on every variant, and the cost of not knowing the length shows up in the arm that used the
  length: `peak_bf`.
- **`_mtx2` (a 2x margin) should NOT separate the arms differently from the oracle**: peak_bf / AC short E2E is 0.10 (burst_v2),
  0.028 (stag_v2_r3), 0.19 (burst_v3), 0.040 (stag_v3_r5.5) against 0.12 / 0.031 / 0.25 / 0.185 for the oracle: it is the same or
  better. An over-declared growth makes the peak planner more conservative; on `stag_v3_r5.5`, where the oracle peak_bf is WORSE
  than sum_bf (6.76 vs 2.13 s, README v3), a 2x margin makes it better (1.51 vs 2.13 s). Hypothesis, not checked: peak_bf at the
  oracle fills the pool to 1.00 and a short waits for pages; the margin leaves some free.
- **`_mt2048` (no information) is the case that can separate them.** peak_bf loses most of its gain on the two bursts (peak_bf /
  AC 0.81 on burst_v2, 0.83 on burst_v3; the oracle 0.12 and 0.25; peak_bf is then worse than sum_bf on burst_v2, 7.41 vs 5.96 s,
  and only 8% better on burst_v3, 6.42 vs 7.00 s). Reading, not checked: with every short planned to grow 2,048 tokens the planner
  can no longer see that the shorts finish within a second, so it plans like the summed fit. On the two staggered workloads it
  keeps almost all of its gain (peak_bf / AC 0.056 on stag_v2_r3, 0.079 on stag_v3_r5.5): there the oracle sum_bf is already as good
  (1.65 and 2.13 s), so the gain is backfill (not blocking the shorts behind a long head), not the length. `sum_bf` degrades most
  on those files (stag_v3_r5.5 2.13 -> 22.3 s, stag_v2_r3 1.65 -> 7.25 s).
- **tok/s does not separate anything** (within 0.6% of the oracle row in every cell): the trace sets it. Long E2E moves the other way
  for the strict summed AC on `burst_v3`: 64.9 s -> 58.6 (`_mtx2`) -> 53.8 s (`_mt2048`), while its mean admitted falls 33.4 -> 24.0
  and its peak KV / pool 0.91 -> 0.68: fewer requests decode together, so each long decodes faster, at the shorts' expense.
- On the GPU `burst_v3` is bound by the 64-row decode cap as much as by KV (Phase 1c finding 2), so its variants are expected
  to be flat there; use `burst_v2` and the two staggered files for the separation.
- `main` ignores `max_tokens` for admission, so its ratio variant / base is the control: it should be 1 within noise.

Run (WORKLOADS can mix bases and variants in one job; run the oracle base under the same arms, `arms_v4.txt` = `main`, `ac`,
`ac_park`, `peak_bf_pc_park`), then the cost of not knowing and the arm ratios on the variants:

```bash
bash experiments/mstar/make_bagel_workloads_v4.sh
REPS=3 WORKLOADS="bagel_burst_v2 bagel_burst_v2_mtx2 bagel_burst_v2_mt2048 bagel_stag_v2_r3 bagel_stag_v2_r3_mtx2 bagel_stag_v2_r3_mt2048" \
  SBATCH_OPTS="-p team1" ./experiments/mstar/submit_matrix.sh -f experiments/mstar/arms_v4.txt
uv run --no-project --with numpy --with pandas --with matplotlib python analysis/mstar_admission.py <result dirs> --out <out>
uv run --no-project --with numpy --with pandas --with matplotlib python analysis/mstar_declared.py <out>/summary.json --md <out>/declared.md
uv run --no-project --with numpy --with pandas --with matplotlib python analysis/mstar_step_gains.py <out>/summary.json \
  ac/main peak_bf_pc_park/ac peak_bf_pc_park/main ac_park/ac --workloads '*_mtx2' '*_mt2048' --md <out>/step_gains_v4.md
```

`mstar_admission.py` reads the client's new `declared_max_tokens` list and `n_overrun_outputs` from bench.json when present (absent
in older runs: declared = intended). New per-case scalars `decl_ratio_mean` and `decl_kv_ratio` (over the completed requests), an
`OVERRUN` flag (a request generated MORE than its intended length; TRUNCATED, fewer, is unchanged), and all-request metrics (E2E
mean / p50 / p99, TTFT p50 / p99 over every completed request, whatever its class; `mid` requests of the public-trace workloads
`pub_mooncake_conv_*` / `pub_azure_conv_*` count there and get their own columns when present). `mstar_declared.py` writes
metric(variant) / metric(base) per arm with the same "≈ when the min-max rep ranges overlap" rule as `mstar_step_gains.py`.
`WORKLOAD_ORDER` is pattern-based: the v1-v3 files first, then the v4 variants (base order, `_mtx2` before `_mt2048`), then the
public traces (each base followed by its variants), then anything else.

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

## Public traces (Mooncake conversation, Azure conversation 2023)

Two public production traces, replayed through the same harness as the synthetic workloads. Raw data is downloaded, never committed;
the converted `data/workloads/pub_*` files are derived and committed (184 MB, 33 MB gzipped, largest file 16.3 MB).

### Provenance

`bash experiments/mstar/fetch_public_traces.sh` downloads both files into `data/public_traces/` (in `.gitignore`), pinned to the
commit that added each file, and checks the SHA-256 (the same bytes as the `main` / `master` branch URLs of the brief). The files are only parsed.

| Trace | File | Rows, span | SHA-256 | License |
|---|---|---|---|---|
| Mooncake (Kimi) FAST'25 conversation, `kvcache-ai/Mooncake` `FAST25-release/traces/conversation_trace.jsonl` @ `3cca71d` (2025-02-21) | `mooncake_conversation.jsonl` | 12,031 requests, 3,537 s | `b8cbb061a85206d729d91cdc2981f43c9e0d99209dce588d3af5f7934408b9df` | Mooncake repository: Apache-2.0 (`LICENSE-APACHE`); `FAST25-release/` states no separate data license |
| Azure LLM inference trace 2023, conversation, `Azure/AzurePublicDataset` `data/AzureLLMInferenceTrace_conv.csv` @ `7909210` (2024-03-24); description `AzureLLMInferenceDataset2023.md` | `azure_conv_2023.csv` | 19,366 requests, 3,502 s | `2f1e5b666d4e3055fdbba98598ce2ec307767b9064e03e2fa46676dbcc7d0bf8` | CC-BY 4.0 (repository `LICENSE`; the dataset page: "CC-BY Attribution License"). Cite Patel et al., "Splitwise", ISCA 2024 |

Mooncake fields: `timestamp` in ms, `input_length`, `output_length`, `hash_ids`. **The block size is 512 tokens** (`FAST25-release/README.md`:
"Remapped prefix block hashes. The block size is 512 tokens. Identical hash IDs indicate reusable prefix KV cache blocks"), checked on the data:
every row has `ceil(input_length / 512)` ids (the last block is partial), 288,500 id references over 182,790 distinct ids (36.6% repeats), the
output is at most 2,000. Azure fields: `TIMESTAMP` (date and time with 7 fraction digits), `ContextTokens`, `GeneratedTokens` (no prefix information).
Citation for Mooncake: Qin et al., "Mooncake: Trading More Storage for Less Computation", USENIX FAST 2025.

### Conversion (`convert_public_trace.py`, `--help` has the full rules)

```bash
python3 experiments/mstar/convert_public_trace.py mooncake --start 1350 --span 600 --rate 0.5 --out data/workloads/pub_mooncake_conv_r0.5.jsonl
bash experiments/mstar/make_public_workloads.sh      # all files below: convert, kv_report.py, variants, meta
python3 experiments/mstar/public_trace_report.py data/workloads/pub_*_r?*.jsonl   # characterisation table (also --sim, --pick-window)
```

- **Window and rate.** Original time `t` is seconds since the first request of the source file. The window is `[start, start + D)` of the ORIGINAL trace
  and a request in it gets the output timestamp `(t - start) / r`: **r multiplies the request rate** (r = 0.5: half the rate, twice the span). With
  `--span S` the window is `D = S * r` original seconds, so every output spans S = 600 s (the last request arrives 0-6 s before 600 s); with
  `--duration D` the window is D seconds as given. Only arrival times are scaled; lengths and prefixes are untouched.
- **Clamps** (BAGEL limits; counted per file in the `.meta.json` and printed): `output_length <= 2000`, and `input_length <= 32768 - 20 - output_length`
  (so `input + 20 + output <= 32768`). No Mooncake output exceeds 2,000, so only the input clamp acts: the Mooncake windows lose 37-139 rows (7%) to it, 1.0-3.3M
  tokens in all (those inputs reach 126k tokens; clamped they carry 32,748 minus their output). Azure needs no clamp (input max 14,050, output max 1,000).
  Azure has 22-76 rows per file with `input_length <= 20`; the replay client pads those to a 1-token user prompt (server-side 21 tokens) and counts them
  in `n_clamped_inputs` (the last column of the characterisation table).
- **hash_ids** (16 tokens per id). Mooncake: block id `b` becomes the 32 ids `b * 32 + j + offset` (j = 0..31), truncated to `ceil(clamped input / 16)`; rows sharing their
  first k blocks share their first 32k ids. Azure has no prefix information, so every request gets its own ids (`offset + 10^8 + 2048 * source row + j`): no sharing, and
  `declare_max_tokens.py --id-offset` can separate a variant from its base (rows without `hash_ids` would build the same tokens from the row index in every file). `--id-offset`
  defaults to 3,000,000,000; Mooncake ids end near 3.003e9, Azure ids start at 3.1e9, the variants add 1e9 (`_mtx2`) and 2e9 (`_mt2048`):
  the six ranges are disjoint (checked), and all are far above the small ids (< 1e6) of the synthetic traces.
- **cls**, per output file on the final KV `input + 20 + output`: `short` if <= the file's p50, `long` if >= its p90, else `mid` (50% / 40% / 10% of the rows).
  The simulator's "long" mask is `cls == long`; `public_trace_report.py --sim` calls it with `cls != short` to get the short class, and with an empty mask for all requests.
- Rows are sorted by timestamp (stable) and validated like `load_trace` in `benchmark/trace_replay.py` (timestamp >= 0, input >= 1, output >= 1; an invalid source row is an error);
  all 24 files were also loaded with the harness's own `load_trace` (rows, `hash_ids` long enough for `ceil((input - 20) / 16)` blocks).
- The `.meta.json` keeps `make_workload.py`'s keys plus `source` (file, SHA-256), `window`, `clamps`, `cls_rule` (the p50 / p90 values), `hash_ids` (in-file prefix reuse) and
  `kv` (from `kv_report.py --pool 262144 --overhead 20 --max-seq-len 32768`, ITL 15 then 20 ms: the file keeps the 20 ms value). A variant's `kv` is its base's (it counts generated
  tokens; arrivals and lengths are identical); its `declared` block has the declared KV.

### Window choice

Both traces drift over their hour, so the first minutes are not typical (Mooncake: fewer, longer requests at the start, 2.7 req/s and 11.4k mean input in minute 0 against ~3.5-4 req/s and ~9-10k mean input
from minute 25 on; Azure: 3.2 req/s in minute 0, 7-8 req/s with p50 input ~420 in minutes 27-36). A window is scored by the RMS of the relative deviations of eight statistics from the whole-trace value (request rate,
input mean / p50 / p90, output mean / p50, final-KV mean, final-KV demand in tokens/s, all after the clamps), and the windows of one source are nested around one centre C
(`--pick-window`; window `[C - 300 r, C + 300 r)` for r in the source's rates), so the files of a source differ mostly in rate. Mooncake C = **1500 s** (mean score 0.014 over r = 0.5 / 0.75 / 1; the best of
294 centres on a 10-s grid, tied with 1530 and 1620; the r = 0.25 control is added at the same centre), Azure C = **2550 s** (mean score 0.081 over r = 1 / 1.25 / 2 / 3, best of 171 centres). Statistics of the chosen
windows beside the whole trace (percentile rank among all windows of that length on a 10-s grid in brackets):

| Source | r | window (orig. s) | rate req/s | input mean | input p50 | input p90 | output mean | output p50 | final KV mean | KV demand tok/s | score |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Mooncake | whole | 0-3,537 | 3.40 | 10,296 | 6,909 | 27,367 | 342.6 | 350 | 10,658 | 36,253 | 0 |
| | 0.25 | [1425, 1575) | 3.49 (p51) | 9,612 (p15) | 5,851 (p3) | 26,241 (p37) | 344.3 (p61) | 348 (p40) | 9,977 (p15) | 34,852 (p23) | 0.067 |
| | 0.5 | [1350, 1650) | 3.45 (p46) | 10,361 (p60) | 6,811 (p54) | 28,256 (p58) | 344.3 (p54) | 354 (p63) | 10,725 (p60) | 37,038 (p69) | 0.016 |
| | 0.75 | [1275, 1725) | 3.41 (p37) | 10,274 (p57) | 6,895 (p56) | 28,138 (p58) | 339.1 (p28) | 347 (p32) | 10,633 (p57) | 36,272 (p50) | 0.011 |
| | 1 | [1200, 1800) | 3.44 (p42) | 10,344 (p61) | 7,113 (p65) | 27,936 (p59) | 340.8 (p33) | 350 (p38) | 10,705 (p61) | 36,771 (p67) | 0.014 |
| Azure conv | whole | 0-3,502 | 5.53 | 1,155 | 1,020 | 2,735 | 211.1 | 129 | 1,386 | 7,664 | 0 |
| | 1 | [2250, 2850) | 5.55 (p50) | 1,069 (p24) | 1,009 (p37) | 2,261 (p25) | 208.3 (p44) | 126 (p44) | 1,298 (p22) | 7,202 (p35) | 0.074 |
| | 1.25 | [2175, 2925) | 5.47 (p45) | 1,038 (p15) | 1,001 (p39) | 2,179 (p18) | 206.3 (p46) | 127 (p47) | 1,265 (p16) | 6,922 (p21) | 0.093 |
| | 2 | [1950, 3150) | 5.57 (p33) | 1,018 (p10) | 984 (p44) | 2,144 (p15) | 202.8 (p59) | 125 (p62) | 1,240 (p6) | 6,911 (p15) | 0.103 |
| | 3 | [1650, 3450) | 5.72 (p5) | 1,101 (p3) | 986 (p51) | 2,584 (p2) | 194.8 (p77) | 118 (p89) | 1,315 (p3) | 7,522 (p3) | 0.054 |

Mooncake is stationary enough that every window is within 2% (RMS) of the whole trace (r = 0.25: a 150-s window, noisier). Azure is not (rate 4.1-7.6 req/s over 10-min windows, a code-like phase in the
middle): the windows have the whole trace's request rate (5.5-5.7 req/s, p33-p50 except r = 3) and the median output length, but their inputs are 5-12% shorter than the whole trace (the whole-trace
p90 is raised by the middle phase), and the r = 3 window (half the trace) has little room to move.

### Candidate files

`data/workloads/pub_mooncake_conv_r{0.25,0.5,0.75,1}` and `pub_azure_conv_r{1,1.25,2,3}`, each with `_mtx2` and `_mt2048` (below) and a `.meta.json`. The three brief rates are
there; r = 0.25 (Mooncake, the no-pressure control: 0.5 already presses on the pool) and r = 1.25 (Azure, the highest rate with decode at the 70% line) were added after the simulator runs.
Demand is over the 600-s span: decode = output tokens / s against ~2,000 tok/s (batch 64); prefill = (input + 20) tokens / s against the assumed 20-40k (the % is of 30k, the
simulator's value). Fluid peak is `kv_report.py` (every request admitted on arrival, a fixed ITL, no queueing and no prefill time): 15 ms / 20 ms. Prefix reuse is inside the file (tokens whose leading
ids an earlier row already has, in 128-token pages: the first row of a prefix always misses, and the server's prefix cache would also have been warm from earlier traffic).

| File | Requests | req/s | Input p50 / p90 / max | Output p50 / p90 / max | Final KV mean / p50 / p90 / max | Prefix reuse | Clamped rows out / in; input <= 20 | Decode tok/s (% of 2k) | Prefill tok/s (% of 30k) | Fluid peak / P (15 / 20 ms) | Total final KV / P |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `pub_mooncake_conv_r0.25` | 524 | 0.87 | 5,850 / 26,241 / 32,747 | 348 / 619 / 2,000 | 9,977 / 6,254 / 26,732 / 32,768 | 11.6% | 0 / 37; 0 | 301 (15%) | 8,412 (28%) | 0.80 / 0.81 | 19.9 |
| `pub_mooncake_conv_r0.5` | 1,036 | 1.73 | 6,810 / 28,256 / 32,747 | 354 / 611 / 2,000 | 10,725 / 7,286 / 28,695 / 32,768 | 21.6% | 0 / 74; 0 | 595 (30%) | 17,924 (60%) | 1.50 / 1.77 | 42.4 |
| `pub_mooncake_conv_r0.75` | 1,535 | 2.56 | 6,895 / 28,138 / 32,747 | 347 / 610 / 2,000 | 10,633 / 7,305 / 28,331 / 32,768 | 25.3% | 0 / 110; 0 | 867 (43%) | 26,336 (88%) | 1.94 / 2.03 | 62.3 |
| `pub_mooncake_conv_r1` | 2,061 | 3.44 | 7,113 / 27,936 / 32,747 | 350 / 609 / 2,000 | 10,705 / 7,502 / 28,147 / 32,768 | 28.4% | 0 / 139; 0 | 1,171 (59%) | 35,600 (119%) | 2.03 / 2.37 | 84.2 |
| `pub_azure_conv_r1` | 3,330 | 5.55 | 1,009 / 2,260 / 7,219 | 126 / 420 / 1,000 | 1,298 / 1,407 / 2,377 / 7,300 | 0 | 0 / 0; 22 | 1,156 (58%) | 6,046 (20%) | 0.18 / 0.23 | 16.5 |
| `pub_azure_conv_r1.25` | 4,105 | 6.84 | 1,001 / 2,179 / 7,219 | 127 / 420 / 1,000 | 1,265 / 1,385 / 2,286 / 7,300 | 0 | 0 / 0; 34 | 1,411 (71%) | 7,241 (24%) | 0.22 / 0.30 | 19.8 |
| `pub_azure_conv_r2` | 6,686 | 11.14 | 984 / 2,144 / 7,219 | 125 / 419 / 1,000 | 1,240 / 1,342 / 2,260 / 7,300 | 0 | 0 / 0; 53 | 2,260 (113%) | 11,562 (39%) | 0.32 / 0.40 | 31.6 |
| `pub_azure_conv_r3` | 10,293 | 17.16 | 986 / 2,584 / 7,219 | 118 / 417 / 1,000 | 1,315 / 1,331 / 2,710 / 7,300 | 0 | 0 / 0; 76 | 3,342 (167%) | 19,224 (64%) | 0.43 / 0.52 | 51.7 |

(Input and output are after the clamps. The Mooncake input max 32,747 is the clamp. Mooncake's whole-trace block reuse is 37%; a 150-600 s window sees 12-28% because the history before the window is missing, so a
replay that starts cold has the same limit; a warm server that served earlier traffic would hit more.)

Over-declared variants (`declare_max_tokens.py`, which this section did not change; the server declares `max_tokens` for admission and stops each request after `output_length` tokens):
`<name>_mtx2` (`--mode x2 --id-offset 1000000000`: `max_tokens = min(2048, ceil(2 * output))`) and `<name>_mt2048` (`--mode fixed --id-offset 2000000000`: 2,048 for every row).
They have their base's arrivals, lengths and cls; the prompts are the same tokens shifted to other ids (no prefix-cache hits across a base and its variants on one server).
Declared KV = sum of `input + 20 + max_tokens`; ratio to the generated KV (`input + 20 + output`):

| base | MB base / mtx2 / mt2048 | declared / generated KV, mtx2 | mt2048 | mean max_tokens / mean output (mtx2, mt2048) |
|---|---|---|---|---|
| pub_mooncake_conv_r0.25 | 3.5 / 3.8 / 3.9 | 1.034 | 1.171 | 1.98, 5.95 |
| pub_mooncake_conv_r0.5 | 7.5 / 8.2 / 8.2 | 1.031 | 1.159 | 1.97, 5.95 |
| pub_mooncake_conv_r0.75 | 11.0 / 12.0 / 12.0 | 1.031 | 1.161 | 1.97, 6.04 |
| pub_mooncake_conv_r1 | 14.9 / 16.3 / 16.3 | 1.031 | 1.159 | 1.97, 6.01 |
| pub_azure_conv_r1 | 2.8 / 3.1 / 3.2 | 1.161 | 2.418 | 2.00, 9.83 |
| pub_azure_conv_r1.25 | 3.4 / 3.8 / 3.8 | 1.163 | 2.456 | 2.00, 9.93 |
| pub_azure_conv_r2 | 5.4 / 6.1 / 6.1 | 1.163 | 2.488 | 2.00, 10.10 |
| pub_azure_conv_r3 | 9.0 / 10.0 / 10.0 | 1.148 | 2.409 | 2.00, 10.51 |

Mooncake's requests are prompt-dominated (a 10.7k final KV is ~96% prompt), so even a 2,048 bound on every request over-books by only 16%; Azure's are short, and `mt2048` over-books 2.4-2.5x.

### Simulator

`analysis/admit_sim_bagel.py` through `public_trace_report.py --sim` (oracle files only: the simulator in this tree reads `max_tokens` as the output length, so the `_mtx2` / `_mt2048` variants are not
simulated; policies as in "Workloads v2"; pool 2,047 pages x 128; prefill 30k tok/s, 8,192 tokens per step; decode step cost from the smoke ITLs, decode in chunks of 64). "short" is `cls == short` (final KV <= the
file's p50), "all" is every request; seconds, mean (p99); out tok/s over the makespan; peak KV / P is the simulated in-flight KV (`open`: the demand a pool would have to carry, with decode slowed by the batch):

| file | policy | unfin / >600 s | short E2E mean (p99) | all E2E mean (p99) | out tok/s | makespan s | peak KV / P |
|---|---|---|---|---|---|---|---|
| pub_mooncake_conv_r0.25 | sum_fifo | 0 / 0 | 6.14 (14.6) | 6.62 (17.8) | 298 | 605 | 0.80 |
| pub_mooncake_conv_r0.25 | sum_bf | 0 / 0 | 6.14 (14.6) | 6.62 (17.8) | 298 | 605 | 0.80 |
| pub_mooncake_conv_r0.25 | peak_fifo | 0 / 0 | 6.14 (14.6) | 6.62 (17.8) | 298 | 605 | 0.80 |
| pub_mooncake_conv_r0.25 | peak_bf | 0 / 0 | 6.14 (14.6) | 6.62 (17.8) | 298 | 605 | 0.80 |
| pub_mooncake_conv_r0.25 | open | 0 / 0 | 6.14 (14.6) | 6.62 (17.8) | 298 | 605 | 0.80 |
| pub_mooncake_conv_r0.5 | sum_fifo | 0 / 0 | 33.83 (63.4) | 34.29 (65.0) | 556 | 642 | 0.99 |
| pub_mooncake_conv_r0.5 | sum_bf | 0 / 0 | 16.16 (47.8) | 24.95 (90.7) | 553 | 645 | 0.99 |
| pub_mooncake_conv_r0.5 | peak_fifo | 0 / 0 | 33.75 (63.1) | 34.26 (66.9) | 556 | 641 | 1.00 |
| pub_mooncake_conv_r0.5 | peak_bf | 0 / 0 | 16.21 (44.3) | 24.93 (96.4) | 554 | 643 | 1.00 |
| pub_mooncake_conv_r0.5 | open | 0 / 0 | 46.07 (137.1) | 47.98 (150.6) | 565 | 631 | 6.16 |
| pub_mooncake_conv_r0.75 | sum_fifo | 0 / 0 | 183.39 (342.1) | 182.00 (341.8) | 555 | 938 | 0.99 |
| pub_mooncake_conv_r0.75 | sum_bf | 0 / 0 | 22.82 (69.9) | 98.32 (518.9) | 550 | 947 | 0.99 |
| pub_mooncake_conv_r0.75 | peak_fifo | 0 / 0 | 183.27 (342.5) | 181.89 (341.3) | 555 | 938 | 1.00 |
| pub_mooncake_conv_r0.75 | peak_bf | 0 / 0 | 24.42 (73.7) | 100.86 (537.2) | 550 | 946 | 1.00 |
| pub_mooncake_conv_r0.75 | open | 0 / 1 | 246.56 (481.8) | 248.78 (514.0) | 570 | 913 | 11.24 |
| pub_mooncake_conv_r1 | sum_fifo | 0 / 223 | 351.45 (666.9) | 348.02 (666.5) | 556 | 1263 | 0.99 |
| pub_mooncake_conv_r1 | sum_bf | 0 / 267 | 48.84 (224.1) | 202.44 (777.9) | 551 | 1276 | 1.00 |
| pub_mooncake_conv_r1 | peak_fifo | 0 / 221 | 351.23 (666.6) | 347.81 (666.3) | 556 | 1262 | 1.00 |
| pub_mooncake_conv_r1 | peak_bf | 0 / 284 | 51.27 (198.6) | 203.48 (790.0) | 551 | 1275 | 1.00 |
| pub_mooncake_conv_r1 | open | 0 / 448 | 414.09 (738.2) | 414.15 (738.6) | 572 | 1229 | 11.39 |
| pub_azure_conv_r1 | sum_fifo | 0 / 0 | 1.81 (7.7) | 3.10 (13.6) | 1150 | 603 | 0.30 |
| pub_azure_conv_r1 | sum_bf | 0 / 0 | 1.81 (7.7) | 3.10 (13.6) | 1150 | 603 | 0.30 |
| pub_azure_conv_r1 | peak_fifo | 0 / 0 | 1.81 (7.7) | 3.10 (13.6) | 1150 | 603 | 0.30 |
| pub_azure_conv_r1 | peak_bf | 0 / 0 | 1.81 (7.7) | 3.10 (13.6) | 1150 | 603 | 0.30 |
| pub_azure_conv_r1 | open | 0 / 0 | 1.81 (7.7) | 3.10 (13.6) | 1150 | 603 | 0.30 |
| pub_azure_conv_r1.25 | sum_fifo | 0 / 0 | 7.43 (43.9) | 13.38 (57.6) | 1387 | 610 | 0.91 |
| pub_azure_conv_r1.25 | sum_bf | 0 / 0 | 7.30 (44.4) | 13.40 (58.2) | 1387 | 610 | 0.91 |
| pub_azure_conv_r1.25 | peak_fifo | 0 / 0 | 7.46 (44.8) | 13.55 (61.1) | 1387 | 610 | 1.00 |
| pub_azure_conv_r1.25 | peak_bf | 0 / 0 | 7.43 (44.9) | 13.57 (61.8) | 1387 | 610 | 1.00 |
| pub_azure_conv_r1.25 | open | 0 / 0 | 7.52 (44.9) | 13.65 (64.1) | 1387 | 610 | 1.23 |
| pub_azure_conv_r2 | sum_fifo | 0 / 0 | 145.24 (341.7) | 173.71 (363.5) | 1429 | 949 | 0.92 |
| pub_azure_conv_r2 | sum_bf | 0 / 0 | 106.05 (255.4) | 160.17 (367.2) | 1427 | 950 | 0.92 |
| pub_azure_conv_r2 | peak_fifo | 0 / 0 | 144.37 (341.7) | 174.31 (365.0) | 1429 | 949 | 1.00 |
| pub_azure_conv_r2 | peak_bf | 0 / 0 | 108.12 (307.3) | 161.97 (367.6) | 1430 | 948 | 1.00 |
| pub_azure_conv_r2 | open | 0 / 8 | 142.38 (360.2) | 221.07 (505.0) | 1439 | 942 | 6.49 |
| pub_azure_conv_r3 | sum_fifo | 0 / 2666 | 355.41 (827.2) | 405.61 (847.7) | 1388 | 1444 | 0.94 |
| pub_azure_conv_r3 | sum_bf | 0 / 2238 | 290.55 (642.3) | 383.62 (852.7) | 1385 | 1448 | 0.94 |
| pub_azure_conv_r3 | peak_fifo | 0 / 2688 | 356.27 (828.9) | 407.87 (851.1) | 1386 | 1447 | 1.00 |
| pub_azure_conv_r3 | peak_bf | 0 / 2238 | 245.66 (682.8) | 370.92 (850.2) | 1388 | 1445 | 1.00 |
| pub_azure_conv_r3 | open | 0 / 3425 | 354.28 (799.0) | 455.87 (928.5) | 1397 | 1435 | 7.91 |

Reading it:
- **Mooncake r = 0.25** is the no-pressure control: the five policies give identical results (short E2E 6.1 s, all 6.6 s; nothing queues for KV); `open` peaks at 0.80 P.
- **Mooncake r = 0.5** is the highest rate the simulator finishes in time (makespan 645 s for a 594-s trace) and the pool binds: `open` would hold 6.2 P, the sum / peak policies sit at 0.99-1.00 P, and
  strict FIFO makes the shorts wait twice as long as backfill (short E2E 33.8 vs 16.2 s). At r = 0.5 peak and sum are within 1% of each other (outputs are only ~3% of a request's final KV, so the summed reservation over-books by little and an
  oracle does not buy peak much; the over-declared variants are where sum and peak can separate).
  The prefill load is the limit: 60% of 30k tok/s (45-90% over 40-20k) on top of decode, which is why r = 0.75 (88%) takes 938-947 s and r = 1 (119%) 1,263-1,276 s, with 221-284 requests past the 600-s server timeout (448 for `open`).
- **Azure r = 1** is the no-pressure control: identical results for all policies (short E2E 1.8 s, all 3.1 s), `open` peaks at 0.30 P.
  **r = 1.25** touches the pool (`open` 1.23 P; the sum / peak policies 0.91-1.00 P) with decode at 71% of 2k, but the policies differ by under 2% in short E2E (7.3-7.5 s): a mild regime in oracle mode.
  **r = 2 and r = 3 are decode-overloaded** (113% and 167% of 2k tok/s; the simulated decode saturates near 1,400 tok/s once prefill steps are interleaved with it, so the makespans are 949 and 1,444 s);
  the pool binds there (0.92-1.00 P) only because the backlog keeps ~190 requests admitted.
  Azure's KV pool is large against its requests (a 1.3k mean final KV: 200 requests fit): the pool binds only when decode is already at 70-85% load, or through over-declared reservations.

Bracketing runs, not committed (the same converter, centre and simulator; `MC_RATES="0.35 0.6" AZ_RATES="1.5" bash experiments/mstar/make_public_workloads.sh` regenerates them):

| file | policy | unfin / >600 s | short E2E mean (p99) | all E2E mean (p99) | out tok/s | makespan s | peak KV / P |
|---|---|---|---|---|---|---|---|
| pub_mooncake_conv_r0.35 | sum_fifo | 0 / 0 | 7.37 (22.2) | 7.91 (22.4) | 398 | 600 | 0.98 |
| pub_mooncake_conv_r0.35 | sum_bf | 0 / 0 | 7.36 (22.3) | 7.91 (22.5) | 398 | 600 | 0.98 |
| pub_mooncake_conv_r0.35 | peak_bf | 0 / 0 | 7.37 (22.3) | 7.92 (22.5) | 398 | 600 | 0.99 |
| pub_mooncake_conv_r0.35 | open | 0 / 0 | 7.41 (23.2) | 7.96 (22.6) | 398 | 600 | 1.23 |
| pub_mooncake_conv_r0.6 | sum_fifo | 0 / 0 | 99.81 (181.0) | 99.66 (181.5) | 558 | 766 | 0.99 |
| pub_mooncake_conv_r0.6 | sum_bf | 0 / 0 | 20.69 (52.7) | 59.25 (317.1) | 554 | 772 | 0.99 |
| pub_mooncake_conv_r0.6 | peak_bf | 0 / 0 | 18.49 (46.4) | 55.91 (362.3) | 555 | 771 | 1.00 |
| pub_mooncake_conv_r0.6 | open | 0 / 1 | 164.87 (391.5) | 166.70 (410.0) | 574 | 744 | 11.27 |
| pub_azure_conv_r1.5 | sum_fifo | 0 / 0 | 47.51 (118.6) | 64.56 (144.1) | 1427 | 715 | 0.91 |
| pub_azure_conv_r1.5 | sum_bf | 0 / 0 | 30.17 (106.2) | 58.58 (150.0) | 1427 | 715 | 0.92 |
| pub_azure_conv_r1.5 | peak_bf | 0 / 0 | 30.54 (112.0) | 59.78 (156.4) | 1427 | 715 | 1.00 |
| pub_azure_conv_r1.5 | open | 0 / 0 | 46.68 (176.8) | 88.42 (310.2) | 1436 | 710 | 5.20 |

**Prefix cache.** The simulator has no prefix cache: it prefills every prompt token and holds KV for every one, so for Mooncake it overstates both the prefill load and the KV pool use
(Azure rows are unique prompts: no bias). `public_trace_report.py --sim --prefix-hits` is the optimistic bound: every prompt is cut by the tokens that an earlier request in the file already holds (whole 128-token pages,
no eviction, no wait for the earlier prefill). Mooncake, short E2E mean (p99) / all E2E mean, seconds, and makespan, plain simulator -> with every hit:

| file | policy | short E2E, plain -> hits | all E2E, plain -> hits | makespan s, plain -> hits | peak KV / P, plain -> hits |
|---|---|---|---|---|---|
| pub_mooncake_conv_r0.25 | sum_fifo | 6.14 (14.6) -> 5.70 (11.8) | 6.62 (17.8) -> 6.14 (16.9) | 605 -> 604 | 0.80 -> 0.77 |
| pub_mooncake_conv_r0.25 | sum_bf | 6.14 (14.6) -> 5.70 (11.8) | 6.62 (17.8) -> 6.14 (16.9) | 605 -> 604 | 0.80 -> 0.77 |
| pub_mooncake_conv_r0.25 | peak_bf | 6.14 (14.6) -> 5.70 (11.8) | 6.62 (17.8) -> 6.14 (16.9) | 605 -> 604 | 0.80 -> 0.77 |
| pub_mooncake_conv_r0.25 | open | 6.14 (14.6) -> 5.70 (11.8) | 6.62 (17.8) -> 6.14 (16.9) | 605 -> 604 | 0.80 -> 0.77 |
| pub_mooncake_conv_r0.5 | sum_fifo | 33.83 (63.4) -> 14.31 (47.5) | 34.29 (65.0) -> 15.22 (47.7) | 642 -> 609 | 0.99 -> 0.99 |
| pub_mooncake_conv_r0.5 | sum_bf | 16.16 (47.8) -> 10.85 (34.9) | 24.95 (90.7) -> 13.18 (55.6) | 645 -> 609 | 0.99 -> 0.99 |
| pub_mooncake_conv_r0.5 | peak_bf | 16.21 (44.3) -> 10.86 (35.8) | 24.93 (96.4) -> 13.18 (63.2) | 643 -> 609 | 1.00 -> 1.00 |
| pub_mooncake_conv_r0.5 | open | 46.07 (137.1) -> 17.02 (82.3) | 47.98 (150.6) -> 18.59 (87.1) | 631 -> 610 | 6.16 -> 5.00 |
| pub_mooncake_conv_r0.75 | sum_fifo | 183.39 (342.1) -> 115.11 (186.7) | 182.00 (341.8) -> 114.62 (187.0) | 938 -> 773 | 0.99 -> 0.99 |
| pub_mooncake_conv_r0.75 | sum_bf | 22.82 (69.9) -> 18.48 (52.0) | 98.32 (518.9) -> 51.72 (440.8) | 947 -> 777 | 0.99 -> 0.99 |
| pub_mooncake_conv_r0.75 | peak_bf | 24.42 (73.7) -> 18.94 (57.2) | 100.86 (537.2) -> 55.05 (400.1) | 946 -> 775 | 1.00 -> 1.00 |
| pub_mooncake_conv_r0.75 | open | 246.56 (481.8) -> 176.18 (432.9) | 248.78 (514.0) -> 179.32 (438.1) | 913 -> 754 | 11.24 -> 11.45 |
| pub_mooncake_conv_r1 | sum_fifo | 351.45 (666.9) -> 243.22 (418.4) | 348.02 (666.5) -> 241.34 (418.7) | 1263 -> 1010 | 0.99 -> 0.99 |
| pub_mooncake_conv_r1 | sum_bf | 48.84 (224.1) -> 29.66 (105.1) | 202.44 (777.9) -> 109.13 (678.0) | 1276 -> 1014 | 1.00 -> 0.99 |
| pub_mooncake_conv_r1 | peak_bf | 51.27 (198.6) -> 28.19 (94.2) | 203.48 (790.0) -> 109.79 (670.2) | 1275 -> 1015 | 1.00 -> 1.00 |
| pub_mooncake_conv_r1 | open | 414.09 (738.2) -> 307.05 (573.1) | 414.15 (738.6) -> 308.78 (598.3) | 1229 -> 985 | 11.39 -> 11.88 |

The truth lies between the two columns. The hits cut the prefill load (r = 0.5: 17.9k -> 14.2k tok/s) and the KV of the shared pages, so the same r pressures less: at r = 0.5 FIFO's short E2E falls from 33.8 to 14.3 s and the backfill gap
shrinks (16.2 -> 10.9 s); r = 0.75 still takes 773 s (every hit assumed). The in-file reuse is itself a floor, because the replay window starts cold.

### Recommendation

| Source | KV binding, GPU not hopelessly overloaded | No-pressure control | Not usable as a single-GPU load |
|---|---|---|---|
| Mooncake conversation | **`pub_mooncake_conv_r0.5`**: prefill 60% of 30k tok/s (decode 30%), the pool binds (sum / peak 0.99-1.00 P, `open` 6.2 P), the run finishes by 645 s; strict FIFO vs backfill separates (short E2E 33.8 vs 16.2 s) | **`pub_mooncake_conv_r0.25`**: all policies identical, `open` peak 0.80 P (a 0.35 bracket run is already at the edge: `open` 1.23 P, policies within 1%) | r = 0.75 and 1: prefill 88% and 119% of 30k tok/s, makespan 1.6x and 2.1x the trace |
| Azure conversation 2023 | **`pub_azure_conv_r1.25`** (decode 71%, `open` 1.23 P): the pool is touched but the oracle policies are within 2%; expect the effect through `_mt2048` (reserves 2.5x the generated KV) rather than through the oracle file | **`pub_azure_conv_r1`**: all policies identical, `open` 0.30 P, decode 58% | r = 2 and 3: decode 113% and 167% of capacity |

For Azure no rate has both a clear KV effect and decode below 70%: the prompts are too short for a 262k-token pool; if a sharper Azure point is needed, r = 1.5 (decode 85%, sum_fifo short E2E 47.5 s against 30.2 s for
sum_bf, makespan 715 s; bracketing run above) is the next rung, at the price of a saturated decode. Mooncake is the workload with real pool pressure: its pressure comes from long prompts and prefill, not from
decode, so the KV and prefill limits arrive together (at r = 0.5 the prefill share is already 60%).

### Caveats

- The simulator's step cost, prefill 30k tok/s and 8,192-token prefill budget are assumptions (see `admit_sim_bagel.py`); for Mooncake prefill steps (0.27 s each) dominate the simulated decode ITL, so its short E2E (6-34 s) mostly reflects
  prefill queueing. Use the policy ORDER and the saturation points, not the seconds.
- `input_length` convention: `benchmark/trace_replay.py` builds a prompt whose templated server-side length is `input_length` (user tokens `input_length - 20`), while `kv_report.py`, the simulator and the clamp here count `input_length + 20`. For these
  files the clamp is therefore 20 tokens conservative and the KV accounting 20 tokens per request above what the server sees (0.2% for Mooncake, 1.5% for Azure).
- One GPU replays the trace at r times its rate: rate scaling stands in for replica scaling (a production deployment spreads this load over many GPUs), and it keeps each request's length and prefix structure.
