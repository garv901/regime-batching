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

The fluid peak is `kv_report.py --itl-ms 15` to `--itl-ms 20` (both keys are in each `.meta.json`): every request admitted on arrival,
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
