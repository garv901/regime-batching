"""vLLM general-plugin entry point (runs in API server, engine core and worker processes).

Env switches:
  REGIME_TELEMETRY_DIR=<dir>   per-step markers/telemetry (baseline and proposed)
  REGIME_ATTN=1                override FLASH_ATTN with the regime-aware split-policy subclass (C3 v1)

Oracle-aware scheduler (C4, not enabled by this plugin; pass to the server):
  --scheduler-cls regime_batching.scheduler.RegimeScheduler   (AsyncScheduler subclass; async stays on)
  REGIME_SCHED_ADMIT=peak|fcfs|sjf peak (default): gate waiting requests whose admission would push the
                                   future peak KV (oracle lengths) over capacity, with backfill; fcfs = stock;
                                   sjf = ablation, waiting ordered by oracle remaining tokens, no KV planning
  REGIME_SCHED_WATERMARK=0.02      fraction of KV tokens kept free in the capacity check
  REGIME_SCHED_SHAPE=off|cg_cap    cg_cap: cap step tokens at the CUDA-graph capture size while a long-KV
                                   decode runs and the prefill backlog is small (avoids the "X regime")
  REGIME_SHAPE_MIN_KV=4096  REGIME_SHAPE_MIN_REMAIN=256  REGIME_SHAPE_MAX_BACKLOG=16384
  REGIME_SHAPE_CAP=<n>             default: compilation_config.max_cudagraph_capture_size
  REGIME_TELEMETRY_DIR also enables <dir>/sched_<pid>.jsonl (one decision record per schedule() call)
"""
import os


def register():
    if os.environ.get("REGIME_TELEMETRY_DIR"):
        from regime_batching import telemetry
        telemetry.install()
    if os.environ.get("REGIME_ATTN") == "1":
        from regime_batching.attn import register_backend
        register_backend()
