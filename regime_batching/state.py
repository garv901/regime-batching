"""Process-local shared state between telemetry, the attention dispatcher and the scheduler."""
from __future__ import annotations

# Set by the regime attention backend's metadata builder on every step (worker process).
LAST_ATTN_DECISION: dict | None = None
# Set by the cudagraph-dispatch wrapper (worker process).
LAST_BATCH_DESC: dict | None = None
# Per-request oracle cache in the worker: req_id -> (prompt_len, max_tokens)
REQ_ORACLE: dict[str, tuple[int, int]] = {}
# Features of the step currently executing (worker process).
CURRENT_FEATURES = None
# Set by RegimeScheduler on every schedule() call (engine-core process).
LAST_SCHED_DECISION: dict | None = None
