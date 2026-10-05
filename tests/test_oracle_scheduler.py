"""CPU tests for regime_batching.oracle / regime_batching.scheduler (real vLLM Scheduler code path)."""
import os
import statistics
import sys
import time
from collections import deque

import numpy as np
import pytest

VLLM_SRC = os.path.join(os.path.dirname(__file__), "..", "third_party", "vllm")
sys.path.append(os.path.abspath(VLLM_SRC))  # for vLLM's own test helpers (tests.v1.core.utils)

from regime_batching.oracle import PeakPlanner, final_len, remaining  # noqa: E402

BS = 16


# ---------------------------------------------------------------- planner vs brute force
def brute_peak(u, r, extra=None):
    """Simulate: each request grows 1 token per step for r steps (r+1 states incl. step 0), then frees."""
    reqs = list(zip(u, r)) + ([extra] if extra else [])
    if not reqs:
        return 0.0
    horizon = max(rr for _, rr in reqs)
    best = 0.0
    for k in range(horizon + 1):
        # request with remaining r is alive through step r-1 (its last token is produced at step r-1);
        # at step k it holds u+k tokens while k < r (r=0: already done), the new one is counted at k<=r-1.
        tot = sum(uu + k for uu, rr in reqs if k < rr)
        best = max(best, tot)
    return best


def test_planner_matches_brute_force():
    rng = np.random.default_rng(0)
    n_fit = 0
    for _ in range(800):
        n = int(rng.integers(0, 9))
        u = rng.integers(1, 200, n)
        r = rng.integers(1, 120, n)
        nu, nr = int(rng.integers(1, 200)), int(rng.integers(1, 120))
        p = PeakPlanner(u, r)
        bf = brute_peak(list(u), list(r))
        assert p.peak() == bf
        bf_new = brute_peak(list(u), list(r), (nu, nr))
        assert p.peak_with(nu, nr) == bf_new
        assert p.fits(nu, nr, bf_new) and (bf_new == 0 or not p.fits(nu, nr, bf_new - 1))


def test_planner_pad_and_blocks():
    p = PeakPlanner([100, 50], [10, 20], pad=BS - 1)
    q = PeakPlanner([100, 50], [10, 20])
    assert p.peak() == q.peak() + 2 * (BS - 1)
    assert q.blocks(BS) == -(-int(q.peak()) // BS)


# ---------------------------------------------------------------- scheduler helpers
@pytest.fixture(scope="module")
def U(tmp_path_factory):
    pytest.importorskip("vllm")
    import json

    import tests.v1.core.utils as u
    # offline: config-only tiny llama dir (scheduler tests never touch weights or the tokenizer)
    d = tmp_path_factory.mktemp("tiny_llama")
    (d / "config.json").write_text(json.dumps(dict(
        architectures=["LlamaForCausalLM"], model_type="llama", hidden_size=64, intermediate_size=128,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4, vocab_size=1000,
        max_position_embeddings=16384, torch_dtype="float16")))
    u.TINY = str(d)
    return u


def make(U, monkeypatch, cls, num_blocks=101, env=None, **kw):
    from regime_batching.scheduler import RegimeScheduler
    for k in list(os.environ):
        if k.startswith("REGIME_SCHED") or k.startswith("REGIME_SHAPE"):
            monkeypatch.delenv(k)
    monkeypatch.delenv("REGIME_TELEMETRY_DIR", raising=False)
    for k, v in (env or {}).items():
        monkeypatch.setenv(k, v)
    from vllm.v1.core.sched.async_scheduler import AsyncScheduler
    monkeypatch.setattr(U, "AsyncScheduler", RegimeScheduler if cls == "regime" else AsyncScheduler)
    return U.create_scheduler(model=U.TINY, skip_tokenizer_init=True, async_scheduling=True, num_blocks=num_blocks, block_size=BS, max_num_seqs=64, **kw)


def mkreqs(U, specs, prefix="r"):
    """specs: list of (prompt, max_tokens) -> requests with ignore_eos (oracle)."""
    out = []
    for i, (pl, mt) in enumerate(specs):
        (r,) = U.create_requests(1, num_tokens=pl, max_tokens=mt, ignore_eos=True, req_ids=[f"{prefix}{i}"],
                                 block_size=BS)
        out.append(r)
    return out


def mro(out):
    from vllm.v1.outputs import ModelRunnerOutput
    ids = list(out.num_scheduled_tokens)
    return ModelRunnerOutput(req_ids=ids, req_id_to_index={r: i for i, r in enumerate(ids)},
                             sampled_token_ids=[[7] for _ in ids], logprobs=None, prompt_logprobs_dict={},
                             pooler_output=[])


def run_trace(sched, reqs, depth=2, max_steps=20000):
    """Pipelined (async-scheduling style) loop; returns per-step scheduled id lists and total preemptions."""
    for r in reqs:
        sched.add_request(r)
    inflight, trace, n = deque(), [], 0
    while n < max_steps:
        out = sched.schedule()
        if out.num_scheduled_tokens:
            inflight.append(out)
            trace.append(sorted(out.num_scheduled_tokens))
        if inflight and (len(inflight) >= depth or not out.num_scheduled_tokens):
            sched.update_from_output(inflight[0], mro(inflight.popleft()))
        n += 1
        if not inflight and not out.num_scheduled_tokens and sched.get_num_unfinished_requests() == 0:
            break
    assert sched.get_num_unfinished_requests() == 0
    return trace, sum(r.num_preemptions for r in reqs)


PREEMPT_SPECS = [(32, 200)] * 12


# ---------------------------------------------------------------- (i) zero preemptions
def test_peak_has_no_preemption_where_fcfs_preempts(U, monkeypatch):
    s = make(U, monkeypatch, "regime", env={"REGIME_SCHED_ADMIT": "fcfs"})
    _, pre_fcfs = run_trace(s, mkreqs(U, PREEMPT_SPECS))
    assert pre_fcfs > 0
    s = make(U, monkeypatch, "regime", env={"REGIME_SCHED_ADMIT": "peak"})
    reqs = mkreqs(U, PREEMPT_SPECS)
    trace, pre_peak = run_trace(s, reqs)
    assert pre_peak == 0
    assert all(r.num_output_tokens == 200 for r in reqs)
    assert s.get_num_unfinished_requests() == 0


def test_peak_mixed_lengths_no_preemption(U, monkeypatch):
    rng = np.random.default_rng(1)
    specs = [(int(rng.integers(16, 200)), int(rng.integers(5, 300))) for _ in range(30)]
    s = make(U, monkeypatch, "regime", num_blocks=151, env={"REGIME_SCHED_ADMIT": "peak"})
    _, pre = run_trace(s, mkreqs(U, specs))
    s2 = make(U, monkeypatch, "regime", num_blocks=151, env={"REGIME_SCHED_ADMIT": "fcfs"})
    _, pre2 = run_trace(s2, mkreqs(U, specs))
    assert pre == 0 and pre2 >= 0


# ---------------------------------------------------------------- (ii) backfill + order restore
def test_backfill_and_queue_order(U, monkeypatch):
    s = make(U, monkeypatch, "regime")
    a, b, c, d, e = mkreqs(U, [(32, 1000), (32, 1000), (32, 20), (32, 1000), (32, 20)])
    s.add_request(a)
    out = s.schedule()
    s.update_from_output(out, mro(out))
    for r in (b, c, d, e):
        s.add_request(r)
    out = s.schedule()
    new = [n.req_id for n in out.scheduled_new_reqs]
    assert new == [c.request_id, e.request_id]  # big ones gated, small ones backfilled
    assert [q.request_id for q in s.waiting] == [b.request_id, d.request_id]  # FCFS order restored
    from regime_batching import state
    assert state.LAST_SCHED_DECISION["n_gated"] == 2
    assert state.LAST_SCHED_DECISION["gated_ids"] == [b.request_id, d.request_id]
    # stock scheduler would have admitted the big ones
    s0 = make(U, monkeypatch, "regime", env={"REGIME_SCHED_ADMIT": "fcfs"})
    a, b, c, d, e = mkreqs(U, [(32, 1000), (32, 1000), (32, 20), (32, 1000), (32, 20)])
    s0.add_request(a)
    out = s0.schedule()
    s0.update_from_output(out, mro(out))
    for r in (b, c, d, e):
        s0.add_request(r)
    assert len(s0.schedule().scheduled_new_reqs) == 4


def test_preempted_stay_in_front(U, monkeypatch):
    s = make(U, monkeypatch, "regime")
    reqs = mkreqs(U, [(32, 1000)] * 3)
    for r in reqs:
        s.add_request(r)
    out = s.schedule()  # admits 1 (second gated: 2 * 1047 > cap), third gated
    assert len(out.scheduled_new_reqs) == 1
    assert [q.request_id for q in s.waiting] == ["r1", "r2"]
    s.update_from_output(out, mro(out))
    # simulate vLLM preempting r0 later: prepended in front of waiting
    r0 = s.running[0]
    s.running.remove(r0)
    s._preempt_request(r0, time.monotonic())
    s.schedule()  # r0 (front, preempted) is re-admitted ahead of the gated r1, r2
    assert [q.request_id for q in s.waiting] == ["r1", "r2"]
    assert [q.request_id for q in s.running] == ["r0"]


# ---------------------------------------------------------------- (iii) fcfs == stock AsyncScheduler
def test_fcfs_identical_to_async_scheduler(U, monkeypatch):
    rng = np.random.default_rng(2)
    specs = [(int(rng.integers(16, 200)), int(rng.integers(5, 300))) for _ in range(30)]
    t_stock, p_stock = run_trace(make(U, monkeypatch, "async", num_blocks=101), mkreqs(U, specs))
    t_reg, p_reg = run_trace(make(U, monkeypatch, "regime", num_blocks=101, env={"REGIME_SCHED_ADMIT": "fcfs"}),
                             mkreqs(U, specs))
    assert p_stock > 0
    assert t_stock == t_reg and p_stock == p_reg


# ---------------------------------------------------------------- (iv) step shaping
def _shape_sched(U, monkeypatch, shape):
    s = make(U, monkeypatch, "regime", num_blocks=5000,
             env={"REGIME_SCHED_ADMIT": "peak", "REGIME_SCHED_SHAPE": shape, "REGIME_SHAPE_CAP": "512"}, max_num_batched_tokens=8192)
    (long,) = mkreqs(U, [(4200, 600)], prefix="long")
    s.add_request(long)
    out = s.schedule()
    s.update_from_output(out, mro(out))
    return s


def test_shape_cap(U, monkeypatch):
    s = _shape_sched(U, monkeypatch, "cg_cap")
    assert s.shape_cap == 512
    assert s.vllm_config.compilation_config.max_cudagraph_capture_size is not None  # default source
    for r in mkreqs(U, [(2000, 10)] * 3, prefix="p"):
        s.add_request(r)
    out = s.schedule()
    assert out.total_num_scheduled_tokens <= 512 and out.total_num_scheduled_tokens > 1
    from regime_batching import state
    assert state.LAST_SCHED_DECISION["shape"] is True and state.LAST_SCHED_DECISION["budget"] == 512
    assert s.max_num_scheduled_tokens == 8192  # restored
    # backlog beyond threshold -> cap removed
    s = _shape_sched(U, monkeypatch, "cg_cap")
    for r in mkreqs(U, [(2000, 10)] * 10, prefix="q"):
        s.add_request(r)
    out = s.schedule()
    assert out.total_num_scheduled_tokens > 512
    assert state.LAST_SCHED_DECISION["shape"] is False
    # shape off -> no cap
    s = _shape_sched(U, monkeypatch, "off")
    for r in mkreqs(U, [(2000, 10)] * 3, prefix="p"):
        s.add_request(r)
    assert s.schedule().total_num_scheduled_tokens > 512


# ---------------------------------------------------------------- telemetry + overhead
def test_decision_file_and_overhead(U, monkeypatch, tmp_path):
    import json
    from vllm.v1.core.sched.async_scheduler import AsyncScheduler
    inner = []
    orig = AsyncScheduler.schedule

    def timed(self, *a, **k):
        t = time.perf_counter()
        o = orig(self, *a, **k)
        inner.append(time.perf_counter() - t)
        return o

    monkeypatch.setattr(AsyncScheduler, "schedule", timed)
    s = make(U, monkeypatch, "regime", env={"REGIME_TELEMETRY_DIR": str(tmp_path)})
    outer = []
    for r in mkreqs(U, PREEMPT_SPECS):
        s.add_request(r)
    inflight = deque()
    for _ in range(400):
        n0 = len(inner)
        t = time.perf_counter()
        out = s.schedule()
        outer.append(time.perf_counter() - t)
        assert len(inner) == n0 + 1
        if out.num_scheduled_tokens:
            inflight.append(out)
        if len(inflight) >= 2 or not out.num_scheduled_tokens:
            if inflight:
                s.update_from_output(inflight[0], mro(inflight.popleft()))
    ovh = [(o - i) * 1e6 for o, i in zip(outer, inner)]
    print(f"\nwrapper overhead us: median={statistics.median(ovh):.1f} p99={np.percentile(ovh, 99):.1f}")
    assert statistics.median(ovh) < 500
    s._writer.close()
    lines = [json.loads(x) for x in open(next(tmp_path.glob("sched_*.jsonl")))]
    assert len(lines) == 400 and lines[0]["admit"] == "peak" and "gated_ids" in lines[0]


def test_oracle_helpers(U):
    (r,) = mkreqs(U, [(10, 5)])
    assert final_len(r) == 15 and remaining(r) == 5
    r.num_output_placeholders = 2
    assert remaining(r) == 3


# ---------------------------------------------------------------- (v) SJF ablation (ordering only)
def test_sjf_orders_by_oracle_remaining(U, monkeypatch):
    s = make(U, monkeypatch, "regime", env={"REGIME_SCHED_ADMIT": "sjf"})
    reqs = mkreqs(U, [(32, 1000), (32, 20), (32, 500), (32, 5)])
    for r in reqs:
        s.add_request(r)
    out = s.schedule()
    assert [n.req_id for n in out.scheduled_new_reqs] == ["r3", "r1", "r2", "r0"]
    # still preempts under KV pressure (no planning) -- unlike peak
    s = make(U, monkeypatch, "regime", env={"REGIME_SCHED_ADMIT": "sjf"})
    _, pre = run_trace(s, mkreqs(U, PREEMPT_SPECS))
    assert pre > 0
