"""CPU tests for the declared-max_tokens (v4) workloads: analysis/admit_sim_bagel.py and experiments/mstar/kv_report.py.

  uv run --no-project --with numpy --with pytest python -m pytest -q tests/test_declared_workloads.py

A row with `max_tokens` is reserved / planned with that value while the request still stops after `output_length` tokens
(an emulated EOS); a row without it is an oracle row (declared == output_length).
"""
import json
import os
import subprocess
import sys

import numpy as np
import pytest

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "analysis"))
import admit_sim_bagel as sim  # noqa: E402

KV_REPORT = os.path.join(ROOT, "experiments", "mstar", "kv_report.py")


def write_trace(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def run_sim(prompts, outs, decl, sched, cap, is_long=(True, False)):
    """Requests all arrive at t = 0 in this order; prompts are the server-side tokens (template included)."""
    n = len(prompts)
    return sim.simulate(np.zeros(n), np.array(prompts), np.array(outs), np.array(is_long), sched=sched, cap=cap,
                        decl=None if decl is None else np.array(decl))


def test_load_trace_reads_declared_and_defaults_to_the_oracle(tmp_path):
    rows = [{"timestamp": 1.0, "input_length": 80, "output_length": 10, "cls": "short"},
            {"timestamp": 0.0, "input_length": 80, "output_length": 10, "cls": "long", "max_tokens": 400},
            {"timestamp": 2.0, "input_length": 80, "output_length": 10, "cls": "short", "max_tokens": None}]
    write_trace(tmp_path / "t.jsonl", rows)
    arr, prompts, outs, is_long, decl = sim.load_trace(str(tmp_path / "t.jsonl"))
    assert list(arr) == [0.0, 1.0, 2.0] and list(prompts) == [100, 100, 100] and list(is_long) == [True, False, False]
    assert list(outs) == [10, 10, 10] and list(decl) == [400, 10, 10]  # sorted by arrival; absent / null -> output_length
    rows[0]["max_tokens"] = 5  # below its output_length: the server would stop it there, not a valid row
    write_trace(tmp_path / "bad.jsonl", rows)
    with pytest.raises(ValueError, match="below its output_length"):
        sim.load_trace(str(tmp_path / "bad.jsonl"))


def test_oracle_rows_are_unchanged_by_the_declared_path():
    for sched in ("sum_fifo", "sum_bf", "peak_fifo", "peak_bf", "open"):
        assert run_sim([100, 100], [5, 5], None, sched, 1000) == run_sim([100, 100], [5, 5], [5, 5], sched, 1000)


def test_sum_reserves_the_declared_bound_but_frees_at_the_real_length():
    # cap 1000: A (long) reserves 100 + 800 = 900 when it declares 800, so B (short, 100 + 5) cannot be admitted beside it
    oracle = run_sim([100, 100], [5, 5], [5, 5], "sum_bf", 1000)
    declared = run_sim([100, 100], [5, 5], [800, 5], "sum_bf", 1000)
    assert oracle["short_ttft_mean"] < oracle["long_e2e_mean"]  # both admitted at once: B's first token comes before A is done
    assert declared["short_ttft_mean"] >= declared["long_e2e_mean"]  # B waits for A to finish ...
    # ... and A finishes after its 5 real tokens, not 800: B's first token follows within a step or two (800 tokens take > 4 s)
    assert declared["short_ttft_mean"] - declared["long_e2e_mean"] < 0.1
    assert declared["unfinished"] == 0 and declared["out_tok_s"] > 0
    # the peak planner assumes growth up to the declared bound too: both would reach 100 + 800 = 900 -> 1800 > 1000
    peak = run_sim([100, 100], [5, 5], [800, 800], "peak_bf", 1000)
    assert peak["short_ttft_mean"] >= peak["long_e2e_mean"]
    assert run_sim([100, 100], [5, 5], [5, 5], "peak_bf", 1000)["short_ttft_mean"] < oracle["long_e2e_mean"] + 1
    # open ignores the declared bound
    assert run_sim([100, 100], [5, 5], [800, 800], "open", 1000) == run_sim([100, 100], [5, 5], None, "open", 1000)


def test_peak_fits_where_the_summed_reservation_does_not():
    # A declares 700 and B 300 more tokens: the sum is 800 + 400 = 1200 > 1000, but B's growth ends (k = 300) long before A's
    # (k = 700), so the future peak is 800 <= 1000: B is admitted at once under peak, kept waiting under sum
    s = run_sim([100, 100], [5, 5], [700, 300], "sum_bf", 1000)
    p = run_sim([100, 100], [5, 5], [700, 300], "peak_bf", 1000)
    assert s["short_ttft_mean"] >= s["long_e2e_mean"] and p["short_ttft_mean"] < p["long_e2e_mean"]


def test_fifo_head_of_line_with_a_declared_head():
    # strict FIFO: a head that does not fit blocks a request behind it that would; backfill lets it pass
    prompts, outs, decl = [100, 100, 100], [5, 5, 5], [800, 800, 5]
    args = (np.zeros(3), np.array(prompts), np.array(outs), np.array([True, True, False]))
    fifo = sim.simulate(*args, sched="sum_fifo", cap=1100, decl=np.array(decl))
    bf = sim.simulate(*args, sched="sum_bf", cap=1100, decl=np.array(decl))
    assert bf["short_e2e_mean"] < fifo["short_e2e_mean"]  # the third request (short) overtakes the blocked second one


# ------------------------------------------------------------------------------------------------ kv_report
def kv_report(tmp_path, rows, *extra):
    write_trace(tmp_path / "w.jsonl", rows)
    json.dump({"path": "w.jsonl", "n": len(rows)}, open(tmp_path / "w.meta.json", "w"))
    r = subprocess.run([sys.executable, KV_REPORT, "--pool", "10000", "--itl-ms", "10", *extra, str(tmp_path / "w.jsonl")],
                       capture_output=True, text=True)
    return r, json.load(open(tmp_path / "w.meta.json"))


ROWS = [{"timestamp": 0.0, "input_length": 100, "output_length": 10, "cls": "short", "max_tokens": 20},
        {"timestamp": 0.0, "input_length": 1000, "output_length": 100, "cls": "long", "max_tokens": 400}]


def test_kv_report_declared_figures(tmp_path):
    r, meta = kv_report(tmp_path, ROWS)
    assert r.returncode == 0, r.stderr
    kv = meta["kv"]
    # generated: short 100 + 20 + 10 = 130, long 1000 + 20 + 100 = 1120; declared: 140 and 1420
    assert kv["total_final_kv_tokens"] == 1250 and kv["declared_total_final_kv_tokens"] == 1560
    assert kv["declared_total_ratio_to_pool"] == pytest.approx(0.156) and kv["decl_kv_ratio"] == pytest.approx(1560 / 1250, abs=1e-3)
    assert kv["per_class"]["short"]["declared_final_kv_tokens"] == 140 and kv["per_class"]["long"]["declared_final_kv_tokens"] == 1420
    assert kv["per_class"]["long"]["declared_ratio_to_pool"] == pytest.approx(0.142)
    assert kv["per_class"]["short"]["decl_kv_ratio"] == pytest.approx(140 / 130, abs=1e-3)
    assert kv["decl_ratio_mean"] == pytest.approx(3.0)  # (20/10 + 400/100) / 2 = 3
    # fluid peaks at 10 ms per token: generated 1160 (t = 0.1 s: the short is at its last token, the long at its 10th);
    # declared lengths 1180 at t = 0.2 s, then the long alone grows to 1420 at 4 s; the reservation 140 + 1420 at t = 0
    assert kv["peak_inflight_kv_ratio_itl10ms"] == pytest.approx(0.116)
    assert kv["peak_declared_inflight_kv_ratio_itl10ms"] == pytest.approx(0.142)
    assert kv["peak_reserved_inflight_kv_ratio_itl10ms"] == pytest.approx(0.156)
    assert "declared_total_ratio_to_pool" in r.stdout and "peak_reserved_inflight_kv_ratio_itl10ms" in r.stdout


def test_kv_report_oracle_file_has_no_declared_keys(tmp_path):
    r, meta = kv_report(tmp_path, [{k: v for k, v in row.items() if k != "max_tokens"} for row in ROWS])
    assert r.returncode == 0, r.stderr
    kv = meta["kv"]
    assert set(kv) == {"pool_tokens", "template_overhead", "max_seq_len", "max_request_kv", "per_class", "total_final_kv_tokens",
                       "total_ratio_to_pool", "peak_inflight_kv_ratio_itl10ms"}
    assert all(set(c) == {"n", "final_kv_tokens", "ratio_to_pool"} for c in kv["per_class"].values())
    assert "declared" not in r.stdout and kv["peak_inflight_kv_ratio_itl10ms"] == pytest.approx(0.116)
    # a null max_tokens counts as declared == output_length, the others keep theirs
    mixed = [dict(ROWS[0]), {k: v for k, v in ROWS[1].items() if k != "max_tokens"}]
    r, meta = kv_report(tmp_path, mixed)
    assert meta["kv"]["declared_total_final_kv_tokens"] == 140 + 1120


def test_kv_report_rejects_invalid_declared_bounds(tmp_path):
    below = [dict(ROWS[0], max_tokens=5)]  # below its output_length
    r, _ = kv_report(tmp_path, below)
    assert r.returncode != 0 and "declared max_tokens 5" in r.stderr
    above = [dict(ROWS[0], max_tokens=4000)]  # above BAGEL's 2,048 cap
    r, _ = kv_report(tmp_path, above)
    assert r.returncode != 0 and "declared max_tokens 4000" in r.stderr
    r, meta = kv_report(tmp_path, above, "--max-declared", "8000")
    assert r.returncode == 0 and meta["kv"]["declared_total_final_kv_tokens"] == 100 + 20 + 4000
