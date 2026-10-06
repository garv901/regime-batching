"""experiments/make_workload.py --long-lo: long outputs ~U[long-lo, long], every other draw unchanged."""
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = ["bimodal", "--rate", "6", "--duration", "60", "--seed", "1", "--prompt-lo", "450", "--prompt", "1000",
        "--short-lo", "16", "--short-hi", "64", "--long", "2000", "--long-prompt", "12000", "--long-prompt-lo", "4000",
        "--frac-long", "0.15", "--max-model-len", "32748"]


def gen(tmp_path, name, *extra):
    out = str(tmp_path / f"{name}.jsonl")
    subprocess.run([sys.executable, os.path.join(ROOT, "experiments", "make_workload.py"), *BASE, "--out", out, *extra],
                   check=True, capture_output=True)
    return [json.loads(line) for line in open(out)]


def test_long_lo_varies_only_the_long_outputs(tmp_path):
    fixed = gen(tmp_path, "fixed")
    varied = gen(tmp_path, "varied", "--long-lo", "256")
    assert len(fixed) == len(varied) and any(r["cls"] == "long" for r in fixed)
    assert {r["output_length"] for r in fixed if r["cls"] == "long"} == {2000}
    outs = [r["output_length"] for r in varied if r["cls"] == "long"]
    assert min(outs) >= 256 and max(outs) <= 2000 and len(set(outs)) > 10
    for a, b in zip(fixed, varied):  # arrivals, classes, prompts and the shorts are the same draws as without the flag
        assert (a["timestamp"], a["cls"], a["input_length"]) == (b["timestamp"], b["cls"], b["input_length"])
        if a["cls"] == "short":
            assert a["output_length"] == b["output_length"]


def test_long_lo_is_reproducible_and_checked(tmp_path):
    assert gen(tmp_path, "a", "--long-lo", "256") == gen(tmp_path, "b", "--long-lo", "256")
    bad = subprocess.run([sys.executable, os.path.join(ROOT, "experiments", "make_workload.py"), *BASE, "--out",
                          str(tmp_path / "bad.jsonl"), "--long-lo", "3000"], capture_output=True)
    assert bad.returncode != 0  # --long-lo above --long
