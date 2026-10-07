"""Compare whole-step census times of two models on the same Stage 1b shapes (default engine config).

Used to check that the Stage 3 model (Llama-3.1-Nemotron-8B-UltraLong-1M, the Llama-3.1-8B architecture with a longer
RoPE table) runs the census steps at the Llama-3.1-8B times, so Stage 1/1b calibrations carry over.
Usage: python3 analysis/model_compare_census.py BASE_RUN_DIR OTHER_RUN_DIR   (each holds probes.jsonl + telemetry/)
"""
import collections, importlib.util, json, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("census_analysis", ROOT / "analysis/regime_census.py")
rc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rc)


def shape_medians(run_dir):
    rc.RUN = Path(run_dir).parent
    probes, steps = rc.load_config(Path(run_dir).name)
    rows, _ = rc.classify_steps("x", probes, steps)
    by = collections.defaultdict(list)
    for r in rows:
        by[rc.shape_key(r)].append((r["gpu_ms"], r["mode"]))
    checks = [json.loads(l) for l in open(Path(run_dir) / "probes.jsonl") if '"long_check"' in l or '"passkey"' in l]
    return {k: (float(np.median([g for g, _ in v])), v[0][1]) for k, v in by.items()}, checks


def main():
    base, _ = shape_medians(sys.argv[1])
    other, checks = shape_medians(sys.argv[2])
    for c in checks:
        print("check:", c)
    common = sorted(set(base) & set(other), key=str)
    ratio = np.array([other[k][0] / base[k][0] for k in common])
    print(f"shapes: base {len(base)}, other {len(other)}, common {len(common)}")
    print(f"step time ratio other/base: median {np.median(ratio):.3f}, p10 {np.percentile(ratio, 10):.3f}, "
          f"p90 {np.percentile(ratio, 90):.3f}, within 5%: {np.mean(np.abs(ratio - 1) <= 0.05):.0%}")
    grp = collections.defaultdict(list)
    for k, x in zip(common, ratio):
        grp[(k[1], base[k][1])].append(x)
    for (kind, mode), v in sorted(grp.items()):
        print(f"  {kind:5s} {mode:9s} n={len(v):3d} median {np.median(v):.3f} "
              f"range {min(v):.3f}-{max(v):.3f} within 5% {np.mean(np.abs(np.array(v) - 1) <= 0.05):.0%}")
    worst = sorted(zip(common, ratio), key=lambda t: -abs(t[1] - 1))[:8]
    for k, x in worst:
        print(f"  far: {k} base {base[k][0]:.2f} ms other {other[k][0]:.2f} ms ratio {x:.3f}")


if __name__ == "__main__":
    main()
