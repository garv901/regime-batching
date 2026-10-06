"""Collect the smoke-test facts from a smoke_bagel.sbatch output dir (default: newest /data/.../logs/mstar/smoke_*).

  python3 experiments/mstar/summarize_smoke.py [DIR]

Prints: KV pool (kv_probe), startup times, GPU memory at the log markers (from the 1 Hz nvidia-smi timeline),
CUDA graph capture lines, and each client phase (raw summary lines + per-request E2E percentiles from results.json).
Log-marker regexes are tolerant: a marker that is missing is reported as missing, never an error.
"""
import glob
import json
import os
import re
import sys
from datetime import datetime

d = sys.argv[1] if len(sys.argv) > 1 else sorted(glob.glob("/data/garv901-55613a/logs/mstar/smoke_*"))[-1]
print(f"## smoke dir {d}")


def read(p):
    return open(p).read() if os.path.exists(p) else ""


print("\n## KV pool\n" + read(f"{d}/kv_probe.txt"))
print("## startup\n" + read(f"{d}/startup.txt"))

TS = re.compile(r"(\d{4})[-/](\d{2})[-/](\d{2})[ T](\d{2}):(\d{2}):(\d{2})")


def ts(line):
    m = TS.search(line)
    return datetime(*map(int, m.groups())) if m else None


mem = []  # (time, used MiB, total MiB)
for line in read(f"{d}/gpu_mem.csv").splitlines():
    f = [x.strip() for x in line.split(",")]
    if len(f) >= 4 and f[2].isdigit():
        mem.append((ts(f[0]), int(f[2]), int(f[3])))


def mem_at(t):
    if t is None or not mem:
        return "n/a"
    best = min(mem, key=lambda m: abs((m[0] - t).total_seconds()))
    return f"{best[1]} MiB used / {best[2]} MiB total -> {best[2] - best[1]} MiB free"


log = read(f"{d}/cold/server.log").splitlines()
marks = {
    "first server log line": lambda l: True,
    "engine loaded (weights on GPU, before capture)": lambda l: "Engine loaded on device" in l,
    "last CUDA graph captured": lambda l: "Captured CUDA graph for" in l,
    "warmup_and_capture done": lambda l: "warmup_and_capture done" in l,
    "API server bound": lambda l: "Starting mstar API server" in l,
}
print("## GPU memory at server-log markers (cold start)")
if mem:
    print(f"baseline before server: {mem[0][1]} MiB; peak over job: {max(m[1] for m in mem)} MiB; "
          f"last sample: {mem[-1][1]} MiB")
for name, pred in marks.items():
    hits = [(ts(l), l) for l in log if pred(l) and ts(l)]
    if not hits:
        print(f"- {name}: marker missing")
        continue
    t, l = (hits[0] if name.startswith("first") or name.startswith("engine") else hits[-1])
    print(f"- {name}: {t:%H:%M:%S}  {mem_at(t)}")

print("\n## CUDA graph capture")
cap = [l for l in log if "Captured CUDA graph for" in l]
print(f"{len(cap)} 'Captured CUDA graph' lines; first/last:")
for l in cap[:1] + cap[-1:]:
    print("  " + l.strip()[:300])
for l in log:
    if "warmup_and_capture done" in l or "run eagerly" in l or "Dropping CUDA graph" in l or "Failed to capture" in l:
        print("  " + l.strip()[:400])
bs = sorted({int(m.group(1)) for l in cap if "decode" in l for m in [re.search(r"bs=(\d+)", l)] if m})
print(f"decode batch sizes captured: {bs} (decode cap = {max(bs) if bs else 'n/a'})")

print("\n## client phases")
for p in sorted(glob.glob(f"{d}/cold/client_*.log")):
    name = os.path.basename(p)[len("client_"):-len(".log")]
    print(f"### {name}")
    for l in read(p).splitlines():
        if re.match(r"(Requests :|TTFT|E2E|ITL|Text tokens|Prompt tokens|Throughput|Total wall|Closed-Loop)", l):
            print("  " + l)
    rj = f"{d}/cold/out_{name}/results.json"
    if os.path.exists(rj):
        j = json.load(open(rj))
        print(f"  per-request jct_ms: mean {j['jct_mean_ms']:.0f} p50 {j['jct_median_ms']:.0f} "
              f"p90 {j['jct_p90_ms']:.0f} p99 {j['jct_p99_ms']:.0f}; completed {j['completed']} failed {j['failed']}")
