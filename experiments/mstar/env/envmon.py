#!/usr/bin/env python3
"""Every --interval s: per running pool step, the CPU time on its cpuset that is NOT ours (outsiders), our usage and PSI,
plus the busy share of CPU groups. Appends JSONL to the out file until the stop file exists.
Interference = (busy(cpus) - step usage) / (n_cpus * dt). The steps and their CPUs are read from the pool job's cgroup.

  nohup python3 experiments/mstar/env/envmon.py --job 13218 > /dev/null 2>&1 &     # on the node of the pool
  touch /data/garv901-55613a/results/mstar/pool/13218/STOP_ENVMON                   # stops it (and speedprobe.py, gate.py)

Row: ts, dt, <group>: busy share 0-1 of the group's CPUs, steps: {step_N: task, cpus, busy, ours, outsider, psi_some, psi_full}
(shares of the step's CPU time; PSI in s of stall per s). A step is reported from its second sample on.
Env defaults: JOB (--job), INT (--interval), OUT (--out).
"""
import argparse
import glob
import json
import os
import time

RESULTS = "/data/garv901-55613a/results/mstar"
GROUPS = ["poolidle=8-35,112-139", "n0rest=36-51,140-155", "n1=52-103,156-207"]   # node1: pool's idle CPUs, rest of NUMA0, NUMA1
HZ = os.sysconf("SC_CLK_TCK")


def parse_list(s):
    out = []
    for part in s.strip().split(","):
        if "-" in part:
            a, b = part.split("-")
            out += range(int(a), int(b) + 1)
        elif part:
            out.append(int(part))
    return out


def cpustat():
    d = {}
    for line in open("/proc/stat"):
        if line.startswith("cpu") and line[3].isdigit():
            f = line.split()
            v = list(map(int, f[1:9]))   # user nice system idle iowait irq softirq steal
            d[int(f[0][3:])] = (sum(v) - v[3] - v[4]) / HZ
    return d


def steps(job_cgroup):
    r = {}
    for s in glob.glob(f"{job_cgroup}/step_[0-9]*/user"):
        try:
            cpus = parse_list(open(f"{s}/cpuset.cpus.effective").read())
            use = int(next(l.split()[1] for l in open(f"{s}/cpu.stat") if l.startswith("usage_usec"))) / 1e6
            psi = {l.split()[0]: int(l.split()[-1].split("=")[1]) / 1e6 for l in open(f"{s}/cpu.pressure")}
            task = None
            for p in open(f"{s}/task_0/cgroup.procs").read().split()[:3]:
                try:
                    env = open(f"/proc/{p}/environ", "rb").read().split(b"\0")
                    task = next((e.split(b"=", 1)[1].decode() for e in env if e.startswith(b"SLURM_JOB_ID=")), None)
                    if task:
                        break
                except OSError:
                    pass
            r[s.split("/")[-2]] = dict(cpus=cpus, use=use, psi=psi, task=task)
        except (OSError, StopIteration, ValueError):
            pass
    return r


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--job", default=os.environ.get("JOB", "13218"), help="pool job id (default $JOB or %(default)s)")
    ap.add_argument("--results", default=RESULTS, help="results root (default %(default)s)")
    ap.add_argument("--pool", help="pool dir (default <results>/pool/<job>)")
    ap.add_argument("--cgroup",
                    help="the pool job's cgroup dir (default /sys/fs/cgroup/system.slice/node1_slurmstepd.scope/job_<job>)")
    ap.add_argument("--interval", type=float, default=float(os.environ.get("INT", 2)),
                    help="seconds between samples (default $INT or %(default)s)")
    ap.add_argument("--group", action="append", metavar="NAME=CPUS",
                    help="CPU group to report, e.g. poolidle=8-35,112-139; repeatable; default " + " ".join(GROUPS))
    ap.add_argument("--out", default=os.environ.get("OUT", "envmon.jsonl"),
                    help="JSONL file, appended; relative = under the pool dir (default $OUT or %(default)s)")
    ap.add_argument("--stop-file", default="STOP_ENVMON",
                    help="exit when this file exists; relative = under the pool dir (default %(default)s)")
    a = ap.parse_args()
    pool = a.pool or os.path.join(a.results, "pool", a.job)
    job_cgroup = a.cgroup or f"/sys/fs/cgroup/system.slice/node1_slurmstepd.scope/job_{a.job}"
    out, stop = os.path.join(pool, a.out), os.path.join(pool, a.stop_file)
    groups = {n: parse_list(c) for n, c in (g.split("=", 1) for g in (a.group or GROUPS))}

    prev_t, prev_c, prev_s = time.time(), cpustat(), steps(job_cgroup)
    with open(out, "a") as f:
        while not os.path.exists(stop):
            time.sleep(a.interval)
            t, c, s = time.time(), cpustat(), steps(job_cgroup)
            dt = t - prev_t
            row = {"ts": round(t, 2), "dt": round(dt, 2)}
            for g, cl in groups.items():
                row[g] = round(sum(c[i] - prev_c[i] for i in cl) / (len(cl) * dt), 4)
            st = {}
            for k, v in s.items():
                if k not in prev_s or prev_s[k]["cpus"] != v["cpus"]:
                    continue
                busy = sum(c[i] - prev_c[i] for i in v["cpus"])
                use = v["use"] - prev_s[k]["use"]
                n = len(v["cpus"]) * dt
                st[k] = {"task": v["task"], "cpus": v["cpus"], "busy": round(busy / n, 4), "ours": round(use / n, 4),
                         "outsider": round(max(busy - use, 0) / n, 4),
                         "psi_some": round((v["psi"]["some"] - prev_s[k]["psi"]["some"]) / dt, 4),
                         "psi_full": round((v["psi"]["full"] - prev_s[k]["psi"]["full"]) / dt, 4)}
            row["steps"] = st
            f.write(json.dumps(row) + "\n")
            f.flush()
            prev_t, prev_c, prev_s = t, c, s


if __name__ == "__main__":
    main()
