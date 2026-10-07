#!/usr/bin/env python3
"""Measured runs that env_valid.py marks invalid (outsider load > LIMIT on the node): one replacement task per original task
(same arm/env, WARMUP, WORKLOADS = its invalid workloads), only for tasks already in pool done/. Azure first.
Records (orig task, run, new task, reason) in the replaced manifest; skips runs already listed there. Without --submit it
only prints the plan (task, label, workloads) and changes nothing.

  LIMIT=15 python3 experiments/mstar/env/requeue_env.py --job 13218 --sar /tmp/sar_poolidle.txt --manifest <results>/manifest_v4b.tsv
  ... --submit      # queue them through experiments/mstar/submit_matrix.sh + pool_sbatch.sh

A new task goes to the pool's held/ (Azure: deferred_azure/, it waits for a quiet node) instead of queue/. Same flags as
env_valid.py (judging), plus the ones below (queueing).
"""
import argparse
import collections
import os
import re
import subprocess

from env_valid import add_args, pool_dir, tsv_rows

REPO = os.path.realpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../.."))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    add_args(ap)
    ap.add_argument("--manifest", default="manifest_v4b.tsv",
                    help="submit_matrix.sh manifest; relative = under the results root (default %(default)s)")
    ap.add_argument("--replaced", default="manifest_v4b_replaced.tsv",
                    help="replacement record; relative = under the results root (default %(default)s)")
    ap.add_argument("--repo", default=REPO, help="repo to run submit_matrix.sh in (default %(default)s)")
    ap.add_argument("--step-cpus", default="2", help="STEP_CPUS of the new tasks (default %(default)s)")
    ap.add_argument("--step-mem", default="16G", help="STEP_MEM of the new tasks (default %(default)s)")
    ap.add_argument("--warmup", default="bagel_burst_v2", help="WARMUP of the new tasks (default %(default)s)")
    ap.add_argument("--dest", default="held", help="pool subdir a new task waits in (default %(default)s)")
    ap.add_argument("--azure-dest", default="deferred_azure", help="the same for Azure workloads (default %(default)s)")
    ap.add_argument("--submit", action="store_true", help="queue the replacement tasks (default: print the plan only)")
    a = ap.parse_args()
    pool = pool_dir(a)
    manifest, repl = os.path.join(a.results, a.manifest), os.path.join(a.results, a.replaced)

    done_runs = set(tuple(l.split("\t")[:2]) for l in open(repl)) if os.path.exists(repl) else set()
    man = {l.split("\t")[0]: l.rstrip("\n").split("\t") for l in open(manifest) if not l.startswith("#")}
    finished = {f[:-5] for f in os.listdir(f"{pool}/done")}
    by_task = collections.OrderedDict()
    for task, _arm, run, m, ok in tsv_rows(a):
        if ok == 1 or task not in finished or task not in man or (task, run) in done_runs:
            continue
        by_task.setdefault(task, []).append((run, float(m)))
    order = sorted(by_task, key=lambda t: (0 if "azure" in by_task[t][0][0] else 1, t))
    env = dict(os.environ, POOL_JOB=a.job, SBATCH="experiments/mstar/pool_sbatch.sh", MANIFEST=manifest,
               POOL_PIN="0", STEP_CPUS=a.step_cpus, STEP_MEM=a.step_mem, REPS="1", WARMUP=a.warmup)
    for t in order:
        jid, label, config, extra = man[t][:4]
        wls = " ".join(re.sub(r"_r\d+$", "", r) for r, _ in by_task[t])
        print(f"{t} {label:16} {wls}")
        if a.submit:
            o = subprocess.run(["./experiments/mstar/submit_matrix.sh", f"{label} {config} {extra}".strip()],
                               cwd=a.repo, env=dict(env, WORKLOADS=wls), capture_output=True, text=True, check=True).stdout
            new = o.split()[0]
            dest = a.azure_dest if "azure" in wls else a.dest   # Azure waits for a quiet node (user, 2026-10-07)
            if os.path.exists(f"{pool}/queue/{new}.task"):
                os.replace(f"{pool}/queue/{new}.task", f"{pool}/{dest}/{new}.task")
            with open(repl, "a") as fh:
                for r, m in by_task[t]:
                    fh.write(f"{t}\t{r}\t{new}\tenv:{m:.1f}%\n")
            print(f"   -> task {new}")


if __name__ == "__main__":
    main()
