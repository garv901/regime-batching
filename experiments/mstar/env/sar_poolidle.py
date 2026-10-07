#!/usr/bin/env python3
"""Mean busy % (100 - %idle) of a CPU set at each sample of a sysstat day file: the sar input of env_valid.py for runs that
started before envmon.py did. Output: one "HH:MM:SS value" line per sample time, sorted. The same as
  LC_ALL=C sar -P ALL -f FILE -s START | awk '$2 ~ /^[0-9]+$/ {...mean of 100-$NF over the CPUs...}' | grep -v Average | sort

  python3 experiments/mstar/env/sar_poolidle.py --file /var/log/sysstat/sa07 --cpus 8-35,112-139 --out <pool>/sar_poolidle.txt

env_valid.py needs --sar-day = the date of that day file.
"""
import argparse
import os
import subprocess
import sys
import time

from envmon import parse_list


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--file", default=time.strftime("/var/log/sysstat/sa%d"),
                    help="sysstat day file (default today's: %(default)s)")
    ap.add_argument("--cpus", default="8-35,112-139",
                    help="CPU list to average, e.g. 8-35,112-139 (default %(default)s)")
    ap.add_argument("--start", default="00:00:00", help="sar -s start time (default %(default)s)")
    ap.add_argument("--out", help="output file (default stdout)")
    a = ap.parse_args()
    cpus = set(parse_list(a.cpus))

    sar = subprocess.run(["sar", "-P", "ALL", "-f", a.file, "-s", a.start], capture_output=True, text=True,
                         env=dict(os.environ, LC_ALL="C"))
    if sar.returncode:
        sys.exit(f"sar failed: {sar.stderr.strip()}")
    s, n = {}, {}
    for line in sar.stdout.splitlines():
        f = line.split()
        if len(f) > 1 and f[1].isdigit() and int(f[1]) in cpus and "Average" not in f[0]:
            s[f[0]] = s.get(f[0], 0) + 100 - float(f[-1])
            n[f[0]] = n.get(f[0], 0) + 1
    lines = [f"{t} {s[t] / n[t]:.6g}" for t in sorted(s)]
    if a.out:
        with open(a.out, "w") as fh:
            fh.write("\n".join(lines) + "\n")
    else:
        print("\n".join(lines))


if __name__ == "__main__":
    main()
