#!/usr/bin/env python3
"""Markdown tables for the S5 A/B (round-robin vs deadline scheduling on the colocated Qwen3-Omni worker).

    python3 analyze.py --run results/mstar/s5/run1 > results/mstar/s5/run1/analysis.md

Reads, per arm (a dir under --run named after the server-side policy):
    <arm>/<case>_r<rep>/client.json     from mixed_client.py: one row per request, with wall-clock times
    <arm>/<case>_r<rep>/telemetry/      the server's step telemetry (a symlink into <arm>/server_<k>/telemetry)
and pools the reps of a case. Standard library only.

Metrics (audio = the measured requests of every case; text = the text-only long-prompt load of the b cases):
    TTFA          time to the first audio chunk
    RTF           end-to-end time / seconds of audio produced (the benchmark's definition; lower is better)
    stall         listener model: playback starts at the first chunk and runs at 1x; a chunk arriving after the audio
                  buffered so far ran out stalls it by the gap. "stalled" = share of requests with any stall
    TTFT, ITL     text time to first token, and gap between consecutive text tokens
    text tok/s    text tokens that ARRIVED inside the audio window / window length (the text load keeps running for it)
    audio s/s     seconds of audio produced / window length
From the telemetry, inside each case's window:
    wait          the time the longest-waiting ready request of a (node, walk) had waited when the scheduler picked,
                  from the `sched_pick` events (the same bookkeeping under both policies); max and p99 over picks at
                  which that walk had a ready request
    GPU share     sum of the walk's step gpu_ms / window
"""
import argparse
import bisect
import glob
import json
import os
import re
import statistics
from collections import defaultdict

ARM_ORDER = ["round_robin", "deadline"]
CASE_ORDER = ["a4", "a8", "b4", "b8"]
CASE_DESC = {
    "a4": "audio only, conc 4",
    "a8": "audio only, conc 8",
    "b4": "audio conc 4 + text conc 8 (1-2k-token prompts)",
    "b8": "audio conc 8 + text conc 8 (1-2k-token prompts)",
}
WALK_LABEL = {
    "Code2Wav/code2wav_chunk": "Code2Wav chunk",
    "Talker/talker_decode": "Talker decode",
    "Talker/talker_prefill": "Talker prefill",
    "Talker/talker_last_prefill": "Talker last prefill",
    "Thinker/thinker_decode": "Thinker decode",
    "Thinker/prefill_text": "Thinker prefill (text)",
}


def pct(vals, p):
    if not vals:
        return None
    v = sorted(vals)
    k = (len(v) - 1) * p / 100.0
    lo = int(k)
    hi = min(lo + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (k - lo)


def fmt(x, nd=2, unit=""):
    return "n/a" if x is None else f"{x:.{nd}f}{unit}"


def delta(base, new):
    if base is None or new is None or base == 0:
        return "n/a"
    return f"{(new - base) / abs(base) * 100:+.0f}%"


# -- loading ----------------------------------------------------------------------------------------------------------


def load_cases(run):
    """arm -> case -> list of (rep, client dict, telemetry dir or None)."""
    out: dict = defaultdict(lambda: defaultdict(list))
    for path in sorted(glob.glob(os.path.join(run, "*", "*_r*", "client.json"))):
        case_dir = os.path.dirname(path)
        arm = os.path.basename(os.path.dirname(case_dir))
        m = re.fullmatch(r"(.+)_r(\d+)", os.path.basename(case_dir))
        if m is None or m.group(1) == "warmup":
            continue
        with open(path) as f:
            client = json.load(f)
        tel = os.path.join(case_dir, "telemetry")
        out[arm][m.group(1)].append((int(m.group(2)), client, os.path.realpath(tel) if os.path.isdir(tel) else None))
    return out


def scan_telemetry(cases):
    """(arm, case, rep) -> {"waits": {walk: [ms]}, "gpu_ms": {walk: ms}, "steps": {walk: n}, "bs": {walk: [bs]},
    "reasons": {reason: n}}, reading each telemetry dir once and keeping only rows inside some case's window."""
    by_dir = defaultdict(list)  # dir -> [(t0, t1, key)]
    for arm, cs in cases.items():
        for case, reps in cs.items():
            for rep, client, tel in reps:
                if tel is not None:
                    w = client["window"]
                    by_dir[tel].append((w["t_start"], w["t_end"], (arm, case, rep)))
    stats = {}
    for tel, wins in by_dir.items():
        wins.sort()
        starts = [w[0] for w in wins]
        for key in (w[2] for w in wins):
            stats[key] = {
                "waits": defaultdict(list), "gpu_ms": defaultdict(float), "steps": defaultdict(int),
                "bs": defaultdict(list), "reasons": defaultdict(int), "picks": 0,
            }
        for fn in sorted(glob.glob(os.path.join(tel, "steps_rank*_pid*.jsonl"))):
            with open(fn, "rb") as f:
                for line in f:
                    if b'"sched_pick"' not in line and b'"kind":"step"' not in line and b'"kind": "step"' not in line:
                        continue
                    try:
                        row = json.loads(line)
                    except ValueError:
                        continue
                    ts = row.get("ts")
                    if ts is None:
                        continue
                    i = bisect.bisect_right(starts, ts) - 1
                    if i < 0 or ts > wins[i][1]:
                        continue
                    st = stats[wins[i][2]]
                    if row.get("kind") == "event" and row.get("ev") == "sched_pick":
                        st["picks"] += 1
                        st["reasons"][row.get("reason")] += 1
                        for walk, ms in (row.get("waits") or {}).items():
                            st["waits"][walk].append(ms)
                    elif row.get("kind") == "step" and row.get("ok"):
                        walk = f"{row['node']}/{row['walk']}"
                        st["steps"][walk] += 1
                        st["bs"][walk].append(row.get("bs") or 0)
                        if row.get("gpu_ms") is not None:
                            st["gpu_ms"][walk] += row["gpu_ms"]
    return stats


# -- metrics ------------------------------------------------------------------------------------------------------------


def case_metrics(reps, tel_stats, arm, case):
    ttfa, rtf, stalls, stalled, gaps = [], [], [], [], []
    ttft, itl = [], []
    audio_s = text_tok = window_s = 0.0
    n_audio = n_text = n_fail = 0
    rep_ttfa_p50 = []
    waits = defaultdict(list)
    gpu_ms = defaultdict(float)
    steps = defaultdict(int)
    bs = defaultdict(list)
    reasons = defaultdict(int)
    for rep, client, _tel in sorted(reps, key=lambda r: r[0]):
        w = client["window"]
        window = w["t_end"] - w["t_start"]
        window_s += window
        rep_ttfa = []
        for r in client["audio"]:
            if not r["ok"]:
                n_fail += 1
                continue
            n_audio += 1
            ttfa.append(r["ttfa"])
            rep_ttfa.append(r["ttfa"])
            if r["rtf"] is not None:
                rtf.append(r["rtf"])
            stalls.append(r["stall_s"])
            stalled.append(1.0 if r["stall_s"] > 0.05 else 0.0)
            gaps.extend(r["gaps"])
            audio_s += r["audio_s"]
        if rep_ttfa:
            rep_ttfa_p50.append(pct(rep_ttfa, 50))
        for r in client["text"]:
            if not r["ok"]:
                if r["start"] < w["t_end"]:
                    n_fail += 1
                continue
            if r["start"] > w["t_end"]:
                continue
            n_text += 1
            ttft.append(r["ttft"])
            itl.extend(r["gaps"])
            t = r["start"] + r["ttft"]
            if w["t_start"] <= t <= w["t_end"]:
                text_tok += 1
            for g in r["gaps"]:
                t += g
                if w["t_start"] <= t <= w["t_end"]:
                    text_tok += 1
        st = tel_stats.get((arm, case, rep))
        if st:
            for walk, vals in st["waits"].items():
                waits[walk].extend(vals)
            for walk, v in st["gpu_ms"].items():
                gpu_ms[walk] += v
            for walk, v in st["steps"].items():
                steps[walk] += v
            for walk, v in st["bs"].items():
                bs[walk].extend(v)
            for k, v in st["reasons"].items():
                reasons[k] += v
    return {
        "n_audio": n_audio, "n_text": n_text, "n_fail": n_fail, "reps": len(reps),
        "ttfa_p50": pct(ttfa, 50), "ttfa_p99": pct(ttfa, 99), "ttfa_mean": statistics.mean(ttfa) if ttfa else None,
        "rep_ttfa_p50": rep_ttfa_p50,
        "rtf_p50": pct(rtf, 50), "rtf_p99": pct(rtf, 99),
        "stall_mean": statistics.mean(stalls) if stalls else None,
        "stalled": statistics.mean(stalled) * 100 if stalled else None,
        "gap_p50": pct(gaps, 50), "gap_p99": pct(gaps, 99),
        "ttft_p50": pct(ttft, 50), "ttft_p99": pct(ttft, 99),
        "itl_p50": pct(itl, 50), "itl_p99": pct(itl, 99),
        "text_tps": text_tok / window_s if window_s and n_text else None,
        "audio_sps": audio_s / window_s if window_s else None,
        "window_s": window_s,
        "waits": waits, "gpu_ms": gpu_ms, "steps": steps, "bs": bs, "reasons": reasons,
    }


# -- report ---------------------------------------------------------------------------------------------------------------


def arms_in(cases):
    arms = [a for a in ARM_ORDER if a in cases] + sorted(a for a in cases if a not in ARM_ORDER)
    return arms


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="results dir holding <arm>/<case>_r<rep>/client.json")
    args = ap.parse_args()

    cases = load_cases(args.run)
    if not cases:
        print(f"no client.json under {args.run}")
        return
    arms = arms_in(cases)
    base = arms[0]
    tel = scan_telemetry(cases)
    M = {(a, c): case_metrics(reps, tel, a, c) for a in arms for c, reps in cases[a].items()}
    all_cases = [c for c in CASE_ORDER if any((a, c) in M for a in arms)]

    print(f"# S5 A/B: {' vs '.join(arms)}\n")
    print(f"Run dir `{args.run}`. Delta columns compare each arm with `{base}`; "
          "for every metric here lower is better except tok/s and audio s/s.\n")

    # sample sizes
    rows = []
    for c in all_cases:
        for a in arms:
            m = M.get((a, c))
            if m:
                rows.append([c, CASE_DESC[c], a, m["reps"], m["n_audio"], m["n_text"], m["n_fail"], fmt(m["window_s"], 0, " s")])
    print("## Samples\n")
    print(table(["case", "load", "arm", "reps", "audio reqs", "text reqs", "failed", "window (all reps)"], rows))

    def block(title, metrics, cases_):
        print(f"\n## {title}\n")
        header = ["case", "metric"] + arms + [f"{a} vs {base}" for a in arms[1:]]
        rows = []
        for c in cases_:
            for label, key, nd, unit, scale in metrics:
                vals = [M.get((a, c), {}).get(key) for a in arms]
                if all(v is None for v in vals):
                    continue
                vals = [None if v is None else v * scale for v in vals]
                rows.append([c, label] + [fmt(v, nd, unit) for v in vals] + [delta(vals[0], v) for v in vals[1:]])
        print(table(header, rows))

    block("Streaming audio (the measured requests)", [
        ("TTFA p50", "ttfa_p50", 2, " s", 1), ("TTFA p99", "ttfa_p99", 2, " s", 1),
        ("RTF p50", "rtf_p50", 2, "", 1), ("RTF p99", "rtf_p99", 2, "", 1),
        ("stall per request (mean)", "stall_mean", 2, " s", 1), ("requests stalled", "stalled", 0, "%", 1),
        ("audio chunk gap p50", "gap_p50", 2, " s", 1), ("audio chunk gap p99", "gap_p99", 2, " s", 1),
        ("audio produced", "audio_sps", 2, " s/s", 1),
    ], all_cases)
    b_cases = [c for c in all_cases if c.startswith("b")]
    if b_cases:
        block("Text-only load (b cases)", [
            ("TTFT p50", "ttft_p50", 2, " s", 1), ("TTFT p99", "ttft_p99", 2, " s", 1),
            ("ITL p50", "itl_p50", 1, " ms", 1000), ("ITL p99", "itl_p99", 1, " ms", 1000),
            ("text tokens", "text_tps", 1, " tok/s", 1),
        ], b_cases)

    rows = []
    for rep_note in all_cases:
        vals = [M.get((a, rep_note), {}).get("rep_ttfa_p50") or [] for a in arms]
        rows.append([rep_note] + [", ".join(f"{v:.2f}" for v in vs) or "n/a" for vs in vals])
    print("\n## Rep-to-rep spread: TTFA p50 of each rep (s)\n")
    print(table(["case"] + arms, rows))

    # per-walk waits and GPU shares
    walks = sorted({w for m in M.values() for w in m["waits"]} | {w for m in M.values() for w in m["gpu_ms"]},
                   key=lambda w: list(WALK_LABEL).index(w) if w in WALK_LABEL else 99)
    for title, getter, nd, unit in (
        ("Longest wait of a ready (node, walk) at scheduling time: max / p99, ms", "wait", 0, ""),
    ):
        print(f"\n## {title}\n")
        header = ["case", "walk"] + [f"{a} max / p99" for a in arms]
        rows = []
        for c in all_cases:
            for w in walks:
                cells = []
                for a in arms:
                    v = M.get((a, c), {}).get("waits", {}).get(w)
                    cells.append(f"{max(v):.0f} / {pct(v, 99):.0f}  (n={len(v)})" if v else "-")
                if any(x != "-" for x in cells):
                    rows.append([c, WALK_LABEL.get(w, w)] + cells)
        print(table(header, rows))

    print("\n## GPU time and step counts per walk (share of the window; steps/s; mean batch)\n")
    header = ["case", "walk"] + [f"{a}" for a in arms]
    rows = []
    for c in all_cases:
        for w in walks:
            cells = []
            for a in arms:
                m = M.get((a, c))
                if not m or not m["steps"].get(w):
                    cells.append("-")
                    continue
                share = m["gpu_ms"].get(w, 0.0) / (m["window_s"] * 1000.0) * 100
                sps = m["steps"][w] / m["window_s"]
                cells.append(f"{share:.0f}% / {sps:.1f}/s / bs {statistics.mean(m['bs'][w]):.1f}")
            if any(x != "-" for x in cells):
                rows.append([c, WALK_LABEL.get(w, w)] + cells)
    print(table(header, rows))

    print("\n## What the policy did (`sched_pick` reasons, counts in the windows)\n")
    rows = []
    for c in all_cases:
        for a in arms:
            m = M.get((a, c))
            if m and m["reasons"]:
                rows.append([c, a] + [", ".join(f"{k}: {v}" for k, v in sorted(m["reasons"].items(), key=lambda kv: -kv[1]))])
    print(table(["case", "arm", "reasons"], rows))


if __name__ == "__main__":
    main()
