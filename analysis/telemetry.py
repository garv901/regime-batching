"""Analysis + visualization of per-step regime telemetry (steps_<pid>.jsonl) and bench.json.

Library:
    load_run(case_dir)            -> (meta, steps, reqs)   warmup filtered, parquet-cached
    summarize(steps)              -> per-regime DataFrame (+ .attrs["overall"]); overall(steps) -> dict
    plot_case(case_dir, out_dir)  -> figures (PNG) + underlying CSVs
    compare(case_dirs, out_dir)   -> aligned ribbons, per-regime bars, TTFT/TPOT/E2E CDFs, markdown table
    export_perfetto(case_dir, out_json)  -> Chrome trace JSON (.gz if the path ends in .gz)
CLI:
    python analysis/telemetry.py run <run_dir>          # everything for every case under run_dir
    python analysis/telemetry.py case <case_dir> [out]  # one case
    python analysis/telemetry.py compare <run_dir>      # comparison figures only
Per-step tracks under the regime ribbon: "exec" (what executed: CUDA-graph mode + attention split config)
and, when sched_<pid>.jsonl exists (RegimeScheduler), "sched" (shape cap / gated admission / preemption).
compare() also writes compare_zoom.png (3 s window with the most baseline X steps, one rectangle per step).

Conventions / data notes
  * Warmup: vLLM dummy/warmup requests have ids starting with "_warmup_" (benchmark ids are
    "cmpl-bench-..."). Request rows with such ids are dropped; steps whose requests are all warmup
    are dropped. If a file has no per-request detail, leading steps with maxL <= 8 are dropped instead.
  * t_rel = seconds since the first benchmark step's t0. t0 is the CPU launch time of the step;
    the GPU interval [t0, t0 + gpu_ms] is therefore approximate (GPU may start later / overlap with
    the next step under async scheduling).
  * Request class (reqs.cls): prompt_len = (kv_len at the first decode step, q_len==1) - 1, falling
    back to kv_len at last appearance if the request never decodes; output_len = final_len - prompt_len;
    cls = "long" if output_len >= 1024 else "short". (Using kv at first appearance would mistake
    chunked prefill for output.) reqs.is_prefill = kv_len <= prompt_len (step still in prefill).
  * steps.exec = what executed (see EXEC_*): attn.path == "x_split" -> our eager+regime split; else cg.mode
    FULL / PIECEWISE; else eager, with fa3_path ("nosplit" = the FA3 1-split straggler path that stock vLLM's
    rule implies, "dyn<k>" = heuristic splits). fa3_path is a replica of stock vLLM's rule, not a measurement.
  * Scheduler join (load_sched/join_sched): each step takes the most recent sched record with t <= step.t0
    (merge_asof, backward) -> steps.sched_shape / sched_gated / sched_preempt / sched_admit. Approximation: with
    async scheduling schedule(n+1) is computed while step n runs, so a step whose launch is delayed past the next
    schedule() call can be paired with the following record; sched records without a step in between (idle
    calls) are not visible per step, so run-level counts (gated, preemptions) are taken from the sched file.
"""
from __future__ import annotations

import glob
import gzip
import json
import os
import re
import sys
from array import array

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.colors import to_rgb  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_ROOT = os.path.join(ROOT, "reports", "figures")

REGIMES = ["P", "M", "X", "Dw", "Dk", "Ds", "Dl", "Dc"]
REGIME_DESC = {
    "P": "prefill-heavy",
    "M": "mixed prefill+decode",
    "X": "mixed, bound by unsplit long-KV decode (FA3 straggler)",
    "Dw": "decode, weight-bound",
    "Dk": "decode, KV-bound",
    "Ds": "decode, skewed",
    "Dl": "decode, low-occupancy long-context",
    "Dc": "decode, large batch",
}
# Okabe-Ito based (colorblind-safe); fixed per regime across all figures.
REGIME_COLOR = {
    "P": "#D55E00", "M": "#E69F00", "X": "#CC79A7", "Dw": "#56B4E9",
    "Dk": "#0072B2", "Ds": "#009E73", "Dl": "#F0E442", "Dc": "#222222",
}
CASE_COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#222222", "#999999"]
IDLE_RGB = (0.94, 0.94, 0.94)
# exec-config labels (order = legend order) and colour-blind-safe palette, fixed across cases
EXEC_ORDER = ["full CUDA graph, FA3 splits<=32", "piecewise CUDA graph, FA3 splits<=32",
              "eager, FA3 heuristic splits", "eager, FA3 1 split (straggler)", "eager + regime split (C3)"]
EXEC_COLOR = dict(zip(EXEC_ORDER, ["#0072B2", "#56B4E9", "#E69F00", "#D55E00", "#009E73"]))
SCHED_COLOR = {"shape": "#7B3294", "gated": "#F0E442", "preempt": "#000000"}
LONG_OUT_TOKENS = 1024
CACHE_VERSION = 3


# ----------------------------------------------------------------------------- loading
def _step_files(case_dir):
    return sorted(glob.glob(os.path.join(case_dir, "telemetry", "steps_*.jsonl")))


def _parse_file(path):
    """Plain-list parse of one jsonl. Returns meta, step rows (dicts w/o reqs), per-request arrays."""
    meta, rows = {}, []
    rid_of, rid_names = {}, []
    a_step, a_rid = array("i"), array("i")
    a_q, a_kv, a_fin, a_rem = array("i"), array("i"), array("i"), array("i")
    with open(path) as f:
        for line in f:
            try:
                d = json.loads(line)
            except json.JSONDecodeError:  # partial last line of a still-running/killed server
                continue
            if d.get("kind") == "meta":
                meta = d
                continue
            if d.get("kind") != "step":
                continue
            reqs = d.pop("reqs", None)
            i = len(rows)
            cg, attn = d.pop("cg", None), d.pop("attn", None)
            d["cg_mode"] = cg["mode"] if cg else "NA"
            d["cg_num_tokens"] = cg["num_tokens"] if cg else -1
            d["cg_num_reqs"] = cg["num_reqs"] if cg else -1
            d["attn_path"] = attn.get("path", "") if attn else ""
            d["attn_splits"] = attn.get("splits", -1) if attn else -1
            d["attn_str"] = json.dumps(attn, separators=(",", ":"), sort_keys=True) if attn else ""
            d["has_reqs"] = reqs is not None
            nw = 0
            for r in reqs or ():
                name = r[0]
                k = rid_of.get(name)
                if k is None:
                    k = rid_of[name] = len(rid_names)
                    rid_names.append(name)
                a_step.append(i); a_rid.append(k)
                a_q.append(r[1]); a_kv.append(r[2]); a_fin.append(r[3]); a_rem.append(r[4])
                nw += name.startswith("_warmup")
            d["n_warm_reqs"] = nw
            rows.append(d)
    reqs = pd.DataFrame({
        "_i": np.frombuffer(a_step, dtype=np.int32), "rid": np.frombuffer(a_rid, dtype=np.int32),
        "q_len": np.frombuffer(a_q, dtype=np.int32), "kv_len": np.frombuffer(a_kv, dtype=np.int32),
        "final_len": np.frombuffer(a_fin, dtype=np.int32), "remaining": np.frombuffer(a_rem, dtype=np.int32),
    })
    return meta, pd.DataFrame(rows), reqs, np.array(rid_names, dtype=object)


def _build(case_dir):
    files = _step_files(case_dir)
    if not files:
        raise FileNotFoundError(f"no telemetry/steps_*.jsonl under {case_dir}")
    # one worker per file; with several (TP/DP) use the biggest (all ranks see the same schedule)
    path = max(files, key=os.path.getsize)
    meta, steps, reqs, names = _parse_file(path)
    meta["n_files"] = len(files)
    if steps.empty:
        return meta, steps, reqs.iloc[:0].assign(step=[], t_rel=[])
    # ---- warmup filtering
    if steps["has_reqs"].any():
        is_warm_name = np.fromiter((n.startswith("_warmup") for n in names), bool, len(names))
        reqs = reqs[~is_warm_name[reqs["rid"].to_numpy()]]
        keep = np.zeros(len(steps), bool)
        keep[np.unique(reqs["_i"].to_numpy())] = True
        n_warm = int((~keep).sum())
    else:
        lead = (steps["maxL"].to_numpy() <= 8)
        keep = ~(np.cumprod(lead).astype(bool))
        n_warm = int((~keep).sum())
    meta["n_warmup_steps_dropped"] = n_warm
    steps = steps[keep].copy()
    steps["_i_old"] = np.flatnonzero(keep)
    steps = steps.reset_index(drop=True)
    t_first = float(steps["t0"].iloc[0])
    meta["t_first"] = t_first
    steps["t_rel"] = steps["t0"] - t_first
    steps["t_end_rel"] = steps["t_rel"] + steps["gpu_ms"] / 1e3
    # ---- requests (long format)
    new_idx = np.full(keep.shape[0], -1, np.int64)
    new_idx[keep] = np.arange(keep.sum())
    reqs = reqs.assign(step=steps["step"].to_numpy()[new_idx[reqs["_i"].to_numpy()]],
                       t_rel=steps["t_rel"].to_numpy()[new_idx[reqs["_i"].to_numpy()]]).drop(columns="_i")
    codes, uniq = pd.factorize(reqs["rid"].to_numpy())
    reqs["rid"] = codes.astype(np.int32)
    names_u = names[uniq]
    dec = reqs["q_len"].to_numpy() == 1
    first_dec_kv = pd.Series(reqs["kv_len"].to_numpy()[dec] - 1).groupby(codes[dec]).first()
    last_kv = pd.Series(reqs["kv_len"].to_numpy()).groupby(codes).last()
    prompt = last_kv.copy()
    prompt.loc[first_dec_kv.index] = first_dec_kv
    p = prompt.reindex(np.arange(len(names_u))).to_numpy()
    fin = pd.Series(reqs["final_len"].to_numpy()).groupby(codes).last().reindex(np.arange(len(names_u))).to_numpy()
    reqs["prompt_len"] = p[codes].astype(np.int32)
    reqs["is_prefill"] = reqs["kv_len"].to_numpy() <= reqs["prompt_len"].to_numpy()
    reqs["cls"] = np.where((fin - p)[codes] >= LONG_OUT_TOKENS, "long", "short")
    reqs["req_id"] = pd.Categorical.from_codes(codes, categories=names_u) if len(names_u) else ""
    reqs = reqs.drop(columns="rid").reset_index(drop=True)
    steps = steps.drop(columns=["_i_old", "has_reqs"])
    steps["exec"] = exec_label(steps["cg_mode"], steps["fa3_path"], steps["attn_path"])
    return meta, steps, reqs


def exec_label(cg_mode, fa3_path, attn_path):
    """Vectorised exec-config label (see module docstring)."""
    cg, fa, at = (np.asarray(x) for x in (cg_mode, fa3_path, attn_path))
    return pd.Series(np.select([at == "x_split", cg == "FULL", cg == "PIECEWISE", fa == "nosplit"],
                               EXEC_ORDER[4:5] + EXEC_ORDER[:2] + EXEC_ORDER[3:4], EXEC_ORDER[2]), dtype=object)


# ----------------------------------------------------------------------------- scheduler records
def load_sched(case_dir, meta):
    """sched_<pid>.jsonl -> DataFrame (t, t_rel, call, admit, n_waiting, n_gated, shape, budget, n_new, n_preempt;
    .attrs['gated_unique'] = distinct gated request ids) restricted to t >= first benchmark step; None if absent."""
    files = glob.glob(os.path.join(case_dir, "telemetry", "sched_*.jsonl"))
    if not files:
        return None
    pid = str(meta.get("pid"))
    path = next((f for f in files if os.path.basename(f) == f"sched_{pid}.jsonl"), max(files, key=os.path.getsize))
    rows, gated = [], set()
    with open(path) as f:
        for line in f:
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            gated.update(d.pop("gated_ids", ()) or ())
            rows.append({k: d.get(k) for k in ("t", "call", "admit", "n_waiting", "n_gated", "shape", "budget", "n_new", "n_preempt")})
    df = pd.DataFrame(rows)
    if df.empty:
        return None
    df = df.sort_values("t").reset_index(drop=True)
    df["t_rel"] = df["t"] - meta["t_first"]
    df = df[df["t_rel"] >= -1.0].reset_index(drop=True)  # drop startup/warmup calls (1 s slack: schedule() precedes t0)
    df.attrs["gated_unique"] = len(gated)
    return df


def join_sched(steps, sched):
    """Add sched_shape/sched_gated/sched_preempt/sched_admit to steps (most recent sched record with t <= t0)."""
    steps = steps.copy()
    if sched is None or steps.empty:
        return steps
    left = steps[["t0"]].rename_axis("_ix").reset_index().sort_values("t0")
    m = pd.merge_asof(left, sched[["t", "shape", "n_gated", "n_preempt", "admit"]], left_on="t0", right_on="t",
                      direction="backward").set_index("_ix").reindex(steps.index)
    steps["sched_shape"] = m["shape"].fillna(False).astype(bool).to_numpy()
    steps["sched_gated"] = m["n_gated"].fillna(0).astype(int).to_numpy()
    steps["sched_preempt"] = m["n_preempt"].fillna(0).astype(int).to_numpy()
    steps["sched_admit"] = m["admit"].fillna("").to_numpy()
    return steps


def load_run(case_dir):
    """Return (meta, steps, reqs) for one case dir; cached as parquet in <case>/telemetry/.cache."""
    files = _step_files(case_dir)
    if not files:
        raise FileNotFoundError(f"no telemetry/steps_*.jsonl under {case_dir}")
    cdir = os.path.join(case_dir, "telemetry", ".cache")
    key = [CACHE_VERSION] + [[os.path.basename(f), os.path.getsize(f), int(os.path.getmtime(f))] for f in files]
    mp, sp, rp = (os.path.join(cdir, n) for n in ("meta.json", "steps.parquet", "reqs.parquet"))
    try:
        with open(mp) as f:
            m = json.load(f)
        if m.pop("_key") == key:
            return m, pd.read_parquet(sp), pd.read_parquet(rp)
    except Exception:
        pass
    meta, steps, reqs = _build(case_dir)
    try:
        os.makedirs(cdir, exist_ok=True)
        steps.to_parquet(sp); reqs.to_parquet(rp)
        with open(mp, "w") as f:
            json.dump({**meta, "_key": key}, f)
    except Exception as e:  # read-only fs etc.: caching is best effort
        print(f"[warn] cache not written: {e}", file=sys.stderr)
    return meta, steps, reqs


# ----------------------------------------------------------------------------- summary
def overall(steps):
    """Whole-run numbers. utilization = sum(gpu_ms) / wall span (can exceed 1 if steps overlap)."""
    wall = float(steps["t_end_rel"].max()) if len(steps) else 0.0
    busy = float(steps["gpu_ms"].sum()) / 1e3
    return {"n_steps": int(len(steps)), "gpu_busy_s": busy, "wall_s": wall,
            "utilization": busy / wall if wall > 0 else float("nan")}


def summarize(steps):
    """Per-regime table (index = regime, ordered like REGIMES) with overall stats in .attrs['overall']."""
    g = steps.groupby("regime")
    tot = steps["gpu_ms"].sum()
    out = pd.DataFrame({
        "n_steps": g.size(),
        "step_share": g.size() / max(len(steps), 1),
        "gpu_time_share": g["gpu_ms"].sum() / tot if tot else np.nan,
        "gpu_ms_mean": g["gpu_ms"].mean(),
        "gpu_ms_p50": g["gpu_ms"].median(),
        "gpu_ms_p99": g["gpu_ms"].quantile(0.99),
        "B_mean": g["num_reqs"].mean(),
        "T_mean": g["T"].mean(),
        "maxL_mean": g["maxL"].mean(),
    })
    for col, pre in (("cg_mode", "cg_"), ("fa3_path", "fa3_")):
        frac = pd.crosstab(steps["regime"], steps[col], normalize="index")
        frac.columns = [pre + str(c) for c in frac.columns]
        out = out.join(frac)
    out = out.reindex([r for r in REGIMES if r in out.index] + [r for r in out.index if r not in REGIMES])
    out.index.name = "regime"
    out.attrs["overall"] = overall(steps)
    return out


# ----------------------------------------------------------------------------- plotting helpers
def _regime_rgb(regimes):
    lut = {r: to_rgb(REGIME_COLOR.get(r, "#888888")) for r in set(regimes)}
    return np.array([lut[r] for r in regimes])


def _ribbon(ax, steps, t_max, width=2400, label=None):
    """Rasterised regime band: each pixel column takes the regime with the most GPU time in that bin."""
    ax.set_ylim(0, 1); ax.set_yticks([])
    img = np.tile(np.array(IDLE_RGB), (1, width, 1))
    if len(steps) and t_max > 0:
        b = np.minimum((steps["t_rel"].to_numpy() / t_max * width).astype(int), width - 1)
        ri = steps["regime"].map({r: i for i, r in enumerate(REGIMES)}).fillna(0).astype(int).to_numpy()
        w = np.zeros((width, len(REGIMES)))
        np.add.at(w, (b, ri), steps["gpu_ms"].to_numpy())
        has = w.sum(1) > 0
        win = w.argmax(1)
        cols = np.array([to_rgb(REGIME_COLOR[r]) for r in REGIMES])
        img[0, has] = cols[win[has]]
    ax.imshow(img, aspect="auto", extent=(0, t_max, 0, 1), interpolation="nearest")
    if label:
        ax.set_ylabel(label, rotation=0, ha="right", va="center", fontsize=9)


def _raster(ax, steps, t_max, rows, width=2400, label=None):
    """Thin raster track(s). rows = list of (per-step category index or -1, rgb palette, how); how 'major' = most
    GPU time in the pixel bin, 'any' = highest category present. One image row per entry."""
    ax.set_ylim(0, 1); ax.set_yticks([])
    img = np.tile(np.array(IDLE_RGB), (len(rows), width, 1))
    if len(steps) and t_max > 0:
        b = np.minimum((steps["t_rel"].to_numpy() / t_max * width).astype(int), width - 1)
        for k, (cat, pal, how) in enumerate(rows):
            ok = cat >= 0
            w = np.zeros((width, len(pal)))
            np.add.at(w, (b[ok], cat[ok]), steps["gpu_ms"].to_numpy()[ok] if how == "major" else 1.0)
            has = w.sum(1) > 0
            win = w.argmax(1) if how == "major" else np.where(w > 0, np.arange(len(pal)), -1).max(1)
            img[k, has] = np.asarray(pal)[win[has]]
    ax.imshow(img, aspect="auto", extent=(0, t_max, 0, 1), interpolation="nearest")
    if label:
        ax.set_ylabel(label, rotation=0, ha="right", va="center", fontsize=7)


def _exec_cat(steps):
    return steps["exec"].map({e: i for i, e in enumerate(EXEC_ORDER)}).fillna(-1).astype(int).to_numpy()


def _exec_track(ax, steps, t_max):
    _raster(ax, steps, t_max, [(_exec_cat(steps), [to_rgb(EXEC_COLOR[e]) for e in EXEC_ORDER], "major")], label="exec")


def _sched_track(ax, steps, sched, t_max):
    """Top half: shape cap active; bottom half: admission gating (any gated in the bin); black ticks: preemptions."""
    pal = [IDLE_RGB, to_rgb(SCHED_COLOR["shape"])]
    gpal = [IDLE_RGB, to_rgb(SCHED_COLOR["gated"])]
    _raster(ax, steps, t_max, [(steps["sched_shape"].astype(int).to_numpy(), pal, "major"),
                               ((steps["sched_gated"] > 0).astype(int).to_numpy(), gpal, "any")], label="sched")
    pre = sched.loc[sched["n_preempt"] > 0, "t_rel"].to_numpy()
    if len(pre):
        ax.vlines(pre, 0, 1, colors=SCHED_COLOR["preempt"], lw=0.8)
    ax.axhline(0.5, color="white", lw=0.5)


def _exec_legend(ax, has_sched, **kw):
    hs = [Patch(color=EXEC_COLOR[e], label=e) for e in EXEC_ORDER]
    if has_sched:
        hs += [Patch(color=SCHED_COLOR["shape"], label="sched: step budget capped to CG size (top)"),
               Patch(color=SCHED_COLOR["gated"], label="sched: admission gated (bottom)"),
               Patch(color=SCHED_COLOR["preempt"], label="sched: preemption (tick)")]
    ax.legend(handles=hs, fontsize=7.5, frameon=False, **kw)  # ax may be an Axes or a Figure


def _regime_legend(ax, present=None, **kw):
    hs = [Patch(color=REGIME_COLOR[r], label=f"{r}: {REGIME_DESC[r]}") for r in REGIMES
          if present is None or r in present]
    ax.legend(handles=hs, fontsize=7.5, frameon=False, **kw)


def _binned(steps, col, how, t_max, nb=1500):
    """Aggregate a step column into nb uniform time bins (for 1e5-step traces)."""
    b = np.minimum((steps["t_rel"].to_numpy() / t_max * nb).astype(int), nb - 1)
    s = steps[col].groupby(b).agg(how).reindex(np.arange(nb))
    return (np.arange(nb) + 0.5) * t_max / nb, s.to_numpy()


def _plain_log_ticks(*axes):
    for a in axes:
        for ax_ in (a.xaxis, a.yaxis):
            if ax_.get_scale() == "log":
                ax_.set_major_formatter(matplotlib.ticker.LogFormatter(labelOnlyBase=False, minor_thresholds=(1.2, 0.3)))
                ax_.set_minor_formatter(matplotlib.ticker.LogFormatter(labelOnlyBase=False, minor_thresholds=(1.2, 0.3)))


def _save(fig, path, df=None):
    _plain_log_ticks(*fig.axes)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    if df is not None:
        df.to_csv(os.path.splitext(path)[0] + ".csv", index=isinstance(df.index, pd.MultiIndex) or df.index.name is not None)


def _title(meta, case):
    m = os.path.basename(str(meta.get("model", "")).rstrip("/"))
    if len(m) > 20 and "snapshots" in str(meta.get("model", "")):
        m = str(meta["model"]).split("models--")[-1].split("/")[0].replace("--", "/")
    return f"{case}  [{m}, cg={meta.get('cudagraph_mode')}]"


# ----------------------------------------------------------------------------- per-case figures
def fig_timeline(steps, meta, case, out, sched=None):
    t_max = float(steps["t_end_rel"].max())
    ns = 1 if sched is not None else 0
    fig, axs = plt.subplots(6 + ns, 1, figsize=(13, 9.2), sharex=True,
                            gridspec_kw={"height_ratios": [0.5, 0.22] + [0.22] * ns + [1.4, 1, 1, 1], "hspace": 0.12})
    _ribbon(axs[0], steps, t_max)
    _exec_track(axs[1], steps, t_max)
    if ns:
        _sched_track(axs[2], steps, sched, t_max)
    axs, ax_trk = axs[1 + ns + 1:], axs[:1 + ns + 1]
    present = set(steps["regime"])
    t, nd = _binned(steps, "n_dec", "mean", t_max); _, npre = _binned(steps, "n_pre", "mean", t_max)
    axs[0].plot(t, nd, color="#0072B2", lw=1, label="n_dec"); axs[0].plot(t, npre, color="#D55E00", lw=1, label="n_pre")
    axs[0].set_ylabel("B (reqs)"); axs[0].legend(fontsize=7, frameon=False, ncol=2, loc="upper right")
    _, T = _binned(steps, "T", "mean", t_max)
    axs[1].plot(t, T, color="#444", lw=1); axs[1].set_ylabel("T (tokens)")
    _, mL = _binned(steps, "maxL", "max", t_max)
    axs[2].plot(t, mL, color="#009E73", lw=1); axs[2].set_ylabel("maxL")
    _, gm = _binned(steps, "gpu_ms", "mean", t_max); _, gx = _binned(steps, "gpu_ms", "max", t_max)
    axs[3].fill_between(t, gm, gx, color="#bbb", step="mid", label="bin max"); axs[3].plot(t, gm, color="#222", lw=1, label="bin mean")
    axs[3].set_yscale("log"); axs[3].set_ylabel("gpu_ms"); axs[3].legend(fontsize=7, frameon=False, ncol=2, loc="upper right")
    _regime_legend(ax_trk[0], present, loc="lower left", bbox_to_anchor=(0, 1.1), ncol=3)
    fig.suptitle("Regime timeline: " + _title(meta, case), fontsize=10, y=1.08)
    axs[3].yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    axs[3].yaxis.set_minor_formatter(matplotlib.ticker.ScalarFormatter())
    axs[3].set_xlabel("t_rel (s)")
    axs[3].xaxis.set_tick_params(labelbottom=True)
    _exec_legend(axs[3], ns > 0, loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=3)
    for a in axs:
        a.grid(alpha=0.3); a.margins(y=0.08)
    ax_trk[0].set_xlim(0, t_max)
    d = pd.DataFrame({"t_rel": t, "n_dec_mean": nd, "n_pre_mean": npre, "T_mean": T, "maxL_max": mL,
                      "gpu_ms_mean": gm, "gpu_ms_max": gx})
    _save(fig, os.path.join(out, "a_regime_timeline.png"), d.set_index("t_rel"))


def fig_fan(steps, meta, case, out):
    t_max = float(steps["t_end_rel"].max())
    d = steps[steps["n_dec"] > 0]
    fig, axs = plt.subplots(2, 1, figsize=(13, 6.5), sharex=True, gridspec_kw={"hspace": 0.1})
    t, mn = _binned(d, "minL", "min", t_max); _, mx = _binned(d, "maxL", "max", t_max)
    _, p50 = _binned(d, "p50L", "mean", t_max); _, mean = _binned(d, "meanL", "mean", t_max)
    ax = axs[0]
    ax.fill_between(t, mn, mx, color="#56B4E9", alpha=0.35, label="min..max")
    ax.plot(t, p50, color="#0072B2", lw=1, label="p50"); ax.plot(t, mean, color="#D55E00", lw=1, label="mean")
    ax.plot(t, mx, color="#222", lw=0.6, label="max"); ax.plot(t, mn, color="#777", lw=0.6, label="min")
    ax.set_yscale("log"); ax.set_ylabel("decode KV length (tokens)"); ax.legend(fontsize=7, frameon=False, ncol=5, loc="lower right", bbox_to_anchor=(1, 1.0))
    ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter()); ax.yaxis.set_minor_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_title("KV-length fan chart: " + _title(meta, case), fontsize=10, loc="left"); ax.grid(alpha=0.3)
    ax = axs[1]
    _, sl = _binned(d, "sumL", "mean", t_max); _, nl = _binned(d, "n_long", "max", t_max)
    ax.plot(t, sl, color="#009E73", lw=1, label="sumL (left)"); ax.set_ylabel("sumL (tokens)"); ax.set_xlabel("t_rel (s)")
    ax2 = ax.twinx(); ax2.step(t, nl, where="mid", color="#CC79A7", lw=1.2, label="n_long (right)")
    ax2.set_ylabel("n_long (long-lane reqs)"); ax2.set_ylim(bottom=0)
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7, frameon=False, loc="upper right"); ax.grid(alpha=0.3)
    ax.set_xlim(0, t_max)
    dd = pd.DataFrame({"t_rel": t, "minL": mn, "p50L_mean": p50, "meanL": mean, "maxL": mx, "sumL_mean": sl, "n_long_max": nl})
    _save(fig, os.path.join(out, "b_kv_fan.png"), dd.set_index("t_rel"))


def fig_breakdown(steps, summ, meta, case, out):
    s = summ.dropna(subset=["gpu_time_share"])
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [1, 1.7], "wspace": 0.25})
    ax = axs[0]
    x = np.array([0, 1]); bottom = np.zeros(2)
    for r in s.index:
        v = np.array([s.loc[r, "step_share"], s.loc[r, "gpu_time_share"]])
        ax.bar(x, v, 0.6, bottom=bottom, color=REGIME_COLOR.get(r, "#888"), edgecolor="white", lw=0.6)
        for xi, vi, bi in zip(x, v, bottom):
            if vi > 0.04:
                ax.text(xi, bi + vi / 2, f"{r} {vi:.0%}", ha="center", va="center", fontsize=8,
                        color="white" if r in ("Dk", "Dc", "Ds", "P", "X") else "black")
        bottom += v
    ax.set_xticks(x, ["share of steps", "share of GPU time"]); ax.set_ylim(0, 1); ax.set_ylabel("fraction")
    ax.set_title("Per-regime breakdown", fontsize=10)
    ax = axs[1]
    regs = [r for r in s.index if (steps["regime"] == r).sum() >= 1]
    data = [steps.loc[steps["regime"] == r, "gpu_ms"].to_numpy() for r in regs]
    pos = np.arange(len(regs))
    big = [i for i, d in enumerate(data) if len(d) >= 3]
    if big:
        vp = ax.violinplot([np.log10(data[i]) for i in big], positions=pos[big], showextrema=False, widths=0.8)
        for b, i in zip(vp["bodies"], big):
            b.set_facecolor(REGIME_COLOR.get(regs[i], "#888")); b.set_edgecolor("#333"); b.set_alpha(0.75)
    ax.boxplot([np.log10(d) for d in data], positions=pos, widths=0.18, showfliers=False,
               medianprops={"color": "k"}, boxprops={"color": "#333"}, whiskerprops={"color": "#333"}, capprops={"color": "#333"})
    ticks = np.array([1, 2, 5, 10, 20, 50, 100, 200, 500, 1000])
    lo, hi = np.log10(steps["gpu_ms"].min()), np.log10(steps["gpu_ms"].max())
    ticks = ticks[(np.log10(ticks) >= lo - 0.15) & (np.log10(ticks) <= hi + 0.15)]
    ax.set_yticks(np.log10(ticks), [str(t) for t in ticks]); ax.set_ylabel("step gpu_ms (log scale)")
    ax.set_xticks(pos, [f"{r}\n(n={len(d)})" for r, d in zip(regs, data)])
    ax.grid(axis="y", alpha=0.3); ax.set_title("Step-time distribution per regime", fontsize=10)
    fig.suptitle(_title(meta, case), fontsize=10, y=1.0)
    _regime_legend(ax, set(regs), loc="upper left", bbox_to_anchor=(1.01, 1.0))
    _save(fig, os.path.join(out, "c_regime_breakdown.png"), s)


def fig_gantt(steps, reqs, meta, case, out, cap=2000, seed=0):
    t_end = steps.set_index("step")["t_end_rel"]
    r = reqs.assign(t_end=reqs["step"].map(t_end))
    g = r.groupby("req_id", observed=True)
    life = g.agg(t_start=("t_rel", "min"), t_stop=("t_end", "max"), cls=("cls", "first"), prompt=("prompt_len", "first"))
    pre = r[r["is_prefill"]].groupby("req_id", observed=True)["t_end"].max()
    life["t_prefill_end"] = pre.reindex(life.index).fillna(life["t_start"])
    life = life.sort_values("t_start")
    n_total = len(life)
    if n_total > cap:
        life = life.iloc[np.sort(np.random.default_rng(seed).choice(n_total, cap, replace=False))]
    life = life.reset_index()
    y = np.arange(len(life))
    fig, ax = plt.subplots(figsize=(13, 6.5))
    col = np.where(life["cls"] == "long", "#D55E00", "#0072B2")
    lw = 1.2 if len(life) > 400 else 2.5
    segs = [[(a, i), (b, i)] for i, (a, b) in enumerate(zip(life["t_start"], life["t_stop"]))]
    ax.add_collection(LineCollection(segs, colors=col, linewidths=lw, alpha=0.8))
    segs = [[(a, i), (b, i)] for i, (a, b) in enumerate(zip(life["t_start"], life["t_prefill_end"]))]
    ax.add_collection(LineCollection(segs, colors="#222", linewidths=lw * 1.4))
    ax.set_xlim(0, float(steps["t_end_rel"].max())); ax.set_ylim(-1, len(life)); ax.invert_yaxis()
    ax.set_xlabel("t_rel (s)"); ax.set_ylabel("request (sorted by first scheduled step)")
    ax.legend(handles=[Patch(color="#0072B2", label=f"short (<{LONG_OUT_TOKENS} output tokens)"),
                       Patch(color="#D55E00", label=f"long (>={LONG_OUT_TOKENS} output tokens)"),
                       Patch(color="#222", label="prefill portion")], fontsize=8, frameon=False, loc="lower left")
    ax.set_title(f"Request Gantt: {_title(meta, case)}  ({len(life)} of {n_total} requests shown; "
                 f"{(life['cls'] == 'long').sum()} long)", fontsize=10)
    _save(fig, os.path.join(out, "d_request_gantt.png"), life)


def fig_scatter(steps, meta, case, out, cap=60000, seed=0):
    d = steps if len(steps) <= cap else steps.sample(cap, random_state=seed)
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.8), gridspec_kw={"wspace": 0.28})
    order = [r for r in REGIMES if r in set(d["regime"])]
    for ax, xc, xl in ((axs[0], "n_dec", "n_dec (decode reqs)"), (axs[1], "sumL", "sumL (total decode KV tokens)")):
        for r in order:
            m = d[d["regime"] == r]
            ax.scatter(m[xc].clip(lower=0.8), m["gpu_ms"], s=9, color=REGIME_COLOR[r], alpha=0.6, lw=0, rasterized=True)
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel(xl); ax.set_ylabel("step gpu_ms"); ax.grid(alpha=0.3)
    axs[0].set_title("gpu_ms vs decode batch", fontsize=10); axs[1].set_title("gpu_ms vs KV volume", fontsize=10)
    ax = axs[2]
    p = d[d["n_pre"] > 0]
    for r in order:
        m = p[p["regime"] == r]
        if len(m):
            ax.scatter(m["x_ratio"], m["gpu_ms"], s=12, color=REGIME_COLOR[r], alpha=0.7, lw=0, rasterized=True)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("x_ratio"); ax.set_ylabel("step gpu_ms"); ax.grid(alpha=0.3)
    ax.set_title(f"steps with prefill (n={len(p)}): x_ratio vs gpu_ms", fontsize=10)
    _regime_legend(axs[2], set(order), loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.suptitle(_title(meta, case) + ("" if len(steps) <= cap else f"  (random {cap} of {len(steps)} steps)"), fontsize=10, y=1.02)
    _save(fig, os.path.join(out, "e_feature_scatter.png"),
          d[["step", "regime", "gpu_ms", "n_dec", "n_pre", "sumL", "x_ratio"]].set_index("step"))


def plot_case(case_dir, out_dir, loaded=None):
    meta, steps, reqs = loaded or load_run(case_dir)
    case = os.path.basename(os.path.normpath(case_dir))
    os.makedirs(out_dir, exist_ok=True)
    sched = load_sched(case_dir, meta)
    steps = join_sched(steps, sched)
    summ = summarize(steps)
    summ.to_csv(os.path.join(out_dir, "summary_by_regime.csv"))
    fig_timeline(steps, meta, case, out_dir, sched)
    fig_fan(steps, meta, case, out_dir)
    fig_breakdown(steps, summ, meta, case, out_dir)
    if len(reqs):
        fig_gantt(steps, reqs, meta, case, out_dir)
    fig_scatter(steps, meta, case, out_dir)
    return summ


# ----------------------------------------------------------------------------- bench.json
def _bench(case_dir):
    """bench.json, or the bench.json.gz copy kept in git."""
    for name, op in (("bench.json", open), ("bench.json.gz", gzip.open)):
        p = os.path.join(case_dir, name)
        if os.path.exists(p):
            with op(p, "rt") as f:
                return json.load(f)
    return None


def _bench_exists(case_dir):
    return any(os.path.exists(os.path.join(case_dir, n)) for n in ("bench.json", "bench.json.gz"))


def bench_arrays(b):
    """Per-request ttft / tpot / e2e in ms (failed requests, i.e. ttft<=0 or error text, dropped)."""
    ttft, lat = np.array(b["ttfts"], float), np.array(b["latencies"], float)
    out = np.array(b["output_lens"], float)
    ok = (ttft > 0) & (lat > 0) & ~np.array([bool(e) for e in b.get("errors", [""] * len(ttft))])
    tpot = np.where(out > 1, (lat - ttft) / np.maximum(out - 1, 1), np.nan)
    return ttft[ok] * 1e3, tpot[ok] * 1e3, lat[ok] * 1e3


# ----------------------------------------------------------------------------- comparison
def _zoom_window(steps, win=3.0):
    """Start of the win-second window holding the most X steps (most steps if there is no X step)."""
    t = steps.loc[steps["regime"] == "X", "t_rel"].to_numpy()
    if not len(t):
        t = steps["t_rel"].to_numpy()
    t = np.sort(t)
    cnt = np.searchsorted(t, t + win, side="right") - np.arange(len(t))
    return float(t[int(cnt.argmax())]), int(cnt.max())


def _step_bars(ax, d, t0, colors, y=0.0, h=1.0):
    """One rectangle per step: x = t_rel, width = gpu_ms (true duration; may overlap the next step if async)."""
    for c in dict.fromkeys(colors):
        m = np.asarray(colors) == c
        ax.broken_barh(list(zip(d["t_rel"].to_numpy()[m], d["gpu_ms"].to_numpy()[m] / 1e3)), (y, h),
                       facecolors=c, edgecolors="white", linewidths=0.4)


def _fig_zoom(cases, steps_of, scheds, path, win=3.0):
    ws, nx = _zoom_window(steps_of[cases[0][0]], win)
    rat = [[1, 0.6] + ([0.6] if scheds[n] is not None else []) + [0.5] for n, _ in cases]
    rat = [r for rr in rat for r in rr][:-1]
    fig, axs = plt.subplots(len(rat), 1, figsize=(13, 0.5 * sum(rat) + 2.4), sharex=True,
                            gridspec_kw={"hspace": 0.12, "height_ratios": rat}, squeeze=False)
    axs = list(axs[:, 0]); k = 0
    for n, _ in cases:
        d = steps_of[n]; d = d[(d["t_end_rel"] >= ws) & (d["t_rel"] <= ws + win)]
        reg = [REGIME_COLOR.get(r, "#888") for r in d["regime"]]
        _step_bars(axs[k], d, ws, reg); axs[k].set_ylabel(f"{n}\n({len(d)} steps)", rotation=0, ha="right", va="center", fontsize=8)
        _step_bars(axs[k + 1], d, ws, [EXEC_COLOR.get(e, "#888") for e in d["exec"]]); axs[k + 1].set_ylabel("exec", rotation=0, ha="right", va="center", fontsize=7)
        k += 2
        if scheds[n] is not None:
            _step_bars(axs[k], d, ws, [SCHED_COLOR["shape"] if v else "#e8e8e8" for v in d["sched_shape"]], 0.5, 0.5)
            _step_bars(axs[k], d, ws, [SCHED_COLOR["gated"] if v > 0 else "#e8e8e8" for v in d["sched_gated"]], 0.0, 0.5)
            sc = scheds[n]; pre = sc.loc[(sc["n_preempt"] > 0) & (sc["t_rel"] >= ws) & (sc["t_rel"] <= ws + win), "t_rel"]
            axs[k].vlines(pre, 0, 1, colors=SCHED_COLOR["preempt"], lw=1.2)
            axs[k].set_ylabel("sched", rotation=0, ha="right", va="center", fontsize=7); k += 1
        if k < len(axs):
            axs[k].axis("off"); k += 1
    for a in axs:
        a.set_ylim(0, 1); a.set_yticks([])
    axs[0].set_xlim(ws, ws + win)
    axs[-1].set_xlabel("t_rel (s)")
    _regime_legend(axs[0], None, loc="lower left", bbox_to_anchor=(0, 1.03), ncol=3)
    _exec_legend(fig, any(v is not None for v in scheds.values()), loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=3)
    fig.suptitle(f"Zoom: {win:.0f} s window at t_rel={ws:.2f} s ({nx} X steps in baseline '{cases[0][0]}'); "
                 "each rectangle = one step, width = gpu_ms", fontsize=10, y=1 + 0.7 / fig.get_figheight())
    _save(fig, path)


def compare(case_dirs, out_dir, loaded=None):
    """Cross-case figures + markdown table. loaded: optional {case_name: (meta, steps, reqs)}."""
    os.makedirs(out_dir, exist_ok=True)
    loaded = dict(loaded or {})
    cases, skipped = [], {}
    for cd in case_dirs:
        name = os.path.basename(os.path.normpath(cd))
        try:
            loaded.setdefault(name, load_run(cd))
            cases.append((name, cd))
        except FileNotFoundError:
            skipped[name] = "no telemetry"
    if not cases:
        return None
    cmap = {n: CASE_COLORS[i % len(CASE_COLORS)] for i, (n, _) in enumerate(cases)}
    t_max = max(float(loaded[n][1]["t_end_rel"].max()) for n, _ in cases)
    # scheduler records joined onto steps (None when the case has no sched file)
    scheds = {n: load_sched(cd, loaded[n][0]) for n, cd in cases}
    steps_of = {n: join_sched(loaded[n][1], scheds[n]) for n, _ in cases}
    # aligned ribbons: per case regime + exec (+ sched) tracks
    rat = [[1, 0.45] + ([0.45] if scheds[n] is not None else []) + [0.35] for n, _ in cases]  # last = spacer
    rat = [r for rr in rat for r in rr][:-1]
    fig, axs = plt.subplots(len(rat), 1, figsize=(13, 0.38 * sum(rat) + 2.6), sharex=True,
                            gridspec_kw={"hspace": 0.12, "height_ratios": rat}, squeeze=False)
    axs = list(axs[:, 0]); k = 0
    for n, _ in cases:
        _ribbon(axs[k], steps_of[n], t_max, label=n); _exec_track(axs[k + 1], steps_of[n], t_max); k += 2
        if scheds[n] is not None:
            _sched_track(axs[k], steps_of[n], scheds[n], t_max); k += 1
        if k < len(axs):
            axs[k].axis("off"); k += 1
    axs[-1].set_xlabel("t_rel (s)")
    _regime_legend(axs[0], None, loc="lower left", bbox_to_anchor=(0, 1.03), ncol=3)
    _exec_legend(fig, any(v is not None for v in scheds.values()), loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=3)
    fig.suptitle("Regime / exec-config / scheduler tracks per step (aligned at first benchmark step)", fontsize=10, y=1 + 0.7 / fig.get_figheight())
    axs[0].set_xlim(0, t_max)
    _save(fig, os.path.join(out_dir, "compare_ribbons.png"))
    _fig_zoom(cases, steps_of, scheds, os.path.join(out_dir, "compare_zoom.png"))
    # per-regime GPU time (seconds and share)
    sums = {n: summarize(loaded[n][1]) for n, _ in cases}
    share = pd.DataFrame({n: s["gpu_time_share"] for n, s in sums.items()}).reindex(REGIMES).fillna(0)
    secs = pd.DataFrame({n: loaded[n][1].groupby("regime")["gpu_ms"].sum() / 1e3 for n, _ in cases}).reindex(REGIMES).fillna(0)
    fig, axs = plt.subplots(1, 2, figsize=(14, 4.2), gridspec_kw={"wspace": 0.2})
    w = 0.8 / len(cases); x = np.arange(len(REGIMES))
    for ax, df, yl in ((axs[0], share, "share of GPU time"), (axs[1], secs, "GPU time (s)")):
        for i, (n, _) in enumerate(cases):
            ax.bar(x + (i - (len(cases) - 1) / 2) * w, df[n], w, color=cmap[n], label=n)
        ax.set_xticks(x, REGIMES); ax.set_ylabel(yl); ax.grid(axis="y", alpha=0.3)
    axs[0].legend(fontsize=8, frameon=False)
    axs[0].set_title("Per-regime GPU time by case", fontsize=10)
    _save(fig, os.path.join(out_dir, "compare_regime_bars.png"), pd.concat({"share": share, "seconds": secs}, axis=1))
    # CDFs
    benches = {n: _bench(cd) for n, cd in cases}
    fig, axs = plt.subplots(1, 3, figsize=(15, 4), gridspec_kw={"wspace": 0.22})
    rows, cdf_rows = [], []
    for n, _ in cases:
        b = benches[n]
        if b is None:
            continue
        arrs = bench_arrays(b)
        for ax, a, lab in zip(axs, arrs, ("TTFT (ms)", "TPOT (ms)", "E2E latency (ms)")):
            a = np.sort(a[~np.isnan(a)])
            if len(a):
                ax.step(a, np.arange(1, len(a) + 1) / len(a), where="post", color=cmap[n], label=n, lw=1.5)
                cdf_rows += [(n, lab, v, (i + 1) / len(a)) for i, v in enumerate(a)]
            ax.set_xlabel(lab); ax.set_xscale("log"); ax.grid(alpha=0.3)
    axs[0].set_ylabel("CDF"); axs[0].legend(fontsize=8, frameon=False, loc="lower right")
    fig.suptitle("Per-request latency CDFs (bench.json)", fontsize=10, y=1.02)
    _save(fig, os.path.join(out_dir, "compare_latency_cdfs.png"),
          pd.DataFrame(cdf_rows, columns=["case", "metric", "value_ms", "cdf"]).set_index("case"))
    # table
    regs_present = [r for r in REGIMES if share.loc[r].sum() > 0]
    hdr = ["case", "reqs", "req/s", "out tok/s", "TTFT p50", "TTFT p99", "TPOT p50", "TPOT p99", "E2E p50", "E2E p99",
           "GPU busy s", "util"] + [f"{r} %GPU" for r in regs_present]
    lines = ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    for n, _ in cases:
        b, ov = benches[n], sums[n].attrs["overall"]
        if b is not None:
            tt, tp, e2 = bench_arrays(b)
            q = lambda a, p: f"{np.nanpercentile(a, p):.1f}" if len(a) else "n/a"
            vals = [f"{b.get('completed', len(tt))}", f"{b['request_throughput']:.2f}", f"{b['output_throughput']:.1f}",
                    q(tt, 50), q(tt, 99), q(tp, 50), q(tp, 99), q(e2, 50), q(e2, 99)]
        else:
            vals = ["n/a"] * 9
        vals += [f"{ov['gpu_busy_s']:.1f}", f"{ov['utilization']:.2f}"] + [f"{share.loc[r, n] * 100:.1f}" for r in regs_present]
        lines.append("| " + " | ".join([n] + vals) + " |")
    for n, why in skipped.items():
        lines.append(f"| {n} | " + " | ".join([why] + [""] * (len(hdr) - 2)) + " |")
    # exec-config share of GPU time + scheduler decisions
    ex = pd.DataFrame({n: steps_of[n].groupby("exec")["gpu_ms"].sum() / steps_of[n]["gpu_ms"].sum() for n, _ in cases}
                      ).reindex(EXEC_ORDER).fillna(0)
    hdr2 = ["case"] + [f"{e} %GPU" for e in EXEC_ORDER if ex.loc[e].sum() > 0] + \
           ["shape-cap steps", "gated calls", "gated reqs (unique)", "preemptions"]
    l2 = ["| " + " | ".join(hdr2) + " |", "|" + "---|" * len(hdr2)]
    for n, _ in cases:
        sc, d = scheds[n], steps_of[n]
        sv = [str(int(d["sched_shape"].sum())), str(int((sc["n_gated"] > 0).sum())), str(sc.attrs["gated_unique"]),
              str(int(sc["n_preempt"].sum()))] if sc is not None else ["n/a"] * 4
        l2.append("| " + " | ".join([n] + [f"{ex.loc[e, n] * 100:.1f}" for e in EXEC_ORDER if ex.loc[e].sum() > 0] + sv) + " |")
    ex.to_csv(os.path.join(out_dir, "compare_exec_share.csv"))
    md = "\n".join(lines) + "\n\n**GPU time by exec config, scheduler decisions**\n\n" + "\n".join(l2) + \
        "\n\nshape-cap steps = steps joined to a sched record with shape=true; gated calls = schedule() calls with n_gated>0; " \
        "preemptions = sum of n_preempt over sched records (n/a = no sched file).\n\nLatencies in ms. TPOT = (E2E - TTFT)/(output_len - 1) per request; percentiles over per-request values.\n"
    with open(os.path.join(out_dir, "compare_table.md"), "w") as f:
        f.write(md)
    return md


# ----------------------------------------------------------------------------- perfetto
def export_perfetto(case_dir, out_json, max_events=200_000, loaded=None):
    """Chrome trace (open in ui.perfetto.dev). ts = CPU launch time t0 (us from first benchmark step),
    dur = gpu_ms; X-event durations are clipped to the next step's start so the track stays well nested
    (true gpu_ms is kept in args). Counter tracks: B, T, sumL, maxL, gpu_ms, n_long. If there are more
    than max_events steps, steps are decimated by a uniform stride."""
    meta, steps, _ = loaded or load_run(case_dir)
    sched = load_sched(case_dir, meta)
    steps = join_sched(steps, sched)
    stride = max(1, int(np.ceil(len(steps) / max_events)))
    s = steps.iloc[::stride]
    ts = (s["t_rel"].to_numpy() * 1e6)
    dur = s["gpu_ms"].to_numpy() * 1e3
    dur = np.minimum(dur, np.append(np.diff(ts) - 1, dur[-1:])) if len(ts) > 1 else dur
    dur = np.maximum(dur, 1.0)
    PID, TID = 1, 1
    ev = [{"ph": "M", "pid": PID, "name": "process_name", "args": {"name": f"regime-batching {os.path.basename(case_dir)}"}},
          {"ph": "M", "pid": PID, "tid": TID, "name": "thread_name", "args": {"name": "regime"}}]
    cols = {k: s[k].to_numpy() for k in ("regime", "num_reqs", "n_dec", "n_pre", "T", "maxL", "sumL", "skew", "fa3_path",
                                         "cg_mode", "attn_str", "gpu_ms", "cpu_ms", "n_long", "step", "exec")}
    if sched is not None:
        cols.update({k: s[k].to_numpy() for k in ("sched_shape", "sched_gated", "sched_preempt")})
        ev.append({"ph": "M", "pid": PID, "tid": 3, "name": "thread_name", "args": {"name": "sched"}})
    ev.append({"ph": "M", "pid": PID, "tid": 2, "name": "thread_name", "args": {"name": "exec"}})
    for i in range(len(s)):
        args = {"step": int(cols["step"][i]), "B": int(cols["num_reqs"][i]), "n_dec": int(cols["n_dec"][i]),
                "n_pre": int(cols["n_pre"][i]), "T": int(cols["T"][i]), "maxL": int(cols["maxL"][i]),
                "skew": round(float(cols["skew"][i]), 3), "fa3_path": cols["fa3_path"][i], "cg": cols["cg_mode"][i],
                "attn": cols["attn_str"][i] or None, "gpu_ms": round(float(cols["gpu_ms"][i]), 3),
                "cpu_ms": round(float(cols["cpu_ms"][i]), 3), "exec": cols["exec"][i]}
        if sched is not None:
            args.update(shape_cap=bool(cols["sched_shape"][i]), n_gated=int(cols["sched_gated"][i]),
                        n_preempt=int(cols["sched_preempt"][i]))
        ev.append({"ph": "X", "pid": PID, "tid": TID, "name": cols["regime"][i], "cat": "regime",
                   "ts": round(float(ts[i]), 1), "dur": round(float(dur[i]), 1), "args": args})
        t = round(float(ts[i]), 1)
        ev.append({"ph": "X", "pid": PID, "tid": 2, "name": cols["exec"][i], "cat": "exec", "ts": t,
                   "dur": round(float(dur[i]), 1), "args": {"step": args["step"]}})
        if sched is not None:
            tags = [x for x, c in (("shape-cap", cols["sched_shape"][i]), ("gated", cols["sched_gated"][i] > 0),
                                   ("preempt", cols["sched_preempt"][i] > 0)) if c]
            ev.append({"ph": "X", "pid": PID, "tid": 3, "name": "+".join(tags) or "none", "cat": "sched", "ts": t,
                       "dur": round(float(dur[i]), 1), "args": {"step": args["step"], "shape_cap": args["shape_cap"],
                                                                "n_gated": args["n_gated"], "n_preempt": args["n_preempt"]}})
        for name, a in (("B", {"n_dec": int(cols["n_dec"][i]), "n_pre": int(cols["n_pre"][i])}),
                        ("T", {"T": int(cols["T"][i])}), ("sumL", {"sumL": float(cols["sumL"][i])}),
                        ("maxL", {"maxL": int(cols["maxL"][i])}), ("gpu_ms", {"gpu_ms": round(float(cols["gpu_ms"][i]), 3)}),
                        ("n_long", {"n_long": int(cols["n_long"][i])})):
            ev.append({"ph": "C", "pid": PID, "name": name, "ts": t, "args": a})
    doc = {"traceEvents": ev, "displayTimeUnit": "ms",
           "metadata": {"decimation_stride": stride, "note": "ts = CPU launch time of step; dur = CUDA-event gpu_ms (clipped)"}}
    os.makedirs(os.path.dirname(os.path.abspath(out_json)), exist_ok=True)
    op = gzip.open if out_json.endswith(".gz") else open
    with op(out_json, "wt") as f:
        json.dump(doc, f, separators=(",", ":"))
    return len(ev)


# ----------------------------------------------------------------------------- CLI
def _md_table(df, floatfmt):
    cols = list(df.columns)
    out = ["| regime | " + " | ".join(cols) + " |", "|" + "---|" * (len(cols) + 1)]
    for idx, row in df.iterrows():
        out.append(f"| {idx} | " + " | ".join(
            (f"{v:{floatfmt.get(c, '.2f')}}" if isinstance(v, (float, np.floating)) else str(v)) for c, v in row.items()) + " |")
    return "\n".join(out)


def _case_dirs(run_dir):
    return [d for d in sorted(glob.glob(os.path.join(run_dir, "*"))) if os.path.isdir(d)
            and (os.path.isdir(os.path.join(d, "telemetry")) or _bench_exists(d))]


def _fail_reason(case_dir):
    p = os.path.join(case_dir, "server.log")
    if not os.path.exists(p):
        return "no server.log"
    with open(p, errors="replace") as f:
        tail = f.readlines()[-40:]
    errs = [re.sub(r"^\(.*?\)\s*", "", l).strip() for l in tail if re.search(r"Error|Exception", l)]
    return errs[-1][:200] if errs else "no telemetry steps"


def run_all(run_dir):
    run_dir = os.path.abspath(run_dir)
    run = os.path.basename(run_dir.rstrip("/"))
    fig_dir = os.path.join(FIG_ROOT, run)
    os.makedirs(fig_dir, exist_ok=True)
    md = [f"# Telemetry summary: {run}\n",
          "Regimes: " + "; ".join(f"**{r}** {d}" for r, d in REGIME_DESC.items()) + ".\n"]
    loaded, ok_dirs = {}, []
    fmt = {"n_steps": ".0f", "step_share": ".3f", "gpu_time_share": ".3f", "gpu_ms_mean": ".2f", "gpu_ms_p50": ".2f",
           "gpu_ms_p99": ".2f", "B_mean": ".1f", "T_mean": ".1f", "maxL_mean": ".0f"}
    for cd in _case_dirs(run_dir):
        case = os.path.basename(cd)
        md.append(f"\n## {case}\n")
        try:
            loaded[case] = load_run(cd)
        except FileNotFoundError:
            md.append(f"No telemetry produced ({_fail_reason(cd)}).\n")
            print(f"[skip] {case}: no telemetry")
            continue
        meta, steps, reqs = loaded[case]
        if steps.empty:
            md.append("Telemetry has no benchmark steps.\n")
            continue
        ok_dirs.append(cd)
        out = os.path.join(fig_dir, case)
        summ = plot_case(cd, out, loaded[case])
        n_ev = export_perfetto(cd, os.path.join(out, "trace.json.gz"), loaded=loaded[case])
        ov = summ.attrs["overall"]
        b = _bench(cd)
        bn = f"; bench.json completed={b['completed']} (distinct request ids in telemetry: {reqs['req_id'].nunique()})" if b else "; no bench.json"
        md.append(f"- steps: {ov['n_steps']} (warmup steps dropped: {meta.get('n_warmup_steps_dropped')}), "
                  f"GPU busy {ov['gpu_busy_s']:.2f} s, wall {ov['wall_s']:.2f} s, utilization {ov['utilization']:.3f}{bn}")
        md.append(f"- cudagraph_mode={meta.get('cudagraph_mode')}, capture_max={meta.get('capture_max')}, "
                  f"cg null steps: {(steps['cg_mode'] == 'NA').sum()}\n")
        md.append(_md_table(summ, fmt) + "\n")
        md.append(f"Figures: `{case}/a_regime_timeline.png`, `b_kv_fan.png`, `c_regime_breakdown.png`, "
                  f"`d_request_gantt.png`, `e_feature_scatter.png`, `trace.json.gz` ({n_ev} events)\n")
        print(f"[ok] {case}: {ov['n_steps']} steps")
    if ok_dirs:
        table = compare(ok_dirs, os.path.join(fig_dir, "compare"), loaded)
        md.append("\n## Comparison\n\n" + (table or "") + "\n![ribbons](compare/compare_ribbons.png)\n")
    with open(os.path.join(fig_dir, "SUMMARY.md"), "w") as f:
        f.write("\n".join(md))
    print("wrote", os.path.join(fig_dir, "SUMMARY.md"))


def main(argv):
    if len(argv) < 3 or argv[1] not in ("run", "case", "compare"):
        print(__doc__)
        return 1
    if argv[1] == "run":
        run_all(argv[2])
    elif argv[1] == "case":
        cd = argv[2]
        out = argv[3] if len(argv) > 3 else os.path.join(FIG_ROOT, os.path.basename(os.path.dirname(os.path.abspath(cd))), os.path.basename(cd))
        plot_case(cd, out)
        export_perfetto(cd, os.path.join(out, "trace.json.gz"))
    else:
        rd = os.path.abspath(argv[2])
        print(compare([d for d in _case_dirs(rd)], os.path.join(FIG_ROOT, os.path.basename(rd), "compare")))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
