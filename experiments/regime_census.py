"""Stage 1b in-engine regime census: drive an in-process vLLM engine step by step so each probe step has exactly
the designed shape (experiments/census_shapes.py), and record whole-step GPU/CPU time with the step telemetry.

Per scenario: start the decodes (prompt = KV length, ignore_eos), prefill-cache the shared prefixes, run 8
decode-only steps, then add the probes one at a time (prompt = cached prefix[:P] + q fresh tokens, max_tokens 1),
3 repeats each. Probe and decode request ids are logged so analysis can join them to the telemetry rows.

Usage (one engine config per process):
  REGIME_TELEMETRY_DIR=<out>/telemetry python experiments/regime_census.py --config default|cap2048 --out <out>
"""
import argparse, hashlib, json, os, sys, time
from pathlib import Path

os.environ.setdefault("VLLM_ALLOW_LONG_MAX_MODEL_LEN", "1")
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from census_shapes import MAX_BATCHED, scenarios  # noqa: E402

MODEL = "meta-llama/Llama-3.1-8B-Instruct"
K = 1024
VOCAB_LO, VOCAB_HI = 1000, 120000


def toks(seed, n):
    return np.random.default_rng(seed).integers(VOCAB_LO, VOCAB_HI, n).tolist()


def seed_of(s):
    return int(hashlib.sha256(s.encode()).hexdigest()[:15], 16)


class Driver:
    def __init__(self, llm, log):
        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt
        self.SP, self.TP = SamplingParams, TokensPrompt
        self.eng = llm.llm_engine
        self.ntok, self.done, self.log, self.known = {}, set(), log, set()
        self.nsteps = 0

    def add(self, rid, ids, max_tokens, **meta):
        sp = self.SP(max_tokens=max_tokens, ignore_eos=True, temperature=0.0, detokenize=False)
        self.eng.add_request(rid, self.TP(prompt_token_ids=ids), sp)
        self.known.add(rid)
        self.log.write(json.dumps(dict(rid=rid, n_prompt=len(ids), t_add=time.time(), **meta)) + "\n")

    def step(self):
        for o in self.eng.step():
            rid = o.request_id if o.request_id in self.known else o.request_id.rsplit("-", 1)[0]
            self.ntok[rid] = len(o.outputs[0].token_ids) if o.outputs else 0
            if o.finished:
                self.done.add(rid)
        self.nsteps += 1

    def run_until(self, pred, max_steps, what):
        for _ in range(max_steps):
            if pred():
                return
            self.step()
        raise RuntimeError(f"timeout waiting for {what}")

    def drain(self):
        for _ in range(200):
            if not self.eng.has_unfinished_requests():
                return
            self.step()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", choices=["default", "cap2048"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--scenarios", default="", help="comma-separated subset (default: all)")
    ap.add_argument("--max-probes", type=int, default=0, help="smoke test: probes per scenario")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    assert os.environ.get("REGIME_TELEMETRY_DIR"), "set REGIME_TELEMETRY_DIR"

    from vllm import LLM
    kw = dict(model=MODEL, max_model_len=262144, enable_prefix_caching=True, gpu_memory_utilization=0.92, seed=0,
              max_num_batched_tokens=MAX_BATCHED, max_num_seqs=1024)
    if a.config == "cap2048":
        kw["compilation_config"] = {"max_cudagraph_capture_size": 2048}
    t0 = time.time()
    llm = LLM(**kw)
    print(f"engine up in {time.time() - t0:.0f}s", flush=True)
    log = open(out / "probes.jsonl", "a", buffering=1)
    log.write(json.dumps(dict(kind="meta", config=a.config, engine_kw={k: str(v) for k, v in kw.items()},
                              loadavg=os.getloadavg(), t=time.time())) + "\n")
    d = Driver(llm, log)
    prefix = {"A": toks(1, 128 * K), "B": toks(2, 128 * K)}
    want = set(a.scenarios.split(",")) if a.scenarios else None

    for scen, dec, probes, pairs in scenarios():
        if want and scen not in want:
            continue
        if a.max_probes:
            probes = probes[:: max(1, len(probes) // a.max_probes)][: a.max_probes]
            pairs = pairs[:1]
        ts = time.time()
        dec_ids = [f"dec|{scen}|{i}" for i in range(len(dec))]
        for i, L in enumerate(dec):
            d.add(dec_ids[i], toks(seed_of(f"{scen}|{i}"), L), 4000, kind="dec", scen=scen, L=L)
        warm = []
        pmax = max([P for P, _ in probes] + ([128 * K] if pairs else []), default=0)
        if pmax:
            warm.append(f"warm|{scen}|A")
            d.add(warm[-1], prefix["A"][:pmax], 1, kind="warm", scen=scen, P=pmax)
        if pairs:
            warm.append(f"warm|{scen}|B")
            d.add(warm[-1], prefix["B"], 1, kind="warm", scen=scen, P=128 * K)
        d.run_until(lambda: all(d.ntok.get(r, 0) >= 1 for r in dec_ids) and all(w in d.done for w in warm),
                    5000, f"{scen} setup")
        log.write(json.dumps(dict(kind="mark", scen=scen, what="decode_only_start", step=d.nsteps)) + "\n")
        for _ in range(8):
            d.step()
        log.write(json.dumps(dict(kind="mark", scen=scen, what="probes_start", step=d.nsteps)) + "\n")
        for rep in range(a.reps):
            for P, q in probes:
                rid = f"probe|{scen}|P{P}|q{q}|r{rep}"
                d.add(rid, prefix["A"][:P] + toks(seed_of(rid), q), 1, kind="probe", scen=scen, P=P, q=q, rep=rep)
                d.run_until(lambda: rid in d.done, 50, rid)
            for q1, q2 in pairs:
                r1, r2 = f"pair|{scen}|{q1}+{q2}|a|r{rep}", f"pair|{scen}|{q1}+{q2}|b|r{rep}"
                d.add(r1, prefix["A"] + toks(seed_of(r1), q1), 1, kind="pair", scen=scen, P=128 * K, q=q1, rep=rep,
                      pair=[q1, q2])
                d.add(r2, prefix["B"] + toks(seed_of(r2), q2), 1, kind="pair", scen=scen, P=128 * K, q=q2, rep=rep,
                      pair=[q1, q2])
                d.run_until(lambda: r1 in d.done and r2 in d.done, 50, r1)
        if dec_ids:
            d.eng.abort_request(dec_ids)
        d.drain()
        print(f"[{a.config}] {scen}: {len(probes)} probes x {a.reps} + {len(pairs)} pairs in {time.time() - ts:.0f}s "
              f"(steps so far {d.nsteps}, load {os.getloadavg()[0]:.0f})", flush=True)
    log.write(json.dumps(dict(kind="end", loadavg=os.getloadavg(), t=time.time())) + "\n")
    log.close()


if __name__ == "__main__":
    main()
