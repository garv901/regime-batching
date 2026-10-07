"""Stage 1b regime census: the step shapes measured in-engine (experiments/regime_census.py) and at kernel level
(microbench/attn_sweep.py family F7). One definition so both jobs measure the same steps.

A scenario is a set of running decodes plus a list of prefill probes. A probe is one request of q new tokens on P
cached tokens (P comes from a shared, prefix-cached token sequence), or a pair of such requests in one step.
Decodes come first in a step (vLLM schedules running requests before waiting ones).
"""
K = 1024
KV_BUDGET = 420_000  # tokens; vLLM holds 443,184 for Llama-3.1-8B at max-model-len 262144, util 0.92 (job 1442)
MAX_BATCHED = 8192   # vLLM default max_num_batched_tokens (server meta)

# name -> list of (count, KV length)
DECODE_SETS = {
    "none": [],
    "1x128k": [(1, 128 * K)],
    "3x128k": [(3, 128 * K)],
    "4x24k": [(4, 24 * K)],
    "31x1k+1x128k": [(31, 1 * K), (1, 128 * K)],
    "32x1k": [(32, 1 * K)],
    "8x8k": [(8, 8 * K)],
    "128x2k": [(128, 2 * K)],
    "256x1k": [(256, 1 * K)],
}
PREFIXES = [0, 8 * K, 32 * K, 128 * K]
Q_BASE = [32, 256, 416, 417, 1024, 1057, 2048, 4096]
Q_NONE_EXTRA = [16, 64, 128, 544, 1056]  # H5 remainders and wave edges, prefill alone
PAIRS = [(512, 512), (544, 512), (513, 513), (528, 528), (1056, 32)]  # two appends on two 128k prefixes


def decode_lens(name):
    return [L for n, L in DECODE_SETS[name] for _ in range(n)]


def q_list(name):
    B = len(decode_lens(name))
    qs = set(Q_BASE) | {MAX_BATCHED - B}
    if 512 - B > 0:
        qs |= {512 - B, 513 - B}  # the step size crosses the CUDA-graph capture size (512)
    if name == "none":
        qs |= set(Q_NONE_EXTRA)
    return sorted(qs)


def prefix_list(name):
    dec = sum(decode_lens(name))
    growth = 150 * len(decode_lens(name))  # decodes grow ~1 token per step over a scenario
    return [P for P in PREFIXES if dec + growth + P + MAX_BATCHED <= KV_BUDGET]


def scenarios():
    """[(scenario, decode KV lengths, [(P, q), ...], [pairs])]"""
    out = []
    for name in DECODE_SETS:
        probes = [(P, q) for P in prefix_list(name) for q in q_list(name)]
        out.append((name, decode_lens(name), probes, PAIRS if name == "none" else []))
    return out


def kernel_specs():
    """F7 kernel-level specs: [(spec name, reqs as [(q, kv)], meta)]."""
    specs, seen = [], set()

    def add(name, reqs, **meta):
        if name not in seen:
            seen.add(name)
            specs.append((name, reqs, meta))

    for scen, dec, probes, pairs in scenarios():
        d = [(1, L) for L in dec]
        if d:
            add(f"c_{scen}_dec", d, scen=scen, P=None, q=0)
        for P, q in probes:
            add(f"c_{scen}_P{P}_q{q}", d + [(q, P + q)], scen=scen, P=P, q=q)
        for q1, q2 in pairs:
            P = 128 * K
            add(f"c_{scen}_pair{q1}+{q2}_P{P}", d + [(q1, P + q1), (q2, P + q2)], scen=scen, P=P, q=q1 + q2,
                pair=[q1, q2])
    # gaps from Stage 1: small appends on long prefixes (the H5 remainder), pairs at 32k, long-context decode-only
    for P in [8 * K, 32 * K, 128 * K]:
        for q in [16, 32, 64, 96, 128, 192, 256, 320, 384]:
            add(f"c_none_P{P}_q{q}", [(q, P + q)], scen="none", P=P, q=q)
    for q1, q2 in PAIRS:
        P = 32 * K
        add(f"c_none_pair{q1}+{q2}_P{P}", [(q1, P + q1), (q2, P + q2)], scen="none", P=P, q=q1 + q2, pair=[q1, q2])
    for B in [1, 2, 4, 8, 16]:
        for L in [64 * K, 128 * K]:
            add(f"c_dec_{B}x{L // K}k", [(1, L)] * B, scen=f"dec_{B}x{L // K}k", P=None, q=0)
    return specs


if __name__ == "__main__":
    sc = scenarios()
    for name, dec, probes, pairs in sc:
        print(f"{name:14s} B={len(dec):3d} P={prefix_list(name)} q={q_list(name)} probes={len(probes)} pairs={len(pairs)}")
    print("in-engine probes:", sum(len(p) + len(pr) for _, _, p, pr in sc), "kernel specs:", len(kernel_specs()))
