"""Attention-kernel config-space microbenchmark (vLLM 0.30, H100).

Batch compositions x kernel configs, executed through vLLM's real attention
backend impls + metadata builders (reusing helpers from
third_party/vllm/benchmarks/attention_benchmarks/runner.py).

Timing: CUDA-graph replay, with a 128MB L2-flush memset before every call
(subtracted afterwards) so every call streams KV from HBM like successive
layers do in a real model.  Output rows are appended to a jsonl checkpoint.
"""
import argparse, json, math, os, random, sys, time, traceback
from pathlib import Path

os.environ.setdefault("VLLM_ALLOW_LONG_MAX_MODEL_LEN", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("VLLM_HAS_FLASHINFER_CUBIN", "1")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "third_party/vllm/benchmarks/attention_benchmarks"))

import numpy as np
import torch

from batch_spec import parse_batch_spec
from common import BenchmarkConfig, MockLayer
from runner import (_create_backend_impl, _create_kv_cache, _create_metadata_builder,
                    _get_backend_config)
import vllm.envs  # noqa
from vllm.config import (AttentionConfig, CacheConfig, CompilationConfig, DeviceConfig,
                         LoadConfig, ModelConfig, ParallelConfig, SchedulerConfig,
                         VllmConfig, set_current_vllm_config)
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.attention.backends.utils import get_supported_kv_cache_layouts, resolve_kv_cache_layout
from vllm.v1.kv_cache_interface import FullAttentionSpec

NQ, NKV, HD, BS = 32, 8, 128, 16
NUM_SM = 132
DEV = torch.device("cuda:0")
MODEL = os.environ.get("ATTN_MODEL", "Qwen/Qwen3-8B")
DT = torch.bfloat16
KV_BYTES_PER_TOKEN = NKV * HD * 2 * 2  # K+V bf16

CONFIGS = ["fa3_auto", "fa3_cg", "fa3_s1", "fa3_s2", "fa3_s4", "fa3_s8", "fa3_s16",
           "fi_xqa", "fi_native", "fi_native_nosplit", "fi_native_fs256", "fi_native_fs1024",
           "triton", "fa3_phase_split"]
ENGINE_OF = {"fa3": "FLASH_ATTN", "fi_xqa": "FI_XQA", "fi_native": "FI_NATIVE", "triton": "TRITON_ATTN"}


def engine_of(cfg):
    if cfg.startswith("fa3"):
        return "FLASH_ATTN"
    if cfg == "fi_xqa":
        return "FI_XQA"
    if cfg.startswith("fi_native"):
        return "FI_NATIVE"
    return "TRITON_ATTN"


BACKEND_OF = {"FLASH_ATTN": "FLASH_ATTN", "FI_XQA": "FLASHINFER", "FI_NATIVE": "FLASHINFER",
              "TRITON_ATTN": "TRITON_ATTN"}

# ---------------------------------------------------------------- specs

def Req(q, kv):
    return (q, kv)


def reqs_from_spec(s):
    return [(r.q_len, r.kv_len) for r in parse_batch_spec(s)]


def build_specs():
    specs = []  # (name, family, reqs)
    # F1 homogeneous decode
    for B in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
        for L in [512, 1024, 2048, 4096, 8192, 16384, 32768]:
            if B * L > 2_500_000:
                continue
            specs.append((f"{B}q1s{L}", "F1_homog", [(1, L)] * B))
    # F2 skewed decode
    for Ls in [542, 2048]:
        for ns in [7, 15, 31, 63, 127]:
            for nl in [1, 2, 4]:
                for Ll in [4512, 16384, 32768]:
                    specs.append((f"{ns}q1s{Ls}_{nl}q1s{Ll}", "F2_skew",
                                  [(1, Ls)] * ns + [(1, Ll)] * nl))
    # F3 mixed
    pf = [(q, q, f"1q{q}s{q}") for q in [256, 416, 512, 1024, 2048]] + [(512, 4608, "1q512s4608")]
    for q, kv, pn in pf:
        for ns in [15, 31, 63]:
            for nl, Ll in [(0, 0), (1, 4512), (1, 16384), (1, 32768)]:
                dec = [(1, 542)] * ns + [(1, Ll)] * nl
                name = f"{pn}_{ns}q1s542" + (f"_{nl}q1s{Ll}" if nl else "")
                specs.append((name, "F3_mixed", dec + [(q, kv)]))  # decodes first (vLLM order)
    # F4 steady-state long-heavy decode
    for oL in [4096, 16384]:
        for B in [16, 32, 64, 128]:
            rng = random.Random(1000 * oL + B)
            Ls = [rng.randint(512, 512 + oL) for _ in range(B)]
            specs.append((f"rand_B{B}_oL{oL // 1024}k", "F4_steady", [(1, L) for L in Ls]))
    # F5 single long sequence
    for L in [4096, 16384, 32768, 65536]:
        specs.append((f"single_1q1s{L}", "F5_single", [(1, L)]))
    return specs


def spec_stats(reqs):
    dec = [kv for q, kv in reqs if q == 1]
    pre = [(q, kv) for q, kv in reqs if q > 1]
    base = np.array(dec if dec else [kv for _, kv in reqs], dtype=np.float64)
    return dict(B_dec=len(dec), n_prefill=len(pre), q_total=sum(q for q, _ in reqs),
                num_actual_tokens=sum(q for q, _ in reqs),
                sumL=int(base.sum()), maxL=int(base.max()), meanL=float(base.mean()),
                cv=float(base.std() / base.mean()), skew=float(base.max() / base.mean()),
                occupancy=len(dec) * NKV / NUM_SM,
                prefill_q=max([q for q, _ in pre], default=0),
                prefill_ctx=max([kv - q for q, kv in pre], default=0),
                kv_tokens_total=sum(kv for _, kv in reqs))

# ---------------------------------------------------------------- engines

def make_vllm_config(num_blocks, use_trtllm=None, max_len=70000, cg=False):
    mc = ModelConfig(model=MODEL, tokenizer=MODEL, trust_remote_code=False, dtype="auto",
                     seed=0, max_model_len=max_len)
    assert mc.dtype == DT, mc.dtype
    cc = CacheConfig(block_size=BS, cache_dtype="auto")
    cc.num_gpu_blocks = num_blocks
    cc.num_cpu_blocks = 0
    sc = SchedulerConfig(max_num_seqs=512, max_num_batched_tokens=8192, max_model_len=max_len,
                         is_encoder_decoder=False, enable_chunked_prefill=True)
    import types
    mc.get_num_layers = types.MethodType(lambda self: 1, mc)
    mc.get_sliding_window_for_layer = types.MethodType(lambda self, i: None, mc)
    mc.get_logits_soft_cap_for_layer = types.MethodType(lambda self, i: 0.0, mc)
    mc.get_sm_scale_for_layer = types.MethodType(lambda self, i: 1.0 / HD ** 0.5, mc)
    mc.get_num_attention_heads = types.MethodType(lambda self, parallel_config=None: NQ, mc)
    mc.get_num_kv_heads = types.MethodType(lambda self, parallel_config=None: NKV, mc)
    mc.get_head_size = types.MethodType(lambda self: HD, mc)
    mc.get_sliding_window = types.MethodType(lambda self: None, mc)
    kw = {}
    if use_trtllm is not None:
        kw["attention_config"] = AttentionConfig(use_trtllm_attention=use_trtllm)
    return VllmConfig(model_config=mc, cache_config=cc, parallel_config=ParallelConfig(tensor_parallel_size=1),
                      scheduler_config=sc, device_config=DeviceConfig(), load_config=LoadConfig(),
                      compilation_config=(CompilationConfig() if cg else CompilationConfig(cudagraph_mode=CUDAGraphMode.NONE)), **kw)


class Engine:
    def __init__(self, name, num_blocks, caches, cg=False):
        self.name = name
        bname = BACKEND_OF[name]
        use_trtllm = {"FI_XQA": None, "FI_NATIVE": False}.get(name)
        self.cfg = make_vllm_config(num_blocks, use_trtllm, cg=cg)
        bc = BenchmarkConfig(backend=bname, batch_spec="", num_layers=1, head_dim=HD, num_q_heads=NQ,
                             num_kv_heads=NKV, block_size=BS, device="cuda:0", dtype=DT)
        with set_current_vllm_config(self.cfg):
            bcfg = _get_backend_config(bname)
            self.backend_class, self.impl, self.layer = _create_backend_impl(bcfg, bc, DEV, DT)
            supported = get_supported_kv_cache_layouts([self.backend_class])
            layout = resolve_kv_cache_layout(self.cfg, [[m.name for m in supported]])
            spec = FullAttentionSpec(block_size=BS, num_kv_heads=NKV, head_size=HD, dtype=DT)
            self.builder = _create_metadata_builder(self.backend_class, spec, self.cfg, DEV, bname)
            if bname not in caches:
                caches[bname] = _create_kv_cache(bc, num_blocks, self.backend_class, DEV, DT, layout)[0]
        self.cache = caches[bname]

    def ctx(self):
        return set_current_vllm_config(self.cfg)


def kv_views(cache, nb):
    """Return (K, V) views each logical (nb, BS, NKV, HD) for any vLLM KV layout."""
    if cache.dim() == 4 and cache.shape[-1] == 2 * HD:  # (nb, NKV, BS, 2*HD) strided: K|V concatenated
        p = cache.permute(0, 2, 1, 3)
        return p[..., :HD], p[..., HD:]
    shp = list(cache.shape)
    used, perm = set(), []
    for target in [2, nb, BS, NKV, HD]:
        for i, s_ in enumerate(shp):
            if i not in used and s_ == target:
                used.add(i); perm.append(i); break
        else:
            raise RuntimeError(f"cannot canonicalize cache shape {shp}")
    c = cache.permute(*perm)
    return c[0], c[1]


def fill_cache(cache, nb):
    K, V = kv_views(cache, nb)
    for start in range(0, nb, 4096):
        n = min(4096, nb - start)
        g = torch.Generator(device="cuda").manual_seed(1234 + start)
        d = torch.randn(2, n, BS, NKV, HD, generator=g, device="cuda", dtype=DT)
        K[start:start + n].copy_(d[0]); V[start:start + n].copy_(d[1])

# ---------------------------------------------------------------- batch metadata

class Batch:
    def __init__(self, reqs, perm, nb_total, seed=0):
        self.reqs = reqs
        qs = [q for q, _ in reqs]; kvs = [kv for _, kv in reqs]
        self.q_lens, self.kv_lens = qs, kvs
        nb_each = [(kv + BS - 1) // BS for kv in kvs]
        assert sum(nb_each) <= nb_total, (sum(nb_each), nb_total)
        maxb = max(nb_each)
        bt = torch.zeros(len(reqs), maxb, dtype=torch.int32)
        off = 0
        for i, n in enumerate(nb_each):
            bt[i, :n] = perm[off:off + n]; off += n
        self.block_table = bt.to(DEV)
        self.total_q = sum(qs)
        g = torch.Generator(device="cuda").manual_seed(seed)
        self.q = torch.randn(self.total_q, NQ, HD, generator=g, device="cuda", dtype=DT)
        self.k = torch.randn(self.total_q, NKV, HD, generator=g, device="cuda", dtype=DT)
        self.v = torch.randn(self.total_q, NKV, HD, generator=g, device="cuda", dtype=DT)
        self.out = torch.zeros(self.total_q, NQ, HD, device="cuda", dtype=DT)

    def common(self, lo=0, hi=None):
        hi = len(self.reqs) if hi is None else hi
        qs = self.q_lens[lo:hi]; kvs = self.kv_lens[lo:hi]
        qsl_cpu = torch.zeros(len(qs) + 1, dtype=torch.int32)
        qsl_cpu[1:] = torch.tensor(qs, dtype=torch.int32).cumsum(0)
        seq_cpu = torch.tensor(kvs, dtype=torch.int32)
        return CommonAttentionMetadata(
            query_start_loc=qsl_cpu.to(DEV), query_start_loc_cpu=qsl_cpu,
            seq_lens=seq_cpu.to(DEV), num_reqs=len(qs), num_actual_tokens=sum(qs),
            max_query_len=max(qs), max_seq_len=max(kvs),
            block_table_tensor=self.block_table[lo:hi],
            slot_mapping=torch.arange(sum(qs), dtype=torch.int64, device=DEV), causal=True,
            seq_lens_cpu_upper_bound=seq_cpu.clone())

# ---------------------------------------------------------------- FA helpers

def fa_metadata(eng, common, nsplits):
    """Build FA metadata the way vLLM does, with max_num_splits forced to nsplits
    (None -> leave builder default = eager/heuristic path), recomputing AOT scheduler metadata."""
    with eng.ctx():
        md = eng.builder.build(common_prefix_len=0, common_attn_metadata=common)
        assert md.max_num_splits == 0 and not eng.builder.use_full_cuda_graph
        if nsplits:
            md.scheduler_metadata = eng.builder._get_scheduler_metadata(
                aot_schedule=eng.builder.aot_schedule, batch_size=common.num_reqs,
                cu_query_lens=common.query_start_loc, max_query_len=common.max_query_len,
                seqlens=common.seq_lens, max_seq_len=common.max_seq_len, causal=True,
                max_num_splits=nsplits)
            md.max_num_splits = nsplits
    return md


def fa_split_info(md, common):
    sm = md.scheduler_metadata
    if sm is None:
        return "no_aot_md"
    B = common.num_reqs
    b_r = (B + 3) // 4 * 4
    nvec = (sm.shape[0] - 1) // b_r
    if nvec < 3:  # no dynamic split vector (prepare + [dyn] + swizzle)
        return (f"static_splits={md.max_num_splits}" if md.max_num_splits else "heuristic_static_bound=1")+";no_dyn_split_vec"
    ns = sm[b_r:b_r + B].cpu().numpy()
    kvs = common.seq_lens.cpu().numpy()
    il = int(kvs.argmax())
    return f"dyn_splits(max={ns.max()},longest={ns[il]},nsplit>1={int((ns>1).sum())}/{B})"

# ---------------------------------------------------------------- timing

_flush_buf = None
FLUSH_US = None
CGREF = None


def flush():
    _flush_buf.zero_()


def _capture(fn, R, with_flush):
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(R):
            if with_flush:
                flush()
            fn()
    return g


def _replay_times(g, R, retries):
    ts = []
    for _ in range(retries):
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record(); g.replay(); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e) * 1000.0 / R)  # us per rep
    return ts


def graph_time(fn, with_flush=True, retries=10, target_ms=12.0):
    fn(); torch.cuda.synchronize()
    g = _capture(fn, 4, with_flush); g.replay(); torch.cuda.synchronize()
    est = np.median(_replay_times(g, 4, 3))
    R = int(min(200, max(2, target_ms * 1000.0 / max(est, 1.0))))
    del g
    g = _capture(fn, R, with_flush); g.replay(); torch.cuda.synchronize()
    return _replay_times(g, R, retries)


def measure_flush():
    global FLUSH_US
    ts = graph_time(lambda: None, with_flush=True)
    FLUSH_US = float(np.median(ts))
    return FLUSH_US


def eager_time(fn):
    from triton.testing import do_bench
    flush_ = flush
    ms = do_bench(fn, warmup=10, rep=40, return_mode="median")
    return ms * 1000.0

# ---------------------------------------------------------------- run one (spec, config)

def run_config(cfg_name, eng, batch, stats, want_eager, want_warm):
    """Returns (fn, notes) after building metadata; caller times it."""
    notes = []
    layer, impl, cache = eng.layer, eng.impl, eng.cache
    q, k, v, out = batch.q, batch.k, batch.v, batch.out
    if cfg_name.startswith("fa3") and cfg_name != "fa3_phase_split":
        ns = None if cfg_name == "fa3_auto" else 32 if cfg_name == "fa3_cg" else int(cfg_name.split("_s")[1])
        common = batch.common()
        md = fa_metadata(eng, common, ns)
        notes.append(fa_split_info(md, common))
        if cfg_name == "fa3_cg" and CGREF is not None and stats["num_actual_tokens"] <= CGREF.builder.max_cudagraph_size:
            with CGREF.ctx():
                mdr = CGREF.builder.build(common_prefix_len=0, common_attn_metadata=common)
            ok = (mdr.max_num_splits == 32 and torch.equal(mdr.scheduler_metadata, md.scheduler_metadata))
            notes.append(f"vllm_cg_builder_md_match={ok}")
        elif cfg_name == "fa3_cg":
            notes.append("tokens>max_cg_size:real vLLM would use auto here")
        def fn():
            impl.forward(layer, q, k, v, cache, md, output=out)
    elif cfg_name == "fa3_phase_split":
        nd = stats["B_dec"]
        cd, cp = batch.common(0, nd), batch.common(nd, len(batch.reqs))
        md_d = fa_metadata(eng, cd, 32)
        md_p = fa_metadata(eng, cp, None)
        notes.append("dec:" + fa_split_info(md_d, cd) + "|pre:" + fa_split_info(md_p, cp))
        qd, kd, vd, od = q[:nd], k[:nd], v[:nd], out[:nd]
        qp, kp, vp, op = q[nd:], k[nd:], v[nd:], out[nd:]
        def fn():
            impl.forward(layer, qd, kd, vd, cache, md_d, output=od)
            impl.forward(layer, qp, kp, vp, cache, md_p, output=op)
    else:
        common = batch.common()
        b = eng.builder
        if cfg_name.startswith("fi_native"):
            b.disable_split_kv = cfg_name == "fi_native_nosplit"
            b.decode_fixed_split_size = {"fi_native_fs256": 256, "fi_native_fs1024": 1024}.get(cfg_name, -1)
        with eng.ctx():
            md = b.build(common_prefix_len=0, common_attn_metadata=common)
        if cfg_name.startswith("fi"):
            dec = getattr(md, "decode", None)
            pre = getattr(md, "prefill", None)
            dk = "none" if dec is None else (f"xqa" if type(dec).__name__ == "FlashInferTrtllmAPIDecode"
                                              and dec.kernel.value == "xqa" else type(dec).__name__)
            notes.append(f"decode_path={dk};prefill_path={'none' if pre is None else type(pre).__name__}"
                         f";use_xqa={b.use_xqa};disable_split_kv={b.disable_split_kv};fixed_split={b.decode_fixed_split_size}")
        def fn():
            with eng.ctx():
                impl.forward(layer, q, k, v, cache, md, output=out)
    return fn, notes


def summarize_times(ts):
    ts = np.array(ts) - FLUSH_US
    return float(np.median(ts)), float(np.mean(ts)), float(np.std(ts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results/microbench/attn_sweep.jsonl"))
    ap.add_argument("--families", default="")
    ap.add_argument("--configs", default=",".join(CONFIGS))
    ap.add_argument("--spec-filter", default="")
    ap.add_argument("--max-specs", type=int, default=0)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--eager-families", default="F2_skew,F3_mixed,F5_single")
    ap.add_argument("--warm-families", default="F5_single")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    specs = build_specs()
    if args.families:
        fams = args.families.split(",")
        specs = [s for s in specs if s[1] in fams]
    if args.spec_filter:
        specs = [s for s in specs if args.spec_filter in s[0]]
    specs = specs[::args.stride]
    if args.max_specs:
        specs = specs[:args.max_specs]
    configs = args.configs.split(",")
    eager_f = set(args.eager_families.split(",")); warm_f = set(args.warm_families.split(","))

    done = set()
    if args.resume and os.path.exists(args.out):
        for l in open(args.out):
            r = json.loads(l); done.add((r["spec"], r["config"]))
    fout = open(args.out, "a")

    nb_total = max(sum((kv + BS - 1) // BS for _, kv in s[2]) for s in specs) + 8
    print(f"specs={len(specs)} configs={configs} nb_total={nb_total} "
          f"({nb_total * BS * KV_BYTES_PER_TOKEN / 1e9:.1f} GB per cache)", flush=True)
    global _flush_buf
    _flush_buf = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=DEV)

    caches, engines = {}, {}
    need = sorted({engine_of(c) for c in configs})
    for en in need:
        engines[en] = Engine(en, nb_total, caches)
        print("engine", en, "ok", type(engines[en].builder).__name__, flush=True)
    global CGREF
    if "FLASH_ATTN" in engines:
        CGREF = Engine("FLASH_ATTN", nb_total, caches, cg=True)
        b = CGREF.builder
        print("CGREF builder: use_full_cuda_graph", b.use_full_cuda_graph, "max_cg_size", b.max_cudagraph_size,
              "max_num_splits", b.max_num_splits, "aot", b.aot_schedule, flush=True)
    for bname, c in caches.items():
        print("filling cache", bname, tuple(c.shape), c.stride(), flush=True)
        fill_cache(c, nb_total)
    torch.cuda.synchronize()
    perm = torch.randperm(nb_total, generator=torch.Generator().manual_seed(7), dtype=torch.int64).to(torch.int32)
    print("flush us (subtracted):", measure_flush(), flush=True)

    t_start = time.time()
    for si, (name, fam, reqs) in enumerate(specs):
        stats = spec_stats(reqs)
        todo = [c for c in configs if (name, c) not in done
                and not (c == "fa3_phase_split" and not (stats["B_dec"] > 0 and stats["n_prefill"] > 0))]
        if not todo:
            continue
        batch = Batch(reqs, perm, nb_total, seed=si)
        ref = None
        # reference first (fa3_auto) if available
        order = sorted(todo, key=lambda c: c != "fa3_auto")
        if "fa3_auto" not in todo and "fa3_auto" in configs:
            pass
        for cname in order:
            row = dict(spec=name, family=fam, config=cname, **stats)
            eng = engines[engine_of(cname)]
            try:
                batch.out.zero_()
                fn, notes = run_config(cname, eng, batch, stats, False, False)
                fn(); torch.cuda.synchronize()
                outc = batch.out.clone()
                if cname == "fa3_auto":
                    ref = outc
                if ref is not None:
                    row["maxdiff"] = float((outc.float() - ref.float()).abs().max())
                    if not math.isfinite(row["maxdiff"]) or row["maxdiff"] > 2e-2:
                        notes.append("MAXDIFF_FLAG")
                try:
                    ts = graph_time(fn)
                    med, mean, std = summarize_times(ts)
                except Exception as e:
                    notes.append(f"graph_failed:{type(e).__name__}:{str(e)[:120]}")
                    torch.cuda.synchronize()
                    from triton.testing import do_bench
                    med = mean = do_bench(fn, warmup=10, rep=40, return_mode="median") * 1000.0
                    std = None
                row.update(time_us_median=med, time_us_mean=mean, time_us_std=std)
                if fam in warm_f:
                    try:
                        tw = graph_time(fn, with_flush=False)
                        row["time_us_warm"] = float(np.median(tw))
                    except Exception as e:
                        notes.append(f"warm_failed:{type(e).__name__}")
                if fam in eager_f:
                    row["eager_time_us"] = eager_time(fn)
                row["bw_GBps"] = stats["kv_tokens_total"] * KV_BYTES_PER_TOKEN / (med * 1e-6) / 1e9
                row["notes"] = ";".join(notes)
            except Exception as e:
                row["notes"] = f"ERROR:{type(e).__name__}:{str(e)[:200]}"
                print("ERR", name, cname, row["notes"], flush=True)
                if "CUDA" in str(e) or "illegal" in str(e).lower():
                    traceback.print_exc()
                    torch.cuda.synchronize()
            fout.write(json.dumps(row) + "\n"); fout.flush()
        del batch, ref
        print(f"[{si+1}/{len(specs)}] {name} ({fam}) done t={time.time()-t_start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
