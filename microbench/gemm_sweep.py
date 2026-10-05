"""GEMM per-M regime sweep for the linear layers of Qwen3-8B / Qwen3-4B / Llama-3.1-8B on H100.

For each (shape, M): time torch.nn.functional.linear (cuBLASLt, what vLLM uses for bf16) under CUDA-graph
replay, report achieved TFLOP/s and GB/s (weights+activations), roofline fraction, and the cuBLAS kernel
name (tile config) chosen. Also times a Triton autotuned matmul as a "best alternative" probe at small M.

Usage: python microbench/gemm_sweep.py --out results/microbench/gemm_sweep.parquet
"""
from __future__ import annotations

import argparse
import os

import pandas as pd
import torch
import triton
import triton.language as tl

PEAK_BW = 3.35e12
PEAK_FLOPS = 989e12

SHAPES = {  # name: (N, K)  (y = x @ W^T, W: [N, K])
    "qwen3-8b.qkv": (6144, 4096), "qwen3-8b.o": (4096, 4096), "qwen3-8b.gate_up": (24576, 4096),
    "qwen3-8b.down": (4096, 12288), "qwen3-8b.lm_head": (151936, 4096),
    "qwen3-4b.qkv": (6144, 2560), "qwen3-4b.o": (2560, 4096), "qwen3-4b.gate_up": (19456, 2560),
    "qwen3-4b.down": (2560, 9728),
    "llama3.1-8b.gate_up": (28672, 4096), "llama3.1-8b.down": (4096, 14336),
}
MS = [1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 160, 192, 256, 320, 384, 512, 768, 1024, 2048, 4096, 8192]


@triton.autotune(
    configs=[triton.Config({"BM": bm, "BN": bn, "BK": bk, "SPLIT_K": sk}, num_warps=w, num_stages=s)
             for bm in (16, 32, 64) for bn in (64, 128, 256) for bk in (64, 128) for sk in (1, 2, 4)
             for w, s in ((4, 4), (8, 3)) if not (bm == 64 and bn == 256)],
    key=["M", "N", "K"], reset_to_zero=["C"],
)
@triton.jit
def _mm(A, B, C, M, N, K, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, SPLIT_K: tl.constexpr):
    pid = tl.program_id(0)
    pid_k = tl.program_id(1)
    num_n = tl.cdiv(N, BN)
    pm = pid // num_n
    pn = pid % num_n
    rm = pm * BM + tl.arange(0, BM)
    rn = pn * BN + tl.arange(0, BN)
    rk = pid_k * BK + tl.arange(0, BK)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BK * SPLIT_K)):
        kk = rk + k * BK * SPLIT_K
        a = tl.load(A + rm[:, None] * K + kk[None, :], mask=(rm[:, None] < M) & (kk[None, :] < K), other=0.0)
        b = tl.load(B + rn[None, :] * K + kk[:, None], mask=(rn[None, :] < N) & (kk[:, None] < K), other=0.0)
        acc += tl.dot(a, b)
    c = C + rm[:, None] * N + rn[None, :]
    mask = (rm[:, None] < M) & (rn[None, :] < N)
    if SPLIT_K == 1:
        tl.store(c, acc.to(tl.bfloat16), mask=mask)
    else:
        tl.atomic_add(c, acc.to(tl.bfloat16), mask=mask, sem="relaxed")


def triton_mm(a, w, c):
    M, K = a.shape
    N = w.shape[0]
    grid = lambda meta: (triton.cdiv(M, meta["BM"]) * triton.cdiv(N, meta["BN"]), meta["SPLIT_K"])
    _mm[grid](a, w, c, M, N, K)
    return c


def cublas_kernel_name(fn) -> str:
    from torch.profiler import ProfilerActivity, profile
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    names = [e.name for e in prof.events() if e.device_type.name == "CUDA"]
    return ";".join(sorted(set(names)))[:300]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/microbench/gemm_sweep.parquet")
    ap.add_argument("--shapes", nargs="*", default=list(SHAPES))
    ap.add_argument("--triton-max-m", type=int, default=256)
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    rows = []
    for name in args.shapes:
        N, K = SHAPES[name]
        w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.02
        for M in MS:
            x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
            fn = lambda: torch.nn.functional.linear(x, w)
            t_ms = triton.testing.do_bench_cudagraph(fn, rep=200)
            byts = (N * K + M * K + M * N) * 2
            flops = 2 * M * N * K
            row = dict(shape=name, N=N, K=K, M=M, impl="cublas", time_us=t_ms * 1e3,
                       tflops=flops / (t_ms * 1e-3) / 1e12, gbps=byts / (t_ms * 1e-3) / 1e9,
                       roofline_frac=max(byts / PEAK_BW, flops / PEAK_FLOPS) / (t_ms * 1e-3),
                       kernel=cublas_kernel_name(fn))
            rows.append(row)
            if M <= args.triton_max_m and N <= 32768:
                c = torch.zeros(M, N, device="cuda", dtype=torch.bfloat16)
                tfn = lambda: triton_mm(x, w, c)
                tfn()
                ref = fn().float()
                err = (c.float() - ref).abs().max().item() if True else 0.0
                t2 = triton.testing.do_bench_cudagraph(lambda: (c.zero_(), triton_mm(x, w, c)), rep=200)
                t_zero = triton.testing.do_bench_cudagraph(lambda: c.zero_(), rep=100)
                t2 = max(t2 - t_zero, 1e-6)
                rows.append(dict(shape=name, N=N, K=K, M=M, impl="triton_autotune", time_us=t2 * 1e3,
                                 tflops=flops / (t2 * 1e-3) / 1e12, gbps=byts / (t2 * 1e-3) / 1e9,
                                 roofline_frac=max(byts / PEAK_BW, flops / PEAK_FLOPS) / (t2 * 1e-3),
                                 kernel=str(_mm.best_config), maxerr=err))
            print(f"{name:22s} M={M:5d} cublas {row['time_us']:8.1f}us {row['gbps']:7.0f}GB/s "
                  f"{row['tflops']:6.1f}TF roof={row['roofline_frac']:.2f}  {row['kernel'][:60]}", flush=True)
        pd.DataFrame(rows).to_parquet(args.out)
    pd.DataFrame(rows).to_parquet(args.out)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
