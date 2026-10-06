# K3 decode split-KV microbenchmark: summary

GPU `NVIDIA H100 80GB HBM3` sm90, torch 2.12.1+cu132, flashinfer 0.7.0.post1, host brev-tffbb7a2o, job(s) 13074, 13083; 652 ok rows, 101 error rows.

Values are the median GPU time of one `run()` (one attention layer), us, KV streamed from HBM (128 MB memset before each call). `!` marks a result that differs from the fp32 reference by more than 0.05 (or NaN).

## Kernels launched by one eager `run()`

| geom | composition | backend | resolved backend | kernels |
|---|---|---|---|---|
| bagel | u8x512 | auto/default | fa2 | `void flashinfer::BatchPrefillWithPagedKVCacheKernel<true, flashinfer::KernelTraits<(flashinfer::MaskMode)0, 16u, 1u, 2u, 8u, 8u, 1u, 4u, (fl`<br>`void flashinfer::PersistentVariableLengthMergeStatesKernel<8u, 16u, 8u, 4u, __nv_bfloat16, __nv_bfloat16, int>(__nv_bfloat16*, float*, int*,` |
| bagel | u8x512 | fa2/default | fa2 | `void flashinfer::BatchPrefillWithPagedKVCacheKernel<true, flashinfer::KernelTraits<(flashinfer::MaskMode)0, 16u, 1u, 2u, 8u, 8u, 1u, 4u, (fl`<br>`void flashinfer::PersistentVariableLengthMergeStatesKernel<8u, 16u, 8u, 4u, __nv_bfloat16, __nv_bfloat16, int>(__nv_bfloat16*, float*, int*,` |
| bagel | u8x512 | fa3/default | fa3 | `void flashinfer::PrefillWithKVCacheKernel<flashinfer::SparseCollectiveMainloop<PagedParams::AdditionalParams, flashinfer::AttentionKernelTra` |
| | | | | ERROR RuntimeError: Error in function 'operator()' at /data/garv901-55613a/k3_fi_cache/.cache/flashinfer/0.7.0.post1 |
| bagel | s1x32768+7x512 | auto/default | fa2 | `void flashinfer::BatchPrefillWithPagedKVCacheKernel<true, flashinfer::KernelTraits<(flashinfer::MaskMode)0, 16u, 1u, 2u, 8u, 8u, 1u, 4u, (fl`<br>`void flashinfer::PersistentVariableLengthMergeStatesKernel<8u, 16u, 8u, 4u, __nv_bfloat16, __nv_bfloat16, int>(__nv_bfloat16*, float*, int*,` |
| bagel | s1x32768+7x512 | fa2/default | fa2 | `void flashinfer::BatchPrefillWithPagedKVCacheKernel<true, flashinfer::KernelTraits<(flashinfer::MaskMode)0, 16u, 1u, 2u, 8u, 8u, 1u, 4u, (fl`<br>`void flashinfer::PersistentVariableLengthMergeStatesKernel<8u, 16u, 8u, 4u, __nv_bfloat16, __nv_bfloat16, int>(__nv_bfloat16*, float*, int*,` |
| bagel | s1x32768+7x512 | fa3/default | fa3 | `void flashinfer::PrefillWithKVCacheKernel<flashinfer::SparseCollectiveMainloop<PagedParams::AdditionalParams, flashinfer::AttentionKernelTra` |
| | | | | ERROR RuntimeError: Error in function 'operator()' at /data/garv901-55613a/k3_fi_cache/.cache/flashinfer/0.7.0.post1 |

## bagel, eager mode (28 layers)

| composition | bs | auto/default | fa2/default | fa2/nosplit | fa2/fix2 | fa2/fix4 | fa2/fix8 | fa2/fix16 | fa2/fix32 | fa2/fix64 | fa3/default | best (any) | best split knob | auto/best knob | auto attn ms/step | share of step saved |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| u1x512 | 1 | 14 | 13 | 15 | 14 | 15 | 15 | 15 | 15 | 15 | 16 | fa2/default 13 | fa2/default 13 | 1.01x | 0.38 | n/a |
| u1x2048 | 1 | 15 | 15 | 31 | 16 | 19 | 23 | 31 | 31 | 31 | 36 | auto/default 15 | fa2/default 15 | 0.98x | 0.41 | n/a |
| u8x512 | 8 | 17 | 17 | 17 | 17 | 17 | 17 | 17 | 17 | 17 | 27 | fa2/fix16 17 | fa2/fix16 17 | 1.01x | 0.47 | n/a |
| u8x2048 | 8 | 30 | 30 | 37 | 30 | 29 | 31 | 37 | 37 | 37 | 67 | fa2/fix4 29 | fa2/fix4 29 | 1.02x | 0.83 | n/a |
| u32x512 | 32 | 31 | 31 | 26 | 31 | 26 | 26 | 26 | 26 | 26 | 77 | fa2/fix32 26 | fa2/fix32 26 | 1.18x | 0.86 | 0.8% of 16 ms |
| u32x2048 | 32 | 68 | 68 | 64 | 72 | 71 | 69 | 64 | 64 | 64 | 226 | fa2/fix64 64 | fa2/fix64 64 | 1.07x | 1.91 | 0.8% of 16 ms |
| u64x512 | 64 | 41 | 42 | 41 | 49 | 41 | 41 | 41 | 42 | 41 | 149 | fa2/fix4 41 | fa2/fix4 41 | 1.00x | 1.16 | 0.0% of 29 ms |
| u64x2048 | 64 | 108 | 108 | 108 | 121 | 119 | 117 | 108 | 108 | 108 | 456 | fa2/fix16 108 | fa2/fix16 108 | 1.00x | 3.02 | 0.0% of 29 ms |
| s1x32768+7x512 | 8 | 48 | 48 | 366 | 52 | 52 | 48 | 50 | 66 | 108 | 434 | auto/default 48 | fa2/default 48 | 1.00x | 1.34 | n/a |
| s1x32768+31x512 | 32 | 57 | 57 | 367 | 61 | 59 | 57 | 60 | 75 | 115 | 493 | auto/default 57 | fa2/default 57 | 1.00x | 1.58 | 0.0% of 16 ms |
| s1x32768+63x512 | 64 | 183 | 182 | 373 | 75 | 71 | 70 | 71 | 86 | 146 | 568 | fa2/fix8 70 | fa2/fix8 70 | 2.60x | 5.11 | 10.8% of 29 ms |
| s1x16384+31x512 | 32 | 44 | 44 | 199 | 48 | 44 | 46 | 52 | 72 | 113 | 284 | fa2/fix4 44 | fa2/fix4 44 | 1.01x | 1.23 | 0.1% of 16 ms |
| s2x32768+30x512 | 32 | 79 | 79 | 358 | 84 | 82 | 80 | 79 | 90 | 118 | 506 | fa2/fix16 79 | fa2/fix16 79 | 1.01x | 2.22 | 0.1% of 16 ms |
| s1x32768+15x512 | 16 | 51 | 51 | 364 | 55 | 53 | 51 | 53 | 68 | 109 | 456 | fa2/fix8 51 | fa2/fix8 51 | 1.01x | 1.43 | n/a |
| s1x32768+63x2048 | 64 | 231 | 231 | 424 | 145 | 140 | 139 | 140 | 140 | 194 | 886 | fa2/fix8 139 | fa2/fix8 139 | 1.66x | 6.47 | 8.8% of 29 ms |
| u8x32768 | 8 | 197 | 198 | 365 | 214 | 207 | 204 | 200 | 198 | 196 | 868 | fa2/fix64 196 | fa2/fix64 196 | 1.01x | 5.53 | n/a |
| u32x8192 | 32 | 199 | 198 | 194 | 215 | 208 | 205 | 201 | 199 | 194 | 782 | fa2/nosplit 194 | fa2/nosplit 194 | 1.03x | 5.56 | 0.9% of 16 ms |

Planner decisions and speed, bagel eager: auto/default (chunk = KV tokens per CTA, tiles = (row, chunk) pairs, CTAs = tiles x kv heads, max CTA = longest KV range one CTA walks, GB/s = KV bytes / time)

| composition | chunk tokens | tiles | max CTA tokens | padded batch | auto GB/s | plan CPU us | launch CPU us |
|---|---|---|---|---|---|---|---|
| u1x512 | 128 | 4 | 128 | 4 | 77 | 76 | 40 |
| u1x2048 | 128 | 16 | 128 | 16 | 283 | 79 | 38 |
| u8x512 | 128 | 32 | 128 | 32 | 496 | 93 | 40 |
| u8x2048 | 256 | 64 | 256 | 64 | 1137 | 92 | 37 |
| u32x512 | 256 | 64 | 256 | 64 | 1087 | 144 | 37 |
| u32x2048 | 1024 | 64 | 1024 | 64 | 1969 | 147 | 42 |
| u64x512 | 512 | 64 | 512 | 64 | 1621 | 214 | 40 |
| u64x2048 | 2048 | 64 | 2048 | 64 | 2488 | 215 | 39 |
| s1x32768+7x512 | 640 | 59 | 640 | 59 | 1553 | 93 | 40 |
| s1x32768+31x512 | 1024 | 63 | 1024 | 63 | 1761 | 148 | 38 |
| s1x32768+63x512 | 11008 | 66 | 11008 | 66 | 729 | 216 | 45 |
| s1x16384+31x512 | 512 | 63 | 512 | 63 | 1504 | 147 | 39 |
| s2x32768+30x512 | 1920 | 66 | 1920 | 66 | 2090 | 167 | 41 |
| s1x32768+15x512 | 768 | 58 | 768 | 58 | 1619 | 114 | 45 |
| s1x32768+63x2048 | 11008 | 66 | 11008 | 66 | 1435 | 223 | 48 |
| u8x32768 | 4096 | 64 | 4096 | 64 | 2720 | 106 | 46 |
| u32x8192 | 4096 | 64 | 4096 | 64 | 2702 | 150 | 45 |

## bagel, graph mode (28 layers)

| composition | bs | auto/default | fa2/default | fa2/nosplit | fa2/fix2 | fa2/fix4 | fa2/fix8 | fa2/fix16 | fa2/fix32 | fa2/fix64 | fa3/default | best (any) | best split knob | auto/best knob | auto attn ms/step | share of step saved |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| u1x512 | 1 | 16 | 16 | 18 | 17 | 18 | 18 | 19 | 19 | 19 | 17 | fa2/default 16 | fa2/default 16 | 1.01x | 0.44 | n/a |
| u1x2048 | 1 | 17 | 17 | 38 | 19 | 21 | 28 | 38 | 38 | 38 | 36 | auto/default 17 | fa2/default 17 | 1.00x | 0.49 | n/a |
| u8x512 | 8 | 19 | 19 | 19 | 20 | 19 | 19 | 20 | 19 | 19 | 28 | auto/default 19 | fa2/default 19 | 1.00x | 0.53 | n/a |
| u8x2048 | 8 | 31 | 30 | 42 | 31 | 32 | 34 | 42 | 42 | 42 | 68 | fa2/default 30 | fa2/default 30 | 1.02x | 0.86 | n/a |
| u32x512 | 32 | 32 | 32 | 28 | 32 | 28 | 28 | 28 | 28 | 28 | 77 | fa2/fix8 28 | fa2/fix8 28 | 1.15x | 0.90 | 0.7% of 16 ms |
| u32x2048 | 32 | 69 | 69 | 66 | - | - | 69 | 67 | 67 | 67 | 226 | fa2/nosplit 66 | fa2/nosplit 66 | 1.04x | 1.93 | 0.5% of 16 ms |
| u64x512 | 64 | 46 | 46 | 44 | - | 44 | 44 | 44 | 44 | 44 | 150 | fa2/fix32 44 | fa2/fix32 44 | 1.05x | 1.29 | 0.2% of 29 ms |
| u64x2048 | 64 | 112 | 112 | 115 | - | - | - | 115 | 115 | 115 | 460 | fa2/default 112 | fa2/default 112 | 1.00x | 3.15 | 0.0% of 29 ms |
| s1x32768+7x512 | 8 | 49 | 48 | 433 | - | - | 49 | 54 | 73 | 125 | 434 | fa2/default 48 | fa2/default 48 | 1.01x | 1.37 | n/a |
| s1x32768+31x512 | 32 | 57 | 57 | 410 | - | - | 57 | 62 | 82 | 133 | 496 | fa2/fix8 57 | fa2/fix8 57 | 1.00x | 1.60 | 0.0% of 16 ms |
| s1x32768+63x512 | 64 | 156 | 155 | 422 | - | - | - | 72 | 95 | 146 | 571 | fa2/fix16 72 | fa2/fix16 72 | 2.15x | 4.35 | 8.0% of 29 ms |
| s1x16384+31x512 | 32 | 45 | 45 | 230 | - | 44 | 46 | 55 | 80 | 132 | 284 | fa2/fix4 44 | fa2/fix4 44 | 1.02x | 1.25 | 0.1% of 16 ms |
| s2x32768+30x512 | 32 | 80 | 80 | 447 | - | - | - | 79 | 91 | 136 | 513 | fa2/fix16 79 | fa2/fix16 79 | 1.01x | 2.24 | 0.1% of 16 ms |
| s1x32768+15x512 | 16 | 52 | 52 | 429 | - | - | 51 | 56 | 76 | 127 | 460 | fa2/fix8 51 | fa2/fix8 51 | 1.01x | 1.44 | n/a |
| s1x32768+63x2048 | 64 | 208 | 208 | 484 | - | - | - | 142 | 144 | 180 | 888 | fa2/fix16 142 | fa2/fix16 142 | 1.47x | 5.82 | 6.4% of 29 ms |
| u8x32768 | 8 | 198 | 198 | 448 | - | - | - | - | 197 | 206 | 868 | fa2/fix32 197 | fa2/fix32 197 | 1.00x | 5.54 | n/a |
| u32x8192 | 32 | 199 | 199 | 203 | - | - | - | - | 199 | 207 | 788 | fa2/fix32 199 | fa2/fix32 199 | 1.00x | 5.57 | 0.1% of 16 ms |

Planner decisions and speed, bagel graph: auto/default (chunk = KV tokens per CTA, tiles = (row, chunk) pairs, CTAs = tiles x kv heads, max CTA = longest KV range one CTA walks, GB/s = KV bytes / time)

| composition | chunk tokens | tiles | max CTA tokens | padded batch | auto GB/s | plan CPU us | launch CPU us |
|---|---|---|---|---|---|---|---|
| u1x512 | 128 | 4 | 128 | 66 | 67 | 70 | 3 |
| u1x2048 | 128 | 16 | 128 | 66 | 242 | 76 | 4 |
| u8x512 | 128 | 32 | 128 | 66 | 440 | 89 | 3 |
| u8x2048 | 256 | 64 | 256 | 66 | 1087 | 104 | 3 |
| u32x512 | 256 | 64 | 256 | 66 | 1049 | 211 | 3 |
| u32x2048 | 1024 | 64 | 1024 | 66 | 1945 | 148 | 5 |
| u64x512 | 512 | 64 | 512 | 91 | 1461 | 251 | 3 |
| u64x2048 | 2048 | 64 | 2048 | 91 | 2387 | 247 | 4 |
| s1x32768+7x512 | 640 | 59 | 640 | 66 | 1527 | 139 | 5 |
| s1x32768+31x512 | 1024 | 63 | 1024 | 66 | 1742 | 155 | 3 |
| s1x32768+63x512 | 11008 | 66 | 11008 | 91 | 856 | 258 | 6 |
| s1x16384+31x512 | 512 | 63 | 512 | 66 | 1480 | 148 | 3 |
| s2x32768+30x512 | 1920 | 66 | 1920 | 66 | 2070 | 172 | 5 |
| s1x32768+15x512 | 768 | 58 | 768 | 66 | 1606 | 117 | 3 |
| s1x32768+63x2048 | 11008 | 66 | 11008 | 91 | 1593 | 224 | 3 |
| u8x32768 | 4096 | 64 | 4096 | 66 | 2713 | 102 | 6 |
| u32x8192 | 4096 | 64 | 4096 | 66 | 2697 | 156 | 4 |

## omni32, eager mode (48 layers)

| composition | bs | auto/default | fa2/default | fa2/nosplit | fa2/fix2 | fa2/fix4 | fa2/fix8 | fa2/fix16 | fa2/fix32 | fa2/fix64 | fa3/default | fa2_notc/default | best (any) | best split knob | auto/best knob | auto attn ms/step | share of step saved |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| u1x512 | 1 | 14 | 14 | 15 | 15 | 15 | 15 | 15 | 15 | 15 | 16 | 17 | fa2/default 14 | fa2/default 14 | 1.00x | 0.66 | n/a |
| u1x2048 | 1 | 15 | 15 | 31 | 16 | 19 | 24 | 31 | 31 | 31 | 36 | 18 | auto/default 15 | fa2/default 15 | 0.98x | 0.72 | n/a |
| u8x512 | 8 | 17 | 17 | 17 | 17 | 17 | 17 | 17 | 17 | 17 | 27 | 18 | fa2/fix64 17 | fa2/fix64 17 | 1.00x | 0.82 | n/a |
| u8x2048 | 8 | 30 | 30 | 37 | 30 | 29 | 31 | 37 | 38 | 37 | 67 | 33 | fa2/fix4 29 | fa2/fix4 29 | 1.02x | 1.43 | n/a |
| u32x512 | 32 | 31 | 31 | 26 | 31 | 26 | 26 | 26 | 26 | 26 | 87 | 34 | fa2/fix64 26 | fa2/fix64 26 | 1.20x | 1.51 | n/a |
| u32x2048 | 32 | 68 | 69 | 64 | 73 | 72 | 69 | 64 | 64 | 64 | 253 | 98 | fa2/nosplit 64 | fa2/nosplit 64 | 1.07x | 3.29 | n/a |
| u64x512 | 64 | 41 | 41 | 41 | 50 | 41 | 42 | 42 | 41 | 41 | 168 | 58 | fa2/nosplit 41 | fa2/nosplit 41 | 1.00x | 1.99 | n/a |
| u64x2048 | 64 | 108 | 108 | 108 | 124 | 121 | 118 | 108 | 108 | 108 | 516 | 186 | fa2/fix32 108 | fa2/fix32 108 | 1.00x | 5.19 | n/a |
| s1x32768+7x512 | 8 | 48 | 48 | 366 | 52 | 52 | 48 | 50 | 66 | 108 | 442 | 60 | auto/default 48 | fa2/fix8 48 | 1.00x | 2.30 | n/a |
| s1x32768+31x512 | 32 | 57 | 56 | 367 | 62 | 59 | 57 | 61 | 75 | 114 | 502 | 77 | fa2/default 56 | fa2/default 56 | 1.00x | 2.72 | n/a |
| s1x32768+63x512 | 64 | 183 | 182 | 373 | 75 | 71 | 70 | 72 | 86 | 146 | 597 | 101 | fa2/fix8 70 | fa2/fix8 70 | 2.61x | 8.79 | n/a |
| s1x16384+31x512 | 32 | 44 | 44 | 199 | 48 | 43 | 46 | 52 | 72 | 113 | 298 | 56 | fa2/fix4 43 | fa2/fix4 43 | 1.01x | 2.11 | n/a |
| s2x32768+30x512 | 32 | 79 | 79 | 358 | 85 | 82 | 80 | 79 | 89 | 119 | 521 | 138 | auto/default 79 | fa2/default 79 | 1.00x | 3.80 | n/a |
| s1x32768+15x512 | 16 | 51 | 51 | 367 | 55 | 53 | 51 | 52 | 69 | 109 | 462 | 67 | fa2/fix8 51 | fa2/fix8 51 | 1.01x | 2.46 | n/a |
| s1x32768+63x2048 | 64 | 232 | 232 | 426 | 148 | 143 | 140 | 141 | 139 | 196 | 966 | 263 | fa2/fix32 139 | fa2/fix32 139 | 1.66x | 11.12 | n/a |
| u8x32768 | 8 | 197 | 197 | 366 | 215 | 208 | 205 | 200 | 198 | 196 | 866 | 340 | fa2/fix64 196 | fa2/fix64 196 | 1.01x | 9.48 | n/a |
| u32x8192 | 32 | 199 | 199 | 193 | 215 | 209 | 206 | 202 | 198 | 193 | 898 | 348 | fa2/fix64 193 | fa2/fix64 193 | 1.03x | 9.54 | n/a |

Planner decisions and speed, omni32 eager: auto/default (chunk = KV tokens per CTA, tiles = (row, chunk) pairs, CTAs = tiles x kv heads, max CTA = longest KV range one CTA walks, GB/s = KV bytes / time)

| composition | chunk tokens | tiles | max CTA tokens | padded batch | auto GB/s | plan CPU us | launch CPU us |
|---|---|---|---|---|---|---|---|
| u1x512 | 128 | 4 | 128 | 4 | 77 | 72 | 42 |
| u1x2048 | 128 | 16 | 128 | 16 | 278 | 71 | 36 |
| u8x512 | 128 | 32 | 128 | 32 | 494 | 91 | 43 |
| u8x2048 | 256 | 64 | 256 | 64 | 1129 | 90 | 39 |
| u32x512 | 256 | 64 | 256 | 64 | 1068 | 124 | 32 |
| u32x2048 | 1024 | 64 | 1024 | 64 | 1960 | 145 | 42 |
| u64x512 | 512 | 64 | 512 | 64 | 1618 | 213 | 34 |
| u64x2048 | 2048 | 64 | 2048 | 64 | 2484 | 228 | 40 |
| s1x32768+7x512 | 640 | 59 | 640 | 59 | 1556 | 95 | 42 |
| s1x32768+31x512 | 1024 | 63 | 1024 | 63 | 1759 | 144 | 39 |
| s1x32768+63x512 | 11008 | 66 | 11008 | 66 | 727 | 218 | 42 |
| s1x16384+31x512 | 512 | 63 | 512 | 63 | 1504 | 154 | 40 |
| s2x32768+30x512 | 1920 | 66 | 1920 | 66 | 2094 | 156 | 38 |
| s1x32768+15x512 | 768 | 58 | 768 | 58 | 1615 | 122 | 41 |
| s1x32768+63x2048 | 11008 | 66 | 11008 | 66 | 1430 | 223 | 45 |
| u8x32768 | 4096 | 64 | 4096 | 64 | 2719 | 93 | 43 |
| u32x8192 | 4096 | 64 | 4096 | 64 | 2702 | 146 | 44 |

## omni32, graph mode (48 layers)

| composition | bs | auto/default | fa2/default | fa2/nosplit | fa2/fix2 | fa2/fix4 | fa2/fix8 | fa2/fix16 | fa2/fix32 | fa2/fix64 | fa3/default | fa2_notc/default | best (any) | best split knob | auto/best knob | auto attn ms/step | share of step saved |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| u1x512 | 1 | 16 | 16 | - | 17 | 18 | 18 | 18 | 18 | 18 | 17 | 21 | fa2/default 16 | fa2/default 16 | 1.01x | 0.75 | n/a |
| u1x2048 | 1 | 18 | 18 | 38 | 18 | 21 | 27 | 38 | 38 | 38 | 36 | 22 | fa2/default 18 | fa2/default 18 | 1.01x | 0.85 | n/a |
| u8x512 | 8 | 19 | 19 | 19 | 20 | 19 | 19 | 19 | 19 | 19 | 27 | 22 | fa2/default 19 | fa2/default 19 | 1.01x | 0.92 | n/a |
| u8x2048 | 8 | 31 | 30 | 42 | 31 | 31 | 34 | 42 | 42 | 42 | 67 | 38 | fa2/default 30 | fa2/default 30 | 1.02x | 1.49 | n/a |
| u32x512 | 32 | 32 | 32 | 28 | 32 | 28 | 28 | 28 | 28 | 28 | 87 | 39 | fa2/fix64 28 | fa2/fix64 28 | 1.15x | 1.54 | n/a |
| u32x2048 | 32 | 69 | 69 | 67 | - | - | 69 | 67 | 67 | 67 | 254 | 107 | fa2/nosplit 67 | fa2/nosplit 67 | 1.04x | 3.33 | n/a |
| u64x512 | 64 | 46 | 46 | 44 | - | 44 | 44 | 44 | 44 | 44 | 167 | 62 | fa2/fix32 44 | fa2/fix32 44 | 1.04x | 2.19 | n/a |
| u64x2048 | 64 | 112 | 112 | 117 | - | - | - | 116 | 115 | 116 | 518 | 204 | auto/default 112 | fa2/default 112 | 1.00x | 5.39 | n/a |
| s1x32768+7x512 | 8 | 48 | 49 | - | - | - | 48 | 53 | 73 | 124 | 442 | 63 | fa2/fix8 48 | fa2/fix8 48 | 1.00x | 2.33 | n/a |
| s1x32768+31x512 | 32 | 57 | 57 | 410 | - | - | 57 | 62 | 83 | 133 | 502 | 78 | fa2/default 57 | fa2/default 57 | 1.01x | 2.74 | n/a |
| s1x32768+63x512 | 64 | 155 | 155 | 427 | - | - | 71 | 73 | 87 | 127 | 597 | 110 | fa2/fix8 71 | fa2/fix8 71 | 2.18x | 7.46 | n/a |
| s1x16384+31x512 | 32 | 44 | 44 | 216 | - | 44 | 47 | 55 | 79 | 131 | 296 | 61 | fa2/fix4 44 | fa2/fix4 44 | 1.00x | 2.14 | n/a |
| s2x32768+30x512 | 32 | 80 | 80 | 446 | - | - | - | 80 | 91 | 136 | 517 | 130 | fa2/fix16 80 | fa2/fix16 80 | 1.00x | 3.83 | n/a |
| s1x32768+15x512 | 16 | 52 | 51 | 428 | - | - | 51 | 56 | 76 | 127 | 466 | 76 | fa2/fix8 51 | fa2/fix8 51 | 1.00x | 2.47 | n/a |
| s1x32768+63x2048 | 64 | 207 | 208 | 482 | - | - | - | 141 | 141 | 180 | 965 | 264 | fa2/fix32 141 | fa2/fix32 141 | 1.47x | 9.96 | n/a |
| u8x32768 | 8 | 198 | 198 | - | - | - | - | - | 198 | 204 | 865 | 341 | fa2/fix32 198 | fa2/fix32 198 | 1.00x | 9.52 | n/a |
| u32x8192 | 32 | 199 | 199 | 202 | - | - | - | - | 199 | 206 | 900 | 388 | fa2/fix32 199 | fa2/fix32 199 | 1.00x | 9.57 | n/a |

Planner decisions and speed, omni32 graph: auto/default (chunk = KV tokens per CTA, tiles = (row, chunk) pairs, CTAs = tiles x kv heads, max CTA = longest KV range one CTA walks, GB/s = KV bytes / time)

| composition | chunk tokens | tiles | max CTA tokens | padded batch | auto GB/s | plan CPU us | launch CPU us |
|---|---|---|---|---|---|---|---|
| u1x512 | 128 | 4 | 128 | 66 | 67 | 73 | 3 |
| u1x2048 | 128 | 16 | 128 | 66 | 237 | 82 | 2 |
| u8x512 | 128 | 32 | 128 | 66 | 436 | 103 | 3 |
| u8x2048 | 256 | 64 | 256 | 66 | 1084 | 99 | 3 |
| u32x512 | 256 | 64 | 256 | 66 | 1045 | 156 | 3 |
| u32x2048 | 1024 | 64 | 1024 | 66 | 1937 | 156 | 3 |
| u64x512 | 512 | 64 | 512 | 95 | 1471 | 220 | 2 |
| u64x2048 | 2048 | 64 | 2048 | 95 | 2393 | 235 | 3 |
| s1x32768+7x512 | 640 | 59 | 640 | 66 | 1536 | 101 | 3 |
| s1x32768+31x512 | 1024 | 63 | 1024 | 66 | 1743 | 160 | 4 |
| s1x32768+63x512 | 11008 | 66 | 11008 | 95 | 857 | 231 | 4 |
| s1x16384+31x512 | 512 | 63 | 512 | 66 | 1485 | 156 | 3 |
| s2x32768+30x512 | 1920 | 66 | 1920 | 66 | 2075 | 154 | 4 |
| s1x32768+15x512 | 768 | 58 | 768 | 66 | 1608 | 126 | 3 |
| s1x32768+63x2048 | 11008 | 66 | 11008 | 95 | 1597 | 244 | 4 |
| u8x32768 | 4096 | 64 | 4096 | 66 | 2708 | 99 | 4 |
| u32x8192 | 4096 | 64 | 4096 | 66 | 2692 | 139 | 3 |

## Skewed compositions where a split knob beats auto by >= 10%

| geom | mode | composition | best knob | auto us | knob us | speedup | saved per step | share |
|---|---|---|---|---|---|---|---|---|
| bagel | eager | s1x32768+63x512 | fa2/fix8 | 183 | 70 | 2.60x | 3.15 ms | 10.8% of 29 ms |
| bagel | eager | s1x32768+63x2048 | fa2/fix8 | 231 | 139 | 1.66x | 2.56 ms | 8.8% of 29 ms |
| bagel | graph | s1x32768+63x512 | fa2/fix16 | 156 | 72 | 2.15x | 2.33 ms | 8.0% of 29 ms |
| bagel | graph | s1x32768+63x2048 | fa2/fix16 | 208 | 142 | 1.47x | 1.85 ms | 6.4% of 29 ms |
| omni32 | eager | s1x32768+63x512 | fa2/fix8 | 183 | 70 | 2.61x | 5.43 ms | n/a |
| omni32 | eager | s1x32768+63x2048 | fa2/fix32 | 232 | 139 | 1.66x | 4.43 ms | n/a |
| omni32 | graph | s1x32768+63x512 | fa2/fix8 | 155 | 71 | 2.18x | 4.04 ms | n/a |
| omni32 | graph | s1x32768+63x2048 | fa2/fix32 | 207 | 141 | 1.47x | 3.19 ms | n/a |

## Configs that raised

| geom | mode | backend/variant | failed / run | error |
|---|---|---|---|---|
| bagel | * | cute-dsl/default | 1 / 1 | ValueError: cute-dsl paged decode supports page_size ∈ {8,16,32,64}, got 128 |
| bagel | * | cutile/default | 1 / 1 | ValueError: No valid config found in search space. Config: namespace(BLOCK_H=1, BLOCK_N=64, occupancy=1) FileNotFoundError: 'tileiras' compiler not found, make  |
| bagel | * | prims-ts/default | 1 / 1 | NotImplementedError: prims-ts decode backend requires kv_layout='HND' |
| bagel | * | trtllm-gen/default | 1 / 1 | RuntimeError: Error in function 'TllmGenFmhaRunner' at /shared/home/garv901-55613a/mstar-worktrees/k3-flashinfer-split/.venv/lib/python3.12/site-packages/flashi |
| bagel | eager | fa2_notc/default | 17 / 17 | RuntimeError: Error in function 'operator()' at /data/garv901-55613a/k3_fi_cache/.cache/flashinfer/0.7.0.post1/90a/generated/batch_decode_with_kv_cache_dtype_q_ |
| bagel | graph | fa2/fix16 | 2 / 17 | RuntimeError: Error in function 'PrefillSplitQOKVIndptr' at /shared/home/garv901-55613a/mstar-worktrees/k3-flashinfer-split/.venv/lib/python3.12/site-packages/f |
| bagel | graph | fa2/fix2 | 12 / 17 | RuntimeError: Error in function 'PrefillSplitQOKVIndptr' at /shared/home/garv901-55613a/mstar-worktrees/k3-flashinfer-split/.venv/lib/python3.12/site-packages/f |
| bagel | graph | fa2/fix4 | 10 / 17 | RuntimeError: Error in function 'PrefillSplitQOKVIndptr' at /shared/home/garv901-55613a/mstar-worktrees/k3-flashinfer-split/.venv/lib/python3.12/site-packages/f |
| bagel | graph | fa2/fix8 | 6 / 17 | RuntimeError: Error in function 'PrefillSplitQOKVIndptr' at /shared/home/garv901-55613a/mstar-worktrees/k3-flashinfer-split/.venv/lib/python3.12/site-packages/f |
| bagel | graph | fa2/nosplit | 1 / 18 | AcceleratorError: CUDA error: an illegal memory access was encountered Search for `cudaErrorIllegalAddress' in https://docs.nvidia.com/cuda/cuda-runtime-api/gro |
| bagel | graph | fa2_notc/default | 17 / 17 | RuntimeError: Error in function 'operator()' at /data/garv901-55613a/k3_fi_cache/.cache/flashinfer/0.7.0.post1/90a/generated/batch_decode_with_kv_cache_dtype_q_ |
| omni32 | graph | fa2/fix16 | 2 / 17 | RuntimeError: Error in function 'PrefillSplitQOKVIndptr' at /shared/home/garv901-55613a/mstar-worktrees/k3-flashinfer-split/.venv/lib/python3.12/site-packages/f |
| omni32 | graph | fa2/fix2 | 12 / 17 | RuntimeError: Error in function 'PrefillSplitQOKVIndptr' at /shared/home/garv901-55613a/mstar-worktrees/k3-flashinfer-split/.venv/lib/python3.12/site-packages/f |
| omni32 | graph | fa2/fix4 | 10 / 17 | RuntimeError: Error in function 'PrefillSplitQOKVIndptr' at /shared/home/garv901-55613a/mstar-worktrees/k3-flashinfer-split/.venv/lib/python3.12/site-packages/f |
| omni32 | graph | fa2/fix8 | 5 / 17 | RuntimeError: Error in function 'PrefillSplitQOKVIndptr' at /shared/home/garv901-55613a/mstar-worktrees/k3-flashinfer-split/.venv/lib/python3.12/site-packages/f |
| omni32 | graph | fa2/nosplit | 3 / 17 | hang: no progress for 180 s in stage 'graph warmup+capture', worker at 100% CPU, killed by the supervisor |

