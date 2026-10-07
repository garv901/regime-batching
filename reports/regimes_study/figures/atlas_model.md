
**decode only**

| shape | T | path | splits | regime_old | x_ratio_old | regime_new | t_default_us | t_best_us | s_best | x_cost | t_sep_us | sep_gain | pack_gain | sm_eff | structural_x |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 64 x 2k decodes (T <= 512, CUDA graph) | 64 | dyn32 | 32 | Dk | 1.0 | Dk | 197.2 | 197.2 | 1 | 1.0 | 197.2 | 1.0 | 1.0 | 0.97 | False |
| 31 x 542 + 1 x 4k (user's 31+1 example) | 32 | dyn32 | 32 | Ds | 1.0 | Ds | 41.1 | 41.1 | 2 | 1.0 | 41.1 | 1.0 | 1.0 | 0.52 | False |
| 3 x 128k decodes | 3 | dyn32 | 32 | Dl | 1.0 | Dl | 576.2 | 576.2 | 2 | 1.0 | 576.2 | 1.0 | 1.0 | 0.91 | False |

**short prefix (chat)**

| shape | T | path | splits | regime_old | x_ratio_old | regime_new | t_default_us | t_best_us | s_best | x_cost | t_sep_us | sep_gain | pack_gain | sm_eff | structural_x |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 512 chunk, 15 short decodes | 527 | nosplit | 1 | P | 1.07 | P | 36.0 | 36.0 | 1 | 1.0 | 41.2 | 0.87 | 1.15 | 0.84 | False |
| 512 chunk + 15 short + 1 x 32k (original X) | 528 | nosplit | 1 | X | 4.98 | X | 428.7 | 73.4 | 16 | 5.84 | 90.6 | 4.73 | 6.62 | 0.12 | True |
| 600 chunk + 4 x 24k (X-heavy workload) | 604 | nosplit | 1 | X | 2.14 | X | 325.1 | 149.6 | 4 | 2.17 | 183.0 | 1.78 | 2.21 | 0.29 | True |
| 416 chunk + 4 x 24k (just below 417) | 420 | dyn32 | 32 | P | 1.0 | P | 149.6 | 149.6 | 4 | 1.0 | 171.9 | 0.87 | 1.0 | 0.93 | False |
| 2048 chunk + 4 x 24k | 2052 | nosplit | 1 | X | 1.74 | X | 325.1 | 172.0 | 2 | 1.89 | 254.0 | 1.28 | 1.96 | 0.5 | True |

**long prefix (agent round)**

| shape | T | path | splits | regime_old | x_ratio_old | regime_new | t_default_us | t_best_us | s_best | x_cost | t_sep_us | sep_gain | pack_gain | sm_eff | structural_x |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1k append on 128k, alone | 1024 | nosplit | 1 | P | 1.0 | A | 3286.4 | 3286.4 | 1 | 1.0 | 3286.4 | 1.0 | 1.03 | 0.97 | False |
| 1k append on 128k + 3 x 128k decodes | 1027 | nosplit | 1 | P | 1.01 | A | 4902.9 | 4902.9 | 1 | 1.0 | 3862.7 | 1.27 | 1.41 | 0.71 | True |
| 256 append on 128k + 3 x 128k decodes | 259 | dyn32 | 32 | P | 1.0 | A | 1648.0 | 1648.0 | 2 | 1.0 | 1404.9 | 1.17 | 1.49 | 0.67 | False |
| 1k append on 32k + 1 x 128k decode | 1025 | nosplit | 1 | X | 1.6 | X | 1671.4 | 1260.5 | 2 | 1.33 | 1054.8 | 1.58 | 1.81 | 0.55 | True |

**H5 wave edges (128k prefix)**

| shape | T | path | splits | regime_old | x_ratio_old | regime_new | t_default_us | t_best_us | s_best | x_cost | t_sep_us | sep_gain | pack_gain | sm_eff | structural_x |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 512 append on 128k | 512 | dyn32 | 32 | P | 1.0 | A | 1641.5 | 1641.5 | 1 | 1.0 | 1641.5 | 1.0 | 1.03 | 0.97 | False |
| 513 append on 128k | 513 | nosplit | 1 | P | 1.0 | A | 3266.3 | 3266.3 | 1 | 1.0 | 3266.3 | 1.0 | 1.93 | 0.52 | False |
| 544 append on 128k | 544 | nosplit | 1 | P | 1.0 | A | 3266.3 | 3266.3 | 1 | 1.0 | 3266.3 | 1.0 | 1.93 | 0.52 | False |
| 1056 append on 128k | 1056 | nosplit | 1 | P | 1.0 | A | 3287.2 | 3287.2 | 1 | 1.0 | 3287.2 | 1.0 | 1.0 | 1.0 | False |
| 1057 append on 128k | 1057 | nosplit | 1 | P | 1.0 | A | 4899.8 | 4899.8 | 1 | 1.0 | 4899.8 | 1.0 | 1.45 | 0.69 | False |

**re-prefill (H4)**

| shape | T | path | splits | regime_old | x_ratio_old | regime_new | t_default_us | t_best_us | s_best | x_cost | t_sep_us | sep_gain | pack_gain | sm_eff | structural_x |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 8k chunk at prefix 0 + 3 x 128k decodes (cold start) | 8195 | nosplit | 1 | X | 1.43 | P | 1671.4 | 1671.4 | 1 | 1.0 | 1504.8 | 1.11 | 1.36 | 0.73 | True |
| 8k chunk at prefix 64k + 3 x 128k decodes | 8195 | nosplit | 1 | P | 1.0 | A | 13919.9 | 13919.9 | 1 | 1.0 | 14382.1 | 0.97 | 1.01 | 0.99 | True |
| 2k chunk at prefix 0 + 3 x 128k decodes | 2051 | nosplit | 1 | X | 2.46 | X | 1671.4 | 573.4 | 4 | 2.91 | 677.9 | 2.47 | 2.93 | 0.23 | True |
