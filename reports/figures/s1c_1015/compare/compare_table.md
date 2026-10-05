| case | reqs | req/s | out tok/s | TTFT p50 | TTFT p99 | TPOT p50 | TPOT p99 | E2E p50 | E2E p99 | GPU busy s | util | P %GPU | X %GPU | Dw %GPU | Ds %GPU | Dl %GPU |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3-4B_xheavy_lc24k_B0 | 2400 | 24.22 | 1133.5 | 52.6 | 1867.4 | 12.0 | 23.5 | 421.7 | 2798.1 | 97.9 | 0.99 | 10.9 | 33.1 | 26.7 | 14.9 | 14.5 |
| Qwen3-4B_xheavy_lc24k_C3 | 2400 | 24.23 | 1133.5 | 45.3 | 1842.1 | 9.6 | 22.0 | 379.0 | 2698.2 | 95.9 | 0.99 | 12.1 | 26.1 | 30.4 | 10.9 | 20.4 |
| Qwen3-8B_xheavy_lc16k_B0 | 2400 | 24.18 | 1131.5 | 65.6 | 1752.7 | 16.1 | 36.6 | 662.7 | 2944.9 | 97.4 | 1.00 | 12.6 | 43.4 | 13.8 | 20.9 | 9.4 |
| Qwen3-8B_xheavy_lc16k_C3 | 2400 | 24.18 | 1131.6 | 67.1 | 1712.3 | 14.7 | 34.3 | 620.8 | 2901.8 | 98.7 | 1.00 | 15.4 | 36.5 | 17.4 | 19.3 | 11.4 |
| Qwen3-8B_xheavy_lc24k_B0 | 2400 | 24.18 | 1131.5 | 89.3 | 3111.6 | 21.3 | 67.5 | 900.4 | 5281.9 | 98.2 | 1.00 | 7.8 | 56.6 | 4.2 | 26.6 | 4.7 |
| Qwen3-8B_xheavy_lc24k_C3 | 2400 | 24.18 | 1131.6 | 67.3 | 3056.4 | 17.4 | 64.1 | 708.8 | 5253.9 | 97.6 | 1.00 | 12.4 | 42.9 | 10.7 | 25.7 | 8.2 |

**GPU time by exec config, scheduler decisions**

| case | full CUDA graph, FA3 splits<=32 %GPU | piecewise CUDA graph, FA3 splits<=32 %GPU | eager, FA3 1 split (straggler) %GPU | eager + regime split (C3) %GPU | shape-cap steps | gated calls | gated reqs (unique) | preemptions |
|---|---|---|---|---|---|---|---|---|
| Qwen3-4B_xheavy_lc24k_B0 | 56.1 | 1.8 | 42.1 | 0.0 | n/a | n/a | n/a | n/a |
| Qwen3-4B_xheavy_lc24k_C3 | 61.8 | 1.9 | 18.1 | 18.2 | n/a | n/a | n/a | n/a |
| Qwen3-8B_xheavy_lc16k_B0 | 44.0 | 1.7 | 54.3 | 0.0 | n/a | n/a | n/a | n/a |
| Qwen3-8B_xheavy_lc16k_C3 | 48.1 | 1.9 | 17.1 | 32.9 | n/a | n/a | n/a | n/a |
| Qwen3-8B_xheavy_lc24k_B0 | 35.5 | 1.5 | 62.9 | 0.0 | n/a | n/a | n/a | n/a |
| Qwen3-8B_xheavy_lc24k_C3 | 44.7 | 1.8 | 12.7 | 40.8 | n/a | n/a | n/a | n/a |

shape-cap steps = steps joined to a sched record with shape=true; gated calls = schedule() calls with n_gated>0; preemptions = sum of n_preempt over sched records (n/a = no sched file).

Latencies in ms. TPOT = (E2E - TTFT)/(output_len - 1) per request; percentiles over per-request values.
