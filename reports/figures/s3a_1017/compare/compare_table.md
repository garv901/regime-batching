| case | reqs | req/s | out tok/s | TTFT p50 | TTFT p99 | TPOT p50 | TPOT p99 | E2E p50 | E2E p99 | GPU busy s | util | P %GPU | M %GPU | X %GPU | Dw %GPU | Ds %GPU | Dl %GPU |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B0 | 2400 | 24.18 | 1131.4 | 78.7 | 3034.5 | 21.0 | 62.2 | 883.7 | 5183.6 | 99.0 | 1.00 | 7.6 | 0.0 | 55.6 | 5.5 | 26.7 | 4.6 |
| cap2048 | 2400 | 24.17 | 1131.0 | 75.9 | 3045.2 | 17.5 | 65.2 | 735.4 | 5179.2 | 99.0 | 1.00 | 48.9 | 0.0 | 6.6 | 11.0 | 25.6 | 7.9 |
| full | 2400 | 24.19 | 1131.6 | 84.3 | 3073.7 | 16.0 | 51.6 | 675.3 | 4766.6 | 98.7 | 1.00 | 59.5 | 0.3 | 7.6 | 14.4 | 11.7 | 6.5 |
| mnbt512 | 2400 | 24.18 | 1131.4 | 78.5 | 4041.2 | 15.5 | 19.8 | 640.2 | 4614.0 | 98.6 | 1.00 | 70.5 | 0.3 | 0.0 | 13.4 | 8.8 | 7.0 |
| shape | 2400 | 24.18 | 1131.5 | 74.7 | 3036.2 | 15.6 | 45.9 | 656.4 | 4632.1 | 98.2 | 1.00 | 59.3 | 0.3 | 7.3 | 14.9 | 11.4 | 6.8 |

**GPU time by exec config, scheduler decisions**

| case | full CUDA graph, FA3 splits<=32 %GPU | piecewise CUDA graph, FA3 splits<=32 %GPU | eager, FA3 1 split (straggler) %GPU | eager + regime split (C3) %GPU | shape-cap steps | gated calls | gated reqs (unique) | preemptions |
|---|---|---|---|---|---|---|---|---|
| B0 | 36.8 | 1.6 | 61.6 | 0.0 | n/a | n/a | n/a | n/a |
| cap2048 | 44.5 | 44.6 | 10.9 | 0.0 | n/a | n/a | n/a | n/a |
| full | 32.6 | 47.0 | 15.9 | 4.5 | 3735 | 0 | 0 | 0 |
| mnbt512 | 29.1 | 70.9 | 0.0 | 0.0 | n/a | n/a | n/a | n/a |
| shape | 33.1 | 46.5 | 20.4 | 0.0 | 3736 | 0 | 0 | 0 |

shape-cap steps = steps joined to a sched record with shape=true; gated calls = schedule() calls with n_gated>0; preemptions = sum of n_preempt over sched records (n/a = no sched file).

Latencies in ms. TPOT = (E2E - TTFT)/(output_len - 1) per request; percentiles over per-request values.
