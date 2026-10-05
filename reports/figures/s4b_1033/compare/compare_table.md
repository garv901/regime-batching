| case | reqs | req/s | out tok/s | TTFT p50 | TTFT p99 | TPOT p50 | TPOT p99 | E2E p50 | E2E p99 | GPU busy s | util | P %GPU | M %GPU | X %GPU | Dw %GPU | Dk %GPU | Ds %GPU | Dl %GPU | Dc %GPU |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B0 | 4784 | 14.39 | 2486.9 | 762.0 | 99216.6 | 37.6 | 153.0 | 6203.9 | 201190.4 | 332.1 | 1.00 | 35.3 | 0.0 | 1.4 | 7.6 | 50.8 | 0.9 | 4.0 | 0.0 |
| full | 4800 | 13.85 | 2389.2 | 93.8 | 5994.6 | 19.6 | 49.0 | 840.1 | 175149.8 | 346.1 | 1.00 | 45.8 | 0.6 | 1.4 | 15.2 | 37.0 | 0.0 | 0.0 | 0.0 |
| peak | 4799 | 13.63 | 2350.6 | 85.4 | 5323.4 | 20.1 | 61.9 | 840.4 | 180228.7 | 351.2 | 1.00 | 34.1 | 0.0 | 1.4 | 15.1 | 49.4 | 0.0 | 0.0 | 0.0 |

**GPU time by exec config, scheduler decisions**

| case | full CUDA graph, FA3 splits<=32 %GPU | piecewise CUDA graph, FA3 splits<=32 %GPU | eager, FA3 1 split (straggler) %GPU | shape-cap steps | gated calls | gated reqs (unique) | preemptions |
|---|---|---|---|---|---|---|---|
| B0 | 63.3 | 0.1 | 36.6 | n/a | n/a | n/a | n/a |
| full | 52.2 | 32.2 | 15.7 | 8332 | 7991 | 36 | 0 |
| peak | 64.5 | 0.3 | 35.2 | 0 | 7992 | 36 | 0 |

shape-cap steps = steps joined to a sched record with shape=true; gated calls = schedule() calls with n_gated>0; preemptions = sum of n_preempt over sched records (n/a = no sched file).

Latencies in ms. TPOT = (E2E - TTFT)/(output_len - 1) per request; percentiles over per-request values.
