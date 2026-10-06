# S5 A/B: round_robin vs deadline vs deadline_pa vs deadline_pa_w50 vs round_robin_b

Run dir `/shared/home/garv901-55613a/regime-batching/regime-batching/results/mstar/s5/run1`. Delta columns compare each arm with `round_robin`; for every metric here lower is better except tok/s and audio s/s.

## Samples

| case | load | arm | reps | audio reqs | text reqs | failed | window (all reps) |
|---|---|---|---|---|---|---|---|
| a4 | audio only, conc 4 | round_robin | 2 | 48 | 0 | 0 | 41 s |
| a4 | audio only, conc 4 | deadline | 2 | 48 | 0 | 0 | 59 s |
| a4 | audio only, conc 4 | deadline_pa | 2 | 48 | 0 | 0 | 44 s |
| a4 | audio only, conc 4 | deadline_pa_w50 | 2 | 48 | 0 | 0 | 41 s |
| a4 | audio only, conc 4 | round_robin_b | 2 | 48 | 0 | 0 | 41 s |
| a8 | audio only, conc 8 | round_robin | 2 | 64 | 0 | 0 | 37 s |
| a8 | audio only, conc 8 | deadline | 2 | 64 | 0 | 0 | 45 s |
| a8 | audio only, conc 8 | deadline_pa | 2 | 64 | 0 | 0 | 37 s |
| a8 | audio only, conc 8 | deadline_pa_w50 | 2 | 64 | 0 | 0 | 37 s |
| a8 | audio only, conc 8 | round_robin_b | 2 | 64 | 0 | 0 | 37 s |
| b4 | audio conc 4 + text conc 8 (1-2k-token prompts) | round_robin | 2 | 48 | 192 | 0 | 82 s |
| b4 | audio conc 4 + text conc 8 (1-2k-token prompts) | deadline | 2 | 48 | 64 | 0 | 67 s |
| b4 | audio conc 4 + text conc 8 (1-2k-token prompts) | deadline_pa | 2 | 48 | 112 | 0 | 67 s |
| b4 | audio conc 4 + text conc 8 (1-2k-token prompts) | deadline_pa_w50 | 2 | 48 | 152 | 0 | 75 s |
| b4 | audio conc 4 + text conc 8 (1-2k-token prompts) | round_robin_b | 2 | 48 | 192 | 0 | 80 s |
| b8 | audio conc 8 + text conc 8 (1-2k-token prompts) | round_robin | 2 | 64 | 144 | 0 | 66 s |
| b8 | audio conc 8 + text conc 8 (1-2k-token prompts) | deadline | 2 | 64 | 48 | 0 | 54 s |
| b8 | audio conc 8 + text conc 8 (1-2k-token prompts) | deadline_pa | 2 | 64 | 88 | 0 | 53 s |
| b8 | audio conc 8 + text conc 8 (1-2k-token prompts) | deadline_pa_w50 | 2 | 64 | 112 | 0 | 59 s |
| b8 | audio conc 8 + text conc 8 (1-2k-token prompts) | round_robin_b | 2 | 64 | 136 | 0 | 65 s |

## Streaming audio (the measured requests)

| case | metric | round_robin | deadline | deadline_pa | deadline_pa_w50 | round_robin_b | deadline vs round_robin | deadline_pa vs round_robin | deadline_pa_w50 vs round_robin | round_robin_b vs round_robin |
|---|---|---|---|---|---|---|---|---|---|---|
| a4 | TTFA p50 | 0.65 s | 1.34 s | 0.74 s | 0.63 s | 0.65 s | +107% | +14% | -3% | -0% |
| a4 | TTFA p99 | 1.17 s | 3.03 s | 0.88 s | 0.93 s | 1.03 s | +160% | -24% | -20% | -11% |
| a4 | RTF p50 | 0.18 | 0.26 | 0.19 | 0.18 | 0.18 | +45% | +8% | +1% | -1% |
| a4 | RTF p99 | 0.30 | 0.36 | 0.42 | 0.25 | 0.42 | +19% | +41% | -16% | +41% |
| a4 | stall per request (mean) | 0.00 s | 0.03 s | 0.00 s | 0.00 s | 0.00 s | n/a | n/a | n/a | n/a |
| a4 | requests stalled | 0% | 6% | 0% | 0% | 0% | n/a | n/a | n/a | n/a |
| a4 | audio chunk gap p50 | 0.26 s | 0.19 s | 0.23 s | 0.25 s | 0.25 s | -26% | -10% | -4% | -1% |
| a4 | audio chunk gap p99 | 0.81 s | 1.84 s | 0.96 s | 0.69 s | 0.60 s | +126% | +17% | -15% | -26% |
| a4 | audio produced | 21.64 s/s | 15.25 s/s | 20.55 s/s | 21.13 s/s | 21.83 s/s | -30% | -5% | -2% | +1% |
| a8 | TTFA p50 | 0.76 s | 1.93 s | 0.95 s | 0.85 s | 0.80 s | +155% | +25% | +12% | +6% |
| a8 | TTFA p99 | 1.33 s | 3.27 s | 1.31 s | 1.36 s | 1.33 s | +145% | -2% | +2% | -0% |
| a8 | RTF p50 | 0.24 | 0.30 | 0.24 | 0.24 | 0.24 | +22% | -1% | -3% | -2% |
| a8 | RTF p99 | 0.62 | 0.67 | 0.63 | 0.65 | 0.64 | +8% | +3% | +5% | +4% |
| a8 | stall per request (mean) | 0.00 s | 0.00 s | 0.00 s | 0.00 s | 0.00 s | n/a | n/a | n/a | n/a |
| a8 | requests stalled | 0% | 0% | 0% | 0% | 0% | n/a | n/a | n/a | n/a |
| a8 | audio chunk gap p50 | 0.40 s | 0.24 s | 0.30 s | 0.36 s | 0.39 s | -41% | -25% | -10% | -2% |
| a8 | audio chunk gap p99 | 0.71 s | 1.84 s | 1.02 s | 1.00 s | 0.72 s | +158% | +43% | +40% | +1% |
| a8 | audio produced | 32.78 s/s | 26.26 s/s | 32.76 s/s | 32.60 s/s | 33.34 s/s | -20% | -0% | -1% | +2% |
| b4 | TTFA p50 | 0.84 s | 1.95 s | 1.01 s | 0.86 s | 0.83 s | +133% | +20% | +2% | -1% |
| b4 | TTFA p99 | 1.76 s | 3.45 s | 2.00 s | 1.74 s | 1.59 s | +96% | +13% | -1% | -10% |
| b4 | RTF p50 | 0.36 | 0.29 | 0.29 | 0.33 | 0.35 | -19% | -19% | -9% | -1% |
| b4 | RTF p99 | 0.83 | 0.43 | 0.87 | 0.45 | 0.75 | -47% | +5% | -46% | -9% |
| b4 | stall per request (mean) | 0.00 s | 0.02 s | 0.00 s | 0.00 s | 0.00 s | n/a | n/a | n/a | n/a |
| b4 | requests stalled | 0% | 6% | 0% | 0% | 0% | n/a | n/a | n/a | n/a |
| b4 | audio chunk gap p50 | 0.58 s | 0.22 s | 0.28 s | 0.47 s | 0.59 s | -63% | -52% | -18% | +1% |
| b4 | audio chunk gap p99 | 1.32 s | 2.00 s | 1.49 s | 1.47 s | 1.16 s | +52% | +13% | +12% | -12% |
| b4 | audio produced | 10.85 s/s | 13.09 s/s | 13.46 s/s | 12.04 s/s | 10.98 s/s | +21% | +24% | +11% | +1% |
| b8 | TTFA p50 | 0.96 s | 2.25 s | 1.13 s | 0.99 s | 0.95 s | +134% | +17% | +3% | -1% |
| b8 | TTFA p99 | 1.87 s | 3.40 s | 2.28 s | 2.32 s | 2.03 s | +82% | +22% | +24% | +9% |
| b8 | RTF p50 | 0.41 | 0.34 | 0.33 | 0.36 | 0.42 | -17% | -18% | -12% | +3% |
| b8 | RTF p99 | 0.78 | 0.90 | 1.20 | 1.02 | 0.91 | +16% | +54% | +31% | +17% |
| b8 | stall per request (mean) | 0.00 s | 0.04 s | 0.00 s | 0.00 s | 0.00 s | n/a | n/a | n/a | n/a |
| b8 | requests stalled | 0% | 9% | 0% | 0% | 0% | n/a | n/a | n/a | n/a |
| b8 | audio chunk gap p50 | 0.70 s | 0.25 s | 0.43 s | 0.55 s | 0.68 s | -64% | -38% | -22% | -2% |
| b8 | audio chunk gap p99 | 1.37 s | 2.00 s | 1.65 s | 1.45 s | 1.42 s | +46% | +20% | +5% | +4% |
| b8 | audio produced | 18.42 s/s | 23.01 s/s | 22.72 s/s | 20.87 s/s | 18.18 s/s | +25% | +23% | +13% | -1% |

## Text-only load (b cases)

| case | metric | round_robin | deadline | deadline_pa | deadline_pa_w50 | round_robin_b | deadline vs round_robin | deadline_pa vs round_robin | deadline_pa_w50 vs round_robin | round_robin_b vs round_robin |
|---|---|---|---|---|---|---|---|---|---|---|
| b4 | TTFT p50 | 0.40 s | 0.67 s | 0.67 s | 0.57 s | 0.39 s | +69% | +68% | +44% | -1% |
| b4 | TTFT p99 | 1.02 s | 1.16 s | 1.19 s | 1.01 s | 0.75 s | +14% | +16% | -1% | -26% |
| b4 | ITL p50 | 17.7 ms | 17.1 ms | 15.1 ms | 14.7 ms | 16.8 ms | -3% | -14% | -17% | -5% |
| b4 | ITL p99 | 86.3 ms | 249.6 ms | 236.4 ms | 93.3 ms | 83.2 ms | +189% | +174% | +8% | -4% |
| b4 | text tokens | 292.2 tok/s | 107.2 tok/s | 200.2 tok/s | 240.0 tok/s | 293.3 tok/s | -63% | -31% | -18% | +0% |
| b8 | TTFT p50 | 0.32 s | 0.68 s | 0.65 s | 0.50 s | 0.35 s | +116% | +105% | +59% | +12% |
| b8 | TTFT p99 | 0.73 s | 1.01 s | 1.43 s | 0.99 s | 0.85 s | +37% | +95% | +35% | +17% |
| b8 | ITL p50 | 18.4 ms | 18.7 ms | 16.7 ms | 16.1 ms | 19.9 ms | +2% | -9% | -12% | +9% |
| b8 | ITL p99 | 116.2 ms | 256.3 ms | 244.5 ms | 96.1 ms | 110.0 ms | +121% | +110% | -17% | -5% |
| b8 | text tokens | 259.9 tok/s | 92.3 tok/s | 194.0 tok/s | 222.0 tok/s | 257.6 tok/s | -64% | -25% | -15% | -1% |

## Rep-to-rep spread: TTFA p50 of each rep (s)

| case | round_robin | deadline | deadline_pa | deadline_pa_w50 | round_robin_b |
|---|---|---|---|---|---|
| a4 | 0.65, 0.65 | 1.59, 1.25 | 0.65, 0.75 | 0.60, 0.70 | 0.65, 0.65 |
| a8 | 0.77, 0.74 | 1.92, 1.97 | 0.94, 0.99 | 0.85, 0.83 | 0.80, 0.81 |
| b4 | 0.83, 0.84 | 1.96, 1.73 | 1.05, 1.01 | 0.83, 0.87 | 0.83, 0.81 |
| b8 | 0.93, 0.99 | 2.22, 2.33 | 1.40, 1.01 | 0.98, 1.03 | 0.93, 0.98 |

## Longest wait of a ready (node, walk) at scheduling time: max / p99, ms

| case | walk | round_robin max / p99 | deadline max / p99 | deadline_pa max / p99 | deadline_pa_w50 max / p99 | round_robin_b max / p99 |
|---|---|---|---|---|---|---|
| a4 | Code2Wav chunk | 401 / 37  (n=487) | 66 / 43  (n=493) | 220 / 213  (n=703) | 357 / 75  (n=568) | 60 / 32  (n=486) |
| a4 | Talker decode | 105 / 34  (n=1243) | 52 / 31  (n=948) | 209 / 204  (n=779) | 71 / 61  (n=905) | 80 / 41  (n=1248) |
| a4 | Talker prefill | 59 / 47  (n=49) | 31 / 24  (n=46) | 0 / 0  (n=40) | 53 / 41  (n=52) | 59 / 47  (n=50) |
| a4 | Talker last prefill | 0 / 0  (n=43) | 15 / 9  (n=42) | 0 / 0  (n=38) | 19 / 18  (n=45) | 0 / 0  (n=44) |
| a4 | Thinker decode | 324 / 41  (n=1018) | 459 / 206  (n=1306) | 254 / 205  (n=693) | 319 / 57  (n=783) | 57 / 42  (n=1015) |
| a4 | Thinker prefill (text) | 48 / 43  (n=47) | 153 / 146  (n=73) | 215 / 213  (n=242) | 65 / 65  (n=106) | 46 / 39  (n=53) |
| a8 | Code2Wav chunk | 74 / 55  (n=659) | 106 / 77  (n=543) | 447 / 216  (n=674) | 411 / 76  (n=686) | 79 / 63  (n=648) |
| a8 | Talker decode | 59 / 41  (n=1302) | 264 / 62  (n=833) | 254 / 204  (n=694) | 324 / 65  (n=931) | 68 / 45  (n=1260) |
| a8 | Talker prefill | 287 / 151  (n=58) | 74 / 56  (n=65) | 53 / 43  (n=47) | 228 / 108  (n=62) | 52 / 50  (n=61) |
| a8 | Talker last prefill | 28 / 24  (n=54) | 37 / 30  (n=47) | 22 / 13  (n=39) | 25 / 23  (n=58) | 27 / 25  (n=55) |
| a8 | Thinker decode | 273 / 47  (n=1118) | 440 / 208  (n=1105) | 244 / 206  (n=644) | 260 / 56  (n=826) | 84 / 47  (n=1054) |
| a8 | Thinker prefill (text) | 254 / 245  (n=63) | 183 / 161  (n=92) | 220 / 214  (n=257) | 70 / 69  (n=154) | 72 / 62  (n=69) |
| b4 | Code2Wav chunk | 247 / 54  (n=533) | 380 / 102  (n=551) | 871 / 485  (n=641) | 711 / 318  (n=722) | 286 / 59  (n=490) |
| b4 | Talker decode | 407 / 50  (n=2118) | 407 / 53  (n=974) | 807 / 417  (n=852) | 388 / 262  (n=1271) | 361 / 91  (n=2073) |
| b4 | Talker prefill | 42 / 41  (n=46) | 122 / 82  (n=48) | 86 / 75  (n=44) | 39 / 36  (n=64) | 54 / 48  (n=48) |
| b4 | Talker last prefill | 19 / 18  (n=44) | 19 / 11  (n=44) | 7 / 5  (n=37) | 31 / 25  (n=51) | 20 / 20  (n=46) |
| b4 | Thinker decode | 694 / 232  (n=2119) | 1125 / 207  (n=1387) | 928 / 426  (n=842) | 662 / 260  (n=1262) | 350 / 188  (n=2074) |
| b4 | Thinker prefill (text) | 323 / 274  (n=194) | 222 / 215  (n=141) | 350 / 220  (n=267) | 362 / 315  (n=191) | 300 / 266  (n=203) |
| b8 | Code2Wav chunk | 337 / 178  (n=673) | 595 / 123  (n=637) | 985 / 429  (n=622) | 645 / 349  (n=749) | 314 / 87  (n=590) |
| b8 | Talker decode | 564 / 158  (n=1747) | 402 / 71  (n=910) | 1257 / 365  (n=747) | 423 / 283  (n=1096) | 343 / 65  (n=1686) |
| b8 | Talker prefill | 53 / 49  (n=54) | 60 / 53  (n=60) | 527 / 510  (n=52) | 211 / 113  (n=68) | 68 / 62  (n=52) |
| b8 | Talker last prefill | 24 / 23  (n=52) | 46 / 26  (n=46) | 27 / 19  (n=41) | 38 / 33  (n=61) | 30 / 28  (n=50) |
| b8 | Thinker decode | 499 / 190  (n=1746) | 1040 / 211  (n=1370) | 784 / 392  (n=749) | 483 / 206  (n=1108) | 651 / 199  (n=1680) |
| b8 | Thinker prefill (text) | 191 / 182  (n=180) | 225 / 217  (n=141) | 274 / 221  (n=256) | 472 / 417  (n=194) | 313 / 298  (n=152) |

## GPU time and step counts per walk (share of the window; steps/s; mean batch)

| case | walk | round_robin | deadline | deadline_pa | deadline_pa_w50 | round_robin_b |
|---|---|---|---|---|---|---|
| a4 | Code2Wav chunk | 3% / 9.9/s / bs 1.2 | 2% / 7.1/s / bs 1.1 | 2% / 3.9/s / bs 2.8 | 3% / 6.6/s / bs 1.7 | 3% / 10.2/s / bs 1.2 |
| a4 | Talker decode | 54% / 74.5/s / bs 3.7 | 73% / 116.7/s / bs 1.7 | 59% / 83.7/s / bs 3.1 | 55% / 76.3/s / bs 3.5 | 53% / 73.8/s / bs 3.7 |
| a4 | Talker prefill | 0% / 1.1/s / bs 1.1 | 0% / 0.7/s / bs 1.1 | 0% / 0.9/s / bs 1.2 | 0% / 1.0/s / bs 1.2 | 0% / 1.1/s / bs 1.1 |
| a4 | Talker last prefill | 1% / 1.1/s / bs 1.1 | 0% / 0.7/s / bs 1.2 | 1% / 0.9/s / bs 1.3 | 1% / 1.0/s / bs 1.2 | 1% / 1.1/s / bs 1.1 |
| a4 | Thinker decode | 27% / 35.0/s / bs 2.1 | 15% / 15.5/s / bs 3.2 | 27% / 35.9/s / bs 1.9 | 28% / 37.4/s / bs 2.0 | 27% / 34.8/s / bs 2.1 |
| a4 | Thinker prefill (text) | 2% / 1.1/s / bs 1.1 | 2% / 0.7/s / bs 1.1 | 3% / 0.9/s / bs 1.2 | 3% / 1.0/s / bs 1.1 | 3% / 1.1/s / bs 1.1 |
| a8 | Code2Wav chunk | 4% / 12.8/s / bs 1.4 | 3% / 9.3/s / bs 1.5 | 4% / 3.7/s / bs 4.8 | 4% / 7.6/s / bs 2.3 | 5% / 12.9/s / bs 1.4 |
| a8 | Talker decode | 44% / 58.2/s / bs 7.1 | 63% / 91.2/s / bs 3.6 | 49% / 65.2/s / bs 6.4 | 46% / 60.7/s / bs 6.8 | 45% / 59.0/s / bs 7.1 |
| a8 | Talker prefill | 1% / 1.5/s / bs 1.2 | 0% / 1.2/s / bs 1.2 | 0% / 1.1/s / bs 1.6 | 0% / 1.3/s / bs 1.3 | 1% / 1.5/s / bs 1.2 |
| a8 | Talker last prefill | 1% / 1.4/s / bs 1.2 | 1% / 1.0/s / bs 1.5 | 1% / 1.0/s / bs 1.7 | 1% / 1.3/s / bs 1.4 | 1% / 1.4/s / bs 1.2 |
| a8 | Thinker decode | 31% / 35.8/s / bs 3.0 | 16% / 13.6/s / bs 6.5 | 30% / 33.0/s / bs 3.3 | 30% / 34.1/s / bs 3.2 | 30% / 34.2/s / bs 3.2 |
| a8 | Thinker prefill (text) | 5% / 1.5/s / bs 1.2 | 5% / 1.2/s / bs 1.2 | 6% / 1.1/s / bs 1.6 | 5% / 1.4/s / bs 1.3 | 5% / 1.5/s / bs 1.2 |
| b4 | Code2Wav chunk | 2% / 5.1/s / bs 1.1 | 2% / 6.4/s / bs 1.1 | 2% / 2.7/s / bs 2.6 | 2% / 4.0/s / bs 1.6 | 2% / 5.0/s / bs 1.2 |
| b4 | Talker decode | 27% / 36.7/s / bs 3.7 | 60% / 94.1/s / bs 1.8 | 39% / 55.1/s / bs 3.1 | 33% / 44.6/s / bs 3.4 | 27% / 36.9/s / bs 3.8 |
| b4 | Talker prefill | 0% / 0.5/s / bs 1.1 | 0% / 0.6/s / bs 1.1 | 0% / 0.6/s / bs 1.3 | 0% / 0.6/s / bs 1.1 | 0% / 0.6/s / bs 1.1 |
| b4 | Talker last prefill | 0% / 0.5/s / bs 1.1 | 0% / 0.6/s / bs 1.1 | 0% / 0.5/s / bs 1.3 | 0% / 0.5/s / bs 1.2 | 0% / 0.6/s / bs 1.1 |
| b4 | Thinker decode | 44% / 36.9/s / bs 8.9 | 19% / 14.1/s / bs 10.7 | 33% / 26.8/s / bs 9.1 | 38% / 30.7/s / bs 9.1 | 45% / 37.1/s / bs 8.9 |
| b4 | Thinker prefill (text) | 17% / 1.4/s / bs 2.0 | 10% / 0.9/s / bs 1.8 | 16% / 1.1/s / bs 2.3 | 18% / 1.3/s / bs 2.0 | 17% / 1.6/s / bs 1.9 |
| b8 | Code2Wav chunk | 3% / 7.0/s / bs 1.4 | 3% / 9.1/s / bs 1.4 | 2% / 2.8/s / bs 4.4 | 3% / 5.0/s / bs 2.2 | 3% / 6.8/s / bs 1.4 |
| b8 | Talker decode | 25% / 32.6/s / bs 7.1 | 57% / 83.3/s / bs 3.5 | 34% / 45.0/s / bs 6.4 | 30% / 39.2/s / bs 6.7 | 25% / 32.3/s / bs 7.1 |
| b8 | Talker prefill | 0% / 0.8/s / bs 1.2 | 0% / 1.0/s / bs 1.2 | 0% / 0.7/s / bs 1.6 | 0% / 0.8/s / bs 1.3 | 0% / 0.8/s / bs 1.3 |
| b8 | Talker last prefill | 1% / 0.8/s / bs 1.3 | 1% / 0.8/s / bs 1.4 | 1% / 0.7/s / bs 1.6 | 1% / 0.8/s / bs 1.3 | 1% / 0.7/s / bs 1.3 |
| b8 | Thinker decode | 41% / 32.9/s / bs 9.7 | 18% / 12.2/s / bs 13.6 | 34% / 26.0/s / bs 10.4 | 37% / 28.4/s / bs 10.2 | 41% / 32.5/s / bs 9.7 |
| b8 | Thinker prefill (text) | 15% / 1.8/s / bs 1.7 | 10% / 1.3/s / bs 1.6 | 17% / 1.2/s / bs 2.4 | 17% / 1.4/s / bs 2.1 | 15% / 1.5/s / bs 2.0 |

## What the policy did (`sched_pick` reasons, counts in the windows)

| case | arm | reasons |
|---|---|---|
| a4 | round_robin | round_robin: 2192 |
| a4 | deadline | slack: 980, only: 835, aged: 22 |
| a4 | deadline_pa | slack: 852, only: 404, aged: 116 |
| a4 | deadline_pa_w50 | only: 783, slack: 578, aged: 212 |
| a4 | round_robin_b | round_robin: 2189 |
| a8 | round_robin | round_robin: 2148 |
| a8 | deadline | slack: 897, only: 623, aged: 33 |
| a8 | deadline_pa | slack: 800, only: 317, aged: 89 |
| a8 | deadline_pa_w50 | slack: 714, only: 524, aged: 282 |
| a8 | round_robin_b | round_robin: 2081 |
| b4 | round_robin | round_robin: 3650 |
| b4 | deadline | slack: 1056, only: 763, aged: 52, priority: 18 |
| b4 | deadline_pa | slack: 747, only: 405, aged: 277 |
| b4 | deadline_pa_w50 | only: 807, slack: 737, aged: 567, priority: 1 |
| b4 | round_robin_b | round_robin: 3574 |
| b8 | round_robin | round_robin: 2853 |
| b8 | deadline | slack: 1140, only: 527, aged: 47, priority: 15 |
| b8 | deadline_pa | slack: 748, only: 307, aged: 205 |
| b8 | deadline_pa_w50 | slack: 764, only: 506, aged: 506, priority: 1 |
| b8 | round_robin_b | round_robin: 2763 |
