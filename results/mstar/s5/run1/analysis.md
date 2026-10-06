# S5 A/B: round_robin vs deadline

Run dir `/shared/home/garv901-55613a/regime-batching/regime-batching/results/mstar/s5/run1`. Delta columns compare each arm with `round_robin`; for every metric here lower is better except tok/s and audio s/s.

## Samples

| case | load | arm | reps | audio reqs | text reqs | failed | window (all reps) |
|---|---|---|---|---|---|---|---|
| a4 | audio only, conc 4 | round_robin | 2 | 48 | 0 | 0 | 41 s |
| a4 | audio only, conc 4 | deadline | 2 | 48 | 0 | 0 | 59 s |
| a8 | audio only, conc 8 | round_robin | 2 | 64 | 0 | 0 | 37 s |
| a8 | audio only, conc 8 | deadline | 2 | 64 | 0 | 0 | 45 s |
| b4 | audio conc 4 + text conc 8 (1-2k-token prompts) | round_robin | 2 | 48 | 192 | 0 | 82 s |
| b4 | audio conc 4 + text conc 8 (1-2k-token prompts) | deadline | 2 | 48 | 64 | 0 | 67 s |
| b8 | audio conc 8 + text conc 8 (1-2k-token prompts) | round_robin | 2 | 64 | 144 | 0 | 66 s |
| b8 | audio conc 8 + text conc 8 (1-2k-token prompts) | deadline | 2 | 64 | 48 | 0 | 54 s |

## Streaming audio (the measured requests)

| case | metric | round_robin | deadline | deadline vs round_robin |
|---|---|---|---|---|
| a4 | TTFA p50 | 0.65 s | 1.34 s | +107% |
| a4 | TTFA p99 | 1.17 s | 3.03 s | +160% |
| a4 | RTF p50 | 0.18 | 0.26 | +45% |
| a4 | RTF p99 | 0.30 | 0.36 | +19% |
| a4 | stall per request (mean) | 0.00 s | 0.03 s | n/a |
| a4 | requests stalled | 0% | 6% | n/a |
| a4 | audio chunk gap p50 | 0.26 s | 0.19 s | -26% |
| a4 | audio chunk gap p99 | 0.81 s | 1.84 s | +126% |
| a4 | audio produced | 21.64 s/s | 15.25 s/s | -30% |
| a8 | TTFA p50 | 0.76 s | 1.93 s | +155% |
| a8 | TTFA p99 | 1.33 s | 3.27 s | +145% |
| a8 | RTF p50 | 0.24 | 0.30 | +22% |
| a8 | RTF p99 | 0.62 | 0.67 | +8% |
| a8 | stall per request (mean) | 0.00 s | 0.00 s | n/a |
| a8 | requests stalled | 0% | 0% | n/a |
| a8 | audio chunk gap p50 | 0.40 s | 0.24 s | -41% |
| a8 | audio chunk gap p99 | 0.71 s | 1.84 s | +158% |
| a8 | audio produced | 32.78 s/s | 26.26 s/s | -20% |
| b4 | TTFA p50 | 0.84 s | 1.95 s | +133% |
| b4 | TTFA p99 | 1.76 s | 3.45 s | +96% |
| b4 | RTF p50 | 0.36 | 0.29 | -19% |
| b4 | RTF p99 | 0.83 | 0.43 | -47% |
| b4 | stall per request (mean) | 0.00 s | 0.02 s | n/a |
| b4 | requests stalled | 0% | 6% | n/a |
| b4 | audio chunk gap p50 | 0.58 s | 0.22 s | -63% |
| b4 | audio chunk gap p99 | 1.32 s | 2.00 s | +52% |
| b4 | audio produced | 10.85 s/s | 13.09 s/s | +21% |
| b8 | TTFA p50 | 0.96 s | 2.25 s | +134% |
| b8 | TTFA p99 | 1.87 s | 3.40 s | +82% |
| b8 | RTF p50 | 0.41 | 0.34 | -17% |
| b8 | RTF p99 | 0.78 | 0.90 | +16% |
| b8 | stall per request (mean) | 0.00 s | 0.04 s | n/a |
| b8 | requests stalled | 0% | 9% | n/a |
| b8 | audio chunk gap p50 | 0.70 s | 0.25 s | -64% |
| b8 | audio chunk gap p99 | 1.37 s | 2.00 s | +46% |
| b8 | audio produced | 18.42 s/s | 23.01 s/s | +25% |

## Text-only load (b cases)

| case | metric | round_robin | deadline | deadline vs round_robin |
|---|---|---|---|---|
| b4 | TTFT p50 | 0.40 s | 0.67 s | +69% |
| b4 | TTFT p99 | 1.02 s | 1.16 s | +14% |
| b4 | ITL p50 | 17.7 ms | 17.1 ms | -3% |
| b4 | ITL p99 | 86.3 ms | 249.6 ms | +189% |
| b4 | text tokens | 292.2 tok/s | 107.2 tok/s | -63% |
| b8 | TTFT p50 | 0.32 s | 0.68 s | +116% |
| b8 | TTFT p99 | 0.73 s | 1.01 s | +37% |
| b8 | ITL p50 | 18.4 ms | 18.7 ms | +2% |
| b8 | ITL p99 | 116.2 ms | 256.3 ms | +121% |
| b8 | text tokens | 259.9 tok/s | 92.3 tok/s | -64% |

## Rep-to-rep spread: TTFA p50 of each rep (s)

| case | round_robin | deadline |
|---|---|---|
| a4 | 0.65, 0.65 | 1.59, 1.25 |
| a8 | 0.77, 0.74 | 1.92, 1.97 |
| b4 | 0.83, 0.84 | 1.96, 1.73 |
| b8 | 0.93, 0.99 | 2.22, 2.33 |

## Longest wait of a ready (node, walk) at scheduling time: max / p99, ms

| case | walk | round_robin max / p99 | deadline max / p99 |
|---|---|---|---|
| a4 | Code2Wav chunk | 401 / 37  (n=487) | 66 / 43  (n=493) |
| a4 | Talker decode | 105 / 34  (n=1243) | 52 / 31  (n=948) |
| a4 | Talker prefill | 59 / 47  (n=49) | 31 / 24  (n=46) |
| a4 | Talker last prefill | 0 / 0  (n=43) | 15 / 9  (n=42) |
| a4 | Thinker decode | 324 / 41  (n=1018) | 459 / 206  (n=1306) |
| a4 | Thinker prefill (text) | 48 / 43  (n=47) | 153 / 146  (n=73) |
| a8 | Code2Wav chunk | 74 / 55  (n=659) | 106 / 77  (n=543) |
| a8 | Talker decode | 59 / 41  (n=1302) | 264 / 62  (n=833) |
| a8 | Talker prefill | 287 / 151  (n=58) | 74 / 56  (n=65) |
| a8 | Talker last prefill | 28 / 24  (n=54) | 37 / 30  (n=47) |
| a8 | Thinker decode | 273 / 47  (n=1118) | 440 / 208  (n=1105) |
| a8 | Thinker prefill (text) | 254 / 245  (n=63) | 183 / 161  (n=92) |
| b4 | Code2Wav chunk | 247 / 54  (n=533) | 380 / 102  (n=551) |
| b4 | Talker decode | 407 / 50  (n=2118) | 407 / 53  (n=974) |
| b4 | Talker prefill | 42 / 41  (n=46) | 122 / 82  (n=48) |
| b4 | Talker last prefill | 19 / 18  (n=44) | 19 / 11  (n=44) |
| b4 | Thinker decode | 694 / 232  (n=2119) | 1125 / 207  (n=1387) |
| b4 | Thinker prefill (text) | 323 / 274  (n=194) | 222 / 215  (n=141) |
| b8 | Code2Wav chunk | 337 / 178  (n=673) | 595 / 123  (n=637) |
| b8 | Talker decode | 564 / 158  (n=1747) | 402 / 71  (n=910) |
| b8 | Talker prefill | 53 / 49  (n=54) | 60 / 53  (n=60) |
| b8 | Talker last prefill | 24 / 23  (n=52) | 46 / 26  (n=46) |
| b8 | Thinker decode | 499 / 190  (n=1746) | 1040 / 211  (n=1370) |
| b8 | Thinker prefill (text) | 191 / 182  (n=180) | 225 / 217  (n=141) |

## GPU time and step counts per walk (share of the window; steps/s; mean batch)

| case | walk | round_robin | deadline |
|---|---|---|---|
| a4 | Code2Wav chunk | 3% / 9.9/s / bs 1.2 | 2% / 7.1/s / bs 1.1 |
| a4 | Talker decode | 54% / 74.5/s / bs 3.7 | 73% / 116.7/s / bs 1.7 |
| a4 | Talker prefill | 0% / 1.1/s / bs 1.1 | 0% / 0.7/s / bs 1.1 |
| a4 | Talker last prefill | 1% / 1.1/s / bs 1.1 | 0% / 0.7/s / bs 1.2 |
| a4 | Thinker decode | 27% / 35.0/s / bs 2.1 | 15% / 15.5/s / bs 3.2 |
| a4 | Thinker prefill (text) | 2% / 1.1/s / bs 1.1 | 2% / 0.7/s / bs 1.1 |
| a8 | Code2Wav chunk | 4% / 12.8/s / bs 1.4 | 3% / 9.3/s / bs 1.5 |
| a8 | Talker decode | 44% / 58.2/s / bs 7.1 | 63% / 91.2/s / bs 3.6 |
| a8 | Talker prefill | 1% / 1.5/s / bs 1.2 | 0% / 1.2/s / bs 1.2 |
| a8 | Talker last prefill | 1% / 1.4/s / bs 1.2 | 1% / 1.0/s / bs 1.5 |
| a8 | Thinker decode | 31% / 35.8/s / bs 3.0 | 16% / 13.6/s / bs 6.5 |
| a8 | Thinker prefill (text) | 5% / 1.5/s / bs 1.2 | 5% / 1.2/s / bs 1.2 |
| b4 | Code2Wav chunk | 2% / 5.1/s / bs 1.1 | 2% / 6.4/s / bs 1.1 |
| b4 | Talker decode | 27% / 36.7/s / bs 3.7 | 60% / 94.1/s / bs 1.8 |
| b4 | Talker prefill | 0% / 0.5/s / bs 1.1 | 0% / 0.6/s / bs 1.1 |
| b4 | Talker last prefill | 0% / 0.5/s / bs 1.1 | 0% / 0.6/s / bs 1.1 |
| b4 | Thinker decode | 44% / 36.9/s / bs 8.9 | 19% / 14.1/s / bs 10.7 |
| b4 | Thinker prefill (text) | 17% / 1.4/s / bs 2.0 | 10% / 0.9/s / bs 1.8 |
| b8 | Code2Wav chunk | 3% / 7.0/s / bs 1.4 | 3% / 9.1/s / bs 1.4 |
| b8 | Talker decode | 25% / 32.6/s / bs 7.1 | 57% / 83.3/s / bs 3.5 |
| b8 | Talker prefill | 0% / 0.8/s / bs 1.2 | 0% / 1.0/s / bs 1.2 |
| b8 | Talker last prefill | 1% / 0.8/s / bs 1.3 | 1% / 0.8/s / bs 1.4 |
| b8 | Thinker decode | 41% / 32.9/s / bs 9.7 | 18% / 12.2/s / bs 13.6 |
| b8 | Thinker prefill (text) | 15% / 1.8/s / bs 1.7 | 10% / 1.3/s / bs 1.6 |

## What the policy did (`sched_pick` reasons, counts in the windows)

| case | arm | reasons |
|---|---|---|
| a4 | round_robin | round_robin: 2192 |
| a4 | deadline | slack: 980, only: 835, aged: 22 |
| a8 | round_robin | round_robin: 2148 |
| a8 | deadline | slack: 897, only: 623, aged: 33 |
| b4 | round_robin | round_robin: 3650 |
| b4 | deadline | slack: 1056, only: 763, aged: 52, priority: 18 |
| b8 | round_robin | round_robin: 2853 |
| b8 | deadline | slack: 1140, only: 527, aged: 47, priority: 15 |
