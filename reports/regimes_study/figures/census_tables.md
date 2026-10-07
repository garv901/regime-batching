# Stage 1b regime census tables

Whole step = one engine step of Llama-3.1-8B on one H100 (GPU time from CUDA events around the step, median of 3 repeats). Attention alternatives are swapped in from the F7 kernel census (32 layers). roofline = max(weights + KV bytes at 3.35 TB/s, FLOPs at 989 TFLOPS); eff = roofline / measured.

## Data checks

- default: 2175 probe/decode steps kept; rejected (shape mismatch): none
- cap2048: 2175 probe/decode steps kept; rejected (shape mismatch): none
- repeat spread (max/min - 1, default config): median 1.9%, share of shapes within 5%: 81%
- pre-registered whole-step model, mean abs error (default config): PIECEWISE 12.4% (n=162), NONE 7.9% (n=221), FULL 4.1% (n=8)

## Qualitative predictions, checked

1. Decode-only: the best attention kernel moves the whole step by at most 1.9% (median 0.4%, 8 decode sets); predicted <= 5%.
2. Steps of 513-2048 tokens: capture 2048 is faster by 0.95-3.36x (median 1.00x, 128 shapes); eager excess median 0.3 ms.
3. X steps: 43 shapes, loss median 1.57; of the 31 at <= 2,048 tokens, capture 2048 alone reaches the best alternative within 5% for 24; best-alternative kinds: {'cap2048': 6, 'split': 20, 'separate': 6, 'cap2048+att': 11}.
4. Remainder on a 122,880-token prefix in a step of <= 512 tokens: 16 / 32 / 64 tokens cost 0.186 / 0.187 / 0.251 ms of attention per layer (predicted <= 0.25 ms); H5 gain (attention, same tokens) 1.14-1.53x over 9 shapes.

## Regimes: where the time goes and what recovers it

Shares are medians over the shapes in the row, of the default-config step. eager = default minus capture-2048 time for the same step, net of the attention change (measured only for 513-2048 tokens). fix: the alternative that recovers the most, by count of shapes with loss >= 1.05.

| regime | step tokens | shapes | scenarios | attention | eager | non-attention | eff (median) | loss median (max) | fix |
|---|---|---|---|---|---|---|---|---|---|
| A | <=512 | 41 | 128x2k, 256x1k, 32x1k, 8x8k, none | 57% |  | 43% | 48% | 1.03 (1.18) | separate (7), split (2) |
| A | 513-2048 | 39 | 128x2k, 256x1k, 32x1k, 8x8k, none | 62% | 0% | 38% | 48% | 1.01 (1.28) | realign (5), cap2048+att (2) |
| A | >2048 | 22 | 128x2k, 256x1k, 32x1k, 8x8k, none | 59% |  | 41% | 60% | 1.00 (1.02) | none |
| A+L | <=512 | 35 | 1x120k, 31x1k+1x120k, 3x120k, 4x24k | 70% |  | 30% | 45% | 1.06 (1.48) | separate (16), split (2) |
| A+L | 513-2048 | 12 | 1x120k, 31x1k+1x120k, 4x24k | 81% | 1% | 18% | 48% | 1.03 (1.45) | cap2048+att (5) |
| A+L | >2048 | 21 | 1x120k, 31x1k+1x120k, 3x120k, 4x24k | 62% |  | 38% | 60% | 1.00 (1.05) | none |
| Dc | <=512 | 1 | 256x1k | 55% |  | 45% | 69% | 1.01 (1.01) | none |
| Dk | <=512 | 1 | 128x2k | 61% |  | 39% | 77% | 1.01 (1.01) | none |
| Dl | <=512 | 4 | 1x120k, 3x120k, 4x24k, 8x8k | 47% |  | 53% | 79% | 1.00 (1.02) | none |
| Ds | <=512 | 1 | 31x1k+1x120k | 53% |  | 47% | 78% | 1.00 (1.00) | none |
| Dw | <=512 | 1 | 32x1k | 24% |  | 76% | 70% | 1.01 (1.01) | none |
| M | <=512 | 4 | 128x2k, 256x1k | 57% |  | 43% | 64% | 1.05 (1.09) | separate (2) |
| P | <=512 | 82 | 128x2k, 1x120k, 256x1k, 31x1k+1x120k, 32x1k, 3x120k, 4x24k, 8x8k, none | 39% |  | 61% | 54% | 1.00 (1.15) | split (3), separate (1) |
| P | 513-2048 | 46 | 128x2k, 256x1k, 32x1k, 4x24k, 8x8k, none | 30% | 2% | 68% | 49% | 1.02 (1.10) | cap2048+att (5) |
| P | >2048 | 40 | 128x2k, 1x120k, 256x1k, 31x1k+1x120k, 32x1k, 3x120k, 4x24k, 8x8k, none | 28% |  | 72% | 59% | 1.00 (1.06) | separate (2) |
| X | 513-2048 | 31 | 1x120k, 31x1k+1x120k, 3x120k, 4x24k | 67% | 1% | 33% | 28% | 1.75 (3.36) | split (13), cap2048+att (11), cap2048 (6), separate (1) |
| X | >2048 | 12 | 1x120k, 31x1k+1x120k, 3x120k, 4x24k | 44% |  | 56% | 45% | 1.34 (1.78) | split (7), separate (5) |

Regime label vs measured loss (shapes):

| label | <1.05 | 1.05-1.15 | 1.15-1.5 | >=1.5 |
|---|---|---|---|---|
| A | 86 | 10 | 6 | 0 |
| A+L | 45 | 9 | 14 | 0 |
| Dc | 1 | 0 | 0 | 0 |
| Dk | 1 | 0 | 0 | 0 |
| Dl | 4 | 0 | 0 | 0 |
| Ds | 1 | 0 | 0 | 0 |
| Dw | 1 | 0 | 0 | 0 |
| M | 2 | 2 | 0 | 0 |
| P | 157 | 10 | 1 | 0 |
| X | 0 | 7 | 14 | 22 |

## Decode-only steps

| decode set | B | KV tokens | step ms | ms per token | roofline ms | eff | attention share | best attention kernel vs FA3 (whole step) |
|---|---|---|---|---|---|---|---|---|
| 1x120k | 1 | 122,880 | 11.7 | 11.662 | 9.3 | 80% | 50% | -0.2% |
| 3x120k | 3 | 368,640 | 22.3 | 7.432 | 18.9 | 85% | 74% | -1.9% |
| 4x24k | 4 | 98,304 | 10.6 | 2.641 | 8.3 | 79% | 44% | -0.0% |
| 31x1k+1x120k | 32 | 154,624 | 13.5 | 0.423 | 10.5 | 78% | 53% | -0.0% |
| 32x1k | 32 | 32,768 | 8.2 | 0.258 | 5.8 | 70% | 24% | -0.8% |
| 8x8k | 8 | 65,536 | 9.3 | 1.162 | 7.0 | 76% | 36% | -0.0% |
| 128x2k | 128 | 262,144 | 19.0 | 0.149 | 14.7 | 77% | 61% | -0.6% |
| 256x1k | 256 | 262,144 | 21.5 | 0.084 | 14.7 | 69% | 55% | -0.9% |

Long-context decode-only attention per layer (kernel level, no in-engine counterpart):

| spec | FA3 (graph, bound 32) ms | best config | best ms | FA3 / best |
|---|---|---|---|---|
| c_dec_1x64k | 0.107 | fi_native | 0.106 | 1.00 |
| c_dec_2x64k | 0.189 | fa3_auto | 0.189 | 1.00 |
| c_dec_4x64k | 0.361 | fa3_s4 | 0.361 | 1.00 |
| c_dec_8x64k | 0.701 | fa3_s8 | 0.701 | 1.00 |
| c_dec_16x64k | 1.379 | fa3_s1 | 1.377 | 1.00 |
| c_dec_1x120k | 0.183 | fi_native | 0.183 | 1.00 |
| c_dec_2x120k | 0.339 | fa3_s16 | 0.339 | 1.00 |
| c_dec_4x120k | 0.661 | fi_native | 0.661 | 1.00 |
| c_dec_8x120k | 1.300 | fa3_s4 | 1.299 | 1.00 |
| c_dec_16x120k | 2.570 | fa3_s1 | 2.567 | 1.00 |

## H5: the aligned chunk now and the remainder later (revised criterion, attention time, measured)

now = the chunk in one step (default config: split bound 32 if the step is <= 512 tokens, else FA3's own choice); later = the aligned chunk plus the remainder (rounded up to the measured grid) in a step of <= 512 tokens. SM efficiency from the tile model. Whole step: in-engine default config, remainder as its own step (conservative) or riding in a step that runs anyway (marginal).

| P | q | aligned + remainder | SM eff (now) | attention now ms/layer | attention later ms/layer | gain | whole step now ms | realign (own step) ms | realign (marginal) ms |
|---|---|---|---|---|---|---|---|---|---|
| 8192 | 513 | 512 + 16 | 53% | 0.219 | 0.156 | 1.40 | 23.5 | 26.5 | 18.1 |
| 8192 | 544 | 512 + 32 | 53% | 0.218 | 0.158 | 1.38 | 23.5 | 25.9 | 18.5 |
| 8192 | 1057 | 1056 + 16 | 72% | 0.329 | 0.290 | 1.14 | 38.0 | 45.4 | 37.0 |
| 32768 | 513 | 512 + 16 | 52% | 0.801 | 0.525 | 1.53 | 42.7 | 40.1 | 30.8 |
| 32768 | 544 | 512 + 32 | 52% | 0.798 | 0.528 | 1.51 | 42.5 | 39.4 | 31.3 |
| 32768 | 1057 | 1056 + 16 | 69% | 1.231 | 1.032 | 1.19 | 68.7 | 69.7 | 60.4 |
| 122880 | 513 | 512 + 16 | 52% | 2.914 | 2.000 | 1.46 | 115.2 | 90.2 | 77.6 |
| 122880 | 544 | 512 + 32 | 52% | 2.925 | 2.000 | 1.46 | 113.3 | 88.9 | 77.9 |
| 122880 | 1057 | 1056 + 16 | 69% | 4.549 | 3.703 | 1.23 | 174.9 | 159.0 | 146.3 |

Other work filling the idle wave: two appends in one launch vs each in its own launch (kernel level).

| pair | P | tiles | together ms/layer | apart ms/layer | apart / together |
|---|---|---|---|---|---|
| 512+512 | 32768 | 256 | 0.933 | 0.921 | 0.99 |
| 544+512 | 32768 | 264 | 0.954 | 1.258 | 1.32 |
| 513+513 | 32768 | 272 | 1.228 | 1.603 | 1.31 |
| 1056+32 | 32768 | 272 | 1.254 | 1.035 | 0.83 |
| 512+512 | 122880 | 256 | 3.436 | 3.627 | 1.06 |
| 544+512 | 122880 | 264 | 3.555 | 4.738 | 1.33 |
| 513+513 | 122880 | 272 | 4.572 | 5.829 | 1.27 |
| 1056+32 | 122880 | 272 | 4.641 | 3.703 | 0.80 |

## Every census step

| scenario | kind | P | q | T | mode | regime | default ms | cap2048 ms | attention ms | roofline ms (bound) | eff | best alternative | loss |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| none | probe | 0 | 16 | 16 | PIECEWISE | P | 8.7 | 8.7 | 0.3 | 4.5 (m) | 51% | none | 1.02 |
| none | probe | 0 | 32 | 32 | PIECEWISE | P | 7.9 | 7.8 | 0.3 | 4.5 (m) | 57% | none | 1.03 |
| none | probe | 0 | 64 | 64 | PIECEWISE | P | 8.0 | 8.7 | 0.4 | 4.5 (m) | 56% | none | 1.03 |
| none | probe | 0 | 128 | 128 | PIECEWISE | P | 8.3 | 8.8 | 0.4 | 4.5 (m) | 54% | none | 1.02 |
| none | probe | 0 | 256 | 256 | PIECEWISE | P | 9.4 | 9.6 | 0.5 | 4.5 (m) | 48% | none | 1.01 |
| none | probe | 0 | 416 | 416 | PIECEWISE | P | 13.8 | 13.6 | 0.6 | 5.9 (c) | 43% | none | 1.01 |
| none | probe | 0 | 417 | 417 | PIECEWISE | P | 12.8 | 12.7 | 0.6 | 5.9 (c) | 46% | none | 1.01 |
| none | probe | 0 | 512 | 512 | PIECEWISE | P | 13.1 | 13.2 | 0.6 | 7.3 (c) | 56% | none | 1.01 |
| none | probe | 0 | 513 | 513 | NONE | P | 15.9 | 14.6 | 0.6 | 7.3 (c) | 46% | cap2048+att | 1.09 |
| none | probe | 0 | 544 | 544 | NONE | P | 16.7 | 15.9 | 0.6 | 7.8 (c) | 46% | cap2048+att | 1.06 |
| none | probe | 0 | 1024 | 1024 | NONE | P | 24.6 | 23.9 | 0.9 | 14.7 (c) | 60% | none | 1.03 |
| none | probe | 0 | 1056 | 1056 | NONE | P | 28.2 | 27.2 | 1.0 | 15.2 (c) | 54% | none | 1.04 |
| none | probe | 0 | 1057 | 1057 | NONE | P | 28.1 | 27.4 | 1.1 | 15.2 (c) | 54% | none | 1.03 |
| none | probe | 0 | 2048 | 2048 | NONE | P | 47.6 | 49.4 | 2.4 | 30.0 (c) | 63% | none | 1.00 |
| none | probe | 0 | 4096 | 4096 | NONE | P | 99.9 | 100.6 | 8.3 | 62.3 (c) | 62% | none | 1.00 |
| none | probe | 0 | 8192 | 8192 | NONE | P | 211.5 | 214.1 | 30.9 | 133.4 (c) | 63% | none | 1.00 |
| none | probe | 8192 | 16 | 16 | PIECEWISE | P | 9.5 | 9.6 | 0.8 | 4.8 (m) | 50% | none | 1.00 |
| none | probe | 8192 | 32 | 32 | PIECEWISE | P | 8.9 | 8.9 | 0.9 | 4.8 (m) | 54% | none | 1.00 |
| none | probe | 8192 | 64 | 64 | PIECEWISE | P | 8.9 | 9.3 | 1.0 | 4.8 (m) | 54% | none | 1.00 |
| none | probe | 8192 | 128 | 128 | PIECEWISE | P | 9.6 | 10.0 | 1.4 | 4.8 (m) | 50% | none | 1.00 |
| none | probe | 8192 | 256 | 256 | PIECEWISE | P | 11.7 | 11.9 | 2.3 | 4.8 (m) | 41% | none | 1.00 |
| none | probe | 8192 | 416 | 416 | PIECEWISE | P | 17.6 | 18.3 | 4.3 | 7.7 (c) | 44% | none | 1.04 |
| none | probe | 8192 | 417 | 417 | PIECEWISE | P | 16.8 | 17.3 | 4.4 | 7.7 (c) | 46% | none | 1.04 |
| none | probe | 8192 | 512 | 512 | PIECEWISE | P | 17.0 | 17.5 | 4.2 | 9.5 (c) | 56% | none | 1.01 |
| none | probe | 8192 | 513 | 513 | NONE | P | 23.5 | 22.1 | 7.0 | 9.5 (c) | 41% | cap2048+att | 1.08 |
| none | probe | 8192 | 544 | 544 | NONE | P | 23.5 | 23.3 | 7.0 | 10.1 (c) | 43% | none | 1.02 |
| none | probe | 8192 | 1024 | 1024 | NONE | P | 31.1 | 31.0 | 8.2 | 19.2 (c) | 62% | none | 1.02 |
| none | probe | 8192 | 1056 | 1056 | NONE | P | 35.9 | 36.2 | 8.5 | 19.8 (c) | 55% | none | 1.00 |
| none | probe | 8192 | 1057 | 1057 | NONE | P | 38.0 | 38.8 | 10.5 | 19.8 (c) | 52% | none | 1.00 |
| none | probe | 8192 | 2048 | 2048 | NONE | P | 63.0 | 62.4 | 17.3 | 38.9 (c) | 62% | none | 1.02 |
| none | probe | 8192 | 4096 | 4096 | NONE | P | 127.8 | 127.8 | 36.8 | 80.1 (c) | 63% | none | 1.00 |
| none | probe | 8192 | 8192 | 8192 | NONE | P | 271.1 | 268.6 | 88.9 | 169.0 (c) | 62% | none | 1.00 |
| none | probe | 32768 | 16 | 16 | PIECEWISE | A | 11.7 | 13.1 | 2.1 | 5.8 (m) | 49% | none | 1.00 |
| none | probe | 32768 | 32 | 32 | PIECEWISE | A | 10.9 | 11.9 | 2.2 | 5.8 (m) | 53% | none | 1.00 |
| none | probe | 32768 | 64 | 64 | PIECEWISE | A | 11.2 | 12.0 | 2.4 | 5.8 (m) | 52% | none | 1.00 |
| none | probe | 32768 | 128 | 128 | PIECEWISE | A | 13.2 | 13.2 | 4.1 | 5.8 (m) | 44% | none | 1.01 |
| none | probe | 32768 | 256 | 256 | PIECEWISE | A | 17.7 | 17.6 | 7.9 | 8.1 (c) | 46% | none | 1.01 |
| none | probe | 32768 | 416 | 416 | PIECEWISE | A | 28.7 | 28.6 | 14.6 | 13.1 (c) | 46% | none | 1.03 |
| none | probe | 32768 | 417 | 417 | PIECEWISE | A | 27.6 | 27.9 | 15.2 | 13.2 (c) | 48% | none | 1.05 |
| none | probe | 32768 | 512 | 512 | PIECEWISE | A | 28.5 | 28.9 | 14.7 | 16.2 (c) | 57% | none | 1.00 |
| none | probe | 32768 | 513 | 513 | NONE | A | 42.7 | 43.8 | 25.6 | 16.2 (c) | 38% | realign | 1.06 |
| none | probe | 32768 | 544 | 544 | NONE | A | 42.5 | 43.2 | 25.5 | 17.2 (c) | 40% | realign | 1.08 |
| none | probe | 32768 | 1024 | 1024 | NONE | A | 52.3 | 52.6 | 28.9 | 32.5 (c) | 62% | none | 1.04 |
| none | probe | 32768 | 1056 | 1056 | NONE | A | 58.0 | 58.7 | 30.9 | 33.5 (c) | 58% | none | 1.01 |
| none | probe | 32768 | 1057 | 1057 | NONE | A | 68.7 | 71.0 | 39.4 | 33.6 (c) | 49% | none | 1.00 |
| none | probe | 32768 | 2048 | 2048 | NONE | A | 105.6 | 108.4 | 62.0 | 65.6 (c) | 62% | none | 1.00 |
| none | probe | 32768 | 4096 | 4096 | NONE | A | 216.7 | 216.4 | 126.2 | 133.4 (c) | 62% | none | 1.01 |
| none | probe | 32768 | 8192 | 8192 | NONE | A | 445.7 | 443.9 | 263.0 | 275.7 (c) | 62% | none | 1.00 |
| none | probe | 122880 | 16 | 16 | PIECEWISE | A | 18.9 | 19.1 | 6.0 | 9.3 (m) | 49% | none | 1.00 |
| none | probe | 122880 | 32 | 32 | PIECEWISE | A | 17.6 | 17.8 | 6.0 | 9.3 (m) | 53% | none | 1.00 |
| none | probe | 122880 | 64 | 64 | PIECEWISE | A | 19.0 | 19.4 | 8.0 | 9.3 (m) | 49% | none | 1.00 |
| none | probe | 122880 | 128 | 128 | PIECEWISE | A | 25.6 | 25.8 | 15.0 | 10.2 (c) | 40% | none | 1.01 |
| none | probe | 122880 | 256 | 256 | PIECEWISE | A | 39.3 | 39.6 | 28.7 | 20.3 (c) | 52% | none | 1.02 |
| none | probe | 122880 | 416 | 416 | PIECEWISE | A | 69.6 | 69.5 | 53.7 | 33.0 (c) | 47% | split | 1.05 |
| none | probe | 122880 | 417 | 417 | PIECEWISE | A | 69.0 | 69.4 | 54.7 | 33.1 (c) | 48% | none | 1.05 |
| none | probe | 122880 | 512 | 512 | PIECEWISE | A | 71.3 | 71.4 | 58.0 | 40.6 (c) | 57% | none | 1.05 |
| none | probe | 122880 | 513 | 513 | NONE | A | 115.2 | 118.7 | 93.3 | 40.7 (c) | 35% | realign | 1.28 |
| none | probe | 122880 | 544 | 544 | NONE | A | 113.3 | 117.5 | 93.6 | 43.2 (c) | 38% | realign | 1.27 |
| none | pair 512+512 | 122880 | 1024 | 1024 | NONE | A | 137.4 | 144.9 | 110.0 | 81.3 (c) | 59% | none | 1.00 |
| none | probe | 122880 | 1024 | 1024 | NONE | A | 135.3 | 138.3 | 100.2 | 81.4 (c) | 60% | cap2048+att | 1.09 |
| none | pair 513+513 | 122880 | 1026 | 1026 | NONE | A | 180.2 | 186.2 | 146.3 | 81.5 (c) | 45% | none | 1.01 |
| none | probe | 122880 | 1056 | 1056 | NONE | A | 140.1 | 143.9 | 112.5 | 84.0 (c) | 60% | none | 1.00 |
| none | pair 544+512 | 122880 | 1056 | 1056 | NONE | A | 142.5 | 148.7 | 113.8 | 83.8 (c) | 59% | none | 1.01 |
| none | pair 528+528 | 122880 | 1056 | 1056 | NONE | A | 178.5 | 187.8 | 148.7 | 83.8 (c) | 47% | none | 1.01 |
| none | probe | 122880 | 1057 | 1057 | NONE | A | 174.9 | 182.7 | 145.6 | 84.1 (c) | 48% | realign | 1.10 |
| none | pair 1056+32 | 122880 | 1088 | 1088 | NONE | A | 179.3 | 187.5 | 148.5 | 86.5 (c) | 48% | none | 1.01 |
| none | probe | 122880 | 2048 | 2048 | NONE | A | 268.8 | 275.8 | 223.7 | 163.4 (c) | 61% | none | 1.01 |
| none | probe | 122880 | 4096 | 4096 | NONE | A | 535.3 | 535.1 | 447.4 | 329.1 (c) | 61% | none | 1.00 |
| none | probe | 122880 | 8192 | 8192 | NONE | A | 1109.9 | 1107.1 | 918.9 | 667.1 (c) | 60% | none | 1.00 |
| 1x120k | dec |  | 0 | 1 | FULL | Dl | 11.7 | 11.7 | 5.9 | 9.3 (m) | 80% | none | 1.00 |
| 1x120k | probe | 0 | 32 | 33 | PIECEWISE | P | 12.0 | 12.0 | 5.8 | 9.3 (m) | 78% | none | 1.00 |
| 1x120k | probe | 0 | 256 | 257 | PIECEWISE | P | 14.2 | 14.2 | 5.9 | 9.3 (m) | 66% | none | 1.00 |
| 1x120k | probe | 0 | 416 | 417 | PIECEWISE | P | 16.8 | 16.8 | 6.0 | 9.3 (m) | 55% | none | 1.00 |
| 1x120k | probe | 0 | 417 | 418 | PIECEWISE | P | 16.8 | 16.8 | 6.0 | 9.3 (m) | 55% | none | 1.00 |
| 1x120k | probe | 0 | 511 | 512 | PIECEWISE | P | 17.4 | 17.3 | 6.1 | 9.3 (m) | 53% | none | 1.00 |
| 1x120k | probe | 0 | 512 | 513 | NONE | X | 62.6 | 18.6 | 46.3 | 9.3 (m) | 15% | cap2048 | 3.36 |
| 1x120k | probe | 0 | 1024 | 1025 | NONE | X | 69.0 | 28.9 | 46.3 | 14.8 (c) | 21% | cap2048 | 2.39 |
| 1x120k | probe | 0 | 1057 | 1058 | NONE | X | 69.8 | 30.7 | 46.5 | 15.3 (c) | 22% | split | 2.34 |
| 1x120k | probe | 0 | 2048 | 2049 | NONE | X | 88.7 | 92.4 | 46.7 | 30.1 (c) | 34% | split | 1.78 |
| 1x120k | probe | 0 | 4096 | 4097 | NONE | X | 130.9 | 132.0 | 47.3 | 62.3 (c) | 48% | separate | 1.34 |
| 1x120k | probe | 0 | 8191 | 8192 | NONE | P | 226.3 | 226.8 | 48.9 | 133.5 (c) | 59% | separate | 1.06 |
| 1x120k | probe | 8192 | 32 | 33 | PIECEWISE | P | 12.4 | 12.3 | 6.2 | 9.6 (m) | 78% | none | 1.00 |
| 1x120k | probe | 8192 | 256 | 257 | PIECEWISE | P | 16.6 | 16.7 | 7.9 | 9.6 (m) | 58% | none | 1.04 |
| 1x120k | probe | 8192 | 416 | 417 | PIECEWISE | P | 18.9 | 18.9 | 8.1 | 9.6 (m) | 51% | none | 1.00 |
| 1x120k | probe | 8192 | 417 | 418 | PIECEWISE | P | 19.1 | 19.2 | 8.2 | 9.6 (m) | 50% | none | 1.00 |
| 1x120k | probe | 8192 | 511 | 512 | PIECEWISE | P | 20.5 | 20.0 | 8.6 | 9.6 (m) | 47% | none | 1.02 |
| 1x120k | probe | 8192 | 512 | 513 | NONE | X | 63.5 | 21.5 | 46.8 | 9.6 (m) | 15% | cap2048 | 2.95 |
| 1x120k | probe | 8192 | 1024 | 1025 | NONE | X | 69.6 | 36.7 | 47.1 | 19.3 (c) | 28% | split | 1.97 |
| 1x120k | probe | 8192 | 1057 | 1058 | NONE | X | 70.3 | 38.9 | 47.0 | 19.9 (c) | 28% | split | 1.93 |
| 1x120k | probe | 8192 | 2048 | 2049 | NONE | X | 89.6 | 93.3 | 47.6 | 39.0 (c) | 44% | split | 1.41 |
| 1x120k | probe | 8192 | 4096 | 4097 | NONE | X | 141.1 | 140.5 | 51.1 | 80.1 (c) | 57% | separate | 1.07 |
| 1x120k | probe | 8192 | 8191 | 8192 | NONE | P | 272.6 | 278.0 | 94.1 | 169.1 (c) | 62% | none | 1.00 |
| 1x120k | probe | 32768 | 32 | 33 | PIECEWISE | A | 13.6 | 14.0 | 7.3 | 10.6 (m) | 78% | none | 1.00 |
| 1x120k | probe | 32768 | 256 | 257 | PIECEWISE | A | 23.5 | 24.1 | 14.4 | 10.6 (m) | 45% | none | 1.04 |
| 1x120k | probe | 32768 | 416 | 417 | PIECEWISE | A | 38.3 | 38.1 | 26.2 | 13.2 (c) | 35% | separate | 1.19 |
| 1x120k | probe | 32768 | 417 | 418 | PIECEWISE | A | 37.7 | 37.7 | 26.3 | 13.3 (c) | 35% | separate | 1.22 |
| 1x120k | probe | 32768 | 511 | 512 | PIECEWISE | A | 39.0 | 42.5 | 27.1 | 16.2 (c) | 42% | separate | 1.22 |
| 1x120k | probe | 32768 | 512 | 513 | NONE | X | 66.0 | 44.5 | 47.7 | 16.3 (c) | 25% | cap2048+att | 1.75 |
| 1x120k | probe | 32768 | 1024 | 1025 | NONE | X | 75.7 | 68.4 | 48.9 | 32.6 (c) | 43% | cap2048+att | 1.23 |
| 1x120k | probe | 32768 | 1057 | 1058 | NONE | X | 77.7 | 68.8 | 49.1 | 33.7 (c) | 43% | cap2048 | 1.13 |
| 1x120k | probe | 32768 | 2048 | 2049 | NONE | A | 115.2 | 117.0 | 71.3 | 65.7 (c) | 57% | none | 1.05 |
| 1x120k | probe | 32768 | 4096 | 4097 | NONE | A | 219.3 | 216.3 | 127.8 | 133.5 (c) | 61% | none | 1.00 |
| 1x120k | probe | 32768 | 8191 | 8192 | NONE | A | 457.8 | 462.4 | 266.6 | 275.8 (c) | 60% | none | 1.00 |
| 1x120k | probe | 122880 | 32 | 33 | PIECEWISE | A | 23.7 | 23.5 | 11.0 | 14.1 (m) | 59% | none | 1.00 |
| 1x120k | probe | 122880 | 256 | 257 | PIECEWISE | A | 64.9 | 65.5 | 50.1 | 20.4 (c) | 31% | separate | 1.32 |
| 1x120k | probe | 122880 | 416 | 417 | PIECEWISE | A | 69.6 | 69.8 | 55.8 | 33.1 (c) | 48% | none | 1.05 |
| 1x120k | probe | 122880 | 417 | 418 | PIECEWISE | A | 71.1 | 70.7 | 56.1 | 33.2 (c) | 47% | none | 1.02 |
| 1x120k | probe | 122880 | 511 | 512 | PIECEWISE | A | 118.1 | 118.0 | 98.8 | 40.7 (c) | 34% | separate | 1.48 |
| 1x120k | probe | 122880 | 512 | 513 | NONE | A | 113.6 | 117.3 | 94.0 | 40.7 (c) | 36% | cap2048+att | 1.45 |
| 1x120k | probe | 122880 | 1024 | 1025 | NONE | A | 137.5 | 139.5 | 114.2 | 81.5 (c) | 59% | none | 1.01 |
| 1x120k | probe | 122880 | 1057 | 1058 | NONE | A | 180.8 | 187.0 | 147.3 | 84.2 (c) | 47% | none | 1.01 |
| 1x120k | probe | 122880 | 2048 | 2049 | NONE | A | 271.0 | 272.4 | 225.5 | 163.5 (c) | 60% | none | 1.00 |
| 1x120k | probe | 122880 | 4096 | 4097 | NONE | A | 540.7 | 540.8 | 448.6 | 329.2 (c) | 61% | none | 1.00 |
| 1x120k | probe | 122880 | 8191 | 8192 | NONE | A | 1107.0 | 1107.5 | 951.3 | 667.1 (c) | 60% | none | 1.03 |
| 3x120k | dec |  | 0 | 3 | FULL | Dl | 22.3 | 22.4 | 16.5 | 18.9 (m) | 85% | none | 1.02 |
| 3x120k | probe | 0 | 32 | 35 | PIECEWISE | P | 22.3 | 22.3 | 16.3 | 18.9 (m) | 85% | none | 1.00 |
| 3x120k | probe | 0 | 256 | 259 | PIECEWISE | P | 24.8 | 24.8 | 16.4 | 18.9 (m) | 76% | none | 1.00 |
| 3x120k | probe | 0 | 416 | 419 | PIECEWISE | P | 27.5 | 27.4 | 16.5 | 18.9 (m) | 69% | none | 1.00 |
| 3x120k | probe | 0 | 417 | 420 | PIECEWISE | P | 27.5 | 27.5 | 16.5 | 18.9 (m) | 69% | none | 1.00 |
| 3x120k | probe | 0 | 509 | 512 | PIECEWISE | P | 28.2 | 28.1 | 16.5 | 18.9 (m) | 67% | none | 1.00 |
| 3x120k | probe | 0 | 510 | 513 | NONE | X | 63.1 | 29.6 | 46.5 | 18.9 (m) | 30% | cap2048+att | 2.14 |
| 3x120k | probe | 0 | 1024 | 1027 | NONE | X | 69.1 | 39.9 | 46.6 | 18.9 (m) | 27% | split | 1.75 |
| 3x120k | probe | 0 | 1057 | 1060 | NONE | X | 69.9 | 41.7 | 46.6 | 18.9 (m) | 27% | split | 1.74 |
| 3x120k | probe | 0 | 2048 | 2051 | NONE | X | 88.8 | 91.5 | 46.8 | 30.3 (c) | 34% | split | 1.49 |
| 3x120k | probe | 0 | 4096 | 4099 | NONE | X | 134.5 | 132.7 | 47.5 | 62.5 (c) | 46% | split | 1.24 |
| 3x120k | probe | 0 | 8189 | 8192 | NONE | P | 230.5 | 237.3 | 52.2 | 133.6 (c) | 58% | none | 1.03 |
| 3x120k | probe | 8192 | 32 | 35 | PIECEWISE | P | 22.7 | 22.7 | 16.6 | 19.2 (m) | 85% | none | 1.00 |
| 3x120k | probe | 8192 | 256 | 259 | PIECEWISE | P | 27.8 | 27.4 | 18.8 | 19.2 (m) | 69% | split | 1.08 |
| 3x120k | probe | 8192 | 416 | 419 | PIECEWISE | P | 27.7 | 27.7 | 17.0 | 19.2 (m) | 69% | none | 1.00 |
| 3x120k | probe | 8192 | 417 | 420 | PIECEWISE | P | 28.2 | 28.0 | 17.0 | 19.2 (m) | 68% | none | 1.00 |
| 3x120k | probe | 8192 | 509 | 512 | PIECEWISE | P | 28.9 | 29.3 | 17.1 | 19.2 (m) | 67% | none | 1.00 |
| 3x120k | probe | 8192 | 510 | 513 | NONE | X | 65.0 | 31.3 | 47.0 | 19.2 (m) | 30% | cap2048 | 2.08 |
| 3x120k | probe | 8192 | 1024 | 1027 | NONE | X | 69.8 | 45.0 | 47.2 | 19.4 (c) | 28% | split | 1.59 |
| 3x120k | probe | 8192 | 1057 | 1060 | NONE | X | 70.7 | 47.4 | 47.4 | 20.0 (c) | 28% | split | 1.57 |
| 3x120k | probe | 8192 | 2048 | 2051 | NONE | X | 93.2 | 96.9 | 48.4 | 39.2 (c) | 42% | split | 1.23 |
| 3x120k | probe | 8192 | 4096 | 4099 | NONE | P | 142.9 | 145.4 | 53.4 | 80.3 (c) | 56% | none | 1.01 |
| 3x120k | probe | 8192 | 8189 | 8192 | NONE | P | 279.2 | 287.4 | 101.7 | 169.2 (c) | 61% | none | 1.00 |
| 3x120k | probe | 32768 | 32 | 35 | PIECEWISE | A | 26.9 | 27.1 | 20.0 | 20.2 (m) | 75% | split | 1.09 |
| 3x120k | probe | 32768 | 256 | 259 | PIECEWISE | A | 37.2 | 37.7 | 27.9 | 20.2 (m) | 54% | separate | 1.12 |
| 3x120k | probe | 32768 | 416 | 419 | PIECEWISE | A | 40.3 | 40.4 | 28.7 | 20.2 (m) | 50% | none | 1.00 |
| 3x120k | probe | 32768 | 417 | 420 | PIECEWISE | A | 40.7 | 41.0 | 29.6 | 20.2 (m) | 50% | none | 1.01 |
| 3x120k | probe | 32768 | 509 | 512 | PIECEWISE | A | 46.5 | 45.4 | 32.8 | 20.2 (m) | 43% | separate | 1.06 |
| 3x120k | probe | 32768 | 510 | 513 | NONE | X | 64.4 | 46.8 | 47.8 | 20.2 (m) | 31% | cap2048+att | 1.44 |
| 3x120k | probe | 32768 | 1024 | 1027 | NONE | X | 74.6 | 69.6 | 52.9 | 32.8 (c) | 44% | separate | 1.12 |
| 3x120k | probe | 32768 | 1057 | 1060 | NONE | X | 77.2 | 71.9 | 54.0 | 33.8 (c) | 44% | split | 1.10 |
| 3x120k | probe | 32768 | 2048 | 2051 | NONE | A | 120.0 | 121.4 | 76.9 | 65.8 (c) | 55% | none | 1.01 |
| 3x120k | probe | 32768 | 4096 | 4099 | NONE | A | 227.3 | 232.7 | 138.5 | 133.7 (c) | 59% | none | 1.00 |
| 3x120k | probe | 32768 | 8189 | 8192 | NONE | A | 466.4 | 468.1 | 275.1 | 275.9 (c) | 59% | none | 1.00 |
| 4x24k | dec |  | 0 | 4 | FULL | Dl | 10.6 | 10.6 | 4.7 | 8.3 (m) | 79% | none | 1.00 |
| 4x24k | probe | 0 | 32 | 36 | PIECEWISE | P | 11.0 | 11.0 | 4.7 | 8.3 (m) | 76% | none | 1.00 |
| 4x24k | probe | 0 | 256 | 260 | PIECEWISE | P | 13.2 | 13.2 | 4.9 | 8.3 (m) | 63% | none | 1.00 |
| 4x24k | probe | 0 | 416 | 420 | PIECEWISE | P | 15.8 | 15.8 | 5.0 | 8.3 (m) | 53% | none | 1.00 |
| 4x24k | probe | 0 | 417 | 421 | PIECEWISE | P | 15.8 | 15.7 | 5.0 | 8.3 (m) | 53% | none | 1.00 |
| 4x24k | probe | 0 | 508 | 512 | PIECEWISE | P | 16.2 | 16.2 | 5.0 | 8.3 (m) | 51% | none | 1.00 |
| 4x24k | probe | 0 | 509 | 513 | NONE | X | 24.1 | 18.0 | 9.7 | 8.3 (m) | 35% | cap2048+att | 1.34 |
| 4x24k | probe | 0 | 1024 | 1028 | NONE | X | 32.3 | 27.4 | 9.8 | 14.8 (c) | 46% | cap2048+att | 1.18 |
| 4x24k | probe | 0 | 1057 | 1061 | NONE | X | 33.7 | 29.8 | 9.8 | 15.3 (c) | 46% | split | 1.15 |
| 4x24k | probe | 0 | 2048 | 2052 | NONE | X | 54.7 | 55.2 | 10.1 | 30.1 (c) | 55% | split | 1.07 |
| 4x24k | probe | 0 | 4096 | 4100 | NONE | P | 101.6 | 102.2 | 12.2 | 62.4 (c) | 61% | none | 1.00 |
| 4x24k | probe | 0 | 8188 | 8192 | NONE | P | 214.5 | 216.5 | 35.0 | 133.5 (c) | 62% | none | 1.01 |
| 4x24k | probe | 8192 | 32 | 36 | PIECEWISE | P | 13.1 | 13.2 | 6.5 | 8.6 (m) | 66% | separate | 1.07 |
| 4x24k | probe | 8192 | 256 | 260 | PIECEWISE | P | 17.4 | 17.6 | 8.3 | 8.7 (m) | 50% | split | 1.15 |
| 4x24k | probe | 8192 | 416 | 420 | PIECEWISE | P | 19.4 | 19.2 | 7.8 | 8.7 (m) | 45% | none | 1.00 |
| 4x24k | probe | 8192 | 417 | 421 | PIECEWISE | P | 18.9 | 18.9 | 7.8 | 8.7 (m) | 46% | none | 1.00 |
| 4x24k | probe | 8192 | 508 | 512 | PIECEWISE | P | 19.6 | 20.1 | 8.2 | 9.6 (c) | 49% | none | 1.00 |
| 4x24k | probe | 8192 | 509 | 513 | NONE | X | 24.4 | 21.2 | 10.2 | 9.6 (c) | 39% | cap2048+att | 1.15 |
| 4x24k | probe | 8192 | 1024 | 1028 | NONE | P | 34.9 | 36.0 | 12.6 | 19.3 (c) | 55% | none | 1.02 |
| 4x24k | probe | 8192 | 1057 | 1061 | NONE | P | 37.9 | 38.1 | 12.2 | 19.9 (c) | 53% | none | 1.02 |
| 4x24k | probe | 8192 | 2048 | 2052 | NONE | P | 66.7 | 67.3 | 21.5 | 39.0 (c) | 58% | none | 1.00 |
| 4x24k | probe | 8192 | 4096 | 4100 | NONE | P | 136.6 | 134.4 | 40.8 | 80.2 (c) | 59% | none | 1.00 |
| 4x24k | probe | 8192 | 8188 | 8192 | NONE | P | 271.4 | 275.2 | 93.4 | 169.0 (c) | 62% | none | 1.01 |
| 4x24k | probe | 32768 | 32 | 36 | PIECEWISE | A | 12.7 | 14.4 | 6.1 | 9.6 (m) | 76% | none | 1.00 |
| 4x24k | probe | 32768 | 256 | 260 | PIECEWISE | A | 22.7 | 21.9 | 13.1 | 9.6 (m) | 42% | none | 1.04 |
| 4x24k | probe | 32768 | 416 | 420 | PIECEWISE | A | 38.6 | 35.9 | 23.9 | 13.3 (c) | 34% | separate | 1.14 |
| 4x24k | probe | 32768 | 417 | 421 | PIECEWISE | A | 38.2 | 35.3 | 24.0 | 13.3 (c) | 35% | separate | 1.18 |
| 4x24k | probe | 32768 | 508 | 512 | PIECEWISE | A | 38.8 | 35.8 | 25.1 | 16.2 (c) | 42% | separate | 1.18 |
| 4x24k | probe | 32768 | 509 | 513 | NONE | A | 41.7 | 37.8 | 23.4 | 16.2 (c) | 39% | cap2048+att | 1.30 |
| 4x24k | probe | 32768 | 1024 | 1028 | NONE | A | 63.3 | 63.8 | 37.9 | 32.6 (c) | 52% | cap2048+att | 1.08 |
| 4x24k | probe | 32768 | 1057 | 1061 | NONE | A | 66.2 | 68.8 | 41.3 | 33.7 (c) | 51% | none | 1.00 |
| 4x24k | probe | 32768 | 2048 | 2052 | NONE | A | 117.5 | 114.6 | 69.2 | 65.7 (c) | 56% | none | 1.04 |
| 4x24k | probe | 32768 | 4096 | 4100 | NONE | A | 217.9 | 219.0 | 128.4 | 133.5 (c) | 61% | none | 1.03 |
| 4x24k | probe | 32768 | 8188 | 8192 | NONE | A | 454.8 | 456.1 | 266.2 | 275.7 (c) | 61% | none | 1.00 |
| 4x24k | probe | 122880 | 32 | 36 | PIECEWISE | A | 27.7 | 27.1 | 14.3 | 13.1 (m) | 47% | split | 1.17 |
| 4x24k | probe | 122880 | 256 | 260 | PIECEWISE | A | 51.0 | 51.5 | 37.4 | 20.4 (c) | 40% | separate | 1.07 |
| 4x24k | probe | 122880 | 416 | 420 | PIECEWISE | A | 69.7 | 69.9 | 56.1 | 33.1 (c) | 48% | none | 1.03 |
| 4x24k | probe | 122880 | 417 | 421 | PIECEWISE | A | 71.4 | 70.8 | 56.5 | 33.2 (c) | 47% | none | 1.00 |
| 4x24k | probe | 122880 | 508 | 512 | PIECEWISE | A | 82.4 | 83.2 | 66.3 | 40.4 (c) | 49% | separate | 1.11 |
| 4x24k | probe | 122880 | 509 | 513 | NONE | A | 82.2 | 82.6 | 64.0 | 40.5 (c) | 49% | cap2048+att | 1.09 |
| 4x24k | probe | 122880 | 1024 | 1028 | NONE | A | 144.8 | 149.9 | 120.6 | 81.6 (c) | 56% | none | 1.05 |
| 4x24k | probe | 122880 | 1057 | 1061 | NONE | A | 177.7 | 184.3 | 145.5 | 84.2 (c) | 47% | none | 1.01 |
| 4x24k | probe | 122880 | 2048 | 2052 | NONE | A | 276.6 | 275.2 | 231.7 | 163.5 (c) | 59% | none | 1.01 |
| 4x24k | probe | 122880 | 4096 | 4100 | NONE | A | 546.6 | 545.2 | 454.4 | 329.2 (c) | 60% | none | 1.00 |
| 4x24k | probe | 122880 | 8188 | 8192 | NONE | A | 1126.7 | 1122.6 | 963.5 | 666.8 (c) | 59% | none | 1.03 |
| 31x1k+1x120k | dec |  | 0 | 32 | FULL | Ds | 13.5 | 13.5 | 7.2 | 10.5 (m) | 78% | none | 1.00 |
| 31x1k+1x120k | probe | 0 | 32 | 64 | PIECEWISE | P | 14.1 | 14.0 | 7.4 | 10.5 (m) | 75% | none | 1.00 |
| 31x1k+1x120k | probe | 0 | 256 | 288 | PIECEWISE | P | 17.3 | 17.1 | 7.4 | 10.5 (m) | 61% | none | 1.00 |
| 31x1k+1x120k | probe | 0 | 416 | 448 | PIECEWISE | P | 23.3 | 23.2 | 7.5 | 10.5 (m) | 45% | none | 1.00 |
| 31x1k+1x120k | probe | 0 | 417 | 449 | PIECEWISE | P | 19.5 | 19.4 | 7.5 | 10.5 (m) | 54% | none | 1.00 |
| 31x1k+1x120k | probe | 0 | 480 | 512 | PIECEWISE | P | 19.1 | 19.1 | 7.5 | 10.5 (m) | 55% | none | 1.00 |
| 31x1k+1x120k | probe | 0 | 481 | 513 | NONE | X | 64.9 | 20.7 | 48.0 | 10.5 (m) | 16% | cap2048+att | 3.13 |
| 31x1k+1x120k | probe | 0 | 1024 | 1056 | NONE | X | 72.6 | 32.9 | 48.1 | 15.3 (c) | 21% | split | 2.23 |
| 31x1k+1x120k | probe | 0 | 1057 | 1089 | NONE | X | 72.4 | 33.5 | 48.2 | 15.8 (c) | 22% | split | 2.24 |
| 31x1k+1x120k | probe | 0 | 2048 | 2080 | NONE | X | 91.6 | 94.8 | 48.3 | 30.6 (c) | 33% | separate | 1.73 |
| 31x1k+1x120k | probe | 0 | 4096 | 4128 | NONE | X | 134.2 | 134.0 | 48.9 | 62.8 (c) | 47% | separate | 1.34 |
| 31x1k+1x120k | probe | 0 | 8160 | 8192 | NONE | P | 229.6 | 227.4 | 50.1 | 133.4 (c) | 58% | separate | 1.06 |
| 31x1k+1x120k | probe | 8192 | 32 | 64 | PIECEWISE | P | 14.5 | 14.5 | 7.7 | 10.8 (m) | 75% | none | 1.00 |
| 31x1k+1x120k | probe | 8192 | 256 | 288 | PIECEWISE | P | 19.5 | 19.4 | 9.3 | 10.9 (m) | 56% | none | 1.03 |
| 31x1k+1x120k | probe | 8192 | 416 | 448 | PIECEWISE | P | 25.6 | 25.5 | 9.8 | 10.9 (m) | 42% | none | 1.00 |
| 31x1k+1x120k | probe | 8192 | 417 | 449 | PIECEWISE | P | 22.3 | 22.0 | 9.9 | 10.9 (m) | 49% | none | 1.00 |
| 31x1k+1x120k | probe | 8192 | 480 | 512 | PIECEWISE | P | 22.2 | 22.3 | 10.0 | 10.9 (m) | 49% | none | 1.00 |
| 31x1k+1x120k | probe | 8192 | 481 | 513 | NONE | X | 66.1 | 23.6 | 48.1 | 10.9 (m) | 16% | cap2048 | 2.80 |
| 31x1k+1x120k | probe | 8192 | 1024 | 1056 | NONE | X | 73.1 | 40.1 | 48.4 | 19.7 (c) | 27% | split | 1.86 |
| 31x1k+1x120k | probe | 8192 | 1057 | 1089 | NONE | X | 72.7 | 42.5 | 48.4 | 20.4 (c) | 28% | split | 1.82 |
| 31x1k+1x120k | probe | 8192 | 2048 | 2080 | NONE | X | 93.3 | 95.7 | 49.0 | 39.5 (c) | 42% | split | 1.37 |
| 31x1k+1x120k | probe | 8192 | 4096 | 4128 | NONE | X | 143.6 | 145.9 | 52.3 | 80.6 (c) | 56% | separate | 1.07 |
| 31x1k+1x120k | probe | 8192 | 8160 | 8192 | NONE | P | 272.3 | 276.3 | 95.2 | 168.8 (c) | 62% | none | 1.00 |
| 31x1k+1x120k | probe | 32768 | 32 | 64 | PIECEWISE | A | 15.6 | 15.6 | 8.8 | 11.8 (m) | 76% | none | 1.00 |
| 31x1k+1x120k | probe | 32768 | 256 | 288 | PIECEWISE | A | 27.2 | 26.8 | 16.0 | 11.8 (m) | 43% | none | 1.04 |
| 31x1k+1x120k | probe | 32768 | 416 | 448 | PIECEWISE | A | 47.6 | 44.2 | 27.5 | 13.7 (c) | 29% | separate | 1.14 |
| 31x1k+1x120k | probe | 32768 | 417 | 449 | PIECEWISE | A | 43.2 | 41.0 | 28.7 | 13.7 (c) | 32% | separate | 1.23 |
| 31x1k+1x120k | probe | 32768 | 480 | 512 | PIECEWISE | A | 43.0 | 40.8 | 28.9 | 15.7 (c) | 37% | separate | 1.21 |
| 31x1k+1x120k | probe | 32768 | 481 | 513 | NONE | X | 65.1 | 42.6 | 48.8 | 15.8 (c) | 24% | cap2048+att | 1.85 |
| 31x1k+1x120k | probe | 32768 | 1024 | 1056 | NONE | X | 76.9 | 69.2 | 50.2 | 33.1 (c) | 43% | cap2048+att | 1.24 |
| 31x1k+1x120k | probe | 32768 | 1057 | 1089 | NONE | X | 78.3 | 71.2 | 50.4 | 34.1 (c) | 44% | cap2048+att | 1.11 |
| 31x1k+1x120k | probe | 32768 | 2048 | 2080 | NONE | A | 119.1 | 120.3 | 73.0 | 66.2 (c) | 56% | none | 1.05 |
| 31x1k+1x120k | probe | 32768 | 4096 | 4128 | NONE | A | 220.7 | 221.1 | 129.6 | 134.0 (c) | 61% | none | 1.00 |
| 31x1k+1x120k | probe | 32768 | 8160 | 8192 | NONE | A | 459.9 | 461.0 | 270.2 | 275.1 (c) | 60% | none | 1.00 |
| 31x1k+1x120k | probe | 122880 | 32 | 64 | PIECEWISE | A | 25.4 | 26.1 | 12.4 | 15.3 (m) | 60% | none | 1.00 |
| 31x1k+1x120k | probe | 122880 | 256 | 288 | PIECEWISE | A | 68.5 | 68.5 | 52.2 | 20.9 (c) | 30% | separate | 1.30 |
| 31x1k+1x120k | probe | 122880 | 416 | 448 | PIECEWISE | A | 75.5 | 76.1 | 57.9 | 33.6 (c) | 44% | none | 1.03 |
| 31x1k+1x120k | probe | 122880 | 417 | 449 | PIECEWISE | A | 74.2 | 73.5 | 58.7 | 33.7 (c) | 45% | none | 1.03 |
| 31x1k+1x120k | probe | 122880 | 480 | 512 | PIECEWISE | A | 75.8 | 76.6 | 60.6 | 38.7 (c) | 51% | none | 1.03 |
| 31x1k+1x120k | probe | 122880 | 481 | 513 | NONE | A | 119.5 | 121.7 | 96.6 | 38.8 (c) | 32% | cap2048+att | 1.43 |
| 31x1k+1x120k | probe | 122880 | 1024 | 1056 | NONE | A | 142.0 | 145.9 | 113.5 | 82.0 (c) | 58% | none | 1.01 |
| 31x1k+1x120k | probe | 122880 | 1057 | 1089 | NONE | A | 180.7 | 189.8 | 150.1 | 84.6 (c) | 47% | none | 1.01 |
| 31x1k+1x120k | probe | 122880 | 2048 | 2080 | NONE | A | 274.1 | 273.6 | 226.4 | 164.0 (c) | 60% | none | 1.00 |
| 31x1k+1x120k | probe | 122880 | 4096 | 4128 | NONE | A | 545.0 | 544.1 | 449.7 | 329.6 (c) | 60% | none | 1.00 |
| 31x1k+1x120k | probe | 122880 | 8160 | 8192 | NONE | A | 1106.2 | 1104.8 | 939.3 | 664.9 (c) | 60% | none | 1.00 |
| 32x1k | dec |  | 0 | 32 | FULL | Dw | 8.2 | 8.2 | 2.0 | 5.8 (m) | 70% | none | 1.01 |
| 32x1k | probe | 0 | 32 | 64 | PIECEWISE | P | 8.6 | 8.6 | 2.1 | 5.8 (m) | 67% | none | 1.00 |
| 32x1k | probe | 0 | 256 | 288 | PIECEWISE | P | 12.0 | 11.9 | 2.2 | 5.8 (m) | 48% | none | 1.01 |
| 32x1k | probe | 0 | 416 | 448 | PIECEWISE | P | 18.6 | 18.2 | 2.3 | 6.4 (c) | 35% | none | 1.01 |
| 32x1k | probe | 0 | 417 | 449 | PIECEWISE | P | 14.3 | 14.3 | 2.3 | 6.4 (c) | 45% | none | 1.01 |
| 32x1k | probe | 0 | 480 | 512 | PIECEWISE | P | 14.0 | 14.0 | 2.4 | 7.3 (c) | 52% | none | 1.01 |
| 32x1k | probe | 0 | 481 | 513 | NONE | P | 16.2 | 15.2 | 2.2 | 7.4 (c) | 45% | cap2048+att | 1.08 |
| 32x1k | probe | 0 | 1024 | 1056 | NONE | P | 27.9 | 27.5 | 2.7 | 15.2 (c) | 55% | none | 1.03 |
| 32x1k | probe | 0 | 1057 | 1089 | NONE | P | 28.5 | 28.8 | 2.7 | 15.7 (c) | 55% | none | 1.00 |
| 32x1k | probe | 0 | 2048 | 2080 | NONE | P | 50.8 | 51.3 | 4.2 | 30.5 (c) | 60% | none | 1.00 |
| 32x1k | probe | 0 | 4096 | 4128 | NONE | P | 102.0 | 103.2 | 9.8 | 62.8 (c) | 62% | none | 1.00 |
| 32x1k | probe | 0 | 8160 | 8192 | NONE | P | 214.3 | 216.0 | 32.6 | 133.3 (c) | 62% | none | 1.01 |
| 32x1k | probe | 8192 | 32 | 64 | PIECEWISE | P | 10.6 | 10.2 | 3.3 | 6.1 (m) | 58% | none | 1.03 |
| 32x1k | probe | 8192 | 256 | 288 | PIECEWISE | P | 14.1 | 14.1 | 4.1 | 6.1 (m) | 43% | none | 1.00 |
| 32x1k | probe | 8192 | 416 | 448 | PIECEWISE | P | 21.6 | 21.9 | 5.8 | 8.2 (c) | 38% | none | 1.01 |
| 32x1k | probe | 8192 | 417 | 449 | PIECEWISE | P | 17.9 | 17.9 | 5.8 | 8.2 (c) | 46% | none | 1.02 |
| 32x1k | probe | 8192 | 480 | 512 | PIECEWISE | P | 17.5 | 17.8 | 5.9 | 9.4 (c) | 54% | none | 1.01 |
| 32x1k | probe | 8192 | 481 | 513 | NONE | P | 19.5 | 19.2 | 5.7 | 9.4 (c) | 48% | none | 1.03 |
| 32x1k | probe | 8192 | 1024 | 1056 | NONE | P | 36.0 | 35.4 | 9.7 | 19.7 (c) | 55% | none | 1.03 |
| 32x1k | probe | 8192 | 1057 | 1089 | NONE | P | 40.1 | 39.9 | 12.3 | 20.3 (c) | 51% | none | 1.02 |
| 32x1k | probe | 8192 | 2048 | 2080 | NONE | P | 67.0 | 68.5 | 18.8 | 39.4 (c) | 59% | none | 1.00 |
| 32x1k | probe | 8192 | 4096 | 4128 | NONE | P | 136.7 | 137.4 | 38.8 | 80.6 (c) | 59% | none | 1.01 |
| 32x1k | probe | 8192 | 8160 | 8192 | NONE | P | 272.9 | 273.2 | 91.0 | 168.8 (c) | 62% | none | 1.01 |
| 32x1k | probe | 32768 | 32 | 64 | PIECEWISE | A | 14.3 | 14.0 | 4.0 | 7.0 (m) | 49% | none | 1.00 |
| 32x1k | probe | 32768 | 256 | 288 | PIECEWISE | A | 23.1 | 21.8 | 9.8 | 8.6 (c) | 37% | none | 1.01 |
| 32x1k | probe | 32768 | 416 | 448 | PIECEWISE | A | 35.6 | 33.8 | 16.7 | 13.6 (c) | 38% | none | 1.03 |
| 32x1k | probe | 32768 | 417 | 449 | PIECEWISE | A | 31.4 | 30.3 | 16.6 | 13.7 (c) | 44% | none | 1.05 |
| 32x1k | probe | 32768 | 480 | 512 | PIECEWISE | A | 31.3 | 30.4 | 17.2 | 15.7 (c) | 50% | none | 1.03 |
| 32x1k | probe | 32768 | 481 | 513 | NONE | A | 33.5 | 32.5 | 16.4 | 15.7 (c) | 47% | cap2048+att | 1.06 |
| 32x1k | probe | 32768 | 1024 | 1056 | NONE | A | 58.5 | 60.1 | 32.2 | 33.0 (c) | 56% | none | 1.01 |
| 32x1k | probe | 32768 | 1057 | 1089 | NONE | A | 72.0 | 73.5 | 41.5 | 34.1 (c) | 47% | none | 1.00 |
| 32x1k | probe | 32768 | 2048 | 2080 | NONE | A | 111.0 | 111.3 | 63.7 | 66.1 (c) | 60% | none | 1.01 |
| 32x1k | probe | 32768 | 4096 | 4128 | NONE | A | 220.7 | 226.6 | 127.0 | 133.9 (c) | 61% | none | 1.00 |
| 32x1k | probe | 32768 | 8160 | 8192 | NONE | A | 455.9 | 455.2 | 264.8 | 275.1 (c) | 60% | none | 1.00 |
| 32x1k | probe | 122880 | 32 | 64 | PIECEWISE | A | 20.9 | 20.5 | 7.5 | 10.6 (m) | 51% | none | 1.00 |
| 32x1k | probe | 122880 | 256 | 288 | PIECEWISE | A | 43.5 | 43.7 | 31.3 | 20.8 (c) | 48% | none | 1.01 |
| 32x1k | probe | 122880 | 416 | 448 | PIECEWISE | A | 75.7 | 75.3 | 55.1 | 33.5 (c) | 44% | none | 1.03 |
| 32x1k | probe | 122880 | 417 | 449 | PIECEWISE | A | 73.5 | 73.3 | 56.7 | 33.6 (c) | 46% | separate | 1.08 |
| 32x1k | probe | 122880 | 480 | 512 | PIECEWISE | A | 73.2 | 73.7 | 58.4 | 38.6 (c) | 53% | none | 1.04 |
| 32x1k | probe | 122880 | 481 | 513 | NONE | A | 74.0 | 75.8 | 58.0 | 38.7 (c) | 52% | none | 1.02 |
| 32x1k | probe | 122880 | 1024 | 1056 | NONE | A | 141.7 | 145.6 | 113.6 | 81.9 (c) | 58% | none | 1.01 |
| 32x1k | probe | 122880 | 1057 | 1089 | NONE | A | 178.1 | 185.8 | 147.2 | 84.6 (c) | 47% | none | 1.00 |
| 32x1k | probe | 122880 | 2048 | 2080 | NONE | A | 273.3 | 274.7 | 224.4 | 163.9 (c) | 60% | none | 1.00 |
| 32x1k | probe | 122880 | 4096 | 4128 | NONE | A | 541.7 | 543.4 | 449.1 | 329.6 (c) | 61% | none | 1.00 |
| 32x1k | probe | 122880 | 8160 | 8192 | NONE | A | 1105.4 | 1110.9 | 931.7 | 664.9 (c) | 60% | none | 1.00 |
| 8x8k | dec |  | 0 | 8 | FULL | Dl | 9.3 | 9.3 | 3.3 | 7.0 (m) | 76% | none | 1.00 |
| 8x8k | probe | 0 | 32 | 40 | PIECEWISE | P | 9.6 | 9.7 | 3.4 | 7.0 (m) | 73% | none | 1.00 |
| 8x8k | probe | 0 | 256 | 264 | PIECEWISE | P | 11.9 | 12.0 | 3.5 | 7.1 (m) | 59% | none | 1.00 |
| 8x8k | probe | 0 | 416 | 424 | PIECEWISE | P | 14.4 | 14.5 | 3.6 | 7.1 (m) | 49% | none | 1.00 |
| 8x8k | probe | 0 | 417 | 425 | PIECEWISE | P | 14.3 | 14.3 | 3.6 | 7.1 (m) | 49% | none | 1.00 |
| 8x8k | probe | 0 | 504 | 512 | PIECEWISE | P | 15.0 | 14.9 | 3.7 | 7.3 (c) | 49% | none | 1.00 |
| 8x8k | probe | 0 | 505 | 513 | NONE | P | 17.8 | 16.3 | 3.9 | 7.4 (c) | 41% | cap2048+att | 1.10 |
| 8x8k | probe | 0 | 1024 | 1032 | NONE | P | 26.5 | 26.9 | 4.0 | 14.9 (c) | 56% | none | 1.00 |
| 8x8k | probe | 0 | 1057 | 1065 | NONE | P | 28.7 | 29.4 | 4.0 | 15.4 (c) | 53% | none | 1.00 |
| 8x8k | probe | 0 | 2048 | 2056 | NONE | P | 49.7 | 50.3 | 4.7 | 30.2 (c) | 61% | none | 1.00 |
| 8x8k | probe | 0 | 4096 | 4104 | NONE | P | 101.8 | 102.1 | 10.9 | 62.4 (c) | 61% | none | 1.00 |
| 8x8k | probe | 0 | 8184 | 8192 | NONE | P | 214.1 | 215.1 | 33.5 | 133.4 (c) | 62% | none | 1.00 |
| 8x8k | probe | 8192 | 32 | 40 | PIECEWISE | P | 11.9 | 12.0 | 5.1 | 7.4 (m) | 62% | split | 1.11 |
| 8x8k | probe | 8192 | 256 | 264 | PIECEWISE | P | 12.9 | 12.9 | 4.3 | 7.4 (m) | 57% | none | 1.00 |
| 8x8k | probe | 8192 | 416 | 424 | PIECEWISE | P | 19.5 | 19.0 | 7.7 | 7.9 (c) | 40% | none | 1.02 |
| 8x8k | probe | 8192 | 417 | 425 | PIECEWISE | P | 19.0 | 19.2 | 7.7 | 7.9 (c) | 42% | none | 1.03 |
| 8x8k | probe | 8192 | 504 | 512 | PIECEWISE | P | 19.4 | 19.4 | 7.8 | 9.5 (c) | 49% | none | 1.03 |
| 8x8k | probe | 8192 | 505 | 513 | NONE | P | 21.7 | 21.8 | 7.6 | 9.5 (c) | 44% | none | 1.02 |
| 8x8k | probe | 8192 | 1024 | 1032 | NONE | P | 34.9 | 34.7 | 12.0 | 19.3 (c) | 55% | none | 1.02 |
| 8x8k | probe | 8192 | 1057 | 1065 | NONE | P | 37.8 | 37.3 | 12.2 | 20.0 (c) | 53% | none | 1.01 |
| 8x8k | probe | 8192 | 2048 | 2056 | NONE | P | 67.8 | 66.6 | 20.4 | 39.1 (c) | 58% | none | 1.00 |
| 8x8k | probe | 8192 | 4096 | 4104 | NONE | P | 134.4 | 132.8 | 40.0 | 80.2 (c) | 60% | none | 1.00 |
| 8x8k | probe | 8192 | 8184 | 8192 | NONE | P | 273.0 | 273.9 | 91.8 | 169.0 (c) | 62% | none | 1.01 |
| 8x8k | probe | 32768 | 32 | 40 | PIECEWISE | A | 16.5 | 16.1 | 6.1 | 8.3 (m) | 50% | separate | 1.05 |
| 8x8k | probe | 32768 | 256 | 264 | PIECEWISE | A | 24.7 | 21.7 | 11.7 | 8.3 (m) | 34% | none | 1.02 |
| 8x8k | probe | 32768 | 416 | 424 | PIECEWISE | A | 31.3 | 29.9 | 17.7 | 13.3 (c) | 43% | none | 1.01 |
| 8x8k | probe | 32768 | 417 | 425 | PIECEWISE | A | 31.9 | 30.4 | 18.0 | 13.3 (c) | 42% | none | 1.03 |
| 8x8k | probe | 32768 | 504 | 512 | PIECEWISE | A | 32.4 | 31.3 | 19.0 | 16.1 (c) | 50% | none | 1.03 |
| 8x8k | probe | 32768 | 505 | 513 | NONE | A | 34.0 | 34.6 | 18.7 | 16.1 (c) | 47% | none | 1.02 |
| 8x8k | probe | 32768 | 1024 | 1032 | NONE | A | 57.7 | 57.3 | 34.0 | 32.7 (c) | 57% | none | 1.02 |
| 8x8k | probe | 32768 | 1057 | 1065 | NONE | A | 68.2 | 71.1 | 39.5 | 33.7 (c) | 49% | none | 1.00 |
| 8x8k | probe | 32768 | 2048 | 2056 | NONE | A | 110.1 | 110.3 | 65.3 | 65.8 (c) | 60% | none | 1.01 |
| 8x8k | probe | 32768 | 4096 | 4104 | NONE | A | 218.1 | 221.1 | 126.8 | 133.6 (c) | 61% | none | 1.00 |
| 8x8k | probe | 32768 | 8184 | 8192 | NONE | A | 453.6 | 450.3 | 267.9 | 275.6 (c) | 61% | none | 1.00 |
| 8x8k | probe | 122880 | 32 | 40 | PIECEWISE | A | 23.9 | 24.7 | 11.3 | 11.9 (m) | 50% | split | 1.10 |
| 8x8k | probe | 122880 | 256 | 264 | PIECEWISE | A | 44.3 | 45.1 | 32.9 | 20.5 (c) | 46% | none | 1.02 |
| 8x8k | probe | 122880 | 416 | 424 | PIECEWISE | A | 72.9 | 73.0 | 57.4 | 33.2 (c) | 45% | none | 1.04 |
| 8x8k | probe | 122880 | 417 | 425 | PIECEWISE | A | 73.0 | 73.5 | 58.6 | 33.3 (c) | 46% | separate | 1.06 |
| 8x8k | probe | 122880 | 504 | 512 | PIECEWISE | A | 75.6 | 75.3 | 57.2 | 40.2 (c) | 53% | none | 1.00 |
| 8x8k | probe | 122880 | 505 | 513 | NONE | A | 75.5 | 78.8 | 59.0 | 40.2 (c) | 53% | none | 1.00 |
| 8x8k | probe | 122880 | 1024 | 1032 | NONE | A | 140.8 | 145.2 | 113.8 | 81.6 (c) | 58% | none | 1.01 |
| 8x8k | probe | 122880 | 1057 | 1065 | NONE | A | 177.4 | 186.0 | 146.7 | 84.2 (c) | 47% | none | 1.00 |
| 8x8k | probe | 122880 | 2048 | 2056 | NONE | A | 271.9 | 270.9 | 225.9 | 163.6 (c) | 60% | none | 1.01 |
| 8x8k | probe | 122880 | 4096 | 4104 | NONE | A | 537.6 | 539.9 | 446.8 | 329.2 (c) | 61% | none | 1.00 |
| 8x8k | probe | 122880 | 8184 | 8192 | NONE | A | 1108.3 | 1108.7 | 940.5 | 666.5 (c) | 60% | none | 1.01 |
| 128x2k | dec |  | 0 | 128 | FULL | Dk | 19.0 | 19.0 | 11.6 | 14.7 (m) | 77% | none | 1.01 |
| 128x2k | probe | 0 | 32 | 160 | PIECEWISE | M | 19.8 | 19.7 | 11.8 | 14.7 (m) | 75% | none | 1.00 |
| 128x2k | probe | 0 | 256 | 384 | PIECEWISE | P | 22.6 | 22.6 | 11.9 | 14.7 (m) | 65% | none | 1.00 |
| 128x2k | probe | 0 | 384 | 512 | PIECEWISE | P | 24.5 | 24.4 | 12.0 | 14.7 (m) | 60% | none | 1.01 |
| 128x2k | probe | 0 | 385 | 513 | NONE | P | 27.0 | 26.0 | 12.0 | 14.7 (m) | 55% | none | 1.04 |
| 128x2k | probe | 0 | 416 | 544 | NONE | P | 27.6 | 26.9 | 12.0 | 14.8 (m) | 53% | none | 1.03 |
| 128x2k | probe | 0 | 417 | 545 | NONE | P | 29.5 | 29.6 | 11.9 | 14.8 (m) | 50% | none | 1.00 |
| 128x2k | probe | 0 | 1024 | 1152 | NONE | P | 38.4 | 38.9 | 12.4 | 16.8 (c) | 44% | none | 1.00 |
| 128x2k | probe | 0 | 1057 | 1185 | NONE | P | 41.1 | 40.7 | 12.4 | 17.3 (c) | 42% | none | 1.02 |
| 128x2k | probe | 0 | 2048 | 2176 | NONE | P | 66.5 | 66.3 | 13.9 | 32.1 (c) | 48% | none | 1.00 |
| 128x2k | probe | 0 | 4096 | 4224 | NONE | P | 117.5 | 118.1 | 19.1 | 64.3 (c) | 55% | none | 1.00 |
| 128x2k | probe | 0 | 8064 | 8192 | NONE | P | 228.8 | 227.7 | 41.8 | 133.1 (c) | 58% | none | 1.00 |
| 128x2k | probe | 8192 | 32 | 160 | PIECEWISE | M | 22.9 | 22.9 | 14.3 | 15.1 (m) | 66% | separate | 1.08 |
| 128x2k | probe | 8192 | 256 | 384 | PIECEWISE | P | 26.6 | 26.4 | 15.2 | 15.1 (m) | 57% | none | 1.05 |
| 128x2k | probe | 8192 | 384 | 512 | PIECEWISE | P | 28.1 | 28.7 | 15.3 | 15.1 (m) | 54% | none | 1.00 |
| 128x2k | probe | 8192 | 385 | 513 | NONE | P | 30.6 | 30.1 | 15.3 | 15.1 (m) | 49% | none | 1.02 |
| 128x2k | probe | 8192 | 416 | 544 | NONE | P | 31.2 | 31.0 | 15.3 | 15.1 (m) | 48% | none | 1.01 |
| 128x2k | probe | 8192 | 417 | 545 | NONE | P | 33.3 | 33.4 | 15.4 | 15.1 (m) | 45% | none | 1.00 |
| 128x2k | probe | 8192 | 1024 | 1152 | NONE | P | 45.8 | 46.5 | 19.8 | 21.3 (c) | 46% | none | 1.01 |
| 128x2k | probe | 8192 | 1057 | 1185 | NONE | P | 50.5 | 50.6 | 22.0 | 21.9 (c) | 43% | none | 1.00 |
| 128x2k | probe | 8192 | 2048 | 2176 | NONE | P | 79.4 | 80.4 | 28.4 | 41.0 (c) | 52% | none | 1.01 |
| 128x2k | probe | 8192 | 4096 | 4224 | NONE | P | 145.3 | 146.9 | 48.3 | 82.1 (c) | 57% | none | 1.01 |
| 128x2k | probe | 8192 | 8064 | 8192 | NONE | P | 286.5 | 288.2 | 98.1 | 168.2 (c) | 59% | none | 1.01 |
| 128x2k | probe | 32768 | 32 | 160 | PIECEWISE | A | 27.5 | 27.4 | 17.6 | 16.0 (m) | 58% | separate | 1.17 |
| 128x2k | probe | 32768 | 256 | 384 | PIECEWISE | A | 37.8 | 36.9 | 24.8 | 16.0 (m) | 42% | separate | 1.18 |
| 128x2k | probe | 32768 | 384 | 512 | PIECEWISE | A | 38.8 | 38.6 | 25.6 | 16.0 (m) | 41% | none | 1.01 |
| 128x2k | probe | 32768 | 385 | 513 | NONE | A | 41.8 | 41.0 | 26.0 | 16.0 (m) | 38% | none | 1.02 |
| 128x2k | probe | 32768 | 416 | 544 | NONE | A | 42.0 | 41.8 | 26.0 | 16.0 (m) | 38% | none | 1.01 |
| 128x2k | probe | 32768 | 417 | 545 | NONE | A | 43.3 | 44.4 | 25.9 | 16.0 (m) | 37% | none | 1.01 |
| 128x2k | probe | 32768 | 1024 | 1152 | NONE | A | 68.5 | 70.1 | 42.0 | 34.6 (c) | 50% | none | 1.02 |
| 128x2k | probe | 32768 | 1057 | 1185 | NONE | A | 81.9 | 83.5 | 51.1 | 35.7 (c) | 44% | none | 1.00 |
| 128x2k | probe | 32768 | 2048 | 2176 | NONE | A | 125.4 | 125.7 | 72.9 | 67.7 (c) | 54% | none | 1.01 |
| 128x2k | probe | 32768 | 4096 | 4224 | NONE | A | 236.3 | 234.8 | 135.8 | 135.5 (c) | 57% | none | 1.01 |
| 128x2k | probe | 32768 | 8064 | 8192 | NONE | A | 458.2 | 461.0 | 269.5 | 273.2 (c) | 60% | none | 1.00 |
| 256x1k | dec |  | 0 | 256 | FULL | Dc | 21.5 | 21.5 | 11.8 | 14.7 (m) | 69% | none | 1.01 |
| 256x1k | probe | 0 | 32 | 288 | PIECEWISE | M | 23.8 | 23.7 | 12.4 | 14.7 (m) | 62% | none | 1.02 |
| 256x1k | probe | 0 | 256 | 512 | PIECEWISE | P | 26.0 | 26.1 | 12.5 | 14.7 (m) | 57% | none | 1.02 |
| 256x1k | probe | 0 | 257 | 513 | NONE | P | 28.6 | 27.7 | 12.5 | 14.7 (m) | 52% | none | 1.05 |
| 256x1k | probe | 0 | 416 | 672 | NONE | P | 32.6 | 32.2 | 12.6 | 14.8 (m) | 45% | none | 1.03 |
| 256x1k | probe | 0 | 417 | 673 | NONE | P | 30.6 | 30.9 | 12.3 | 14.8 (m) | 48% | none | 1.01 |
| 256x1k | probe | 0 | 1024 | 1280 | NONE | P | 43.4 | 43.3 | 12.8 | 18.8 (c) | 43% | none | 1.02 |
| 256x1k | probe | 0 | 1057 | 1313 | NONE | P | 44.2 | 45.3 | 12.9 | 19.2 (c) | 44% | none | 1.00 |
| 256x1k | probe | 0 | 2048 | 2304 | NONE | P | 67.8 | 68.7 | 14.3 | 34.0 (c) | 50% | none | 1.00 |
| 256x1k | probe | 0 | 4096 | 4352 | NONE | P | 123.9 | 123.3 | 20.0 | 66.3 (c) | 54% | none | 1.00 |
| 256x1k | probe | 0 | 7936 | 8192 | NONE | P | 227.3 | 227.4 | 38.9 | 132.7 (c) | 58% | none | 1.00 |
| 256x1k | probe | 8192 | 32 | 288 | PIECEWISE | M | 27.8 | 27.7 | 15.2 | 15.1 (m) | 54% | separate | 1.09 |
| 256x1k | probe | 8192 | 256 | 512 | PIECEWISE | P | 29.6 | 30.2 | 15.6 | 15.1 (m) | 51% | none | 1.05 |
| 256x1k | probe | 8192 | 257 | 513 | NONE | P | 32.2 | 31.4 | 15.8 | 15.1 (m) | 47% | none | 1.04 |
| 256x1k | probe | 8192 | 416 | 672 | NONE | P | 36.6 | 36.0 | 16.0 | 15.1 (m) | 41% | none | 1.03 |
| 256x1k | probe | 8192 | 417 | 673 | NONE | P | 34.7 | 35.3 | 16.0 | 15.1 (m) | 43% | none | 1.01 |
| 256x1k | probe | 8192 | 1024 | 1280 | NONE | P | 51.0 | 52.4 | 20.3 | 23.2 (c) | 46% | none | 1.01 |
| 256x1k | probe | 8192 | 1057 | 1313 | NONE | P | 55.9 | 56.2 | 22.4 | 23.8 (c) | 43% | none | 1.01 |
| 256x1k | probe | 8192 | 2048 | 2304 | NONE | P | 82.3 | 83.6 | 28.6 | 42.9 (c) | 52% | none | 1.01 |
| 256x1k | probe | 8192 | 4096 | 4352 | NONE | P | 150.5 | 151.7 | 49.1 | 84.1 (c) | 56% | none | 1.00 |
| 256x1k | probe | 8192 | 7936 | 8192 | NONE | P | 287.2 | 286.3 | 97.0 | 167.2 (c) | 58% | none | 1.01 |
| 256x1k | probe | 32768 | 32 | 288 | PIECEWISE | A | 31.9 | 31.9 | 18.5 | 16.0 (m) | 50% | separate | 1.17 |
| 256x1k | probe | 32768 | 256 | 512 | PIECEWISE | A | 40.4 | 40.4 | 25.2 | 16.0 (m) | 40% | separate | 1.17 |
| 256x1k | probe | 32768 | 257 | 513 | NONE | A | 42.5 | 41.9 | 25.4 | 16.0 (m) | 38% | none | 1.01 |
| 256x1k | probe | 32768 | 416 | 672 | NONE | A | 47.6 | 47.3 | 26.3 | 17.2 (c) | 36% | none | 1.02 |
| 256x1k | probe | 32768 | 417 | 673 | NONE | A | 45.7 | 47.0 | 26.1 | 17.2 (c) | 38% | none | 1.01 |
| 256x1k | probe | 32768 | 1024 | 1280 | NONE | A | 73.4 | 75.2 | 42.3 | 36.5 (c) | 50% | none | 1.02 |
| 256x1k | probe | 32768 | 1057 | 1313 | NONE | A | 86.4 | 88.9 | 50.4 | 37.6 (c) | 44% | none | 1.00 |
| 256x1k | probe | 32768 | 2048 | 2304 | NONE | A | 128.6 | 127.0 | 74.0 | 69.6 (c) | 54% | none | 1.02 |
| 256x1k | probe | 32768 | 4096 | 4352 | NONE | A | 240.2 | 242.2 | 137.6 | 137.4 (c) | 57% | none | 1.01 |
| 256x1k | probe | 32768 | 7936 | 8192 | NONE | A | 456.9 | 459.1 | 265.4 | 270.6 (c) | 59% | none | 1.00 |

## Steps grouped by the alternative that recovers the most

| best alternative | steps | median loss | max loss | scenarios |
|---|---|---|---|---|
| none | 298 | 1.01 | 1.05 | 128x2k, 1x120k, 256x1k, 31x1k+1x120k, 32x1k, 3x120k, 4x24k, 8x8k, none |
| separate | 34 | 1.16 | 1.73 | 128x2k, 1x120k, 256x1k, 31x1k+1x120k, 32x1k, 3x120k, 4x24k, 8x8k |
| split | 27 | 1.41 | 2.34 | 1x120k, 31x1k+1x120k, 3x120k, 4x24k, 8x8k, none |
| cap2048+att | 23 | 1.18 | 3.13 | 1x120k, 31x1k+1x120k, 32x1k, 3x120k, 4x24k, 8x8k, none |
| cap2048 | 6 | 2.60 | 3.36 | 1x120k, 31x1k+1x120k, 3x120k |
| realign | 5 | 1.10 | 1.28 | none |
