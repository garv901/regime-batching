### bagel_burst_v2

| step | reps | short E2E mean | short E2E p99 | short TTFT p99 | long E2E mean | out tok/s | all E2E mean | all E2E p50 | all E2E p99 | all TTFT p50 | all TTFT p99 | host ms/step | host ms/step @400+ | planner µs p50 | KV free lag p50 | KV free lag p99 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ac / main | 3/1 | 0.05 (1 rep) | 0.13 (1 rep) | 0.11 (1 rep) | - | 12.04 (1 rep) | 0.05 (1 rep) | 0.00 (1 rep) | 0.14 (1 rep) | 0.00 (1 rep) | 0.12 (1 rep) | 1.70 (1 rep) | - | - | 0.46 (1 rep) | 0.55 (1 rep) |
| ac_park / ac | 3/3 | 0.49 | 0.74 | 0.72 | 0.73 | 1.00 ≈ | 0.51 | 0.35 | 0.74 | 0.37 | 0.73 | 0.68 | 0.38 | 0.19 | 0.49 | 0.81 ≈ |
| peak_bf_pc_park / ac | 3/3 | 0.04 | 0.08 | 0.05 | 0.74 | 1.00 ≈ | 0.10 | 0.33 | 0.77 | 0.34 | 0.07 | 0.66 | - | 0.71 | 0.43 | 0.54 |
| peak_bf_pc_park / ac_park | 3/3 | 0.09 | 0.11 | 0.07 | 1.02 ≈ | 1.00 ≈ | 0.20 | 0.93 ≈ | 1.03 ≈ | 0.93 ≈ | 0.10 | 0.96 ≈ | - | 3.73 | 0.88 ≈ | 0.67 ≈ |
| peak_bf_pc_park / main | 3/1 | 0.00 (1 rep) | 0.01 (1 rep) | 0.01 (1 rep) | - | 12.04 (1 rep) | 0.00 (1 rep) | 0.00 (1 rep) | 0.11 (1 rep) | 0.00 (1 rep) | 0.01 (1 rep) | 1.12 (1 rep) | - | - | 0.20 (1 rep) | 0.30 (1 rep) |

### bagel_stag_v3_r5.5

| step | reps | short E2E mean | short E2E p99 | short TTFT p99 | long E2E mean | out tok/s | all E2E mean | all E2E p50 | all E2E p99 | all TTFT p50 | all TTFT p99 | host ms/step | host ms/step @400+ | planner µs p50 | KV free lag p50 | KV free lag p99 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ac / main | 3/1 | 1.25 (1 rep) | 1.74 (1 rep) | 2.28 (1 rep) | 0.93 (1 rep) | 1.00 (1 rep) | 1.00 (1 rep) | 0.99 (1 rep) | 0.86 (1 rep) | 1.00 (1 rep) | 1.13 (1 rep) | 1.13 (1 rep) | - | - | 0.97 (1 rep) | 0.95 (1 rep) |
| ac_park / ac | 3/3 | 1.16 ≈ | 1.17 ≈ | 1.23 ≈ | 1.04 ≈ | 1.00 ≈ | 1.07 ≈ | 1.05 ≈ | 1.04 ≈ | 1.07 ≈ | 1.23 ≈ | 1.04 ≈ | - | 1.03 ≈ | 1.05 ≈ | 1.09 ≈ |
| peak_bf_pc_park / ac | 3/3 | 0.82 ≈ | 0.75 ≈ | 0.62 ≈ | 1.05 ≈ | 1.00 ≈ | 1.00 ≈ | 1.00 ≈ | 1.08 ≈ | 0.94 ≈ | 0.86 ≈ | 1.04 ≈ | - | 2.43 | 1.05 ≈ | 1.03 ≈ |
| peak_bf_pc_park / ac_park | 3/3 | 0.71 ≈ | 0.64 ≈ | 0.51 ≈ | 1.01 ≈ | 1.00 ≈ | 0.93 ≈ | 0.96 ≈ | 1.04 ≈ | 0.88 ≈ | 0.70 ≈ | 1.00 ≈ | - | 2.36 | 0.99 ≈ | 0.94 ≈ |
| peak_bf_pc_park / main | 3/1 | 1.03 (1 rep) | 1.30 (1 rep) | 1.42 (1 rep) | 0.98 (1 rep) | 1.00 (1 rep) | 0.99 (1 rep) | 0.99 (1 rep) | 0.93 (1 rep) | 0.94 (1 rep) | 0.97 (1 rep) | 1.17 (1 rep) | - | - | 1.02 (1 rep) | 0.98 (1 rep) |

### bagel_burst_v2_mtx2

| step | reps | short E2E mean | short E2E p99 | short TTFT p99 | long E2E mean | out tok/s | all E2E mean | all E2E p50 | all E2E p99 | all TTFT p50 | all TTFT p99 | host ms/step | host ms/step @400+ | planner µs p50 | KV free lag p50 | KV free lag p99 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ac / main | 3/1 | 0.04 (1 rep) | 0.12 (1 rep) | 0.10 (1 rep) | - | 11.33 (1 rep) | 0.04 (1 rep) | 0.00 (1 rep) | 0.13 (1 rep) | 0.00 (1 rep) | 0.11 (1 rep) | 1.37 (1 rep) | - | - | 0.28 (1 rep) | 0.90 (1 rep) |
| ac_park / ac | 3/3 | 0.72 | 0.85 | 0.83 | 0.86 | 1.00 ≈ | 0.74 | 0.71 | 0.86 | 0.78 | 0.83 | 0.88 ≈ | 0.59 | 0.46 | 0.74 | 0.71 |
| peak_bf_pc_park / ac | 3/3 | 0.06 | 0.11 | 0.07 | 0.84 ≈ | 1.00 ≈ | 0.13 | 0.65 | 0.85 ≈ | 0.67 | 0.08 | 0.79 ≈ | - | 1.48 ≈ | 0.58 | 0.58 |
| peak_bf_pc_park / ac_park | 3/3 | 0.08 | 0.13 | 0.08 | 0.98 ≈ | 1.00 ≈ | 0.18 | 0.91 | 0.99 ≈ | 0.86 | 0.10 | 0.90 ≈ | - | 3.25 | 0.79 | 0.81 ≈ |
| peak_bf_pc_park / main | 3/1 | 0.00 (1 rep) | 0.01 (1 rep) | 0.01 (1 rep) | - | 11.33 (1 rep) | 0.01 (1 rep) | 0.00 (1 rep) | 0.11 (1 rep) | 0.00 (1 rep) | 0.01 (1 rep) | 1.09 (1 rep) | - | - | 0.16 (1 rep) | 0.52 (1 rep) |

### bagel_burst_v2_mt2048

| step | reps | short E2E mean | short E2E p99 | short TTFT p99 | long E2E mean | out tok/s | all E2E mean | all E2E p50 | all E2E p99 | all TTFT p50 | all TTFT p99 | host ms/step | host ms/step @400+ | planner µs p50 | KV free lag p50 | KV free lag p99 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ac / main | 3/1 | 0.09 (1 rep) | 0.13 (1 rep) | 0.12 (1 rep) | - | 13.21 (1 rep) | 0.09 (1 rep) | 0.06 (1 rep) | 0.13 (1 rep) | 0.06 (1 rep) | 0.13 (1 rep) | 13.82 (1 rep) | - | - | 0.83 (1 rep) | 0.76 (1 rep) |
| ac_park / ac | 3/3 | 0.45 ≈ | 0.78 ≈ | 0.78 ≈ | 0.79 ≈ | 1.05 | 0.47 ≈ | 0.27 ≈ | 0.79 ≈ | 0.23 ≈ | 0.79 ≈ | 0.14 ≈ | 0.59 ≈ | 0.81 ≈ | 0.77 ≈ | 1.85 ≈ |
| peak_bf_pc_park / ac | 3/3 | 0.62 ≈ | 0.79 ≈ | 0.77 ≈ | 1.00 ≈ | 1.05 ≈ | 0.64 ≈ | 0.58 ≈ | 0.94 ≈ | 0.48 ≈ | 0.77 ≈ | 0.62 ≈ | 0.66 ≈ | 0.73 | 1.15 ≈ | 1.99 ≈ |
| peak_bf_pc_park / ac_park | 3/3 | 1.37 ≈ | 1.02 ≈ | 1.00 ≈ | 1.27 ≈ | 1.00 ≈ | 1.36 ≈ | 2.15 ≈ | 1.19 ≈ | 2.09 ≈ | 0.98 ≈ | 4.49 ≈ | 1.12 ≈ | 0.90 ≈ | 1.50 ≈ | 1.08 ≈ |
| peak_bf_pc_park / main | 3/1 | 0.06 (1 rep) | 0.10 (1 rep) | 0.10 (1 rep) | - | 13.91 (1 rep) | 0.06 (1 rep) | 0.03 (1 rep) | 0.12 (1 rep) | 0.03 (1 rep) | 0.10 (1 rep) | 8.56 (1 rep) | - | - | 0.95 (1 rep) | 1.51 (1 rep) |

### bagel_stag_v3_r5.5_mtx2

| step | reps | short E2E mean | short E2E p99 | short TTFT p99 | long E2E mean | out tok/s | all E2E mean | all E2E p50 | all E2E p99 | all TTFT p50 | all TTFT p99 | host ms/step | host ms/step @400+ | planner µs p50 | KV free lag p50 | KV free lag p99 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ac / main | 3/1 | 0.05 (1 rep) | 0.05 (1 rep) | 0.05 (1 rep) | 0.32 (1 rep) | 2.32 (1 rep) | 0.07 (1 rep) | 1.08 (1 rep) | 0.10 (1 rep) | 0.94 (1 rep) | 0.05 (1 rep) | 1.35 (1 rep) | - | - | 0.92 (1 rep) | 1.05 (1 rep) |
| ac_park / ac | 3/3 | 0.26 ≈ | 0.27 ≈ | 0.24 ≈ | 0.79 ≈ | 1.04 ≈ | 0.49 ≈ | 0.78 ≈ | 0.68 ≈ | 0.69 ≈ | 0.24 ≈ | 0.89 ≈ | - | 0.97 ≈ | 0.89 ≈ | 1.07 ≈ |
| peak_bf_pc_park / ac | 3/3 | 0.13 | 0.08 | 0.03 | 0.74 ≈ | 1.04 ≈ | 0.39 | 0.64 | 0.64 ≈ | 0.41 | 0.06 | 0.75 ≈ | - | 2.29 | 0.85 ≈ | 0.98 ≈ |
| peak_bf_pc_park / ac_park | 3/3 | 0.52 | 0.30 | 0.13 | 0.93 ≈ | 1.00 ≈ | 0.81 | 0.83 | 0.93 ≈ | 0.59 | 0.25 | 0.84 ≈ | - | 2.37 | 0.95 ≈ | 0.92 ≈ |
| peak_bf_pc_park / main | 3/1 | 0.01 (1 rep) | 0.00 (1 rep) | 0.00 (1 rep) | 0.24 (1 rep) | 2.40 (1 rep) | 0.03 (1 rep) | 0.70 (1 rep) | 0.06 (1 rep) | 0.39 (1 rep) | 0.00 (1 rep) | 1.01 (1 rep) | - | - | 0.78 (1 rep) | 1.03 (1 rep) |

### bagel_stag_v3_r5.5_mt2048

| step | reps | short E2E mean | short E2E p99 | short TTFT p99 | long E2E mean | out tok/s | all E2E mean | all E2E p50 | all E2E p99 | all TTFT p50 | all TTFT p99 | host ms/step | host ms/step @400+ | planner µs p50 | KV free lag p50 | KV free lag p99 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ac / main | 3/1 | 0.01 (1 rep) | 0.01 (1 rep) | 0.01 (1 rep) | 0.08 (1 rep) | 7.35 (1 rep) | 0.01 (1 rep) | 0.00 (1 rep) | 0.07 (1 rep) | 0.00 (1 rep) | 0.01 (1 rep) | 1.01 (1 rep) | - | - | 0.39 (1 rep) | 0.60 (1 rep) |
| ac_park / ac | 3/3 | 1.65 ≈ | 1.30 ≈ | 1.34 ≈ | 1.11 ≈ | 1.00 ≈ | 1.29 ≈ | 2.44 ≈ | 1.11 ≈ | 2.62 ≈ | 1.34 ≈ | 1.21 ≈ | - | 1.04 ≈ | 1.10 ≈ | 1.13 ≈ |
| peak_bf_pc_park / ac | 3/3 | 0.75 ≈ | 0.78 ≈ | 0.71 ≈ | 1.09 ≈ | 1.00 ≈ | 0.97 ≈ | 1.08 ≈ | 1.11 ≈ | 1.10 ≈ | 0.80 ≈ | 1.16 ≈ | - | 2.06 | 1.16 ≈ | 1.19 ≈ |
| peak_bf_pc_park / ac_park | 3/3 | 0.46 ≈ | 0.60 ≈ | 0.53 ≈ | 0.98 ≈ | 1.00 ≈ | 0.76 ≈ | 0.44 ≈ | 1.01 ≈ | 0.42 ≈ | 0.60 ≈ | 0.96 ≈ | - | 1.99 | 1.05 ≈ | 1.05 ≈ |
| peak_bf_pc_park / main | 3/1 | 0.00 (1 rep) | 0.01 (1 rep) | 0.01 (1 rep) | 0.09 (1 rep) | 7.35 (1 rep) | 0.01 (1 rep) | 0.00 (1 rep) | 0.07 (1 rep) | 0.00 (1 rep) | 0.01 (1 rep) | 1.17 (1 rep) | - | - | 0.45 (1 rep) | 0.72 (1 rep) |

### pub_mooncake_conv_r0.5

| step | reps | short E2E mean | short E2E p99 | short TTFT p99 | long E2E mean | out tok/s | all E2E mean | all E2E p50 | all E2E p99 | all TTFT p50 | all TTFT p99 | host ms/step | host ms/step @400+ | planner µs p50 | KV free lag p50 | KV free lag p99 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ac / main | 3/2 | 0.81 | 0.74 | 0.61 | 0.48 | 1.01 | 0.71 | 0.79 | 0.28 | 1.08 ≈ | 0.14 | 0.60 ≈ | - | - | 0.90 ≈ | 1.08 ≈ |
| ac_park / ac | 3/3 | 1.03 ≈ | 1.07 ≈ | 1.03 ≈ | 1.03 ≈ | 1.00 ≈ | 1.03 ≈ | 1.03 ≈ | 1.03 ≈ | 1.03 ≈ | 1.04 ≈ | 0.99 ≈ | - | 1.01 ≈ | 1.01 ≈ | 1.07 ≈ |
| peak_bf_pc_park / ac | 3/3 | 0.91 | 0.95 ≈ | 0.53 | 1.03 ≈ | 1.00 ≈ | 0.96 | 0.97 ≈ | 1.05 ≈ | 0.79 | 1.16 | 0.85 ≈ | - | 1.92 | 0.99 ≈ | 1.02 ≈ |
| peak_bf_pc_park / ac_park | 3/3 | 0.88 | 0.89 | 0.52 | 1.01 ≈ | 1.00 ≈ | 0.93 | 0.94 | 1.02 ≈ | 0.77 | 1.12 | 0.86 | - | 1.89 | 0.98 ≈ | 0.95 ≈ |
| peak_bf_pc_park / main | 3/2 | 0.74 | 0.71 | 0.33 | 0.50 | 1.01 | 0.68 | 0.77 | 0.29 | 0.85 | 0.17 | 0.51 | - | - | 0.88 | 1.09 ≈ |

### pub_mooncake_conv_r0.5_mtx2

| step | reps | short E2E mean | short E2E p99 | short TTFT p99 | long E2E mean | out tok/s | all E2E mean | all E2E p50 | all E2E p99 | all TTFT p50 | all TTFT p99 | host ms/step | host ms/step @400+ | planner µs p50 | KV free lag p50 | KV free lag p99 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ac / main | 3/2 | 0.85 | 0.81 | 0.78 | 0.52 | 1.00 | 0.76 | 0.83 | 0.42 | 1.04 ≈ | 0.24 | 0.63 ≈ | - | - | 0.93 ≈ | 1.23 |
| ac_park / ac | 3/3 | 1.05 ≈ | 1.03 ≈ | 1.01 ≈ | 1.03 ≈ | 1.00 ≈ | 1.04 ≈ | 1.05 ≈ | 1.02 ≈ | 1.03 ≈ | 1.03 ≈ | 1.10 ≈ | - | 1.01 ≈ | 1.02 ≈ | 0.93 ≈ |
| peak_bf_pc_park / ac | 3/3 | 0.91 | 0.92 | 0.52 | 1.03 ≈ | 1.00 ≈ | 0.96 ≈ | 0.98 ≈ | 1.05 ≈ | 0.80 | 1.14 | 0.96 ≈ | - | 1.88 | 0.99 ≈ | 0.94 ≈ |
| peak_bf_pc_park / ac_park | 3/3 | 0.87 | 0.90 | 0.52 | 1.00 ≈ | 1.00 ≈ | 0.92 | 0.93 | 1.03 ≈ | 0.77 | 1.12 | 0.88 ≈ | - | 1.85 | 0.98 ≈ | 1.01 ≈ |
| peak_bf_pc_park / main | 3/2 | 0.78 | 0.75 | 0.41 | 0.54 | 1.00 | 0.73 | 0.81 | 0.44 | 0.83 | 0.28 | 0.60 | - | - | 0.93 | 1.15 ≈ |

### pub_mooncake_conv_r0.5_mt2048

| step | reps | short E2E mean | short E2E p99 | short TTFT p99 | long E2E mean | out tok/s | all E2E mean | all E2E p50 | all E2E p99 | all TTFT p50 | all TTFT p99 | host ms/step | host ms/step @400+ | planner µs p50 | KV free lag p50 | KV free lag p99 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ac / main | 3/2 | 0.07 | 0.10 | 0.05 | 0.12 | 1.62 | 0.08 | 0.05 | 0.10 | 0.02 ≈ | 0.05 | 0.56 ≈ | - | - | 0.77 | 1.27 |
| ac_park / ac | 3/3 | 0.95 ≈ | 0.93 | 0.96 ≈ | 0.96 ≈ | 1.00 | 0.96 ≈ | 0.97 ≈ | 0.93 ≈ | 0.96 ≈ | 0.95 ≈ | 0.77 ≈ | - | 1.01 ≈ | 0.94 ≈ | 1.05 ≈ |
| peak_bf_pc_park / ac | 3/3 | 0.86 | 0.84 | 0.52 | 1.07 ≈ | 1.00 | 0.94 ≈ | 0.97 ≈ | 1.01 ≈ | 0.81 | 1.19 | 0.89 ≈ | - | 1.68 | 1.00 ≈ | 1.05 ≈ |
| peak_bf_pc_park / ac_park | 3/3 | 0.91 | 0.90 | 0.54 | 1.11 | 1.00 ≈ | 0.98 ≈ | 0.99 ≈ | 1.09 | 0.84 | 1.25 | 1.15 ≈ | - | 1.66 | 1.06 ≈ | 1.00 ≈ |
| peak_bf_pc_park / main | 3/2 | 0.06 | 0.09 | 0.03 | 0.13 | 1.62 | 0.08 | 0.05 | 0.10 | 0.02 | 0.06 | 0.50 | - | - | 0.77 | 1.33 ≈ |

### pub_azure_conv_r1.25

| step | reps | short E2E mean | short E2E p99 | short TTFT p99 | long E2E mean | out tok/s | all E2E mean | all E2E p50 | all E2E p99 | all TTFT p50 | all TTFT p99 | host ms/step | host ms/step @400+ | planner µs p50 | KV free lag p50 | KV free lag p99 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ac / main | 3/1 | 1.68 (1 rep) | 1.96 (1 rep) | 3.46 (1 rep) | 1.65 (1 rep) | 0.96 (1 rep) | 1.73 (1 rep) | 1.48 (1 rep) | 1.88 (1 rep) | 1.30 (1 rep) | 3.13 (1 rep) | 1.65 (1 rep) | - | - | 1.35 (1 rep) | 2.26 (1 rep) |
| ac_park / ac | 3/3 | 0.86 ≈ | 0.85 ≈ | 0.68 ≈ | 0.85 ≈ | 1.01 ≈ | 0.87 ≈ | 0.90 ≈ | 0.86 ≈ | 0.91 ≈ | 0.68 ≈ | 0.92 ≈ | - | 1.01 ≈ | 0.88 ≈ | 0.45 |
| peak_bf_pc_park / ac | 3/3 | 0.68 | 0.58 | 0.32 | 0.67 | 1.03 | 0.66 | 0.78 ≈ | 0.57 | 0.84 ≈ | 0.32 | 0.71 ≈ | - | 1.36 | 0.80 ≈ | 0.43 |
| peak_bf_pc_park / ac_park | 3/3 | 0.79 ≈ | 0.68 ≈ | 0.48 ≈ | 0.79 ≈ | 1.02 ≈ | 0.76 ≈ | 0.86 ≈ | 0.67 ≈ | 0.92 ≈ | 0.47 ≈ | 0.77 ≈ | - | 1.35 | 0.91 ≈ | 0.96 ≈ |
| peak_bf_pc_park / main | 3/1 | 1.15 (1 rep) | 1.13 (1 rep) | 1.12 (1 rep) | 1.11 (1 rep) | 0.99 (1 rep) | 1.15 (1 rep) | 1.15 (1 rep) | 1.08 (1 rep) | 1.09 (1 rep) | 1.00 (1 rep) | 1.16 (1 rep) | - | - | 1.08 (1 rep) | 0.97 (1 rep) |

