# mstar KV admission: main, admission control, and the passes on top of it

Diagrams for the KV admission study on mstar (BAGEL-7B-MoT, 1 H100 per job). Read the groups in order.

1. `01_main_vs_ac/` - why main stalls when the KV pool fills, and how the admission control of Gaurav Shah (AC) prevents the stall.
   - `01_main_oom_loop` - flowchart: the fail-hold-retry loop of main. A held request keeps its pages.
   - `02_ac_gate` - flowchart: the AC gate. Reservation = prompt - cached + max_tokens, strict FIFO, summed fit test.
   - `03_pool_pages_over_time` - line chart (illustrative shape): main pins the pool full, AC admits in a staircase.
2. `02_admission_policy/` - Phase 1 of our branch: a different admission policy on top of AC.
   - `01_sum_vs_peak_fit` - the summed fit adds the peaks. The peak fit takes the highest point of the summed projection.
   - `02_easy_backfill` - Gantt: a short request behind the head runs in the gap if it does not delay the head.
   - `03_bankers_grant_guard` - flowchart: the safe-state test before the pool gives a page to an admitted request.
3. `03_host_time_passes/` - Phase 1c: less host time, and a KV release at completion.
   - `01_plan_cache` - before/after: pass C keeps one cached row per admitted request.
   - `02_park_queued_requests` - state machine: pass D parks a refused request until the admission-state key changes.
   - `03_release_kv_at_completion` - sequence: pass E sends RELEASE_KV at completion, before the client reads the output.
4. `04_results/` - the ablation matrix (jobs 13086-13106), 7 arms, 3 reps, 4 workloads.
   - `01_short_e2e_by_arm` - bar chart, log scale: short-request E2E for each arm and workload.
   - `02_pass_vs_parent` - ratio chart: each pass against its parent arm, with the noise verdict.
   - `03_burst_v3_row_cap` - why the peak fit is worse on burst_v3: the 64-row decode capture cap.

Each diagram folder holds the source `.html`, an exported `.svg` (diagram only) and a 2x `.png` preview with fallback fonts. The numbers on the diagrams are the measured values from the study report. Curves marked "illustrative" show the shape of the mechanism, not measured data.
