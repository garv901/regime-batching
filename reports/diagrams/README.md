# Diagrams index

- `flaw2_512_step_boundary/` - why a 512-token step boundary disables both CUDA graphs and KV splits (see its README.md)
  - `01_two_paths` - decision flowchart: fast vs slow path at T = 512
  - `02_heuristic_misjudgment` - what the FA3 split heuristic counts vs what the work is
  - `03_straggler_inside_one_layer` - SM-over-time view, 407 us vs 85 us per layer
  - `04_steps_default_vs_shaping` - step timeline and measured results
- `mstar_kv_admission/` - KV admission on mstar: why main stalls, how admission control (AC) prevents it, and the passes we built on AC (see its README.md)
  - `01_main_vs_ac` - main's fail-hold-retry loop, the AC gate, pool pages over time
  - `02_admission_policy` - summed fit vs peak fit, EASY backfill, Banker's grant guard
  - `03_host_time_passes` - plan cache (C), park queued requests (D), KV release at completion (E)
  - `04_results` - short E2E by arm, each pass vs its parent, the burst_v3 64-row cap

Each folder holds the source `.html` (open it in a browser), an exported `.svg` and a `.png` (2x, transparent
background). The exports hold the diagram only, without the page header and cards. The SVG loads its fonts from
Google Fonts, so offline tools substitute other fonts; the PNG has the correct fonts. Exception: the PNGs in
`mstar_kv_admission/` were rasterized with `rsvg-convert` without those fonts and use fallback fonts.
