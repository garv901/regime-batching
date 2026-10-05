# Diagrams index

- `flaw2_512_step_boundary/` - why a 512-token step boundary disables both CUDA graphs and KV splits (see its README.md)
  - `01_two_paths` - decision flowchart: fast vs slow path at T = 512
  - `02_heuristic_misjudgment` - what the FA3 split heuristic counts vs what the work is
  - `03_straggler_inside_one_layer` - SM-over-time view, 407 us vs 85 us per layer
  - `04_steps_default_vs_shaping` - step timeline and measured results
