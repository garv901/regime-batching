# Flaw 2: the 512-token step boundary

Reading order: 01, 02, 03, 04. Colors are consistent: orange = slow path (default vLLM above 512 tokens), blue = fast path.

1. `01_two_paths/two_paths.html` - Flowchart. A step with T tokens branches on T <= 512. One boundary switches off two optimizations at once (CUDA graph replay and KV splits). Cites flash_attn.py:805-815 and the split-buffer memory reason.
2. `02_heuristic_misjudgment/heuristic_misjudgment.html` - Contrast. Left: what the FA3 heuristic counts (tiles = 8 x ceil(max_q/32), threshold 105.6, 417-token cutoff). Right: the real work, many short tiles plus 8 very long tiles for the long decode. Right for uniform batches, wrong for mixed ones.
3. `03_straggler_inside_one_layer/straggler_inside_one_layer.html` - SM-over-time view of one attention layer. 1 split: 407 us with 124 SMs idle while 8 stream the long KV. Split: 85 us. Phase widths are illustrative; only the totals are measured.
4. `04_steps_default_vs_shaping/steps_default_vs_shaping.html` - Step timeline. Default makes a ~36 ms eager unsplit X step; step shaping cuts the prompt into steps of <= 512 tokens. Includes the measured results panel.
