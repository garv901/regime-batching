"""Print the BAGEL KV pool as the server will build it (model KV spec + the config YAML's `resources:` overrides).

mstar main does not log the KV pool size at startup, so this reads it from the same objects the server uses; no GPU
or weights needed (only llm_config.json from the HF cache). Run with the venv of the worktree under test, from it:
  cd $MSTAR_DIR && .venv/bin/python -I /path/to/kv_probe.py [configs/bagel_single_gpu.yaml]
"""
import sys

import yaml

from mstar.engine.resources.spec import apply_yaml_overrides
from mstar.model.registry import get_model_class

path = sys.argv[1] if len(sys.argv) > 1 else "configs/bagel_single_gpu.yaml"
specs = get_model_class("bagel")(model_path_hf="ByteDance-Seed/BAGEL-7B-MoT", cache_dir=None).get_node_resources()
apply_yaml_overrides(specs, yaml.safe_load(open(path)))
kv = next(s for s in specs if s.resource_key == "kv").config
per_token = kv.num_layers * 2 * kv.num_kv_heads * kv.head_dim * 2  # K and V, bf16
print(kv)
print(f"kv_bytes_per_token={per_token} ({per_token / 1024:.0f} KiB)  bytes_per_page={per_token * kv.page_size} "
      f"({per_token * kv.page_size / 2**20:.1f} MiB)  max_num_pages={kv.max_num_pages} page_size={kv.page_size}  "
      f"pool_tokens={kv.max_num_pages * kv.page_size}  pool_GiB={per_token * kv.max_num_pages * kv.page_size / 2**30:.2f}")
