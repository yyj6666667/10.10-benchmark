#!/usr/bin/env python3
"""Use MegaTrain's causal SDPA mask in its single-GPU forward paths."""

from __future__ import annotations

import argparse
from pathlib import Path


DEFAULT = Path("/mnt/hurry-bench/src/megatrain/infinity/model/cpu_master.py")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", type=Path, default=DEFAULT)
    path = parser.parse_args().path
    source = path.read_text()
    old = """        mask = attention_mask.to(ctx.device)
        layer_kwargs = self._build_layer_kwargs(mask, cache_position, position_ids, position_embeddings)
"""
    new = """        mask = attention_mask.to(ctx.device)
        if mask is not None and mask.dim() == 2 and self.config.attn_implementation != 'flash_attention_2':
            mask = self._prepare_4d_causal_mask(mask, self.config.dtype, T)
        layer_kwargs = self._build_layer_kwargs(mask, cache_position, position_ids, position_embeddings)
"""
    occurrences = source.count(old)
    if occurrences == 0 and source.count(new) == 2:
        print(f"already patched: {path}")
        return
    if occurrences != 2:
        raise RuntimeError(f"expected two single-GPU mask sites in {path}, found {occurrences}")
    path.write_text(source.replace(old, new))
    print(f"installed single-GPU causal masks at two sites: {path}")


if __name__ == "__main__":
    main()
