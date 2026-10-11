#!/usr/bin/env python3
"""Avoid preloading unused GPU embedding/head in the optional offload mode."""

from __future__ import annotations

import argparse
from pathlib import Path


DEFAULT = Path("/mnt/hurry-bench/src/megatrain/infinity/model/mp_worker.py")
OLD = """    # GPU modules
    ctx.emb_gpu = copy.deepcopy(shared_state.embedding).to(device)
    ctx.norm_gpu = copy.deepcopy(shared_state.norm).to(device) if shared_state.norm else None
    ctx.lm_head_gpu = copy.deepcopy(shared_state.lm_head).to(device)

    if shared_state.tied_lm_head and hasattr(ctx.lm_head_gpu, "weight"):
        ctx.lm_head_gpu.weight = ctx.emb_gpu.weight
"""
NEW = """    # GPU modules. In the optional offload mode, each large module is
    # constructed only at the point where its computation needs it.
    offload_emb_head = os.environ.get('MEGATRAIN_OFFLOAD_EMB_HEAD') == '1'
    if offload_emb_head and shared_state.tied_lm_head:
        raise RuntimeError('embedding/head offload requires untied weights')
    ctx.emb_gpu = None if offload_emb_head else copy.deepcopy(shared_state.embedding).to(device)
    ctx.norm_gpu = copy.deepcopy(shared_state.norm).to(device) if shared_state.norm else None
    ctx.lm_head_gpu = None if offload_emb_head else copy.deepcopy(shared_state.lm_head).to(device)

    if shared_state.tied_lm_head and hasattr(ctx.lm_head_gpu, "weight"):
        ctx.lm_head_gpu.weight = ctx.emb_gpu.weight
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", type=Path, default=DEFAULT)
    path = parser.parse_args().path
    source = path.read_text()
    if NEW in source:
        print(f"already patched: {path}")
        return
    if source.count(OLD) != 1 or "MEGATRAIN_OFFLOAD_EMB_HEAD" not in source:
        raise RuntimeError("expected one previously lifecycle-patched worker")
    path.write_text(source.replace(OLD, NEW))
    print(f"patched initial GPU module residency: {path}")


if __name__ == "__main__":
    main()
