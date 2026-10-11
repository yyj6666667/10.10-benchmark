#!/usr/bin/env python3
"""Release inactive CUDA cache before Qwen's large layer backward allocation."""

from __future__ import annotations

import argparse
from pathlib import Path


DEFAULT = Path("/mnt/hurry-bench/src/megatrain/infinity/model/mp_worker.py")
MARKER = "MEGATRAIN_BACKWARD_CACHE_EVICTION"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", type=Path, default=DEFAULT)
    path = parser.parse_args().path
    source = path.read_text()
    if MARKER in source:
        print(f"already patched: {path}")
        return

    old = "                grads = torch.autograd.grad(\n"
    new = (
        "                if release_layer_slot:\n"
        "                    # MEGATRAIN_BACKWARD_CACHE_EVICTION: the output head\n"
        "                    # left cached blocks that crowd layer backward.\n"
        "                    torch.cuda.empty_cache()\n"
        "\n"
        + old
    )
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"expected one layer backward call, found {count}")
    if "release_layer_slot = os.environ.get('MEGATRAIN_RELEASE_LAYER_SLOT_DURING_HEAD')" not in source:
        raise RuntimeError("head slot release patch must be installed first")
    path.write_text(source.replace(old, new))
    print(f"installed optional backward cache eviction: {path}")


if __name__ == "__main__":
    main()
