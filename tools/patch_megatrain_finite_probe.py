#!/usr/bin/env python3
"""Install an opt-in finite-value probe in the pinned MegaTrain worker copy.

The probe is inert unless MEGATRAIN_FINITE_PROBE=1. Non-finite loss returns
without a backward pass so the benchmark harness can stop before AdamW.
"""

from __future__ import annotations

import argparse
from pathlib import Path


DEFAULT = Path("/mnt/hurry-bench/src/megatrain/infinity/model/mp_worker.py")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", type=Path, default=DEFAULT)
    args = parser.parse_args()
    path = args.path
    source = path.read_text()
    if "BenchFinite" in source:
        print(f"already patched: {path}")
        return

    patches = (
        (
            "import logging\n",
            "import logging\nimport os\n",
        ),
        (
            "logger = logging.getLogger(__name__)\n",
            """logger = logging.getLogger(__name__)


def _bench_finite_probe(rank, label, value):
    if rank != 0 or os.environ.get("MEGATRAIN_FINITE_PROBE") != "1":
        return
    finite = bool(value.isfinite().all().item())
    magnitude = float(value.detach().float().abs().max().item()) if finite else float("nan")
    logger.error(f"BenchFinite {label}: finite={finite} absmax={magnitude:.6g}")
""",
        ),
        (
            "    hidden = ctx.emb_gpu(input_ids_gpu)\n    del input_ids_gpu\n",
            "    hidden = ctx.emb_gpu(input_ids_gpu)\n"
            "    _bench_finite_probe(rank, 'embedding', hidden)\n"
            "    del input_ids_gpu\n",
        ),
        (
            "                out = gpu_layer(hidden, **layer_kwargs)\n"
            "                hidden = out[0] if isinstance(out, tuple) else out\n",
            "                out = gpu_layer(hidden, **layer_kwargs)\n"
            "                hidden = out[0] if isinstance(out, tuple) else out\n"
            "                _bench_finite_probe(rank, f'layer {i}', hidden)\n",
        ),
        (
            "    if not torch.isfinite(torch.tensor(loss_val)):\n"
            "        logger.error(f\"Worker {rank}: Loss is {loss_val}!\")\n\n"
            "    loss.backward()\n",
            "    if not torch.isfinite(torch.tensor(loss_val)):\n"
            "        logger.error(f\"Worker {rank}: Loss is {loss_val}!\")\n"
            "        return WorkerResult(\n"
            "            loss_val=loss_val, total_tokens=B * T, valid_tokens=total_valid_tokens\n"
            "        )\n\n"
            "    loss.backward()\n",
        ),
    )
    for old, new in patches:
        if old not in source:
            raise RuntimeError(f"expected MegaTrain source fragment missing: {old[:80]!r}")
        source = source.replace(old, new, 1)
    path.write_text(source)
    print(f"installed finite-value probe: {path}")


if __name__ == "__main__":
    main()
