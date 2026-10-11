#!/usr/bin/env python3
"""Allocate MegaTrain's flat weights and gradients directly in shared memory.

The opt-in path keeps the original distinct master parameters, H2D flats,
and gradient buffers. It only avoids building each flat or zero gradient in
private memory and then copying it into shared memory.
"""

from __future__ import annotations

import argparse
from pathlib import Path


DEFAULT = Path("/mnt/hurry-bench/src/megatrain/infinity/model/mp_state.py")
MARKER = "MEGATRAIN_DIRECT_SHARED_INIT"


def replace(source: str, old: str, new: str) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"expected one patch site, found {count}: {old[:80]!r}")
    return source.replace(old, new)


def patch(source: str) -> str:
    if MARKER in source:
        return source
    source = replace(
        source,
        "import ctypes\nimport logging\nimport torch\n",
        "import ctypes\nimport json\nimport logging\nimport math\nimport os\nimport time\nimport torch\n",
    )
    source = replace(
        source,
        "class SharedState:\n",
        '''def _empty_shared(shape, dtype):
    """Make a CPU tensor whose first allocation is its shared storage."""
    numel = math.prod(shape)
    nbytes = numel * torch.empty((), dtype=dtype).element_size()
    storage = torch.UntypedStorage._new_shared(nbytes, device='cpu')
    return torch.empty(0, dtype=dtype).set_(storage, 0, (numel,), (1,)).view(shape)


class SharedState:
''',
    )
    source = replace(
        source,
        "        self.device_ids = config.devices\n\n"
        "        # --- Move all CPU master module params to shared memory ---\n",
        "        self.device_ids = config.devices\n"
        "        self.direct_shared_init = os.environ.get('MEGATRAIN_DIRECT_SHARED_INIT') == '1'\n"
        "        init_started = time.perf_counter()\n"
        "\n"
        "        def mark(stage):\n"
        "            print('[shared_init] ' + json.dumps({\n"
        "                'stage': stage,\n"
        "                'elapsed_sec': round(time.perf_counter() - init_started, 3),\n"
        "                'direct_shared_init': self.direct_shared_init,\n"
        "            }), flush=True)\n"
        "\n"
        "        # --- Move all CPU master module params to shared memory ---\n",
    )
    source = replace(
        source,
        "        if cpu_master_model.rotary_emb is not None:\n"
        "            cpu_master_model.rotary_emb.share_memory()\n\n"
        "        # --- Shared-memory weight flat buffers (read by workers for H2D) ---\n",
        "        if cpu_master_model.rotary_emb is not None:\n"
        "            cpu_master_model.rotary_emb.share_memory()\n"
        "        mark('parameters_shared')\n\n"
        "        # --- Shared-memory weight flat buffers (read by workers for H2D) ---\n",
    )
    source = replace(
        source,
        "            flat = torch.empty(numel, dtype=config.dtype)\n"
        "            offset = 0\n",
        "            flat = (_empty_shared((numel,), config.dtype) if self.direct_shared_init\n"
        "                    else torch.empty(numel, dtype=config.dtype))\n"
        "            offset = 0\n",
    )
    source = replace(
        source,
        "            flat.share_memory_()\n"
        "            self.layer_shared_flats.append(flat)\n\n"
        "        # --- Shared-memory gradient accumulators ---\n",
        "            if not self.direct_shared_init:\n"
        "                flat.share_memory_()\n"
        "            self.layer_shared_flats.append(flat)\n"
        "        mark('flats_shared')\n\n"
        "        # --- Shared-memory gradient accumulators ---\n",
    )
    source = replace(
        source,
        "            grad = torch.zeros_like(p.data, device='cpu')\n"
        "            grad.share_memory_()\n"
        "            p.grad = grad\n"
        "            self._param_grads.append(grad)\n\n"
        "        # --- Model metadata (picklable scalars/lists) ---\n",
        "            if self.direct_shared_init:\n"
        "                grad = _empty_shared(tuple(p.shape), p.dtype)\n"
        "                grad.zero_()\n"
        "            else:\n"
        "                grad = torch.zeros_like(p.data, device='cpu')\n"
        "                grad.share_memory_()\n"
        "            p.grad = grad\n"
        "            self._param_grads.append(grad)\n"
        "        mark('gradients_shared')\n\n"
        "        # --- Model metadata (picklable scalars/lists) ---\n",
    )
    source = replace(
        source,
        "        self.result_queues = [_mp_ctx.Queue() for _ in range(self.world_size)]\n\n"
        "    @staticmethod\n",
        "        self.result_queues = [_mp_ctx.Queue() for _ in range(self.world_size)]\n"
        "        mark('state_ready')\n\n"
        "    @staticmethod\n",
    )
    return source


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", type=Path, default=DEFAULT)
    path = parser.parse_args().path
    source = path.read_text()
    updated = patch(source)
    if updated == source:
        print(f"already patched: {path}")
        return
    compile(updated, str(path), "exec")
    path.write_text(updated)
    print(f"installed direct shared-memory initialization: {path}")


if __name__ == "__main__":
    main()
