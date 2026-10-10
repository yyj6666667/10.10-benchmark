#!/usr/bin/env python3
"""Copy layer-specific buffers when MegaTrain reuses GPU layer templates.

GLM router correction biases are checkpointed buffers. MegaTrain streams
parameters into a template but otherwise keeps the representative layer's
buffers for every layer with the same structure. This patch restores the
current layer's buffer values before its forward call.
"""

from __future__ import annotations

import argparse
from pathlib import Path


DEFAULT = Path("/mnt/hurry-bench/src/megatrain/infinity/model")


def patch(path: Path, old: str, new: str) -> None:
    source = path.read_text()
    if "for (cpu_name, cpu_buffer)" in source:
        print(f"already patched: {path}")
        return
    if source.count(old) != 1:
        raise RuntimeError(f"expected one exact unflatten block in {path}")
    path.write_text(source.replace(old, new))
    print(f"installed per-layer buffer sync: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", nargs="?", type=Path, default=DEFAULT)
    directory = parser.parse_args().directory
    copy_buffers = """    for (cpu_name, cpu_buffer), (gpu_name, gpu_buffer) in zip(
        cpu_layer.named_buffers(), gpu_layer.named_buffers(), strict=True
    ):
        if cpu_name != gpu_name or cpu_buffer.shape != gpu_buffer.shape:
            raise RuntimeError(f"layer buffer mismatch: {cpu_name} != {gpu_name}")
        gpu_buffer.data.copy_(cpu_buffer.data, non_blocking=True)

"""
    worker = directory / "mp_worker.py"
    worker_old = """    offset = 0
    for p in gpu_layer.parameters():
        numel = p.numel()
        p.data.copy_(flat[offset:offset + numel].view(p.shape))
        offset += numel

    ctx.buffer_free_events[buffer_idx].record(ctx.compute_stream)
"""
    worker_new = worker_old.replace(
        "    ctx.buffer_free_events[buffer_idx].record(ctx.compute_stream)",
        "    cpu_layer = shared_state.cpu_layers[layer_idx]\n"
        + copy_buffers
        + "    ctx.buffer_free_events[buffer_idx].record(ctx.compute_stream)",
    )
    patch(worker, worker_old, worker_new)

    master = directory / "cpu_master.py"
    master_old = """        offset = 0
        for p in gpu_layer.parameters():
            numel = p.numel()
            p.data.copy_(flat[offset:offset + numel].view(p.shape))
            offset += numel

        ctx.buffer_free_events[buffer_idx].record(ctx.compute_stream)
"""
    master_new = master_old.replace(
        "        ctx.buffer_free_events[buffer_idx].record(ctx.compute_stream)",
        "        cpu_layer = self.cpu_layers[layer_idx]\n"
        + "\n".join("    " + line if line else "" for line in copy_buffers.splitlines())
        + "\n        ctx.buffer_free_events[buffer_idx].record(ctx.compute_stream)",
    )
    patch(master, master_old, master_new)


if __name__ == "__main__":
    main()
