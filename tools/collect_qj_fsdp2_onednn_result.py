#!/usr/bin/env python3
"""Extract the eight requested columns from a completed qj5090 FSDP2 run."""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path


FIELDS = [
    "Seq",
    "单卡 GPU 峰值 (GiB)",
    "CPU RSS 峰值 (GB)",
    "Step (s)",
    "TPS",
    "Forward (s)",
    "Backward (s)",
    "Optimizer (s)",
]


def collect(run: Path) -> list[str]:
    if (run / "exit_code.txt").read_text().strip() != "0":
        raise RuntimeError(f"training did not exit successfully: {run}")
    timing = json.loads((run / "case/step_timing.json").read_text())
    resources = json.loads((run / "resources.json").read_text())
    if timing["steps_completed"] != timing["steps_requested"]:
        raise RuntimeError(f"incomplete step timing: {run}")
    if len(timing["rows"]) != timing["steps_requested"]:
        raise RuntimeError(f"missing per-step records: {run}")
    if not all(math.isfinite(row["loss"]) for row in timing["rows"]):
        raise RuntimeError(f"non-finite training loss: {run}")
    if resources["exit_code"] != 0:
        raise RuntimeError(f"resource monitor failed: {run}")
    stable = timing["stable"]
    if not stable["step_sec"] or not math.isfinite(stable["step_sec"]):
        raise RuntimeError(f"missing stable step timing: {run}")
    return [
        str(timing["sequence_length"]),
        f'{resources["peak_single_gpu_used_gib"]:.2f}',
        f'{resources["peak_process_tree_rss_gb"]:.2f}',
        f'{stable["step_sec"]:.3f}',
        f'{stable["tps"]:.2f}',
        f'{stable["forward_sec"]:.3f}',
        f'{stable["backward_sec"]:.3f}',
        f'{stable["optimizer_sec"]:.3f}',
    ]


def main() -> None:
    writer = csv.writer(sys.stdout)
    writer.writerow(FIELDS)
    writer.writerow(collect(Path(sys.argv[1])))


if __name__ == "__main__":
    main()
