"""Extract the standard eight-column KT performance row from one qj5090 run."""

from __future__ import annotations

import argparse
import csv
import json
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


def collect(run: Path) -> dict[str, str]:
    status = (run / "exit_code.txt").read_text().strip()
    if status != "0":
        raise ValueError(f"run failed with exit code {status}: {run}")
    timing = json.loads((run / "step_timing/rank_0/step_timing.json").read_text())
    resources = json.loads((run / "resources.json").read_text())
    config = json.loads((run / "run_config.json").read_text())
    if timing["num_stable_steps"] != config["steps"] - config["warmup_steps"]:
        raise ValueError(f"incomplete timing data: {run}")
    if resources["exit_code"] != 0:
        raise ValueError(f"resource monitor reported failure: {run}")
    stable = timing["aggregate_stable"]
    step = stable["step_total_sec"]["mean_sec"]
    return dict(
        zip(
            FIELDS,
            [
                str(config["sequence_length"]),
                f'{resources["peak_single_gpu_used_gib"]:.2f}',
                f'{resources["peak_process_tree_rss_gb"]:.2f}',
                f"{step:.3f}",
                f'{config["tokens_per_step"] / step:.2f}',
                f'{stable["forward_sec"]["mean_sec"]:.3f}',
                f'{stable["backward_sec"]["mean_sec"]:.3f}',
                f'{stable["optimizer_sec"]["mean_sec"]:.3f}',
            ],
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", type=Path)
    args = parser.parse_args()
    rows = sorted((collect(run) for run in args.runs), key=lambda row: int(row["Seq"]))
    writer = csv.DictWriter(sys.stdout, fieldnames=FIELDS)
    writer.writeheader()
    writer.writerows(rows)


if __name__ == "__main__":
    main()
