#!/usr/bin/env python3
"""Accept two complete timed steps from a user-requested early stop."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

from collect_qj_fsdp2_onednn_result import FIELDS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("result_table", type=Path)
    args = parser.parse_args()
    run = args.run_dir
    timing = json.loads((run / "case/step_timing.json").read_text())
    resources = json.loads((run / "resources.json").read_text())
    stop = json.loads((run / "planned_stop_after_step.json").read_text())
    status = int((run / "exit_code.txt").read_text().strip())
    if status == 0 or resources["exit_code"] == 0:
        raise RuntimeError("this collector is only for an intentional early stop")
    if stop["stop_after_completed_step"] != 4 or timing["warmup_steps"] != 2:
        raise RuntimeError("expected two warmup plus two measured steps")
    rows = timing["rows"]
    if len(rows) != 4 or [row["step"] for row in rows] != [1, 2, 3, 4]:
        raise RuntimeError("the run does not contain exactly four complete steps")
    if not all(math.isfinite(row["loss"]) for row in rows):
        raise RuntimeError("nonfinite loss in a completed step")
    if any(not math.isfinite(row["step_sec"]) or row["step_sec"] <= 0 for row in rows):
        raise RuntimeError("invalid step duration")
    log = (run / "launcher.log").read_text(errors="replace")
    if "torch.OutOfMemoryError" in log or "CUDA out of memory" in log:
        raise RuntimeError("an OOM cannot be accepted as a planned stop")
    measured = rows[2:4]
    mean = lambda field: statistics.fmean(row[field] for row in measured)
    step = mean("step_sec")
    seq = int(timing["sequence_length"])
    batch = int(timing["global_batch_size"])
    selection = {
        "method": "two completed warmup steps plus two completed measured steps",
        "original_steps_requested": timing["steps_requested"],
        "completed_steps": len(rows),
        "measured_step_numbers": [3, 4],
        "planned_stop": stop,
        "process_exit_code": status,
        "stable": {
            "step_sec": step,
            "tps": batch * seq / step,
            "forward_sec": mean("forward_sec"),
            "backward_sec": mean("backward_sec"),
            "optimizer_sec": mean("optimizer_sec"),
        },
        "losses": [row["loss"] for row in rows],
    }
    (run / "selected_timing.json").write_text(json.dumps(selection, indent=2) + "\n")
    output = [
        str(seq),
        f'{resources["peak_single_gpu_used_gib"]:.2f}',
        f'{resources["peak_process_tree_rss_gb"]:.2f}',
        f'{step:.3f}',
        f'{batch * seq / step:.2f}',
        f'{mean("forward_sec"):.3f}',
        f'{mean("backward_sec"):.3f}',
        f'{mean("optimizer_sec"):.3f}',
    ]
    performance = run / "performance.csv"
    if performance.exists():
        raise RuntimeError(f"performance output already exists: {performance}")
    with performance.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(FIELDS)
        writer.writerow(output)
    args.result_table.parent.mkdir(parents=True, exist_ok=True)
    with args.result_table.open("a", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(output)
    print(dict(zip(FIELDS, output)))


if __name__ == "__main__":
    main()
