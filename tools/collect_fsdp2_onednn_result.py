#!/usr/bin/env python3
"""Extract the requested eight-column result from a completed sap4 case."""

import argparse
import csv
import json
import math
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    args = parser.parse_args()
    root = args.run_root
    if (root / "exit_code.txt").read_text().strip() != "0":
        raise RuntimeError("case did not exit successfully")
    timing = json.loads((root / "case" / "step_timing.json").read_text())
    if timing["steps_completed"] != timing["steps_requested"] or not timing["stable"]["step_sec"]:
        raise RuntimeError("case did not complete all requested stable steps")
    if not all(math.isfinite(row["loss"]) for row in timing["rows"]):
        raise RuntimeError("non-finite loss in completed case")

    with (root / "monitor.csv").open(newline="") as handle:
        samples = list(csv.DictReader(handle))
    if not samples:
        raise RuntimeError("no resource samples")
    cpu_peak = max(float(sample["proc_ram_gb"] or 0) for sample in samples)
    gpu_peak_mib = max(
        float(sample[f"proc_gpu{gpu}_mem_mb"] or 0)
        for sample in samples for gpu in range(8)
    )
    if cpu_peak <= 0 or gpu_peak_mib <= 0:
        raise RuntimeError("resource monitor missed the training process")

    stable = timing["stable"]
    result = {
        "Seq": timing["sequence_length"],
        "单卡 GPU 峰值 (GiB)": gpu_peak_mib / 1024,
        "CPU RSS 峰值 (GB)": cpu_peak,
        "Step (s)": stable["step_sec"],
        "TPS": stable["tps"],
        "Forward (s)": stable["forward_sec"],
        "Backward (s)": stable["backward_sec"],
        "Optimizer (s)": stable["optimizer_sec"],
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
