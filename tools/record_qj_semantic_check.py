#!/usr/bin/env python3
"""Add a completed fixed-batch loss check to the qj5090 result table."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import tempfile
from pathlib import Path


FIELDS = ("run_id", "model", "backend", "host", "sequence_length",
          "num_gpus", "steps", "first_loss", "last_loss", "status")
MODELS = {"glm": "GLM-4.5-Air", "qwen": "Qwen3.5-122B-A10B"}
BACKENDS = {"megatrain", "zero_offload", "pytorch_fsdp2_onednn_cpu"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=MODELS)
    parser.add_argument("backend", choices=BACKENDS)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--table", type=Path,
                        default=Path("/mnt/hurry-bench/results/semantic_checks.csv"))
    args = parser.parse_args()
    run = args.run_dir
    if (run / "exit_code.txt").read_text().strip() != "0":
        raise RuntimeError(f"semantic run failed: {run}")
    check = json.loads((run / "loss_check.json").read_text())
    initial = check.get("initial_loss", check["losses"][0])
    final = check.get("final_loss", check["losses"][-1])
    if not (check["passed"] and math.isfinite(initial) and math.isfinite(final)
            and final < initial):
        raise RuntimeError(f"loss did not decrease: {run}")
    config = json.loads((run / "run_config.json").read_text())
    row = {
        "run_id": run.name, "model": MODELS[args.model],
        "backend": args.backend, "host": "qj5090",
        "sequence_length": str(config["sequence_length"]), "num_gpus": "8",
        "steps": str(config["steps"]), "first_loss": str(initial),
        "last_loss": str(final), "status": "SUCCESS",
    }
    args.table.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    if args.table.exists():
        with args.table.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
    rows = [old for old in rows if old["run_id"] != run.name]
    rows.append(row)
    rows.sort(key=lambda item: (item["model"], item["backend"], item["run_id"]))
    with tempfile.NamedTemporaryFile("w", dir=args.table.parent, delete=False,
                                     newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
        temp = stream.name
    os.replace(temp, args.table)
    print(row)


if __name__ == "__main__":
    main()
