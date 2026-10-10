#!/usr/bin/env python3
"""Record an observed qj5090 case failure for the current benchmark chart."""

from __future__ import annotations

import argparse
import csv
import os
import re
import tempfile
from pathlib import Path


FIELDS = ("model", "backend", "sequence_length", "status", "source", "note")
MODELS = {"glm": "GLM-4.5-Air-106B", "qwen": "Qwen3.5-122B-A10B"}
BACKENDS = {"ktransformers", "pytorch_onednn", "megatrain", "zero_offload"}
GPU_OOM = re.compile(r"CUDA out of memory|torch\.OutOfMemoryError|cudaErrorMemoryAllocation", re.I)
ROOT_CAUSES = (
    (re.compile(r"Watchdog caught collective operation timeout"),
     "NCCL collective timed out after 600 s"),
    (re.compile(r"FileNotFoundError:.*'torchrun'"),
     "nested torchrun launch failed"),
    (re.compile(r"Too many open files"),
     "process file descriptor limit reached"),
    (re.compile(r"Signal 9 \(SIGKILL\) received by PID"),
     "rank received SIGKILL during checkpoint load; host memory exhausted"),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=MODELS)
    parser.add_argument("backend", choices=BACKENDS)
    parser.add_argument("sequence_length", type=int)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("exit_code", type=int)
    parser.add_argument("--table", type=Path,
                        default=Path("/mnt/hurry-bench/results/case_status.csv"))
    args = parser.parse_args()
    if args.exit_code == 0:
        parser.error("only failed cases belong in the status table")
    if not args.run_dir.is_dir():
        parser.error(f"run directory is missing: {args.run_dir}")

    logs = [path for path in (args.run_dir / "launcher.log",
                            args.run_dir / "train.log") if path.exists()]
    evidence = logs[0] if logs else args.run_dir / "exit_code.txt"
    status = "FAILED"
    diagnostic = "training exited without a complete timed result"
    for log in logs:
        for line in log.read_text(errors="replace").splitlines():
            if GPU_OOM.search(line):
                status = "OOM"
                diagnostic = line.strip()[:160]
                evidence = log
                break
        if status == "OOM":
            break
    if status == "FAILED" and logs:
        lines = logs[-1].read_text(errors="replace").splitlines()
        for pattern, description in ROOT_CAUSES:
            if any(pattern.search(line) for line in lines):
                diagnostic = description
                break
        else:
            relevant = [line.strip() for line in lines
                        if any(word in line for word in
                               ("Error", "SIGKILL", "Killed", "Traceback"))]
            if relevant:
                diagnostic = relevant[-1][:160]
    row = {
        "model": MODELS[args.model], "backend": args.backend,
        "sequence_length": str(args.sequence_length), "status": status,
        "source": f"qj5090:{evidence}",
        "note": f"exit={args.exit_code}; {diagnostic}",
    }
    args.table.parent.mkdir(parents=True, exist_ok=True)
    existing = []
    if args.table.exists():
        with args.table.open(newline="") as stream:
            existing = list(csv.DictReader(stream))
    key = (row["model"], row["backend"], row["sequence_length"])
    existing = [old for old in existing
                if (old["model"], old["backend"], old["sequence_length"]) != key]
    existing.append(row)
    existing.sort(key=lambda item: (item["model"], item["backend"],
                                    int(item["sequence_length"])))
    with tempfile.NamedTemporaryFile("w", dir=args.table.parent, delete=False,
                                     newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(existing)
        temp = stream.name
    os.replace(temp, args.table)
    print(row)


if __name__ == "__main__":
    main()
