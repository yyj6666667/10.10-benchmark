#!/usr/bin/env python3
"""Mark longer qj5090 cases skipped after a sequence-independent base failure."""

from __future__ import annotations

import argparse
import csv
import os
import tempfile
from pathlib import Path

from record_qj_case_status import BACKENDS, FIELDS, MODELS


SEQUENCES = (32, 64, 128, 256, 512, 1024, 2048, 4096)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=MODELS)
    parser.add_argument("backend", choices=BACKENDS)
    parser.add_argument("--after", type=int, choices=SEQUENCES, required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--reason", required=True)
    parser.add_argument(
        "--replace-existing-blocked", action="store_true",
        help="refresh older BLOCKED rows after a new observed limiting case",
    )
    parser.add_argument(
        "--table", type=Path,
        default=Path("/mnt/hurry-bench/results/case_status.csv"),
    )
    args = parser.parse_args()
    if not args.source.startswith("qj5090:"):
        parser.error("source must identify the qj5090 evidence path")
    if not Path(args.source.removeprefix("qj5090:")).exists():
        parser.error("source evidence does not exist")
    with args.table.open(newline="") as stream:
        existing = list(csv.DictReader(stream))
    base = (MODELS[args.model], args.backend, str(args.after))
    if not any(
        (row["model"], row["backend"], row["sequence_length"]) == base
        and row["status"] in {"OOM", "FAILED", "BLOCKED"}
        for row in existing
    ):
        parser.error("the base case must have a recorded failure or limit")
    keys = {
        (row["model"], row["backend"], row["sequence_length"])
        for row in existing
    }
    refreshed = 0
    if args.replace_existing_blocked:
        for row in existing:
            if (row["model"] == MODELS[args.model]
                    and row["backend"] == args.backend
                    and int(row["sequence_length"]) > args.after
                    and row["status"] == "BLOCKED"):
                row["source"] = args.source
                row["note"] = args.reason
                refreshed += 1
    added = []
    for seq in SEQUENCES:
        if seq <= args.after:
            continue
        key = MODELS[args.model], args.backend, str(seq)
        if key in keys:
            continue
        row = dict(zip(FIELDS, (*key, "BLOCKED", args.source, args.reason)))
        existing.append(row)
        added.append(row)
    existing.sort(key=lambda row: (row["model"], row["backend"],
                                   int(row["sequence_length"])))
    with tempfile.NamedTemporaryFile(
        "w", dir=args.table.parent, delete=False, newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(existing)
        temp = stream.name
    os.replace(temp, args.table)
    print(f"marked {len(added)} skipped cases as BLOCKED; refreshed {refreshed}")


if __name__ == "__main__":
    main()
