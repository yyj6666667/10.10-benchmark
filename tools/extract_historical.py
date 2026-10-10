#!/usr/bin/env python3
"""Extract the original analyses' step time and TPS tables into one CSV."""

import csv
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    "Qwen3.5-122B-A10B": ROOT / "references/qwen3.5-122b-historical-analysis.md",
    "GLM-4.5-Air": ROOT / "references/glm-4.5-air-historical-analysis.md",
}
BACKENDS = {
    "ktransformers": "ktransformers",
    "deepspeed": "deepspeed",
    "megatrain": "megatrain",
    "pytorch torch": "pytorch_torch",
}
OUTPUT = ROOT / "results/historical_throughput.csv"


def cells(line: str) -> list[str]:
    return [value.strip() for value in line.strip().strip("|").split("|")]


rows = []
for model, source in SOURCES.items():
    backend = None
    columns = None
    for line in source.read_text(encoding="utf-8").splitlines():
        if line.startswith("#"):
            heading = re.sub(r"^#+\s*", "", line).lower()
            backend = next(
                (name for prefix, name in BACKENDS.items() if heading.startswith(prefix)),
                None,
            )
            columns = None
            continue
        if backend is None or not line.startswith("|"):
            columns = None
            continue
        values = cells(line)
        if values[0].lower() == "seq" and "Step (s)" in values and "TPS" in values:
            columns = values
            continue
        if columns is None or not values[0].isdigit():
            continue
        step = values[columns.index("Step (s)")]
        tps = values[columns.index("TPS")]
        status = values[columns.index("状态")] if "状态" in columns else "SUCCESS"
        rows.append({
            "source": source.relative_to(ROOT).as_posix(),
            "model": model,
            "backend": backend,
            "sequence_length": int(values[0]),
            "step_seconds": "" if step == "-" else step,
            "tokens_per_second": "" if tps == "-" else tps,
            "status": status,
        })

if len(rows) != 60:
    raise RuntimeError(f"Expected 60 historical rows, found {len(rows)}")

with OUTPUT.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)

print(f"Wrote {len(rows)} rows to {OUTPUT}")
