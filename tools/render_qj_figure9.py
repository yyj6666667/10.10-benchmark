#!/usr/bin/env python3
"""Render the current qj5090 full-finetuning results without historical data."""

from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
SEQUENCES = (32, 64, 128, 256, 512, 1024, 2048, 4096)
MODELS = (
    ("Qwen3.5-122B-A10B", "qwen", "qwen35_122b"),
    ("GLM-4.5-Air-106B", "glm", "glm45_air"),
)
BACKENDS = (
    ("zero_offload", "ZeRO-Offload", "#4f75b5", "///"),
    ("megatrain", "MegaTrain", "#8766a5", "++"),
    ("pytorch_onednn", "PyTorch (FSDP2 + oneDNN)", "#8a8a8a", ".."),
    ("ktransformers", "KT-FT", "#56a369", "xx"),
)
DETAIL_FIELDS = (
    "gpu_peak_gib",
    "cpu_rss_peak_gb",
    "step_seconds",
    "tps",
    "forward_seconds",
    "backward_seconds",
    "optimizer_seconds",
)
RESULT_FIELDS = (
    "model",
    "backend",
    "sequence_length",
    "status",
    *DETAIL_FIELDS,
    "host",
    "source",
    "note",
)
METRIC_COLUMNS = {
    "gpu_peak_gib": "单卡 GPU 峰值 (GiB)",
    "cpu_rss_peak_gb": "CPU RSS 峰值 (GB)",
    "step_seconds": "Step (s)",
    "tps": "TPS",
    "forward_seconds": "Forward (s)",
    "backward_seconds": "Backward (s)",
    "optimizer_seconds": "Optimizer (s)",
}


def read_performance(path: Path) -> dict[int, dict[str, str]]:
    if not path.exists():
        return {}
    rows: dict[int, dict[str, str]] = {}
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            seq = int(row["Seq"])
            if seq not in SEQUENCES:
                raise ValueError(f"unexpected sequence length {seq} in {path}")
            measurements = {key: row[column] for key, column in METRIC_COLUMNS.items()}
            values = {key: float(value) for key, value in measurements.items()}
            if not all(math.isfinite(value) and value > 0 for value in values.values()):
                raise ValueError(f"invalid measurement at seq={seq} in {path}")
            expected_tps = 8 * seq / values["step_seconds"]
            if abs(values["tps"] - expected_tps) / expected_tps > 0.002:
                raise ValueError(f"TPS differs from batch=8 at seq={seq} in {path}")
            # The case runner appends reruns; the latest completed case wins.
            rows[seq] = measurements
    return rows


def read_statuses() -> dict[tuple[str, str, int], dict[str, str]]:
    path = RESULTS / "qj5090_case_status.csv"
    if not path.exists():
        return {}
    statuses = {}
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            key = row["model"], row["backend"], int(row["sequence_length"])
            if row["status"] not in {"OOM", "FAILED", "BLOCKED"}:
                raise ValueError(f"invalid status override: {row}")
            if not row["source"]:
                raise ValueError(f"status lacks evidence source: {row}")
            statuses[key] = row
    return statuses


def assemble() -> list[dict[str, str]]:
    statuses = read_statuses()
    rows = []
    for model, kt_stem, fsdp_stem in MODELS:
        measurements = {
            "ktransformers": read_performance(RESULTS / f"qj5090_{kt_stem}_kt_performance.csv"),
            "pytorch_onednn": read_performance(
                RESULTS / f"qj5090_{fsdp_stem}_fsdp2_onednn_performance.csv"
            ),
            "megatrain": read_performance(RESULTS / f"qj5090_{kt_stem}_megatrain_performance.csv"),
            "zero_offload": read_performance(RESULTS / f"qj5090_{kt_stem}_zero_offload_performance.csv"),
        }
        sources = {
            "ktransformers": f"results/qj5090_{kt_stem}_kt_performance.csv",
            "pytorch_onednn": f"results/qj5090_{fsdp_stem}_fsdp2_onednn_performance.csv",
            "megatrain": f"results/qj5090_{kt_stem}_megatrain_performance.csv",
            "zero_offload": f"results/qj5090_{kt_stem}_zero_offload_performance.csv",
        }
        for backend, *_ in BACKENDS:
            for seq in SEQUENCES:
                row = {
                    "model": model,
                    "backend": backend,
                    "sequence_length": str(seq),
                    "status": "NOT_MEASURED",
                    **{field: "" for field in DETAIL_FIELDS},
                    "host": "qj5090",
                    "source": "",
                    "note": "",
                }
                if seq in measurements[backend]:
                    row.update(measurements[backend][seq])
                    row["status"] = "SUCCESS"
                    row["source"] = sources[backend]
                elif (model, backend, seq) in statuses:
                    row.update(statuses[model, backend, seq])
                rows.append(row)
    return rows


def plot(rows: list[dict[str, str]]) -> None:
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 15, "axes.labelsize": 11})
    plt.rcParams["svg.hashsalt"] = "qj5090-figure9"
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 5.8), sharey=True, layout="constrained")
    success = [float(row["tps"]) for row in rows if row["status"] == "SUCCESS"]
    ymax = max(25, max(success, default=0) * 1.18)
    x = np.arange(len(SEQUENCES))
    width = 0.20
    legend_handles = []
    for ax, (model, _, _) in zip(axes, MODELS):
        model_rows = {(row["backend"], int(row["sequence_length"])): row
                      for row in rows if row["model"] == model}
        for backend_idx, (backend, label, color, hatch) in enumerate(BACKENDS):
            bar_x = x + (backend_idx - 1.5) * width
            values = [float(model_rows[backend, seq]["tps"])
                      if model_rows[backend, seq]["status"] == "SUCCESS" else 0.0
                      for seq in SEQUENCES]
            bars = ax.bar(bar_x, values, width=width, color=color, edgecolor="#252b32",
                          linewidth=0.65, hatch=hatch, label=label)
            if ax is axes[0]:
                legend_handles.append(bars)
            for xpos, seq in zip(bar_x, SEQUENCES):
                if model_rows[backend, seq]["status"] == "OOM":
                    ax.text(xpos, ymax * 0.012, "×", color="#d62728", fontsize=14,
                            ha="center", va="bottom", weight="bold")
        ax.set_title(model, weight="bold")
        ax.set_xticks(x, ("32", "64", "128", "256", "512", "1K", "2K", "4K"))
        ax.set_ylim(0, ymax)
        ax.grid(axis="y", linestyle="--", alpha=0.28)
        ax.set_axisbelow(True)
        ax.set_xlabel("Sequence length (tokens / batch)")
    axes[0].set_ylabel("Throughput (tokens/s)")
    fig.legend(legend_handles, [item[1] for item in BACKENDS], loc="upper center",
               ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.09))
    fig.suptitle("qj5090 · 8 × RTX 5090 · BF16 full fine-tuning · global batch 8", y=1.16,
                 fontsize=12)
    fig.text(0.5, -0.025,
             "Blank = not measured; red × = observed OOM. No historical measurements included.",
             ha="center", fontsize=9, color="#4a4a4a")
    for extension in ("png", "svg"):
        options = {"metadata": {"Date": None}} if extension == "svg" else {}
        fig.savefig(RESULTS / f"qj5090_figure9.{extension}", dpi=220,
                    bbox_inches="tight", **options)
    plt.close(fig)


def main() -> None:
    rows = assemble()
    with (RESULTS / "qj5090_figure9.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    plot(rows)
    measured = sum(row["status"] == "SUCCESS" for row in rows)
    print(f"Rendered {measured}/{len(rows)} measured bars")


if __name__ == "__main__":
    main()
