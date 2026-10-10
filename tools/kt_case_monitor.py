"""Run one benchmark command and sample its process-tree RSS and GPU usage."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import psutil


def gpu_used_mib() -> dict[str, int]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    return {
        fields[0].strip(): int(fields[1].strip())
        for line in result.stdout.splitlines()
        if len(fields := line.split(",")) == 2
    }


def process_tree_rss_bytes(pid: int) -> tuple[int, int]:
    try:
        parent = psutil.Process(pid)
        processes = [parent, *parent.children(recursive=True)]
    except psutil.NoSuchProcess:
        return 0, 0
    total = 0
    count = 0
    for process in processes:
        try:
            total += process.memory_info().rss
            count += 1
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return total, count


def host_available_bytes() -> int:
    with Path("/proc/meminfo").open() as stream:
        for line in stream:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    raise RuntimeError("MemAvailable missing from /proc/meminfo")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--interval", type=float, default=2.0)
    parser.add_argument("--gpu-ids", required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command and args.command[0] == "--":
        args.command.pop(0)
    if not args.command:
        parser.error("missing command after --")

    args.output.mkdir(parents=True, exist_ok=True)
    ids = args.gpu_ids.split(",")
    baseline = gpu_used_mib()
    peak_gpu = {gpu: baseline[gpu] for gpu in ids}
    peak_rss = 0
    minimum_host_available = host_available_bytes()
    start = time.monotonic()
    child = subprocess.Popen(args.command, start_new_session=True)

    def forward_signal(signum: int, _frame: object) -> None:
        if child.poll() is None:
            os.killpg(child.pid, signum)

    signal.signal(signal.SIGTERM, forward_signal)
    signal.signal(signal.SIGINT, forward_signal)
    samples = args.output / "resource_samples.jsonl"
    try:
        with samples.open("w", encoding="utf-8") as handle:
            while True:
                rss, process_count = process_tree_rss_bytes(child.pid)
                try:
                    gpu = gpu_used_mib()
                except (subprocess.SubprocessError, ValueError):
                    gpu = {}
                peak_rss = max(peak_rss, rss)
                available = host_available_bytes()
                minimum_host_available = min(minimum_host_available, available)
                for gpu_id in ids:
                    peak_gpu[gpu_id] = max(peak_gpu[gpu_id], gpu.get(gpu_id, 0))
                handle.write(
                    json.dumps(
                        {
                            "elapsed_sec": round(time.monotonic() - start, 3),
                            "process_count": process_count,
                            "process_tree_rss_bytes": rss,
                            "host_available_bytes": available,
                            "gpu_used_mib": gpu,
                        }
                    )
                    + "\n"
                )
                handle.flush()
                if child.poll() is not None:
                    break
                time.sleep(args.interval)
    finally:
        status = child.wait()
        (args.output / "resources.json").write_text(
            json.dumps(
                {
                    "method": "process_tree_sum_rss_and_nvidia_smi_2s_samples",
                    "gpu_ids": ids,
                    "baseline_gpu_used_mib": {gpu: baseline[gpu] for gpu in ids},
                    "peak_gpu_used_mib": peak_gpu,
                    "peak_single_gpu_used_gib": max(peak_gpu.values()) / 1024,
                    "peak_process_tree_rss_gb": peak_rss / 1e9,
                    "minimum_host_available_gb": minimum_host_available / 1e9,
                    "elapsed_sec": time.monotonic() - start,
                    "exit_code": status,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    return status


if __name__ == "__main__":
    sys.exit(main())
