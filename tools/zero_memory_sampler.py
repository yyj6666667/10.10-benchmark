"""Sample real host and per-rank memory during a ZeRO benchmark case."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import psutil


def proc_memory(pid: int) -> dict:
    result = {"pid": pid}
    try:
        for line in Path(f"/proc/{pid}/smaps_rollup").read_text().splitlines():
            key, _, value = line.partition(":")
            if key in {"Rss", "Pss", "Pss_Anon", "Pss_File", "Pss_Shmem", "Locked"}:
                result[f"{key.lower()}_kib"] = int(value.split()[0])
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmLck:"):
                result["vmlck_kib"] = int(line.split()[1])
                break
        result["cgroup"] = Path(f"/proc/{pid}/cgroup").read_text().strip().split(":")[-1]
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        result["gone"] = True
    return result


def cgroup_memory(path: str) -> dict:
    root = Path("/sys/fs/cgroup") / path.lstrip("/")
    result = {"path": path}
    for name in ("memory.current", "memory.max", "memory.events"):
        try:
            content = (root / name).read_text().strip()
            result[name] = (
                {key: int(value) for key, value in (line.split() for line in content.splitlines())}
                if name == "memory.events"
                else int(content) if content.isdecimal() else content
            )
        except (FileNotFoundError, PermissionError):
            pass
    return result


def sample(parent: psutil.Process) -> dict:
    meminfo = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, _, value = line.partition(":")
        if key in {"MemAvailable", "MemFree", "Cached", "Buffers", "SwapFree"}:
            meminfo[f"{key}_kib"] = int(value.split()[0])
    ranks = []
    try:
        children = parent.children(recursive=True)
    except psutil.NoSuchProcess:
        children = []
    for child in children:
        try:
            args = child.cmdline()
            if (
                "-m" in args
                and args.index("-m") + 1 < len(args)
                and args[args.index("-m") + 1] == "kt_bench_entry"
                and not any(Path(arg).name == "torchrun" for arg in args[:2])
            ):
                ranks.append(proc_memory(child.pid))
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    ranks.sort(key=lambda rank: rank["pid"])
    cgroups = sorted({rank["cgroup"] for rank in ranks if "cgroup" in rank})
    gpu = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,used_gpu_memory", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=10, check=False,
    )
    return {
        "time": time.time(), "meminfo": meminfo, "ranks": ranks,
        "sum_rank_pss_kib": sum(rank.get("pss_kib", 0) for rank in ranks),
        "cgroups": [cgroup_memory(path) for path in cgroups],
        "gpu_compute_apps": gpu.stdout.strip().splitlines(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=5)
    args = parser.parse_args()
    parent = psutil.Process(args.parent)
    with args.output.open("w", buffering=1) as output:
        while parent.is_running():
            output.write(json.dumps(sample(parent)) + "\n")
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
