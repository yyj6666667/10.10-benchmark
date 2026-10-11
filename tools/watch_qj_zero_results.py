"""Push completed qj5090 ZeRO measurement tables from an isolated clone."""

from __future__ import annotations

import argparse
import csv
import io
import subprocess
import time
from pathlib import Path


FILES = (
    "glm_zero_offload_performance.csv",
    "qwen_zero_offload_performance.csv",
    "case_status.csv",
    "semantic_checks.csv",
)
DESTINATIONS = {
    "glm_zero_offload_performance.csv": "qj5090_glm_zero_offload_performance.csv",
    "qwen_zero_offload_performance.csv": "qj5090_qwen_zero_offload_performance.csv",
    "case_status.csv": "qj5090_zero_offload_case_status.csv",
    "semantic_checks.csv": "qj5090_zero_offload_semantic_checks.csv",
}


def command(*args: str, cwd: Path | None = None, check: bool = True) -> subprocess.CompletedProcess:
    result = subprocess.run(args, cwd=cwd, text=True, capture_output=True, check=False, timeout=120)
    if check and result.returncode:
        raise RuntimeError(f"{' '.join(args)}: {result.stderr.strip()}")
    return result


def fetch_tables(clone: Path) -> list[Path]:
    changed = []
    for name in FILES:
        source = f"/mnt/hurry-bench/results/{name}"
        exists = command("ssh", "-J", "sap1", "qj5090", f"test -f {source}", check=False)
        if exists.returncode:
            continue
        destination = clone / "results" / DESTINATIONS.get(name, name)
        temporary = clone / "results" / f".{name}.incoming"
        command("scp", "-o", "ProxyJump=sap1", f"qj5090:{source}", str(temporary))
        if name in {"case_status.csv", "semantic_checks.csv"}:
            with temporary.open(newline="") as stream:
                reader = csv.DictReader(stream)
                fields = reader.fieldnames
                rows = list(reader)
            rows = [row for row in rows if row["backend"] == "zero_offload"]
            if name == "case_status.csv":
                successful = set()
                for model, label in (
                    ("glm", "GLM-4.5-Air-106B"),
                    ("qwen", "Qwen3.5-122B-A10B"),
                ):
                    table = clone / "results" / f"qj5090_{model}_zero_offload_performance.csv"
                    if table.exists():
                        with table.open(newline="") as stream:
                            successful.update((label, row["Seq"]) for row in csv.DictReader(stream))
                rows = [
                    row for row in rows
                    if (row["model"], row["sequence_length"]) not in successful
                ]
            buffer = io.StringIO(newline="")
            writer = csv.DictWriter(buffer, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
            temporary.write_text(buffer.getvalue())
        if not destination.exists() or destination.read_bytes() != temporary.read_bytes():
            temporary.replace(destination)
            changed.append(destination.relative_to(clone))
        else:
            temporary.unlink()
    return changed


def queue_running() -> bool:
    remote = (
        "pid=$(cat /mnt/hurry-bench/logs/zero-offload-recovery-20261011.pid 2>/dev/null); "
        "test -n \"$pid\" && ps -p \"$pid\" -o stat= | grep -qv '^Z'"
    )
    return command("ssh", "-J", "sap1", "qj5090", remote, check=False).returncode == 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clone", type=Path, default=Path("/tmp/qj-zero-sync-20261011"))
    parser.add_argument("--origin", default="https://github.com/yyj6666667/10.10-benchmark.git")
    parser.add_argument("--interval", type=int, default=30)
    args = parser.parse_args()
    clone = args.clone
    if not clone.exists():
        command("git", "clone", "--branch", "main", args.origin, str(clone))
    elif not (clone / ".git").is_dir():
        raise RuntimeError(f"Not a dedicated Git clone: {clone}")
    (clone / "results").mkdir(exist_ok=True)

    while True:
        command("git", "fetch", "origin", "main", cwd=clone)
        command("git", "reset", "--hard", "origin/main", cwd=clone)
        changed = fetch_tables(clone)
        if changed:
            command("git", "add", *(str(path) for path in changed), cwd=clone)
            command("git", "commit", "-m", "Sync qj5090 ZeRO measurements", cwd=clone)
            for attempt in range(5):
                pushed = command("git", "push", "origin", "main", cwd=clone, check=False)
                if pushed.returncode == 0:
                    print(f"pushed {', '.join(map(str, changed))}", flush=True)
                    break
                command("git", "pull", "--rebase", "origin", "main", cwd=clone)
            else:
                raise RuntimeError(f"Could not push measurement update: {pushed.stderr}")
        if not queue_running():
            print("queue ended; final result sync complete", flush=True)
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
