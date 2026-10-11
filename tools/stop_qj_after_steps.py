#!/usr/bin/env python3
"""Stop an already-running torchrun after the requested complete step."""

from __future__ import annotations

import argparse
import json
import os
import signal
import time
from datetime import datetime, timezone
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("torchrun_pid", type=int)
    parser.add_argument("--after", type=int, default=4)
    args = parser.parse_args()
    timing_path = args.run_dir / "case/step_timing.json"
    expected = str(args.run_dir).encode()
    while True:
        try:
            command = Path(f"/proc/{args.torchrun_pid}/cmdline").read_bytes()
        except FileNotFoundError:
            raise SystemExit("torchrun exited before the requested step")
        if expected not in command or os.getpgid(args.torchrun_pid) != args.torchrun_pid:
            raise SystemExit("torchrun PID does not match this run's process group")
        try:
            timing = json.loads(timing_path.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            time.sleep(0.25)
            continue
        rows = timing["rows"]
        if len(rows) >= args.after and rows[args.after - 1]["step"] == args.after:
            record = {
                "reason": "User requested two warmup plus two measured steps",
                "stop_after_completed_step": args.after,
                "torchrun_pid": args.torchrun_pid,
                "observed_loss": rows[args.after - 1]["loss"],
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            }
            (args.run_dir / "planned_stop_after_step.json").write_text(
                json.dumps(record, indent=2) + "\n"
            )
            os.killpg(args.torchrun_pid, signal.SIGTERM)
            print(json.dumps(record), flush=True)
            return
        time.sleep(0.25)


if __name__ == "__main__":
    main()
