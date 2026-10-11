#!/usr/bin/env python3
"""Inject an opt-in, rank-local PyTorch profiler into MegaTrain's GPU worker."""

from __future__ import annotations

import argparse
from pathlib import Path


DEFAULT = Path("/mnt/hurry-bench/src/megatrain/infinity/model/mp_worker.py")
CHANGES = (
    (
        "import copy\nimport threading\n",
        "import copy\nfrom contextlib import nullcontext\nimport threading\n",
    ),
    (
        """    cmd_q = shared_state.cmd_queues[rank]
    result_q = shared_state.result_queues[rank]

    while True:
""",
        """    cmd_q = shared_state.cmd_queues[rank]
    result_q = shared_state.result_queues[rank]
    profile_capture = None
    if (os.environ.get('MEGATRAIN_PROFILE_DIR') and
            rank == int(os.environ.get('MEGATRAIN_PROFILE_RANK', '0'))):
        from megatrain_profile_hooks import MegaTrainWorkerProfiler
        profile_capture = MegaTrainWorkerProfiler(ctx, globals())
    train_step = 0

    while True:
""",
    ),
    (
        """        elif cmd.type == WorkerCommandType.FORWARD_BACKWARD:
            result = _run_forward_backward(
                rank, ctx, shared_state, grad_queue, cmd)
            # Wait for all grad tasks to complete before signaling main
            grad_queue.join()
            result_q.put(result)
""",
        """        elif cmd.type == WorkerCommandType.FORWARD_BACKWARD:
            train_step += 1
            capture = (profile_capture.capture_step(train_step, tuple(cmd.input_ids.shape))
                       if profile_capture is not None else nullcontext())
            with capture:
                result = _run_forward_backward(
                    rank, ctx, shared_state, grad_queue, cmd)
                # Include gradient D2H and CPU accumulation in the trace.
                grad_queue.join()
            result_q.put(result)
""",
    ),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", type=Path, default=DEFAULT)
    args = parser.parse_args()
    source = args.path.read_text()
    if "MegaTrainWorkerProfiler(ctx, globals())" in source:
        print(f"already patched: {args.path}")
        return
    for old, new in CHANGES:
        if source.count(old) != 1:
            raise RuntimeError(f"expected one patch location, found {source.count(old)}")
        source = source.replace(old, new)
    args.path.write_text(source)
    print(f"patched GPU worker profiler: {args.path}")


if __name__ == "__main__":
    main()
