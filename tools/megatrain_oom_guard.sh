#!/usr/bin/env bash
# Stop a MegaTrain parent that is stuck waiting for workers after CUDA OOM.
set -Eeuo pipefail
monitor_pid=$1
log=$2
while kill -0 "$monitor_pid" 2>/dev/null; do
    if [[ -f $log ]] && rg -q 'torch.OutOfMemoryError|CUDA out of memory' "$log"; then
        child=$(pgrep -P "$monitor_pid" -f 'megatrain_full_ft.py' | head -n 1 || true)
        if [[ -n $child ]]; then
            echo "worker CUDA OOM; stopping MegaTrain process group pgid=$child at $(date -Is)"
            kill -KILL -- "-$child" 2>/dev/null || true
        fi
        exit 0
    fi
    sleep 5
done
