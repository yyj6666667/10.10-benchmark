#!/usr/bin/env bash
# Run the low-VRAM MegaTrain sweep after the oneDNN 4K retries release GPUs.
set -Eeuo pipefail

root=/mnt/hurry-bench
previous_pid=$(cat "$root/logs/fsdp2-4k-after-megatrain.pid")
while [[ -n $(ps -p "$previous_pid" -o stat= 2>/dev/null | grep -v '^Z' || true) ]]; do
    sleep 30
done

echo "BEGIN low-VRAM MegaTrain sweep at $(date -Is)"
MEGATRAIN_OFFLOAD_EMB_HEAD=1 MEGATRAIN_SKIP_SINGLE_GPU_DIAGNOSTIC=1 \
    bash "$root/tools/run_qj_megatrain_after_fsdp.sh"
