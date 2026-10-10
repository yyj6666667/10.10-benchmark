#!/usr/bin/env bash
# Retry MegaTrain after the independent FSDP2 + oneDNN sweep releases qj5090.
set -Eeuo pipefail

root=/mnt/hurry-bench
previous_pid=$(cat "$root/logs/fsdp2-onednn-queue-20261011.pid")
while [[ -n $(ps -p "$previous_pid" -o stat= 2>/dev/null | grep -v '^Z' || true) ]]; do
    sleep 30
done
while [[ -n $(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits) ]]; do
    sleep 15
done
sleep 20
MEGATRAIN_REPEAT_BATCH=1 bash "$root/tools/run_qj_megatrain_case.sh" glm45_air 32 8 2
