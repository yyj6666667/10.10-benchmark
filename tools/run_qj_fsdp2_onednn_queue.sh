#!/usr/bin/env bash
# Run the 8/2 FSDP2 + CPU oneDNN sweep after the queued KT sweep.
set -Eeuo pipefail

root=/mnt/hurry-bench
qwen_pid_file=$root/logs/qwen-kt-queue-20261011.pid
[[ -f $qwen_pid_file ]] || { echo "missing Qwen KT queue PID file" >&2; exit 1; }
qwen_pid=$(cat "$qwen_pid_file")
[[ $qwen_pid =~ ^[1-9][0-9]*$ ]] || exit 2

exec 9>"$root/logs/fsdp2-onednn-queue.lock"
flock -n 9 || { echo "FSDP2 oneDNN queue already active" >&2; exit 3; }

while [[ -n $(ps -p "$qwen_pid" -o stat= 2>/dev/null | grep -v '^Z' || true) ]]; do
    sleep 30
done
while [[ -n $(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits) ]]; do
    sleep 15
done
sleep 20

for model_kind in glm45_air qwen35_122b; do
    for seq_len in 32 64 128 256 512 1024 2048 4096; do
        echo "BEGIN oneDNN model=$model_kind seq=$seq_len at $(date -Is)"
        if bash "$root/tools/run_qj_fsdp2_onednn_case.sh" "$model_kind" "$seq_len" 8 2; then
            echo "PASS oneDNN model=$model_kind seq=$seq_len at $(date -Is)"
        else
            status=$?
            echo "FAIL oneDNN model=$model_kind seq=$seq_len exit=$status at $(date -Is)" >&2
        fi
        while [[ -n $(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits) ]]; do
            sleep 15
        done
        sleep 20
    done
done
echo "COMPLETE FSDP2 oneDNN sweep at $(date -Is)"
