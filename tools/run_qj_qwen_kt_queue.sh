#!/usr/bin/env bash
# Start Qwen only after the active GLM KT 4096 process releases all eight GPUs.
set -Eeuo pipefail

root=/mnt/hurry-bench
glm_pid_file=$root/logs/glm-kt-seq4096-retry-20261010_155731.pid
[[ -f $glm_pid_file ]] || { echo "missing GLM PID file" >&2; exit 1; }
glm_pid=$(cat "$glm_pid_file")
[[ $glm_pid =~ ^[1-9][0-9]*$ ]] || exit 2

exec 9>"$root/logs/qwen-kt-queue.lock"
flock -n 9 || { echo "Qwen KT queue already active" >&2; exit 3; }

while [[ -n $(ps -p "$glm_pid" -o stat= 2>/dev/null | grep -v '^Z' || true) ]]; do
    sleep 30
done
while [[ -n $(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits) ]]; do
    sleep 15
done
sleep 15

echo "BEGIN Qwen KT fixed-batch loss check at $(date -Is)"
KT_BENCH_MODE=semantic bash "$root/tools/run_qj_kt_case.sh" 512 4 qwen
echo "PASS Qwen KT fixed-batch loss check at $(date -Is)"

for seq_len in 32 64 128 256 512 1024 2048 4096; do
    sleep 15
    echo "BEGIN Qwen KT seq=$seq_len at $(date -Is)"
    if bash "$root/tools/run_qj_kt_case.sh" "$seq_len" 4 qwen; then
        echo "PASS Qwen KT seq=$seq_len at $(date -Is)"
    else
        status=$?
        echo "FAIL Qwen KT seq=$seq_len exit=$status at $(date -Is)" >&2
        exit "$status"
    fi
done
echo "COMPLETE Qwen KT sweep at $(date -Is)"
