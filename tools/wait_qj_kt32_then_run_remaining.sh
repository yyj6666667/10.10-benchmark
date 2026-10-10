#!/usr/bin/env bash
# Run the remaining GLM lengths one at a time after the first case succeeds.
set -Eeuo pipefail

first_log=/mnt/hurry-bench/logs/first-case-watcher.log
first_pid=$(cat /mnt/hurry-bench/logs/first-case-watcher.pid)
while ! rg -q '/mnt/hurry-bench/runs/glm_kt_seq32_8gpu_.* status=0$' "$first_log"; do
    if ! kill -0 "$first_pid" 2>/dev/null; then
        echo "first KT case exited without a successful result" >&2
        exit 1
    fi
    sleep 30
done

for seq_len in 64 128 256 512 1024 2048 4096; do
    sleep 15
    echo "starting GLM KT seq=$seq_len at $(date -Is)"
    /mnt/hurry-bench/tools/run_qj_kt_case.sh "$seq_len" 15
done
echo "all GLM KT cases completed at $(date -Is)"
