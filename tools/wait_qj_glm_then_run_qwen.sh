#!/usr/bin/env bash
# After the GLM sweep, validate Qwen loss and run its eight benchmark lengths.
set -Eeuo pipefail

glm_log=/mnt/hurry-bench/logs/remaining-cases-watcher.log
glm_pid=$(cat /mnt/hurry-bench/logs/remaining-cases-watcher.pid)
while ! rg -q '^all GLM KT cases completed at ' "$glm_log"; do
    if ! kill -0 "$glm_pid" 2>/dev/null; then
        echo "GLM KT sweep exited before completion" >&2
        exit 1
    fi
    sleep 30
done

sleep 15
echo "starting Qwen KT semantic check at $(date -Is)"
KT_BENCH_MODE=semantic /mnt/hurry-bench/tools/run_qj_kt_case.sh 512 15 qwen

for seq_len in 32 64 128 256 512 1024 2048 4096; do
    sleep 15
    echo "starting Qwen KT seq=$seq_len at $(date -Is)"
    /mnt/hurry-bench/tools/run_qj_kt_case.sh "$seq_len" 15 qwen
done
echo "all Qwen KT cases completed at $(date -Is)"
