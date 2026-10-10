#!/usr/bin/env bash
# Start the first qj5090 KT case after both concurrent downloads succeed.
set -Eeuo pipefail

glm_download_log=/home/yyj/model-downloads/glm45-air-20261010/download.log
glm_download_pid=3396081
qwen_download_log=/mnt/hurry-bench/logs/qwen35-122b-bf16-20261010/download.log
qwen_download_pid_file=/mnt/hurry-bench/logs/qwen35-122b-bf16-20261010/download.pid

wait_for_download() {
    local name=$1 log=$2 pid=$3
    while ! rg -q '^SUCCESS ' "$log"; do
        if ! kill -0 "$pid" 2>/dev/null; then
            echo "$name checkpoint download exited without SUCCESS" >&2
            exit 1
        fi
        sleep 30
    done
    echo "$name download complete: $(date -Is)"
}

wait_for_download GLM "$glm_download_log" "$glm_download_pid"
wait_for_download Qwen "$qwen_download_log" "$(cat "$qwen_download_pid_file")"
/mnt/hurry-bench/tools/run_qj_kt_case.sh 32 15
