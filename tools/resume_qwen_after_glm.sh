#!/usr/bin/env bash
# Resume the paused Qwen transfer only after the GLM ModelScope download succeeds.
set -Eeuo pipefail

glm_log=/home/yyj/model-downloads/glm45-air-20261010/download.log
glm_pid=3396081
qwen_pid_file=/mnt/hurry-bench/logs/qwen35-122b-bf16-20261010/download.pid

while ! rg -q '^SUCCESS ' "$glm_log"; do
    if ! kill -0 "$glm_pid" 2>/dev/null; then
        echo "GLM download exited without SUCCESS; Qwen remains paused" >&2
        exit 1
    fi
    sleep 15
done

qwen_pid=$(cat "$qwen_pid_file")
qwen_cmd=$(ps -o args= -p "$qwen_pid")
[[ $qwen_cmd == *'/mnt/hurry-bench/tools/download_qwen_bf16.py'* ]] || {
    echo "Qwen downloader is no longer the expected process: $qwen_pid" >&2
    exit 1
}
qwen_state=$(ps -o stat= -p "$qwen_pid" | xargs)
if [[ $qwen_state == T* || $qwen_state == t* ]]; then
    kill -CONT "$qwen_pid"
    echo "Qwen download resumed after GLM SUCCESS at $(date -Is): PID $qwen_pid"
else
    echo "Qwen download already running after GLM SUCCESS at $(date -Is): PID $qwen_pid"
fi
