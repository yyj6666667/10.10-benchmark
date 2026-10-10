#!/usr/bin/env bash
# Start the first qj5090 KT case only after the existing ModelScope download succeeds.
set -Eeuo pipefail

download_log=/home/yyj/model-downloads/glm45-air-20261010/download.log
download_pid=3396081
while ! rg -q '^SUCCESS ' "$download_log"; do
    if ! kill -0 "$download_pid" 2>/dev/null; then
        echo "GLM checkpoint download exited without SUCCESS" >&2
        exit 1
    fi
    sleep 30
done

echo "GLM download complete: $(date -Is)"
/mnt/hurry-bench/tools/run_qj_kt_case.sh 32 15
