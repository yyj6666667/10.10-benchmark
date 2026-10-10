#!/usr/bin/env bash
# Keep the small GLM KT result table and sweep log in the local benchmark repo.
set -Eeuo pipefail

repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
result=$repo/results/qj5090_glm_kt_performance.csv
local_log=$repo/runs/kt-qj5090-20261010/glm_sweep/sweep.log
remote_result=/mnt/hurry-bench/results/glm_kt_performance.csv
remote_log=/mnt/hurry-bench/logs/glm-kt-sweep-20261010.log
remote_pid=/mnt/hurry-bench/logs/glm-kt-sweep-20261010.pid

mkdir -p "$(dirname "$result")" "$(dirname "$local_log")"
exec 9>"$repo/runs/kt-qj5090-20261010/glm_sweep/sync.lock"
flock -n 9 || { echo "GLM KT result sync is already running" >&2; exit 3; }

while :; do
    if scp -q -o BatchMode=yes -o ConnectTimeout=15 -J sap1 \
        "qj5090:$remote_result" "$result.pending"; then
        mv -f -- "$result.pending" "$result"
    else
        rm -f -- "$result.pending"
        echo "$(date -Is) result copy failed; retrying" >&2
    fi
    if scp -q -o BatchMode=yes -o ConnectTimeout=15 -J sap1 \
        "qj5090:$remote_log" "$local_log.pending"; then
        mv -f -- "$local_log.pending" "$local_log"
    else
        rm -f -- "$local_log.pending"
        echo "$(date -Is) sweep log copy failed; retrying" >&2
    fi

    if [[ -f $local_log ]] && rg -q '^COMPLETE GLM KT sweep ' "$local_log"; then
        echo "$(date -Is) GLM KT sweep complete; results synchronized"
        exit 0
    fi
    if [[ -f $local_log ]] && rg -q '^FAIL GLM KT ' "$local_log"; then
        echo "$(date -Is) GLM KT sweep failed; available results synchronized" >&2
        exit 1
    fi
    if [[ ${KT_SWEEP_SYNC_ONCE:-0} == 1 ]]; then
        exit 0
    fi
    if ! ssh -o BatchMode=yes -o ConnectTimeout=15 -J sap1 qj5090 \
        "test -f '$remote_pid' && kill -0 \"\$(cat '$remote_pid')\"" 2>/dev/null; then
        echo "$(date -Is) sweep process missing; inspect $local_log" >&2
        exit 1
    fi
    sleep 60
done
