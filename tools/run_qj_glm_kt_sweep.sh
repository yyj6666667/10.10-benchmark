#!/usr/bin/env bash
# Queue the eight GLM KT full-finetuning cases on qj5090, one at a time.
set -Eeuo pipefail

root=/mnt/hurry-bench
mkdir -p "$root/logs"
exec 9>"$root/logs/glm-kt-sweep.lock"
flock -n 9 || { echo "GLM KT sweep is already running" >&2; exit 3; }

export KT_BENCH_MODE=benchmark
export KT_BENCH_OWNER_OMP_THREADS=64

for seq_len in 32 64 128 256 512 1024 2048 4096; do
    echo "BEGIN GLM KT seq=$seq_len at $(date -Is)"
    if bash "$root/tools/run_qj_kt_case.sh" "$seq_len" 15 glm; then
        echo "PASS GLM KT seq=$seq_len at $(date -Is)"
    else
        status=$?
        echo "FAIL GLM KT seq=$seq_len exit=$status at $(date -Is)" >&2
        exit "$status"
    fi
    sleep 20
    if [[ $seq_len == 32 ]]; then
        echo "BEGIN GLM KT fixed-batch loss check seq=512 at $(date -Is)"
        if KT_BENCH_MODE=semantic bash "$root/tools/run_qj_kt_case.sh" 512 15 glm; then
            echo "PASS GLM KT fixed-batch loss check at $(date -Is)"
        else
            status=$?
            echo "FAIL GLM KT fixed-batch loss check exit=$status at $(date -Is)" >&2
            exit "$status"
        fi
        sleep 20
    fi
done

echo "COMPLETE GLM KT sweep at $(date -Is)"
