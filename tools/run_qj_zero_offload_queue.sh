#!/usr/bin/env bash
# Run ZeRO-3 CPU offload after MegaTrain releases the eight qj5090 GPUs.
set -Eeuo pipefail
root=/mnt/hurry-bench
previous_pid=$(cat "$root/logs/megatrain-queue-20261011.pid")
exec 9>"$root/logs/zero-offload-queue.lock"
flock -n 9 || exit 3
while [[ -n $(ps -p "$previous_pid" -o stat= 2>/dev/null | grep -v '^Z' || true) ]]; do
    sleep 30
done

latest_run() {
    local model=$1 seq_len=$2
    find "$root/runs" -maxdepth 1 -type d \
        -name "${model}_zero_offload_seq${seq_len}_8gpu_*" \
        -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-
}

record_failure() {
    local model=$1 seq_len=$2 status=$3 run
    run=$(latest_run "$model" "$seq_len")
    if [[ -n $run && -f $run/exit_code.txt && $(cat "$run/exit_code.txt") == "$status" ]]; then
        /mnt/data/venvs/sft-yyj-pro/bin/python \
            "$root/tools/record_qj_case_status.py" "$model" zero_offload \
            "$seq_len" "$run" "$status"
    fi
}

for model in glm qwen; do
    while [[ -n $(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits) ]]; do
        sleep 15
    done
    sleep 20
    echo "BEGIN ZeRO fixed-batch check model=$model at $(date -Is)"
    if ZERO_BENCH_MODE=semantic bash "$root/tools/run_qj_zero_offload_case.sh" \
        "$model" 32 8 2; then
        echo "PASS ZeRO fixed-batch check model=$model at $(date -Is)"
    else
        status=$?
        record_failure "$model" 32 "$status"
        echo "FAIL ZeRO fixed-batch check model=$model exit=$status at $(date -Is)" >&2
        exit "$status"
    fi

    for seq_len in 32 64 128 256 512 1024 2048 4096; do
        while [[ -n $(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits) ]]; do
            sleep 15
        done
        sleep 20
        echo "BEGIN ZeRO model=$model seq=$seq_len at $(date -Is)"
        if bash "$root/tools/run_qj_zero_offload_case.sh" "$model" "$seq_len" 8 2; then
            echo "PASS ZeRO model=$model seq=$seq_len at $(date -Is)"
        else
            status=$?
            record_failure "$model" "$seq_len" "$status"
            echo "FAIL ZeRO model=$model seq=$seq_len exit=$status at $(date -Is)" >&2
            if (( seq_len == 32 )); then exit "$status"; fi
        fi
    done
done
echo "COMPLETE ZeRO queue at $(date -Is)"
