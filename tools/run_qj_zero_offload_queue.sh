#!/usr/bin/env bash
# Run ZeRO-3 CPU offload after MegaTrain releases the eight qj5090 GPUs.
set -Eeuo pipefail
root=/mnt/hurry-bench
previous_pid_file=${ZERO_WAIT_PID_FILE:-$root/logs/megatrain-queue-20261011.pid}
previous_pid=$(cat "$previous_pid_file")
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

mark_later() {
    local model=$1 seq_len=$2 run source
    run=$(latest_run "$model" "$seq_len")
    [[ -n $run && -f $run/exit_code.txt ]] || return 0
    source=$run/launcher.log
    [[ -f $run/host_memory_guard.txt ]] && source=$run/host_memory_guard.txt
    /mnt/data/venvs/sft-yyj-pro/bin/python \
        "$root/tools/mark_qj_unrun_cases.py" "$model" zero_offload \
        --after "$seq_len" --source "qj5090:$source" \
        --reason "Not run: ZeRO-3 parameter and optimizer CPU offload failed at seq$seq_len on qj5090" \
        --replace-existing-blocked
}

for model in glm qwen; do
    wait_for_resources() {
        while true; do
            compute_apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)
            available_kib=$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)
            [[ -z $compute_apps && $available_kib -ge 1050000000 ]] && break
            sleep 15
        done
        sleep 20
    }
    wait_for_resources
    echo "BEGIN ZeRO fixed-batch check model=$model at $(date -Is)"
    if ZERO_BENCH_MODE=semantic bash "$root/tools/run_qj_zero_offload_case.sh" \
        "$model" 32 4 2; then
        echo "PASS ZeRO fixed-batch check model=$model at $(date -Is)"
    else
        status=$?
        record_failure "$model" 32 "$status"
        mark_later "$model" 32
        echo "FAIL ZeRO fixed-batch check model=$model exit=$status at $(date -Is)" >&2
        continue
    fi

    for seq_len in 32 64 128 256 512 1024 2048 4096; do
        wait_for_resources
        echo "BEGIN ZeRO model=$model seq=$seq_len at $(date -Is)"
        if bash "$root/tools/run_qj_zero_offload_case.sh" "$model" "$seq_len" 4 2; then
            echo "PASS ZeRO model=$model seq=$seq_len at $(date -Is)"
        else
            status=$?
            record_failure "$model" "$seq_len" "$status"
            echo "FAIL ZeRO model=$model seq=$seq_len exit=$status at $(date -Is)" >&2
            run=$(latest_run "$model" "$seq_len")
            if (( seq_len == 32 )) || [[ -f $run/host_memory_guard.txt ]] \
                || rg -q 'torch.OutOfMemoryError|CUDA out of memory' "$run/launcher.log"; then
                mark_later "$model" "$seq_len"
                break
            fi
        fi
    done
done
echo "COMPLETE ZeRO queue at $(date -Is)"
