#!/usr/bin/env bash
# Complete only missing qj5090 MegaTrain points after the active GLM 512 case.
set -Eeuo pipefail

root=/mnt/hurry-bench
python=/mnt/data/venvs/sft-yyj-pro/bin/python
first_pid_file=$root/logs/megatrain-resume-glm512-20261011.pid
exec 9>"$root/logs/megatrain-missing-20261011.lock"
flock -n 9 || exit 3

while [[ -f $first_pid_file ]]; do
    first_pid=$(cat "$first_pid_file")
    [[ -n $(ps -p "$first_pid" -o stat= 2>/dev/null | grep -v '^Z' || true) ]] || break
    sleep 20
done

wait_for_resources() {
    while true; do
        local apps available
        apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)
        available=$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)
        [[ -z $apps && $available -ge 1100000000 ]] && break
        sleep 15
    done
    sleep 20
}

table_for() {
    local model=$1
    if [[ $model == glm45_air ]]; then
        echo "$root/results/glm_megatrain_performance.csv"
    else
        echo "$root/results/qwen_megatrain_performance.csv"
    fi
}

has_result() {
    local table=$1 seq=$2
    [[ -f $table ]] && awk -F, -v seq="$seq" 'NR > 1 && $1 == seq {found=1} END {exit !found}' "$table"
}

latest_run() {
    local model=$1 seq=$2
    find "$root/runs" -maxdepth 1 -type d -name "${model}_megatrain_seq${seq}_8gpu_*" \
        -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-
}

record_failure() {
    local model=$1 short=$2 seq=$3 status=$4 run
    run=$(latest_run "$model" "$seq")
    if [[ -n $run && -f $run/exit_code.txt && $(cat "$run/exit_code.txt") == "$status" ]]; then
        "$python" "$root/tools/record_qj_case_status.py" "$short" megatrain "$seq" "$run" "$status" >&2
    fi
    [[ -n $run ]] && echo "$run"
}

mark_later_if_limited() {
    local short=$1 seq=$2 run=$3 source
    [[ -n $run ]] || return 0
    source=$run/launcher.log
    [[ -f $run/host_memory_guard.txt ]] && source=$run/host_memory_guard.txt
    if [[ -f $run/host_memory_guard.txt ]] || rg -q 'torch.OutOfMemoryError|CUDA out of memory' "$run/launcher.log"; then
        "$python" "$root/tools/mark_qj_unrun_cases.py" "$short" megatrain \
            --after "$seq" --source "qj5090:$source" \
            --reason "Not run: MegaTrain resource limit at seq$seq on qj5090" \
            --replace-existing-blocked
        return 1
    fi
}

run_case() {
    local model=$1 short=$2 seq=$3 status run
    wait_for_resources
    echo "BEGIN MegaTrain model=$model seq=$seq at $(date -Is)"
    if MEGATRAIN_OFFLOAD_EMB_HEAD=1 bash "$root/tools/run_qj_megatrain_case.sh" "$model" "$seq" 4 2; then
        echo "PASS MegaTrain model=$model seq=$seq at $(date -Is)"
        return 0
    else
        status=$?
    fi
    run=$(record_failure "$model" "$short" "$seq" "$status")
    echo "FAIL MegaTrain model=$model seq=$seq exit=$status at $(date -Is)" >&2
    if ! mark_later_if_limited "$short" "$seq" "$run"; then
        return 1
    fi
    return 2
}

glm_table=$(table_for glm45_air)
for seq in 512 1024 2048 4096; do
    if has_result "$glm_table" "$seq"; then
        echo "SKIP measured GLM seq=$seq"
        continue
    fi
    # The independently launched 512 case may already have failed. Do not
    # repeat an identical resource failure while continuing the other model.
    if [[ $seq == 512 ]]; then
        previous=$(latest_run glm45_air 512)
        if [[ -n $previous && -f $previous/exit_code.txt ]]; then
            status=$(cat "$previous/exit_code.txt")
            record_failure glm45_air glm 512 "$status" >/dev/null
            echo "GLM seq=512 already failed: $previous"
            mark_later_if_limited glm 512 "$previous" || true
            break
        fi
    fi
    if run_case glm45_air glm "$seq"; then
        continue
    else
        status=$?
        [[ $status == 1 ]] && break
    fi
done

qwen_table=$(table_for qwen35_122b)
if ! has_result "$qwen_table" 32; then
    wait_for_resources
    echo "BEGIN MegaTrain Qwen fixed-batch check at $(date -Is)"
    if MEGATRAIN_OFFLOAD_EMB_HEAD=1 MEGATRAIN_REPEAT_BATCH=1 \
        bash "$root/tools/run_qj_megatrain_case.sh" qwen35_122b 32 4 2; then
        echo "PASS MegaTrain Qwen fixed-batch check at $(date -Is)"
    else
        status=$?
        record_failure qwen35_122b qwen 32 "$status" >/dev/null
        echo "FAIL MegaTrain Qwen fixed-batch check exit=$status at $(date -Is)" >&2
        exit "$status"
    fi
fi

for seq in 32 64 128 256 512 1024 2048 4096; do
    if has_result "$qwen_table" "$seq"; then
        echo "SKIP measured Qwen seq=$seq"
        continue
    fi
    if run_case qwen35_122b qwen "$seq"; then
        continue
    else
        status=$?
        [[ $status == 1 ]] && break
    fi
done
echo "COMPLETE MegaTrain missing-case queue at $(date -Is)"
