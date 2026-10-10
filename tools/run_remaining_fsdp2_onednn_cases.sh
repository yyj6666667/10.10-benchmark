#!/usr/bin/env bash
# Continue the approved sap4 15/5 sweep after GLM 2048 exits successfully.
set -Eeuo pipefail

root=/home/qujing/outputs/onednn_20261010
tools_dir="$root/fsdp2_review"
first_run="$root/fsdp2-glm2048-formal-20261010_0514"
first_unit=fsdp2-glm2048-formal-20261010_0514.service

while [[ ! -f "$first_run/exit_code.txt" ]]; do
    state=$(systemctl --user show "$first_unit" -p ActiveState --value)
    if [[ "$state" != active ]]; then
        echo "GLM 2048 stopped without a final exit code" >&2
        exit 1
    fi
    sleep 15
done
[[ $(<"$first_run/exit_code.txt") == 0 ]] || {
    echo "GLM 2048 failed; remaining cases were not started" >&2
    exit 1
}

run_case() {
    local model_kind=$1 sequence_length=$2 run_name=$3
    if nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits |
        awk '$1 > 1024 {busy=1} END {exit !busy}'; then
        echo "GPU is occupied before $run_name" >&2
        exit 1
    fi
    if ! awk '/MemAvailable:/ {exit ($2 > 1500000000 ? 0 : 1)}' /proc/meminfo; then
        echo "Less than 1.5 TB RAM is available before $run_name" >&2
        exit 1
    fi
    echo "Starting $run_name"
    /bin/bash "$tools_dir/run_fsdp2_onednn_case.sh" \
        "$model_kind" "$sequence_length" "$root/$run_name" 15 5
    [[ $(<"$root/$run_name/exit_code.txt") == 0 ]]
    echo "Completed $run_name"
}

run_case glm45_air 4096 fsdp2-glm4096-formal-20261010
run_case qwen35_122b 2048 fsdp2-qwen2048-formal-20261010
run_case qwen35_122b 4096 fsdp2-qwen4096-formal-20261010
