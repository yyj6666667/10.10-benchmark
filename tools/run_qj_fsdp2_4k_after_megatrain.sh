#!/usr/bin/env bash
# Retry both 4K oneDNN cases after the isolated MegaTrain queue releases GPUs.
set -Eeuo pipefail

root=/mnt/hurry-bench
python=/mnt/data/venvs/sft-yyj-pro/bin/python
previous_pid=$(cat "$root/logs/megatrain-after-fsdp-20261011.pid")
exec 9>"$root/logs/fsdp2-4k-after-megatrain.lock"
flock -n 9 || exit 3

while [[ -n $(ps -p "$previous_pid" -o stat= 2>/dev/null | grep -v '^Z' || true) ]]; do
    sleep 30
done

wait_for_resources() {
    while true; do
        compute_apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)
        available_kib=$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)
        [[ -z $compute_apps && $available_kib -ge 1400000000 ]] && break
        sleep 15
    done
    sleep 20
}

for model_kind in glm45_air qwen35_122b; do
    wait_for_resources
    echo "BEGIN oneDNN $model_kind seq4096 at $(date -Is)"
    if FSDP2_COMPRESS_CPU_GRADIENTS=1 \
        bash "$root/tools/run_qj_fsdp2_onednn_case.sh" "$model_kind" 4096 4 2; then
        echo "PASS oneDNN $model_kind seq4096 at $(date -Is)"
    else
        status=$?
        run=$(find "$root/runs" -maxdepth 1 -type d \
            -name "${model_kind}_fsdp2_onednn_seq4096_8gpu_*" \
            -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-)
        model=glm
        [[ $model_kind == qwen35_122b ]] && model=qwen
        "$python" "$root/tools/record_qj_case_status.py" \
            "$model" pytorch_onednn 4096 "$run" "$status"
        echo "FAIL oneDNN $model_kind seq4096 exit=$status at $(date -Is)" >&2
    fi
done
echo "COMPLETE oneDNN 4K retry queue at $(date -Is)"
