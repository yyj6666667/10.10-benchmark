#!/usr/bin/env bash
# Run the 8/2 FSDP2 + CPU oneDNN sweep after ZeRO releases the GPUs.
set -Eeuo pipefail

root=/mnt/hurry-bench
zero_pid_file=$root/logs/zero-offload-queue-20261011.pid
[[ -f $zero_pid_file ]] || { echo "missing ZeRO queue PID file" >&2; exit 1; }
zero_pid=$(cat "$zero_pid_file")
[[ $zero_pid =~ ^[1-9][0-9]*$ ]] || exit 2

exec 9>"$root/logs/fsdp2-onednn-queue.lock"
flock -n 9 || { echo "FSDP2 oneDNN queue already active" >&2; exit 3; }

while [[ -n $(ps -p "$zero_pid" -o stat= 2>/dev/null | grep -v '^Z' || true) ]]; do
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
wait_for_resources

for model_kind in glm45_air qwen35_122b; do
    echo "BEGIN oneDNN fixed-batch loss check model=$model_kind at $(date -Is)"
    if FSDP2_REPEAT_BATCH=1 bash "$root/tools/run_qj_fsdp2_onednn_case.sh" \
        "$model_kind" 32 8 2; then
        echo "PASS oneDNN fixed-batch loss check model=$model_kind at $(date -Is)"
    else
        status=$?
        run=$(find "$root/runs" -maxdepth 1 -type d \
            -name "${model_kind}_fsdp2_onednn_seq32_8gpu_*" \
            -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-)
        if [[ -n $run && -f $run/exit_code.txt && $(cat "$run/exit_code.txt") == "$status" ]]; then
            short=glm
            [[ $model_kind == qwen35_122b ]] && short=qwen
            /mnt/data/venvs/sft-yyj-pro/bin/python \
                "$root/tools/record_qj_case_status.py" "$short" pytorch_onednn \
                32 "$run" "$status"
        fi
        echo "FAIL oneDNN fixed-batch loss check model=$model_kind exit=$status at $(date -Is)" >&2
        exit "$status"
    fi
    for seq_len in 32 64 128 256 512 1024 2048 4096; do
        echo "BEGIN oneDNN model=$model_kind seq=$seq_len at $(date -Is)"
        if bash "$root/tools/run_qj_fsdp2_onednn_case.sh" "$model_kind" "$seq_len" 8 2; then
            echo "PASS oneDNN model=$model_kind seq=$seq_len at $(date -Is)"
        else
            status=$?
            run=$(find "$root/runs" -maxdepth 1 -type d \
                -name "${model_kind}_fsdp2_onednn_seq${seq_len}_8gpu_*" \
                -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-)
            if [[ -n $run && -f $run/exit_code.txt && $(cat "$run/exit_code.txt") == "$status" ]]; then
                short=glm
                [[ $model_kind == qwen35_122b ]] && short=qwen
                /mnt/data/venvs/sft-yyj-pro/bin/python \
                    "$root/tools/record_qj_case_status.py" "$short" pytorch_onednn \
                    "$seq_len" "$run" "$status"
            fi
            echo "FAIL oneDNN model=$model_kind seq=$seq_len exit=$status at $(date -Is)" >&2
        fi
        wait_for_resources
    done
done
echo "COMPLETE FSDP2 oneDNN sweep at $(date -Is)"
