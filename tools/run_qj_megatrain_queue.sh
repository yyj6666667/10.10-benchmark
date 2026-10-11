#!/usr/bin/env bash
# Continue MegaTrain after the separately queued GLM seq32 diagnostic.
set -Eeuo pipefail
root=/mnt/hurry-bench
first_pid=$(cat "$root/logs/megatrain-glm32-after-kt-20261011.pid")
exec 9>"$root/logs/megatrain-queue.lock"
flock -n 9 || exit 3
while [[ -n $(ps -p "$first_pid" -o stat= 2>/dev/null | grep -v '^Z' || true) ]]; do
    sleep 30
done

first_run=$(find "$root/runs" -maxdepth 1 -type d \
    -name 'glm45_air_megatrain_seq32_8gpu_*' -printf '%T@ %p\n' \
    | sort -nr | head -1 | cut -d' ' -f2-)
if [[ -z $first_run || ! -f $first_run/exit_code.txt ]]; then
    echo 'first GLM MegaTrain run did not reach a recorded result' >&2
    exit 1
fi
if [[ $(cat "$first_run/exit_code.txt") != 0 || ! -f $first_run/loss_check.json ]]; then
    /mnt/data/venvs/sft-yyj-pro/bin/python \
        "$root/tools/record_qj_case_status.py" glm megatrain 32 \
        "$first_run" "$(cat "$first_run/exit_code.txt")" || true
    echo "first GLM MegaTrain semantic check failed: $first_run" >&2
    exit 1
fi

for model in glm45_air qwen35_122b; do
    if [[ $model == glm45_air ]]; then
        lengths=(32 64 128 256 512 1024 2048 4096)
        short=glm
    else
        lengths=(32 64 128 256 512 1024 2048 4096)
        short=qwen
    fi
    for seq_len in "${lengths[@]}"; do
        if [[ $model == qwen35_122b && $seq_len == 32 ]]; then
            echo "BEGIN MegaTrain Qwen fixed-batch loss check at $(date -Is)"
            if MEGATRAIN_REPEAT_BATCH=1 bash "$root/tools/run_qj_megatrain_case.sh" "$model" 32 4 2; then
                echo "PASS MegaTrain Qwen fixed-batch loss check at $(date -Is)"
            else
                echo "FAIL MegaTrain Qwen fixed-batch loss check at $(date -Is)" >&2
                exit 1
            fi
        fi
        while [[ -n $(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits) ]]; do
            sleep 15
        done
        sleep 20
        echo "BEGIN MegaTrain model=$model seq=$seq_len at $(date -Is)"
        if bash "$root/tools/run_qj_megatrain_case.sh" "$model" "$seq_len" 4 2; then
            echo "PASS MegaTrain model=$model seq=$seq_len at $(date -Is)"
        else
            status=$?
            run=$(find "$root/runs" -maxdepth 1 -type d \
                -name "${model}_megatrain_seq${seq_len}_8gpu_*" -printf '%T@ %p\n' \
                | sort -nr | head -1 | cut -d' ' -f2-)
            if [[ -n $run && -f $run/exit_code.txt && $(cat "$run/exit_code.txt") == "$status" ]]; then
                /mnt/data/venvs/sft-yyj-pro/bin/python \
                    "$root/tools/record_qj_case_status.py" "$short" megatrain \
                    "$seq_len" "$run" "$status"
            fi
            echo "FAIL MegaTrain model=$model seq=$seq_len exit=$status at $(date -Is)" >&2
            if (( seq_len == 32 )); then break; fi
        fi
    done
done
echo "COMPLETE MegaTrain queue at $(date -Is)"
