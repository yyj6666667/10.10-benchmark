#!/usr/bin/env bash
# Diagnose GLM with one GPU, then try the independent Qwen MegaTrain sweep.
set -Eeuo pipefail

root=/mnt/hurry-bench
python=/mnt/sft_yyj_yyj/venvs/qwen3-30b-a3b-deepspeed-cpuadam-py311/bin/python
previous_pid=$(cat "$root/logs/fsdp2-onednn-compressed-queue.pid")
exec 9>"$root/logs/megatrain-after-fsdp.lock"
flock -n 9 || exit 3

wait_for_resources() {
    while true; do
        compute_apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)
        available_kib=$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)
        [[ -z $compute_apps && $available_kib -ge 1100000000 ]] && break
        sleep 15
    done
    sleep 20
}

while [[ -n $(ps -p "$previous_pid" -o stat= 2>/dev/null | grep -v '^Z' || true) ]]; do
    sleep 30
done
wait_for_resources

stamp=$(date +%Y%m%d_%H%M%S)
diagnostic=$root/runs/glm45_air_megatrain_single_gpu_diagnostic_${stamp}_$$
mkdir -p "$diagnostic"
"$python" - "$diagnostic" <<'PY'
import json,sys
from pathlib import Path
Path(sys.argv[1],"run_config.json").write_text(json.dumps({
  "host":"qj5090","backend":"megatrain","model_kind":"glm45_air",
  "sequence_length":32,"global_batch":1,"devices":[0],
  "diagnostic_forward_only":True,"precision":"bf16"
},indent=2)+"\n")
PY
sha256sum "$root/tools/megatrain_full_ft.py" \
    "$root/src/megatrain/infinity/model/cpu_master.py" \
    "$root/data/fft_real_100.json" \
    /mnt/data/models/GLM-4.5-Air/model.safetensors.index.json \
    > "$diagnostic/checksums.txt"
env PYTHONPATH="$root/tools:$root/src/megatrain:$root/src/transformers:$root/src/llamafactory-overlay" \
    "$python" -c 'import torch, transformers, infinity; print(torch.__version__, transformers.__version__, infinity.__file__)' \
    > "$diagnostic/versions.txt"
nvidia-smi -L >> "$diagnostic/versions.txt"
command=("$python" "$root/tools/megatrain_full_ft.py"
    --model-kind glm45_air --dataset-dir "$root/data"
    --output-dir "$diagnostic/case" --sequence-length 32
    --steps 2 --warmup-steps 1 --devices 0 --diagnostic-forward-only)
printf '%q ' "${command[@]}" > "$diagnostic/command.txt"
printf '\n' >> "$diagnostic/command.txt"
echo "BEGIN MegaTrain GLM single-GPU diagnostic at $(date -Is)"
if env CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=64 MKL_NUM_THREADS=64 \
    OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
    PYTHONPATH="$root/tools:$root/src/megatrain:$root/src/transformers:$root/src/llamafactory-overlay" \
    "$python" "$root/tools/kt_case_monitor.py" --output "$diagnostic" \
    --gpu-ids 0 -- "${command[@]}" \
    > "$diagnostic/launcher.log" 2>&1; then
    echo 0 > "$diagnostic/exit_code.txt"
    echo "PASS MegaTrain GLM single-GPU diagnostic at $(date -Is)"
else
    status=$?
    echo "$status" > "$diagnostic/exit_code.txt"
    echo "FAIL MegaTrain GLM single-GPU diagnostic exit=$status at $(date -Is)" >&2
fi

wait_for_resources
echo "BEGIN MegaTrain Qwen fixed-batch loss check at $(date -Is)"
if MEGATRAIN_REPEAT_BATCH=1 bash "$root/tools/run_qj_megatrain_case.sh" \
    qwen35_122b 32 8 2; then
    echo "PASS MegaTrain Qwen fixed-batch loss check at $(date -Is)"
else
    status=$?
    run=$(find "$root/runs" -maxdepth 1 -type d \
        -name 'qwen35_122b_megatrain_seq32_8gpu_*' -printf '%T@ %p\n' \
        | sort -nr | head -1 | cut -d' ' -f2-)
    "$python" "$root/tools/record_qj_case_status.py" qwen megatrain 32 \
        "$run" "$status"
    source=$run/launcher.log
    [[ -f $run/host_memory_guard.txt ]] && source=$run/host_memory_guard.txt
    "$python" "$root/tools/mark_qj_unrun_cases.py" qwen megatrain \
        --after 32 --source "qj5090:$source" \
        --reason 'Not run: seq32 fixed-batch semantic check failed before valid timing'
    echo "FAIL MegaTrain Qwen fixed-batch loss check exit=$status at $(date -Is)" >&2
    exit "$status"
fi

for seq_len in 32 64 128 256 512 1024 2048 4096; do
    wait_for_resources
    echo "BEGIN MegaTrain Qwen seq=$seq_len at $(date -Is)"
    if bash "$root/tools/run_qj_megatrain_case.sh" qwen35_122b "$seq_len" 8 2; then
        echo "PASS MegaTrain Qwen seq=$seq_len at $(date -Is)"
    else
        status=$?
        run=$(find "$root/runs" -maxdepth 1 -type d \
            -name "qwen35_122b_megatrain_seq${seq_len}_8gpu_*" \
            -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-)
        "$python" "$root/tools/record_qj_case_status.py" qwen megatrain \
            "$seq_len" "$run" "$status"
        echo "FAIL MegaTrain Qwen seq=$seq_len exit=$status at $(date -Is)" >&2
        if (( seq_len == 32 )); then break; fi
    fi
done
echo "COMPLETE MegaTrain Qwen queue at $(date -Is)"
