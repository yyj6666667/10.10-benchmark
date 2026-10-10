#!/usr/bin/env bash
# One eight-GPU FSDP2 + CPU oneDNN case on qj5090.
set -Eeuo pipefail

if (( $# < 2 || $# > 4 )); then
    echo "usage: $0 glm45_air|qwen35_122b SEQUENCE_LENGTH [STEPS [WARMUP_STEPS]]" >&2
    exit 2
fi
model_kind=$1
seq_len=$2
steps=${3:-8}
warmup=${4:-2}
profile=${FSDP2_PROFILE:-0}
gpu_state_offload=${FSDP2_GPU_STATE_OFFLOAD:-}
[[ $model_kind == glm45_air || $model_kind == qwen35_122b ]] || exit 2
[[ $seq_len =~ ^[1-9][0-9]*$ && $steps =~ ^[1-9][0-9]*$ && $warmup =~ ^[0-9]+$ ]] || exit 2
[[ $profile == 0 || $profile == 1 ]] || { echo "FSDP2_PROFILE must be 0 or 1" >&2; exit 2; }
if [[ -z $gpu_state_offload ]]; then
    if (( seq_len >= 4096 )); then gpu_state_offload=1; else gpu_state_offload=0; fi
fi
[[ $gpu_state_offload == 0 || $gpu_state_offload == 1 ]] || { echo "FSDP2_GPU_STATE_OFFLOAD must be 0 or 1" >&2; exit 2; }
(( steps > warmup )) || { echo "steps must exceed warmup" >&2; exit 2; }

root=/mnt/hurry-bench
tools_dir=$root/tools
case $model_kind in
    glm45_air) checkpoint=/mnt/data/models/GLM-4.5-Air; model_label=GLM-4.5-Air ;;
    qwen35_122b) checkpoint=/mnt/data/models/Qwen3.5-122B-A10B; model_label=Qwen3.5-122B-A10B ;;
esac
python=/mnt/data/venvs/sft-yyj-pro/bin/python
torchrun=/mnt/data/venvs/sft-yyj-pro/bin/torchrun
gpu_ids=0,1,2,3,4,5,6,7
for file in "$checkpoint/config.json" "$checkpoint/model.safetensors.index.json" \
    "$root/data/fft_real_100.json" "$root/data/dataset_info.json" \
    "$tools_dir/fsdp2_onednn_full_ft.py" "$tools_dir/kt_case_monitor.py" \
    "$tools_dir/collect_qj_fsdp2_onednn_result.py" \
    "$root/src/llamafactory-overlay/llamafactory/__init__.py" \
    "$root/src/transformers/transformers/__init__.py"; do
    [[ -f $file ]] || { echo "missing $file" >&2; exit 1; }
done

"$python" - "$checkpoint" <<'PY'
import json
import sys
from pathlib import Path
root = Path(sys.argv[1])
index = json.loads((root / "model.safetensors.index.json").read_text())
shards = set(index["weight_map"].values())
missing = [name for name in shards if not (root / name).is_file() or (root / name).stat().st_size == 0]
if missing:
    raise SystemExit(f"checkpoint incomplete: {len(missing)}/{len(shards)} shards missing")
print(f"checkpoint: {len(shards)} shards present")
PY

stamp=$(date +%Y%m%d_%H%M%S)
run_dir=$root/runs/${model_kind}_fsdp2_onednn_seq${seq_len}_8gpu_${stamp}_$$
mkdir -p "$run_dir"
"$python" - "$run_dir" "$seq_len" "$steps" "$warmup" "$checkpoint" "$model_label" "$model_kind" "$profile" "$gpu_state_offload" <<'PY'
import json
import sys
from pathlib import Path
run, seq, steps, warmup, checkpoint, model_label, model_kind, profile, gpu_state_offload = sys.argv[1:]
Path(run, "run_config.json").write_text(json.dumps({
    "host": "qj5090",
    "model": model_label,
    "model_kind": model_kind,
    "checkpoint": checkpoint,
    "backend": "pytorch_fsdp2_onednn_cpu",
    "sequence_length": int(seq),
    "steps": int(steps),
    "warmup_steps": int(warmup),
    "profile": profile == "1",
    "dp": 8, "tp": 1,
    "global_batch": 8,
    "precision": "bf16_compute_fp32_master",
    "threads_per_rank": 8,
    "gradient_checkpointing": True,
    "gpu_optimizer_state_offload": gpu_state_offload == "1",
    "max_grad_norm": 1.0,
}, indent=2) + "\n")
PY
sha256sum "$tools_dir/fsdp2_onednn_full_ft.py" "$tools_dir/run_qj_fsdp2_onednn_case.sh" \
    "$tools_dir/kt_case_monitor.py" "$tools_dir/collect_qj_fsdp2_onednn_result.py" \
    "$root/data/fft_real_100.json" "$checkpoint/config.json" \
    "$checkpoint/model.safetensors.index.json" > "$run_dir/checksums.txt"
env PYTHONPATH="$root/src/transformers:$root/src/llamafactory-overlay${PYTHONPATH:+:$PYTHONPATH}" \
    "$python" -c 'import torch, transformers, llamafactory; print("torch",torch.__version__); print("transformers",transformers.__version__,transformers.__file__); print("llamafactory",llamafactory.__file__); print("cuda",torch.version.cuda); print("mkldnn",torch.backends.mkldnn.is_available())' > "$run_dir/versions.txt"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader,nounits > "$run_dir/gpu_before.csv"
free -b > "$run_dir/memory_before.txt"

cmd=("$torchrun" --standalone --nproc_per_node=8 "$tools_dir/fsdp2_onednn_full_ft.py"
    --model-kind "$model_kind" --model-path "$checkpoint"
    --dataset-dir "$root/data" --dataset-name fft_real_100
    --output-dir "$run_dir/case" --sequence-length "$seq_len"
    --steps "$steps" --warmup-steps "$warmup" --threads-per-rank 8
    --learning-rate 1e-5 --max-grad-norm 1.0 --gradient-checkpointing)
if [[ ${FSDP2_REPEAT_BATCH:-0} == 1 ]]; then cmd+=(--repeat-batch); fi
if [[ $profile == 1 ]]; then cmd+=(--profile); fi
if [[ $gpu_state_offload == 1 ]]; then cmd+=(--offload-gpu-optimizer-state); fi
{
    printf '%q' "${cmd[0]}"
    printf ' %q' "${cmd[@]:1}"
    printf '\n'
} > "$run_dir/command.txt"
if [[ ${FSDP2_DRY_RUN:-0} == 1 ]]; then
    echo "$run_dir dry-run"
    exit 0
fi

gpu_state=$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits)
compute_apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)
[[ -z $compute_apps ]] || {
    echo "another GPU compute process is active: $compute_apps" >&2
    exit 3
}
for gpu in {0..7}; do
    used=$(awk -F, -v gpu="$gpu" '$1 + 0 == gpu {gsub(/ /, "", $2); print $2}' <<< "$gpu_state")
    [[ -n $used && $used -lt 512 ]] || {
        echo "GPU $gpu occupied: ${used:-unknown} MiB" >&2
        exit 3
    }
done
available_kib=$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)
[[ $available_kib -ge 1400000000 ]] || {
    echo "insufficient host RAM: $available_kib KiB available" >&2
    exit 3
}

set +e
env CUDA_VISIBLE_DEVICES="$gpu_ids" OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
    OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE \
    TOKENIZERS_PARALLELISM=false HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    PYTHONPATH="$root/src/transformers:$root/src/llamafactory-overlay${PYTHONPATH:+:$PYTHONPATH}" \
    PYTORCH_ALLOC_CONF=expandable_segments:True \
    "$python" "$tools_dir/kt_case_monitor.py" --output "$run_dir" --gpu-ids "$gpu_ids" -- \
    "${cmd[@]}" > "$run_dir/launcher.log" 2>&1
status=$?
set -e
printf '%s\n' "$status" > "$run_dir/exit_code.txt"
if [[ $status -eq 0 && ${FSDP2_REPEAT_BATCH:-0} != 1 && $profile != 1 ]]; then
    "$python" "$tools_dir/collect_qj_fsdp2_onednn_result.py" "$run_dir" > "$run_dir/performance.csv"
    mkdir -p "$root/results"
    table=$root/results/${model_kind}_fsdp2_onednn_performance.csv
    if [[ ! -e $table ]]; then
        cp "$run_dir/performance.csv" "$table"
    else
        tail -n +2 "$run_dir/performance.csv" >> "$table"
    fi
fi
echo "$run_dir status=$status"
exit "$status"
