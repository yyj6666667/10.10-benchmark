#!/usr/bin/env bash
# Run one isolated KT full fine-tuning case on sapphire4.
set -Eeuo pipefail

if [[ $# -ne 5 ]]; then
    echo "usage: $0 qwen|glm SEQ benchmark|semantic GPU_IDS STEPS" >&2
    exit 2
fi

model=$1
seq=$2
mode=$3
gpu_ids=$4
steps=$5
[[ $seq =~ ^[1-9][0-9]*$ && $steps =~ ^[1-9][0-9]*$ ]] || exit 2
[[ $mode == benchmark || $mode == semantic ]] || exit 2
[[ $gpu_ids =~ ^[0-7](,[0-7])*$ ]] || exit 2
IFS=, read -r -a selected_gpus <<< "$gpu_ids"
num_gpus=${#selected_gpus[@]}

root=/home/qujing/outputs/kt_rerun_20261010
shared=/home/qujing/outputs/glm45_kt_profiler_20261007/overlay
llamafactory=/home/qujing/outputs/glm45_kt_profiler_20261007/llamafactory_20260730/src
patched=/home/qujing/outputs/glm45_kt_clip_opt_20261007/overlay
python=/mnt/data2/wbw/conda/envs/Kllama/bin/python
accelerate_bin=/mnt/data2/wbw/conda/envs/Kllama/bin/accelerate
dataset_dir=/mnt/data2/wbw/Ktransformers-development/FFTtest/dataset

case "$model" in
    qwen)
        project=/mnt/data2/wbw/Ktransformers-development/FFTtest/Qwen3.5-122B-A10B
        checkpoint=/mnt/data2/models/Qwen3.5-122B-A10B
        config_source="$project/configs/train_full_bf16_qwen35_122b.yaml"
        accelerate_source="$project/configs/accelerate_ktransformers_bf16_8gpu.yaml"
        template=qwen3
        kernel_overlay="$root/qwen_overlay"
        ;;
    glm)
        project=/mnt/data2/wbw/Ktransformers-development/FFTtest/GLM-4.5-Air
        checkpoint=/mnt/data2/models/GLM-4.5-Air
        config_source=/home/qujing/outputs/glm45_kt_clip_opt_20261007/run_optimized/train_config.yaml
        accelerate_source=/home/qujing/outputs/glm45_kt_clip_opt_20261007/run_optimized/accelerate_config.yaml
        template=glm4_moe
        kernel_overlay="$patched"
        ;;
    *) echo "unknown model: $model" >&2; exit 2 ;;
esac

for path in "$checkpoint/config.json" "$dataset_dir/dataset_info.json" "$config_source" "$accelerate_source" "$project/finetune_train_with_timing.py" "$patched/accelerate/accelerator.py" "$kernel_overlay/kt_kernel/grad_norm.py" "$kernel_overlay/kt_kernel/sft/arch.py"; do
    [[ -f $path ]] || { echo "missing $path" >&2; exit 1; }
done

# Do not launch onto another user's GPU. This also rejects GPU memory used by
# services with zero instantaneous utilization.
gpu_state=$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits)
for gpu in "${selected_gpus[@]}"; do
    used=$(awk -F, -v gpu="$gpu" '$1 + 0 == gpu {gsub(/ /, "", $2); print $2}' <<< "$gpu_state")
    util=$(awk -F, -v gpu="$gpu" '$1 + 0 == gpu {gsub(/ /, "", $3); print $3}' <<< "$gpu_state")
    max_existing=512
    if [[ $gpu == 0 && ${KT_RERUN_ALLOW_SHARED_GPU0:-0} == 1 ]]; then
        # sapphire4 GPU 0 hosts a long-lived idle service using ~10.5 GiB.
        max_existing=11000
    fi
    [[ -n $used && $used -lt $max_existing && ${util:-100} -le 5 ]] || {
        echo "GPU $gpu is occupied (${used:-unknown} MiB, util ${util:-unknown}%)" >&2
        exit 3
    }
done

available_kib=$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)
[[ $available_kib -ge 1800000000 ]] || { echo "insufficient MemAvailable: $available_kib KiB" >&2; exit 3; }

stamp=$(date +%Y%m%d_%H%M%S)
run_dir="$root/${model}_${mode}_seq${seq}_${num_gpus}gpu_${stamp}"
mkdir -p "$run_dir"
printf '%s\n' "$gpu_state" > "$run_dir/gpu_state_before.csv"
free -h > "$run_dir/memory_before.txt"

"$python" - "$config_source" "$accelerate_source" "$run_dir" "$model" "$mode" "$seq" "$steps" "$num_gpus" "$checkpoint" "$dataset_dir" "$template" "$gpu_ids" "$kernel_overlay" <<'PY'
import json
import sys
from pathlib import Path
import yaml

(train_source, accel_source, output, model, mode, seq, steps, num_gpus,
 checkpoint, dataset_dir, template, gpu_ids, kernel_overlay) = sys.argv[1:]
output = Path(output)
with open(train_source) as f:
    train = yaml.safe_load(f)
with open(accel_source) as f:
    accel = yaml.safe_load(f)
train.update(
    model_name_or_path=checkpoint,
    kt_weight_path=checkpoint,
    dataset="fft_real_100",
    dataset_dir=dataset_dir,
    template=template,
    cutoff_len=int(seq),
    output_dir=str(output / "model_output"),
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    max_steps=int(steps),
    max_samples=int(num_gpus) if mode == "semantic" else 100,
    logging_strategy="steps" if mode == "semantic" else "no",
    logging_steps=1,
    save_strategy="no",
    report_to="none",
    seed=42,
    data_seed=42,
    pure_bf16=True,
    use_kt=True,
)
if mode == "semantic":
    # Avoid forking DataLoader workers after the CPU owner has mapped weights.
    train["dataloader_num_workers"] = 0
    train["dataloader_persistent_workers"] = False
    train.pop("dataloader_prefetch_factor", None)
    train["logging_nan_inf_filter"] = False
train.setdefault("kt_config", dict(
    enabled=True, kt_backend="AMXBF16", kt_num_threads=80,
    kt_tp_enabled=True, kt_threadpool_count=2, kt_max_cache_depth=2,
    kt_share_backward_bb=True,
))
accel["num_processes"] = int(num_gpus)
with open(output / "train_config.yaml", "w") as f:
    yaml.safe_dump(train, f, sort_keys=False)
with open(output / "accelerate_config.yaml", "w") as f:
    yaml.safe_dump(accel, f, sort_keys=False)
with open(output / "run_config.json", "w") as f:
    json.dump(dict(model=model, mode=mode, sequence_length=int(seq),
                   num_gpus=int(num_gpus), gpu_ids=gpu_ids, steps=int(steps),
                   max_samples=train["max_samples"],
                   dataset=train["dataset"], checkpoint=checkpoint,
                   kernel_overlay=kernel_overlay,
                   logging_strategy=train["logging_strategy"]), f, indent=2)
PY

sha256sum "$project/finetune_train_with_timing.py" "$patched/accelerate/accelerator.py" "$kernel_overlay/kt_kernel/grad_norm.py" "$kernel_overlay/kt_kernel/sft/arch.py" "$run_dir/train_config.yaml" "$run_dir/accelerate_config.yaml" > "$run_dir/checksums.txt"
sha256sum "$kernel_overlay"/kt_kernel/kt_kernel_ext*.so >> "$run_dir/checksums.txt"
"$python" -c 'import torch, transformers, accelerate; print("torch", torch.__version__, "transformers", transformers.__version__, "accelerate", accelerate.__version__)' > "$run_dir/versions.txt"
nvidia-smi -L >> "$run_dir/versions.txt"

if [[ ${KT_RERUN_DRY_RUN:-0} == 1 ]]; then
    echo "$run_dir dry-run"
    exit 0
fi

unit="kt-rerun-${model}-${mode}-${seq}-${stamp}"
scope_path="/sys/fs/cgroup/user.slice/user-1000.slice/user@1000.service/app.slice/${unit}.scope"
monitor_metrics() {
    while :; do
        printf '%s\n' "$(date -Is)"
        awk '/MemAvailable:/ {print "MemAvailable_KiB," $2}' /proc/meminfo
        if [[ -r $scope_path/memory.current ]]; then
            printf 'CgroupCurrent_bytes,%s\n' "$(cat "$scope_path/memory.current")"
            printf 'CgroupPeak_bytes,%s\n' "$(cat "$scope_path/memory.peak")"
        fi
        nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits
        sleep 5
    done
}
monitor_metrics > "$run_dir/metrics.log" 2>&1 &
monitor_pid=$!
trap 'kill "$monitor_pid" 2>/dev/null || true' EXIT

warmup=5
(( steps > warmup )) || warmup=$((steps - 1))
tokens_per_step=$((num_gpus * seq))
set +e
(
    cd "$project" || exit 1
    systemd-run --user --scope --quiet --unit "$unit" --property MemoryAccounting=yes -- \
        "$python" "$shared/resource_scope_exec.py" \
        --profile server --numa-nodes 0,1 --output-dir "$run_dir" -- \
        env \
        DISABLE_VERSION_CHECK=1 USE_KT=1 ACCELERATE_USE_KT=true \
        ACCELERATE_KT_TRAIN_MODE=full ACCELERATE_KT_LORA_RANK=0 \
        ACCELERATE_KT_LORA_ALPHA=0 KT_FINETUNE_MODE=full \
        FFT_TRAINING_BACKEND=kt FFT_PRECISION=bf16 FFT_FINETUNING_TYPE=full \
        FFT_TEXT_ONLY=1 FFT_SKIP_FINAL_SAVE=1 \
        FFT_STEP_TIMING_OUT_DIR="$run_dir/step_timing" \
        FFT_STEP_TIMING_WARMUP_STEPS="$warmup" \
        FFT_STEP_TIMING_TOKENS_PER_STEP="$tokens_per_step" \
        FFT_DISABLE_PERF_PROBES=1 FFT_CUDA_EMPTY_CACHE_AFTER_PREPARE=0 \
        FFT_CPU_THREADS=2 FFT_KT_OWNER_THREADS=80 FFT_KT_NON_OWNER_THREADS=2 \
        KT_BACKWARD_TIMING=off KT_SFT_PROFILE=0 \
        KT_REUSE_CHECKPOINT_FORWARD=1 KT_REUSE_CHECKPOINT_FORWARD_DISTRIBUTED=1 \
        DS_PROBE_MODE=off ACCELERATE_KT_MODEL_MAX_LENGTH="$seq" \
        OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
        NUMEXPR_NUM_THREADS=2 BLIS_NUM_THREADS=2 OMP_DYNAMIC=FALSE \
        MKL_DYNAMIC=FALSE ACCELERATE_KT_OMP_NUM_THREADS=2 \
        TOKENIZERS_PARALLELISM=false HF_DATASETS_OFFLINE=1 \
        TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        CUDA_VISIBLE_DEVICES="$gpu_ids" \
        PYTHONPATH="$project:$kernel_overlay:$patched:$shared:$llamafactory${PYTHONPATH:+:$PYTHONPATH}" \
        "$accelerate_bin" launch --config_file "$run_dir/accelerate_config.yaml" \
            -m finetune_train_with_timing train "$run_dir/train_config.yaml"
) > "$run_dir/train.log" 2>&1
status=$?
set -e
printf '%s\n' "$status" > "$run_dir/exit_code.txt"
kill "$monitor_pid" 2>/dev/null || true
trap - EXIT
echo "$run_dir status=$status"
exit "$status"
