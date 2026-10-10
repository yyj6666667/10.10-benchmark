#!/usr/bin/env bash
# qj5090: one isolated 8-GPU KT full fine-tuning benchmark case.
set -Eeuo pipefail

if [[ $# -ne 2 ]]; then
    echo "usage: $0 SEQ STEPS" >&2
    exit 2
fi
seq_len=$1
steps=$2
[[ $seq_len =~ ^[1-9][0-9]*$ && $steps =~ ^[1-9][0-9]*$ ]] || exit 2

root=/mnt/hurry-bench
tools_dir=$root/tools
checkpoint=/mnt/data/models/GLM-4.5-Air
python=/mnt/data/venvs/sft-yyj-pro/bin/python
accelerate=/mnt/data/venvs/sft-yyj-pro/bin/accelerate
llamafactory=/mnt/sft_yyj_yyj/kt-v2-final/LLaMA-Factory/src
overlay=$root/py-overlay-contract
gpu_ids=0,1,2,3,4,5,6,7

for path in "$checkpoint/config.json" \
    "$root/data/fft_real_100.json" "$root/data/dataset_info.json" \
    "$overlay/kt_kernel/grad_norm.py" "$tools_dir/step_phase_timer.py" \
    "$tools_dir/kt_glm_train_base.yaml" "$tools_dir/kt_glm_accelerate_base.yaml"; do
    [[ -f $path ]] || { echo "missing $path" >&2; exit 1; }
done

if [[ ${KT_BENCH_DRY_RUN:-0} != 1 ]]; then
[[ -f "$checkpoint/model.safetensors.index.json" ]] || {
    echo "missing $checkpoint/model.safetensors.index.json" >&2
    exit 1
}
"$python" - "$checkpoint" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
index = json.loads((root / "model.safetensors.index.json").read_text())
shards = sorted(set(index["weight_map"].values()))
missing = [name for name in shards if not (root / name).is_file() or (root / name).stat().st_size == 0]
if missing:
    raise SystemExit(f"model incomplete: {len(missing)}/{len(shards)} shards missing or empty")
print(f"model checkpoint: {len(shards)} shards present")
PY
fi

gpu_state=$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits)
for gpu in {0..7}; do
    used=$(awk -F, -v gpu="$gpu" '$1 + 0 == gpu {gsub(/ /, "", $2); print $2}' <<< "$gpu_state")
    util=$(awk -F, -v gpu="$gpu" '$1 + 0 == gpu {gsub(/ /, "", $3); print $3}' <<< "$gpu_state")
    [[ -n $used && $used -lt 512 && ${util:-100} -le 5 ]] || {
        echo "GPU $gpu occupied: ${used:-unknown} MiB, ${util:-unknown}%" >&2
        exit 3
    }
done
available_kib=$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)
[[ $available_kib -ge 500000000 ]] || { echo "insufficient RAM: $available_kib KiB" >&2; exit 3; }

stamp=$(date +%Y%m%d_%H%M%S)
run_dir=$root/runs/glm_kt_seq${seq_len}_8gpu_${stamp}
mkdir -p "$run_dir"
printf '%s\n' "$gpu_state" > "$run_dir/gpu_before.csv"
free -h > "$run_dir/memory_before.txt"

"$python" - "$tools_dir" "$run_dir" "$checkpoint" "$root/data" "$seq_len" "$steps" <<'PY'
import json
import sys
from pathlib import Path

import yaml

tools, run, checkpoint, data, seq_len, steps = sys.argv[1:]
tools, run = Path(tools), Path(run)
train = yaml.safe_load((tools / "kt_glm_train_base.yaml").read_text())
train.update(
    model_name_or_path=checkpoint,
    kt_weight_path=checkpoint,
    dataset="fft_real_100",
    dataset_dir=data,
    template="glm4_moe",
    cutoff_len=int(seq_len),
    output_dir=str(run / "model_output"),
    max_samples=100,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    max_steps=int(steps),
    logging_strategy="no",
    save_strategy="no",
    report_to="none",
    max_grad_norm=1.0,
    pure_bf16=True,
    gradient_checkpointing=True,
    use_kt=True,
)
train["kt_config"].update(kt_backend="AMXBF16", kt_num_threads=64)
accel = yaml.safe_load((tools / "kt_glm_accelerate_base.yaml").read_text())
(run / "train_config.yaml").write_text(yaml.safe_dump(train, sort_keys=False))
(run / "accelerate_config.yaml").write_text(yaml.safe_dump(accel, sort_keys=False))
(run / "run_config.json").write_text(json.dumps({
    "model": "GLM-4.5-Air",
    "checkpoint": checkpoint,
    "sequence_length": int(seq_len),
    "steps": int(steps),
    "warmup_steps": 5,
    "global_batch": 8,
    "tokens_per_step": 8 * int(seq_len),
    "host": "qj5090",
    "gpu_ids": list(range(8)),
    "precision": "bf16",
    "gradient_clipping": 1.0,
}, indent=2) + "\n")
PY

sha256sum "$root/src/kt-kernel-contract/bf16_sumsq.cpp" \
    "$overlay/kt_kernel/grad_norm.py" \
    "$overlay"/kt_kernel/kt_kernel_ext*.so \
    "$root/src/accelerate/accelerate/accelerator.py" \
    "$root/src/transformers/transformers/trainer.py" \
    "$tools_dir/step_phase_timer.py" \
    "$root/data/fft_real_100.json" \
    "$checkpoint/config.json" \
    "$checkpoint/model.safetensors.index.json" \
    "$run_dir/train_config.yaml" "$run_dir/accelerate_config.yaml" \
    > "$run_dir/checksums.txt"
cp "$tools_dir/kt_qj_source_revisions.md" "$run_dir/source_revisions.md"
PYTHONPATH="$overlay:$root/src/accelerate:$root/src/transformers:$tools_dir:$llamafactory${PYTHONPATH:+:$PYTHONPATH}" \
    "$python" -c 'import inspect, torch, kt_kernel, accelerate, transformers; from kt_kernel.grad_norm import bf16_sumsq; print("torch",torch.__version__); print("kt_kernel",inspect.getfile(kt_kernel)); print("accelerate",inspect.getfile(accelerate)); print("transformers",inspect.getfile(transformers)); print("bf16_sumsq",inspect.getfile(bf16_sumsq))' \
    > "$run_dir/versions.txt"
nvidia-smi -L >> "$run_dir/versions.txt"

if [[ ${KT_BENCH_DRY_RUN:-0} == 1 ]]; then
    echo "$run_dir dry-run"
    exit 0
fi

set +e
env \
    DISABLE_VERSION_CHECK=1 USE_KT=1 ACCELERATE_USE_KT=true \
    ACCELERATE_KT_TRAIN_MODE=full ACCELERATE_KT_LORA_RANK=0 \
    ACCELERATE_KT_LORA_ALPHA=0 KT_FINETUNE_MODE=full \
    FFT_TRAINING_BACKEND=kt FFT_PRECISION=bf16 FFT_FINETUNING_TYPE=full \
    FFT_TEXT_ONLY=1 FFT_STEP_TIMING_OUT_DIR="$run_dir/step_timing" \
    FFT_STEP_TIMING_WARMUP_STEPS=5 \
    FFT_STEP_TIMING_TOKENS_PER_STEP="$((8 * seq_len))" \
    FFT_DISABLE_PERF_PROBES=1 FFT_CUDA_EMPTY_CACHE_AFTER_PREPARE=0 \
    FFT_CPU_THREADS=2 FFT_KT_OWNER_THREADS=64 FFT_KT_NON_OWNER_THREADS=2 \
    KT_BACKWARD_TIMING=off KT_SFT_PROFILE=0 \
    KT_REUSE_CHECKPOINT_FORWARD=1 KT_REUSE_CHECKPOINT_FORWARD_DISTRIBUTED=1 \
    DS_PROBE_MODE=off ACCELERATE_KT_MODEL_MAX_LENGTH="$seq_len" \
    OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
    NUMEXPR_NUM_THREADS=2 BLIS_NUM_THREADS=2 OMP_DYNAMIC=FALSE \
    MKL_DYNAMIC=FALSE ACCELERATE_KT_OMP_NUM_THREADS=2 \
    TOKENIZERS_PARALLELISM=false HF_DATASETS_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    HF_DATASETS_CACHE="$run_dir/datasets_cache" \
    CUDA_VISIBLE_DEVICES="$gpu_ids" \
    PYTHONPATH="$overlay:$root/src/accelerate:$root/src/transformers:$tools_dir:$llamafactory${PYTHONPATH:+:$PYTHONPATH}" \
    "$python" "$tools_dir/kt_case_monitor.py" --output "$run_dir" --gpu-ids "$gpu_ids" -- \
    "$accelerate" launch --config_file "$run_dir/accelerate_config.yaml" \
    -m kt_bench_entry train "$run_dir/train_config.yaml" \
    > "$run_dir/train.log" 2>&1
status=$?
set -e
printf '%s\n' "$status" > "$run_dir/exit_code.txt"
echo "$run_dir status=$status"
exit "$status"
