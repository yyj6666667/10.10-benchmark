#!/usr/bin/env bash
# One eight-GPU ZeRO-3 parameter and optimizer CPU-offload full fine-tuning case.
set -Eeuo pipefail

if (( $# < 2 || $# > 4 )); then
    echo "usage: $0 glm|qwen SEQ [STEPS [WARMUP]]" >&2
    exit 2
fi
model=$1
seq_len=$2
steps=${3:-8}
warmup=${4:-2}
[[ $model == glm || $model == qwen ]] || exit 2
[[ $seq_len =~ ^[1-9][0-9]*$ && $steps =~ ^[1-9][0-9]*$ && $warmup =~ ^[0-9]+$ ]] || exit 2
(( steps > warmup )) || exit 2

root=/mnt/hurry-bench
python=/mnt/sft_yyj_yyj/venvs/qwen3-30b-a3b-deepspeed-cpuadam-py311/bin/python
torchrun=/mnt/sft_yyj_yyj/venvs/qwen3-30b-a3b-deepspeed-cpuadam-py311/bin/torchrun
checkpoint=/mnt/data/models/GLM-4.5-Air
template=glm4_moe
if [[ $model == qwen ]]; then
    checkpoint=/mnt/data/models/Qwen3.5-122B-A10B
    template=qwen3
fi
expected_params=106852245504
[[ $model == qwen ]] && expected_params=122111526912
for path in "$checkpoint/model.safetensors.index.json" \
    "$root/tools/zero3_cpu_offload.json" "$root/tools/kt_bench_entry.py" \
    "$root/tools/record_qj_semantic_check.py" \
    "$root/tools/step_phase_timer.py" "$root/tools/kt_case_monitor.py" \
    "$root/tools/host_memory_guard.sh" \
    "$root/src/llamafactory-overlay/llamafactory/cli.py"; do
    [[ -f $path ]] || { echo "missing $path" >&2; exit 1; }
done
compute_apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)
[[ -z $compute_apps ]] || { echo "GPU compute process active: $compute_apps" >&2; exit 3; }
available_kib=$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)
[[ $available_kib -ge 1050000000 ]] || { echo "host RAM too low: $available_kib KiB" >&2; exit 3; }

stamp=$(date +%Y%m%d_%H%M%S)
mode=${ZERO_BENCH_MODE:-benchmark}
[[ $mode == benchmark || $mode == semantic ]] || exit 2
run_dir=$root/runs/${model}_zero_offload_seq${seq_len}_8gpu_${stamp}_$$_${mode}
mkdir -p "$run_dir"
"$python" - "$root" "$run_dir" "$checkpoint" "$template" "$model" "$seq_len" "$steps" "$warmup" "$mode" <<'PY'
import json, sys
from pathlib import Path
import yaml
root, run, checkpoint, template, model, seq, steps, warmup, mode = sys.argv[1:]
root, run = Path(root), Path(run)
train = yaml.safe_load((root / 'tools/kt_glm_train_base.yaml').read_text())
for key in ('use_kt', 'kt_weight_path', 'kt_config'):
    train.pop(key, None)
train.update(
    model_name_or_path=checkpoint, dataset='fft_real_100',
    dataset_dir=str(root / 'data'), template=template,
    cutoff_len=int(seq), max_samples=8 if mode == 'semantic' else 100,
    output_dir=str(run / 'model_output'),
    per_device_train_batch_size=1, gradient_accumulation_steps=1,
    max_steps=int(steps), logging_strategy='steps', logging_steps=1,
    save_strategy='no', report_to='none', max_grad_norm=1.0,
    pure_bf16=False, bf16=True, gradient_checkpointing=True,
    dataloader_num_workers=0, dataloader_persistent_workers=False,
    preprocessing_num_workers=1,
    deepspeed=str(root / 'tools/zero3_cpu_offload.json'),
    freeze_vision_tower=True, freeze_multi_modal_projector=True,
    freeze_language_model=False,
    logging_nan_inf_filter=False,
)
train.pop('dataloader_prefetch_factor', None)
(run / 'train_config.yaml').write_text(yaml.safe_dump(train, sort_keys=False))
(run / 'run_config.json').write_text(json.dumps({
    'host': 'qj5090', 'backend': 'deepspeed_zero3_cpu_offload',
    'model_kind': model, 'checkpoint': checkpoint,
    'sequence_length': int(seq), 'steps': int(steps),
    'warmup_steps': int(warmup), 'tokens_per_step': 8 * int(seq),
    'global_batch': 8, 'precision': 'bf16', 'dataset': 'fft_real_100',
    'gradient_clipping': 1.0,
    'optimizer_offload': 'cpu', 'parameter_offload': 'cpu',
    'host_available_guard_gib': 20,
    'mode': mode,
}, indent=2) + '\n')
PY
sha256sum "$root/tools/zero3_cpu_offload.json" "$root/tools/kt_bench_entry.py" \
    "$root/tools/step_phase_timer.py" "$root/data/fft_real_100.json" \
    "$checkpoint/config.json" "$checkpoint/model.safetensors.index.json" \
    "$run_dir/train_config.yaml" > "$run_dir/checksums.txt"
env PYTHONPATH="$root/tools:$root/src/transformers:$root/src/llamafactory-overlay" \
    "$python" -c 'import torch, deepspeed, accelerate, transformers, llamafactory; print(torch.__version__, deepspeed.__version__, accelerate.__version__, transformers.__version__, llamafactory.__file__)' \
    > "$run_dir/versions.txt"
nvidia-smi -L >> "$run_dir/versions.txt"

cmd=("$torchrun" --standalone --nproc_per_node=8 -m kt_bench_entry "$run_dir/train_config.yaml")
printf '%q ' "${cmd[@]}" > "$run_dir/command.txt"
printf '\n' >> "$run_dir/command.txt"
set +e
bash "$root/tools/host_memory_guard.sh" "$$" "$run_dir" 20 \
    > "$run_dir/host_memory_guard.log" 2>&1 &
guard_pid=$!
env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 \
    TOKENIZERS_PARALLELISM=false HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    PYTHONPATH="$root/tools:$root/src/transformers:$root/src/llamafactory-overlay" \
    FFT_STEP_TIMING_OUT_DIR="$run_dir/step_timing" \
    FFT_STEP_TIMING_WARMUP_STEPS="$warmup" \
    FFT_STEP_TIMING_TOKENS_PER_STEP="$((8 * seq_len))" \
    FFT_TRAINING_BACKEND=deepspeed DISABLE_VERSION_CHECK=1 \
    FFT_EXPECTED_TRAINABLE_PARAMS="$expected_params" \
    PYTORCH_ALLOC_CONF=expandable_segments:True \
    "$python" "$root/tools/kt_case_monitor.py" --output "$run_dir" \
    --gpu-ids 0,1,2,3,4,5,6,7 -- "${cmd[@]}" \
    > "$run_dir/launcher.log" 2>&1
status=$?
kill -TERM "$guard_pid" 2>/dev/null || true
wait "$guard_pid" 2>/dev/null || true
set -e
printf '%s\n' "$status" > "$run_dir/exit_code.txt"
if [[ $status -eq 0 && $mode == semantic ]]; then
    if ! "$python" - "$run_dir" <<'PY'
import json, math, re, sys
from pathlib import Path
run = Path(sys.argv[1])
text = (run / 'launcher.log').read_text(errors='replace')
losses = [float(value) for value in re.findall(r"'loss':\s*'?([0-9.eE+-]+)'?", text)]
passed = len(losses) >= 2 and all(map(math.isfinite, losses)) and losses[-1] < losses[0]
(run / 'loss_check.json').write_text(json.dumps({
    'losses': losses, 'initial_loss': losses[0] if losses else None,
    'final_loss': losses[-1] if losses else None, 'passed': passed,
    'fixed_batch': True,
}, indent=2) + '\n')
print(f'semantic loss: {losses[0] if losses else None} -> '
      f'{losses[-1] if losses else None}; passed={passed}')
if not passed:
    raise SystemExit('fixed-batch loss did not decrease')
PY
    then
        status=4
        printf '%s\n' "$status" > "$run_dir/exit_code.txt"
    fi
fi
if [[ $status -eq 0 && $mode == semantic ]]; then
    "$python" "$root/tools/record_qj_semantic_check.py" \
        "$model" zero_offload "$run_dir"
fi
if [[ $status -eq 0 && $mode == benchmark ]]; then
    "$python" "$root/tools/collect_qj_kt_result.py" "$run_dir" \
        > "$run_dir/performance.csv"
    table=$root/results/${model}_zero_offload_performance.csv
    if [[ ! -e $table ]]; then
        cp "$run_dir/performance.csv" "$table"
    else
        tail -n +2 "$run_dir/performance.csv" >> "$table"
    fi
fi
echo "$run_dir status=$status"
exit "$status"
