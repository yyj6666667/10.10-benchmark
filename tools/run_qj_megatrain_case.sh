#!/usr/bin/env bash
# One isolated eight-GPU MegaTrain full fine-tuning case on qj5090.
set -Eeuo pipefail

if (( $# < 2 || $# > 4 )); then
    echo "usage: $0 glm45_air|qwen35_122b SEQ [STEPS [WARMUP]]" >&2
    exit 2
fi
model_kind=$1
seq_len=$2
steps=${3:-8}
warmup=${4:-2}
[[ $model_kind == glm45_air || $model_kind == qwen35_122b ]] || exit 2
[[ $seq_len =~ ^[1-9][0-9]*$ && $steps =~ ^[1-9][0-9]*$ && $warmup =~ ^[0-9]+$ ]] || exit 2
(( steps > warmup )) || exit 2

root=/mnt/hurry-bench
python=/mnt/sft_yyj_yyj/venvs/qwen3-30b-a3b-deepspeed-cpuadam-py311/bin/python
export TORCH_EXTENSIONS_DIR=$root/torch_extensions
mkdir -p "$TORCH_EXTENSIONS_DIR"
checkpoint=/mnt/data/models/GLM-4.5-Air
[[ $model_kind == qwen35_122b ]] && checkpoint=/mnt/data/models/Qwen3.5-122B-A10B
for path in "$checkpoint/model.safetensors.index.json" \
    "$root/tools/megatrain_full_ft.py" "$root/tools/kt_case_monitor.py" \
    "$root/tools/collect_qj_fsdp2_onednn_result.py" \
    "$root/src/megatrain/infinity/model/cpu_master.py"; do
    [[ -f $path ]] || { echo "missing $path" >&2; exit 1; }
done

compute_apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)
[[ -z $compute_apps ]] || { echo "GPU compute process active: $compute_apps" >&2; exit 3; }
available_kib=$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)
[[ $available_kib -ge 1100000000 ]] || { echo "host RAM too low: $available_kib KiB" >&2; exit 3; }

# Build and validate the CPUAdam kernel before allocating hundreds of GB.
"$python" - <<'PY'
import torch
from deepspeed.ops.adam import DeepSpeedCPUAdam
p = torch.nn.Parameter(torch.tensor([0.1], dtype=torch.bfloat16))
o = DeepSpeedCPUAdam([p], lr=0.01, fp32_optimizer_states=False)
p.grad = torch.ones_like(p)
o.step()
if not p.item() < 0.1:
    raise RuntimeError('BF16 CPUAdam update did not change the weight')
print('[preflight] BF16 CPUAdam update OK')
PY

stamp=$(date +%Y%m%d_%H%M%S)
mode=benchmark
[[ ${MEGATRAIN_REPEAT_BATCH:-0} == 1 ]] && mode=semantic
run_dir=$root/runs/${model_kind}_megatrain_seq${seq_len}_8gpu_${stamp}_$$_${mode}
mkdir -p "$run_dir"
"$python" - "$run_dir" "$model_kind" "$seq_len" "$steps" "$warmup" "$mode" <<'PY'
import json, sys
from pathlib import Path
run, model, seq, steps, warmup, mode = sys.argv[1:]
Path(run, 'run_config.json').write_text(json.dumps({
    'host': 'qj5090', 'backend': 'megatrain', 'model_kind': model,
    'sequence_length': int(seq), 'global_batch': 8, 'precision': 'bf16',
    'steps': int(steps), 'warmup_steps': int(warmup),
    'mode': mode,
    'checkpoint_interval': 1, 'num_grad_slabs': 2,
    'dataset': 'fft_real_100', 'gradient_clipping': 1.0,
}, indent=2) + '\n')
PY
sha256sum "$root/tools/megatrain_full_ft.py" \
    "$root/src/megatrain/infinity/model/cpu_master.py" \
    "$root/src/megatrain/infinity/model/mp_state.py" \
    "$root/data/fft_real_100.json" "$checkpoint/config.json" \
    "$checkpoint/model.safetensors.index.json" > "$run_dir/checksums.txt"
env PYTHONPATH="$root/tools:$root/src/megatrain:$root/src/transformers:$root/src/llamafactory-overlay" \
    "$python" -c 'import torch, transformers, infinity; print(torch.__version__, transformers.__version__, infinity.__file__)' \
    > "$run_dir/versions.txt"
nvidia-smi -L >> "$run_dir/versions.txt"

cmd=("$python" "$root/tools/megatrain_full_ft.py"
    --model-kind "$model_kind" --dataset-dir "$root/data"
    --output-dir "$run_dir/case" --sequence-length "$seq_len"
    --steps "$steps" --warmup-steps "$warmup")
if [[ ${MEGATRAIN_REPEAT_BATCH:-0} == 1 ]]; then cmd+=(--repeat-batch); fi
printf '%q ' "${cmd[@]}" > "$run_dir/command.txt"
printf '\n' >> "$run_dir/command.txt"

set +e
env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 \
    TORCH_EXTENSIONS_DIR="$root/torch_extensions" \
    TOKENIZERS_PARALLELISM=false HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    PYTHONPATH="$root/tools:$root/src/megatrain:$root/src/transformers:$root/src/llamafactory-overlay" \
    PYTORCH_ALLOC_CONF=expandable_segments:True \
    "$python" "$root/tools/kt_case_monitor.py" --output "$run_dir" \
    --gpu-ids 0,1,2,3,4,5,6,7 -- "${cmd[@]}" \
    > "$run_dir/launcher.log" 2>&1
status=$?
set -e
printf '%s\n' "$status" > "$run_dir/exit_code.txt"
if [[ $status -eq 0 && ${MEGATRAIN_REPEAT_BATCH:-0} != 1 ]]; then
    "$python" "$root/tools/collect_qj_fsdp2_onednn_result.py" "$run_dir" \
        > "$run_dir/performance.csv"
    if [[ $model_kind == glm45_air ]]; then
        table=$root/results/glm_megatrain_performance.csv
    else
        table=$root/results/qwen_megatrain_performance.csv
    fi
    if [[ ! -e $table ]]; then
        cp "$run_dir/performance.csv" "$table"
    else
        tail -n +2 "$run_dir/performance.csv" >> "$table"
    fi
fi
if [[ $status -eq 0 && ${MEGATRAIN_REPEAT_BATCH:-0} == 1 ]]; then
    if ! "$python" - "$run_dir" <<'PY'
import json, math, sys
from pathlib import Path
run = Path(sys.argv[1])
timing = json.loads((run / 'case/step_timing.json').read_text())
losses = [row['loss'] for row in timing['rows']]
passed = len(losses) == timing['steps_requested'] and all(map(math.isfinite, losses)) and losses[-1] < losses[0]
(run / 'loss_check.json').write_text(json.dumps({
    'losses': losses, 'initial_loss': losses[0], 'final_loss': losses[-1],
    'passed': passed, 'repeat_batch': True,
}, indent=2) + '\n')
print(f'semantic loss: {losses[0]:.6f} -> {losses[-1]:.6f}; passed={passed}')
if not passed:
    raise SystemExit('fixed-batch loss did not decrease')
PY
    then
        status=4
        printf '%s\n' "$status" > "$run_dir/exit_code.txt"
    fi
fi
echo "$run_dir status=$status"
exit "$status"
