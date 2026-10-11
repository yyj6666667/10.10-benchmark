#!/usr/bin/env bash
# One isolated eight-GPU MegaTrain full fine-tuning case on qj5090.
set -Eeuo pipefail

# PyTorch's file-descriptor sharing keeps one handle per shared tensor.
# GLM has more than 17,000 tensors, plus shared gradients and layer flats.
ulimit -n 131072

if (( $# < 2 || $# > 4 )); then
    echo "usage: $0 glm45_air|qwen35_122b SEQ [STEPS [WARMUP]]" >&2
    exit 2
fi
model_kind=$1
seq_len=$2
steps=${3:-4}
warmup=${4:-2}
offload_emb_head=${MEGATRAIN_OFFLOAD_EMB_HEAD:-0}
profile_dir=${MEGATRAIN_PROFILE_DIR:-}
profile_step=${MEGATRAIN_PROFILE_STEP:-3}
profile_rank=${MEGATRAIN_PROFILE_RANK:-0}
[[ $model_kind == glm45_air || $model_kind == qwen35_122b ]] || exit 2
[[ $offload_emb_head == 0 || $offload_emb_head == 1 ]] || exit 2
[[ $seq_len =~ ^[1-9][0-9]*$ && $steps =~ ^[1-9][0-9]*$ && $warmup =~ ^[0-9]+$ ]] || exit 2
release_layer_slot=0
[[ $model_kind == qwen35_122b ]] && release_layer_slot=1
release_layer_slot=${MEGATRAIN_RELEASE_LAYER_SLOT_DURING_HEAD:-$release_layer_slot}
[[ $release_layer_slot == 0 || $release_layer_slot == 1 ]] || exit 2
[[ $release_layer_slot == 0 || $offload_emb_head == 1 ]] || exit 2
checkpoint_interval=${MEGATRAIN_CHECKPOINT_INTERVAL:-$(( seq_len >= 4096 ? 2 : 1 ))}
[[ $checkpoint_interval =~ ^[1-9][0-9]*$ ]] || exit 2
num_grad_slabs=$((2 * checkpoint_interval))
(( steps > warmup )) || exit 2
if [[ -n $profile_dir ]]; then
    [[ $profile_step =~ ^[1-9][0-9]*$ && $profile_rank =~ ^[0-7]$ ]] || exit 2
    (( profile_step <= steps )) || exit 2
fi

root=/mnt/hurry-bench
python=/mnt/sft_yyj_yyj/venvs/qwen3-30b-a3b-deepspeed-cpuadam-py311/bin/python
export TORCH_EXTENSIONS_DIR=$root/torch_extensions
mkdir -p "$TORCH_EXTENSIONS_DIR"
checkpoint=/mnt/data/models/GLM-4.5-Air
[[ $model_kind == qwen35_122b ]] && checkpoint=/mnt/data/models/Qwen3.5-122B-A10B
for path in "$checkpoint/model.safetensors.index.json" \
    "$root/tools/megatrain_full_ft.py" "$root/tools/kt_case_monitor.py" \
    "$root/tools/record_qj_semantic_check.py" \
    "$root/tools/collect_qj_fsdp2_onednn_result.py" \
    "$root/tools/host_memory_guard.sh" \
    "$root/tools/megatrain_oom_guard.sh" \
    "$root/src/megatrain/infinity/model/cpu_master.py" \
    "$root/src/megatrain/infinity/model/mp_worker.py"; do
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
if [[ -n $profile_dir && $mode != semantic ]]; then
    echo "profiling requires MEGATRAIN_REPEAT_BATCH=1" >&2
    exit 2
fi
run_dir=$root/runs/${model_kind}_megatrain_seq${seq_len}_8gpu_${stamp}_$$_${mode}
mkdir -p "$run_dir"
"$python" - "$run_dir" "$model_kind" "$seq_len" "$steps" "$warmup" "$mode" "$offload_emb_head" "$release_layer_slot" "$profile_dir" "$profile_step" "$profile_rank" "$checkpoint_interval" "$num_grad_slabs" <<'PY'
import json, sys
from pathlib import Path
run, model, seq, steps, warmup, mode, offload_emb_head, release_layer_slot, profile_dir, profile_step, profile_rank, checkpoint_interval, num_grad_slabs = sys.argv[1:]
Path(run, 'run_config.json').write_text(json.dumps({
    'host': 'qj5090', 'backend': 'megatrain', 'model_kind': model,
    'sequence_length': int(seq), 'global_batch': 8, 'precision': 'bf16',
    'steps': int(steps), 'warmup_steps': int(warmup),
    'mode': mode,
    'offload_embedding_head_gpu': offload_emb_head == '1',
    'release_layer_slot_during_head': release_layer_slot == '1',
    'checkpoint_interval': int(checkpoint_interval),
    'num_grad_slabs': int(num_grad_slabs),
    'dataset': 'fft_real_100', 'gradient_clipping': 1.0,
    'host_available_guard_gib': 20,
    'profile_dir': profile_dir or None,
    'profile_step': int(profile_step) if profile_dir else None,
    'profile_rank': int(profile_rank) if profile_dir else None,
}, indent=2) + '\n')
PY
sha256sum "$root/tools/megatrain_full_ft.py" \
    "$root/src/megatrain/infinity/model/cpu_master.py" \
    "$root/src/megatrain/infinity/model/mp_worker.py" \
    "$root/src/megatrain/infinity/model/mp_state.py" \
    "$root/data/fft_real_100.json" "$checkpoint/config.json" \
    "$checkpoint/model.safetensors.index.json" > "$run_dir/checksums.txt"
if [[ -n $profile_dir ]]; then
    sha256sum "$root/tools/megatrain_profile_hooks.py" \
        "$root/tools/patch_megatrain_worker_profiler.py" \
        >> "$run_dir/checksums.txt"
fi
env PYTHONPATH="$root/tools:$root/src/megatrain:$root/src/transformers:$root/src/llamafactory-overlay" \
    "$python" -c 'import torch, transformers, infinity; print(torch.__version__, transformers.__version__, infinity.__file__)' \
    > "$run_dir/versions.txt"
nvidia-smi -L >> "$run_dir/versions.txt"

cmd=("$python" "$root/tools/megatrain_full_ft.py"
    --model-kind "$model_kind" --dataset-dir "$root/data"
    --output-dir "$run_dir/case" --sequence-length "$seq_len"
    --steps "$steps" --warmup-steps "$warmup"
    --checkpoint-interval "$checkpoint_interval"
    --num-grad-slabs "$num_grad_slabs")
if [[ ${MEGATRAIN_REPEAT_BATCH:-0} == 1 ]]; then cmd+=(--repeat-batch); fi
printf '%q ' "${cmd[@]}" > "$run_dir/command.txt"
printf '\n' >> "$run_dir/command.txt"

set +e
bash "$root/tools/host_memory_guard.sh" "$$" "$run_dir" 20 \
    > "$run_dir/host_memory_guard.log" 2>&1 &
guard_pid=$!
env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 \
    MEGATRAIN_OFFLOAD_EMB_HEAD="$offload_emb_head" \
    MEGATRAIN_RELEASE_LAYER_SLOT_DURING_HEAD="$release_layer_slot" \
    MEGATRAIN_PROFILE_DIR="$profile_dir" \
    MEGATRAIN_PROFILE_STEP="$profile_step" MEGATRAIN_PROFILE_RANK="$profile_rank" \
    TORCH_EXTENSIONS_DIR="$root/torch_extensions" \
    TOKENIZERS_PARALLELISM=false HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    PYTHONPATH="$root/tools:$root/src/megatrain:$root/src/transformers:$root/src/llamafactory-overlay" \
    PYTORCH_ALLOC_CONF=expandable_segments:True \
    "$python" "$root/tools/kt_case_monitor.py" --output "$run_dir" \
    --gpu-ids 0,1,2,3,4,5,6,7 -- "${cmd[@]}" \
    > "$run_dir/launcher.log" 2>&1 &
monitor_pid=$!
bash "$root/tools/megatrain_oom_guard.sh" "$monitor_pid" "$run_dir/launcher.log" \
    > "$run_dir/oom_guard.log" 2>&1 &
oom_guard_pid=$!
wait "$monitor_pid"
status=$?
kill -TERM "$oom_guard_pid" 2>/dev/null || true
wait "$oom_guard_pid" 2>/dev/null || true
kill -TERM "$guard_pid" 2>/dev/null || true
wait "$guard_pid" 2>/dev/null || true
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
passed = (len(losses) == timing['steps_requested']
          and len(losses) >= 3
          and all(map(math.isfinite, losses))
          and losses[-1] < losses[1]
          and losses[-1] < losses[0])
(run / 'loss_check.json').write_text(json.dumps({
    'losses': losses, 'initial_loss': losses[0], 'final_loss': losses[-1],
    'passed': passed, 'repeat_batch': True,
}, indent=2) + '\n')
print(f'semantic loss: {losses[0]:.6f} -> {losses[-1]:.6f}; passed={passed}')
if not passed:
    raise SystemExit('fixed-batch loss did not continue decreasing after the first update')
PY
    then
        status=4
        printf '%s\n' "$status" > "$run_dir/exit_code.txt"
    fi
fi
if [[ $status -eq 0 && ${MEGATRAIN_REPEAT_BATCH:-0} == 1 ]]; then
    short=glm
    [[ $model_kind == qwen35_122b ]] && short=qwen
    "$python" "$root/tools/record_qj_semantic_check.py" \
        "$short" megatrain "$run_dir"
fi
echo "$run_dir status=$status"
exit "$status"
