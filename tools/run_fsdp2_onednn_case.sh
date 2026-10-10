#!/usr/bin/env bash
# Run one isolated sap4 FSDP2 + oneDNN case and sample its resource use.
set -Eeuo pipefail

if (( $# < 3 || $# > 5 )); then
    echo "usage: $0 qwen35_122b|glm45_air SEQ OUTPUT_DIR [STEPS [WARMUP]]" >&2
    exit 2
fi
model_kind=$1
sequence_length=$2
output_dir=$3
steps=${4:-15}
warmup=${5:-5}
[[ "$model_kind" == qwen35_122b || "$model_kind" == glm45_air ]] || exit 2
[[ "$sequence_length" =~ ^[0-9]+$ && "$steps" =~ ^[0-9]+$ && "$warmup" =~ ^[0-9]+$ ]] || exit 2
[[ "$output_dir" == /* ]] || { echo 'OUTPUT_DIR must be absolute' >&2; exit 2; }
[[ ! -e "$output_dir/exit_code.txt" ]] || { echo 'case already finished' >&2; exit 2; }

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
trainer="$script_dir/fsdp2_onednn_full_ft.py"
monitor=/mnt/data2/wbw/Ktransformers-development/FFTtest/Qwen3.5-35B-A3B/monitor.py
torchrun=/mnt/data2/wbw/conda/envs/Onednn/bin/torchrun
monitor_python=/mnt/data2/wbw/conda/envs/Deepspeed/bin/python3
dataset_dir=/mnt/data2/wbw/Ktransformers-development/FFTtest/dataset
mkdir -p "$output_dir"

sha256sum "$trainer" "$monitor" "$0" > "$output_dir/source_checksums.txt"
/mnt/data2/wbw/conda/envs/Onednn/bin/python3 -c 'import torch, transformers; print("torch", torch.__version__); print("transformers", transformers.__version__); print("cuda", torch.version.cuda); print("mkldnn", torch.backends.mkldnn.is_available())' > "$output_dir/versions.txt"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader,nounits > "$output_dir/gpu_before.csv"
free -b > "$output_dir/memory_before.txt"

cmd=("$torchrun" --standalone --nproc_per_node=8 "$trainer"
    --model-kind "$model_kind" --dataset-dir "$dataset_dir" --dataset-name fft_real_100
    --output-dir "$output_dir/case" --sequence-length "$sequence_length"
    --steps "$steps" --warmup-steps "$warmup" --threads-per-rank 12
    --learning-rate "${FSDP2_LEARNING_RATE:-1e-5}"
    --no-gradient-checkpointing)
if [[ "${FSDP2_REPEAT_BATCH:-0}" == 1 ]]; then
    cmd+=(--repeat-batch)
fi
printf '%q ' "${cmd[@]}" > "$output_dir/command.txt"
printf '\n' >> "$output_dir/command.txt"

env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 OPENBLAS_NUM_THREADS=1 \
    PYTORCH_ALLOC_CONF=expandable_segments:True \
    "${cmd[@]}" > "$output_dir/launcher.log" 2>&1 &
launcher_pid=$!
printf '%s\n' "$launcher_pid" > "$output_dir/launcher.pid"
"$monitor_python" "$monitor" --out "$output_dir/monitor.csv" \
    --fifo "$output_dir/monitor.fifo" --interval 2 --pid "$launcher_pid" \
    > "$output_dir/monitor.log" 2>&1 &
monitor_pid=$!

stop_monitor() {
    kill -TERM "$monitor_pid" 2>/dev/null || true
    wait "$monitor_pid" 2>/dev/null || true
}
trap stop_monitor EXIT
trap 'kill -TERM "$launcher_pid" 2>/dev/null || true; exit 143' TERM
set +e
wait "$launcher_pid"
result=$?
set -e
printf '%s\n' "$result" > "$output_dir/exit_code.txt"
exit "$result"
