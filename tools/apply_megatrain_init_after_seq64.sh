#!/usr/bin/env bash
# Apply the optional GPU-context change between completed GLM cases.
set -Eeuo pipefail
root=/mnt/hurry-bench
python=/mnt/sft_yyj_yyj/venvs/qwen3-30b-a3b-deepspeed-cpuadam-py311/bin/python
worker=$root/src/megatrain/infinity/model/mp_worker.py

while true; do
    run=$(find "$root/runs" -maxdepth 1 -type d \
        -name 'glm45_air_megatrain_seq64_8gpu_20261011_*_benchmark' \
        -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-)
    if [[ -n $run && -f $run/exit_code.txt ]]; then
        if [[ -z $(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits) ]]; then
            "$python" "$root/tools/patch_megatrain_init_residency.py" "$worker"
            "$python" -m py_compile "$worker"
            sha256sum "$worker"
            echo "applied after $run exit=$(cat "$run/exit_code.txt") at $(date -u -Is)"
            exit 0
        fi
    fi
    sleep 1
done
