#!/usr/bin/env bash
# Pull completed qj5090 result rows and publish the refreshed private chart.
set -Eeuo pipefail

root=$(cd "$(dirname "$0")/.." && pwd)
cd "$root"
export MPLCONFIGDIR=/tmp/mpl-qj-results-watcher
mkdir -p "$MPLCONFIGDIR"

while true; do
    scratch=$(mktemp -d /tmp/qj-results-sync.XXXXXX)
    if scp -q -o BatchMode=yes -o ConnectTimeout=10 \
        -o ServerAliveInterval=10 -o ServerAliveCountMax=2 -J sap1 -r \
        qj5090:/mnt/hurry-bench/results/. "$scratch/"; then
        changed=0
        paths=()
        for stem in glm_kt qwen_kt glm45_air_fsdp2_onednn qwen35_122b_fsdp2_onednn \
            glm_megatrain qwen_megatrain glm_zero_offload qwen_zero_offload; do
            source=$scratch/${stem}_performance.csv
            [[ -f $source ]] || continue
            target=results/qj5090_${stem}_performance.csv
            normalized=$scratch/${stem}_normalized.csv
            python - "$source" "$normalized" <<'PY'
import sys
from pathlib import Path
Path(sys.argv[2]).write_text(Path(sys.argv[1]).read_text().replace("\r\n", "\n"))
PY
            if [[ ! -f $target ]] || ! cmp -s "$normalized" "$target"; then
                cp "$normalized" "$target"
                changed=1
            fi
            paths+=("$target")
        done
        if [[ -f $scratch/case_status.csv ]]; then
            source=$scratch/case_status.csv
            target=results/qj5090_case_status.csv
            normalized=$scratch/case_status_normalized.csv
            python - "$source" "$normalized" <<'PY'
import sys
from pathlib import Path
Path(sys.argv[2]).write_text(Path(sys.argv[1]).read_text().replace("\r\n", "\n"))
PY
            if [[ ! -f $target ]] || ! cmp -s "$normalized" "$target"; then
                cp "$normalized" "$target"
                changed=1
            fi
            paths+=("$target")
        fi
        if [[ -f $scratch/semantic_checks.csv ]]; then
            source=$scratch/semantic_checks.csv
            target=results/qj5090_semantic_checks.csv
            normalized=$scratch/semantic_checks_normalized.csv
            python - "$source" "$normalized" <<'PY'
import sys
from pathlib import Path
Path(sys.argv[2]).write_text(Path(sys.argv[1]).read_text().replace("\r\n", "\n"))
PY
            if [[ ! -f $target ]] || ! cmp -s "$normalized" "$target"; then
                cp "$normalized" "$target"
                changed=1
            fi
            paths+=("$target")
        fi
        if (( changed )); then
            if python tools/render_qj_figure9.py; then
                git add -- "${paths[@]}" results/qj5090_figure9.csv \
                    results/qj5090_figure9.png results/qj5090_figure9.svg \
                    results/qj5090_performance_tables.md
                if ! git diff --cached --quiet; then
                    git commit -m "Update qj5090 full fine-tuning measurements"
                fi
            else
                echo "chart validation failed; waiting for a complete result table" >&2
            fi
        fi
        if [[ $(git rev-list --count origin/main..HEAD) -gt 0 ]]; then
            if ! git -c credential.helper='!gh auth git-credential' push origin main; then
                echo "result push failed at $(date -Is); retrying" >&2
            fi
        fi
    else
        echo "remote result sync failed at $(date -Is); retrying" >&2
    fi
    rm -r -- "$scratch"
    sleep 45
done
