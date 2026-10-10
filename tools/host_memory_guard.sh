#!/usr/bin/env bash
# Stop one qj5090 benchmark case if host RAM falls below its safety reserve.
set -Eeuo pipefail

if (( $# < 2 || $# > 3 )); then
    echo "usage: $0 RUNNER_PID RUN_DIR [MIN_AVAILABLE_GIB]" >&2
    exit 2
fi
runner=$1
run_dir=$2
reserve_gib=${3:-20}
[[ $runner =~ ^[1-9][0-9]*$ && $reserve_gib =~ ^[1-9][0-9]*$ ]] || exit 2
[[ -d $run_dir ]] || exit 2
reserve_kib=$((reserve_gib * 1048576))

while kill -0 "$runner" 2>/dev/null; do
    available_kib=$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)
    if (( available_kib < reserve_kib )); then
        monitor=$(pgrep -P "$runner" -f 'kt_case_monitor.py' | head -1 || true)
        torchrun=''
        if [[ -n $monitor ]]; then
            torchrun=$(pgrep -P "$monitor" -f 'torchrun.*kt_bench_entry' | head -1 || true)
        fi
        printf 'host_memory_guard: MemAvailable=%s KiB at %s; reserve=%s GiB; runner=%s; torchrun=%s\n' \
            "$available_kib" "$(date -u -Is)" "$reserve_gib" "$runner" "$torchrun" \
            > "$run_dir/host_memory_guard.txt"
        if [[ -n $torchrun ]]; then
            pgid=$(ps -o pgid= -p "$torchrun" | tr -d ' ')
            [[ -n $pgid && $pgid != $$ ]] || exit 3
            kill -TERM -- "-$pgid" 2>/dev/null || true
        elif [[ -n $monitor ]]; then
            kill -TERM "$monitor" 2>/dev/null || true
        else
            kill -TERM "$runner" 2>/dev/null || true
        fi
        exit 0
    fi
    sleep 2
done
