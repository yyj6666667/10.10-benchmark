#!/usr/bin/env bash
# Experimental GLM seq4096 alias-slot patch: semantic gate, then throughput.
set -Eeuo pipefail

root=/mnt/hurry-bench
case_script=$root/tools/run_qj_megatrain_glm4096_alias_case.sh
[[ -f $case_script ]] || { echo "missing $case_script" >&2; exit 1; }

export MEGATRAIN_ALIAS_LAYER_SLOTS=1
export MEGATRAIN_OFFLOAD_EMB_HEAD=1
export MEGATRAIN_RELEASE_LAYER_SLOT_DURING_HEAD=0
export MEGATRAIN_DIRECT_SHARED_INIT=0
export MEGATRAIN_CHECKPOINT_INTERVAL=2
unset MEGATRAIN_PROFILE_DIR MEGATRAIN_PROFILE_STEP MEGATRAIN_PROFILE_RANK

echo '[alias-slots patch] GLM-4.5-Air seq4096: fixed-batch loss check'
MEGATRAIN_REPEAT_BATCH=1 bash "$case_script" glm45_air 4096 4 2

echo '[alias-slots patch] semantic check passed; measuring throughput'
MEGATRAIN_REPEAT_BATCH=0 bash "$case_script" glm45_air 4096 4 2

echo "[alias-slots patch] $root/results/glm_megatrain_alias_slots_patch_performance.csv"
