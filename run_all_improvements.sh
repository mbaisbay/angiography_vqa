#!/usr/bin/env bash
# =============================================================================
# Run All Improvement Experiments — Sequential on Single GPU (5090)
# =============================================================================
#
# Usage:
#   bash run_all_improvements.sh              # run all 9 experiments
#   bash run_all_improvements.sh 2>&1 | tee improvements.log  # with full log
#
# Each experiment is run as a separate process for:
#   - Per-experiment log files (runs/logs/exp*.log)
#   - Better GPU memory cleanup between runs
#   - Ability to continue if one experiment fails
# =============================================================================

set -euo pipefail

export CUDA_VISIBLE_DEVICES=0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

LOG_DIR="runs/logs"
mkdir -p "$LOG_DIR"

EXPERIMENTS=(1 2 3 4 5 6 7 8 9)
EXP_NAMES=(
    "exp01_baseline"
    "exp02_clahe"
    "exp03_hec"
    "exp04_res640"
    "exp05_res768"
    "exp06_p2_head"
    "exp07_no_mosaic_mixup"
    "exp08_combined_best"
    "exp09_optuna_tpe"
)
EXP_DESCRIPTIONS=(
    "Baseline (amp=true, lrf=0.01)"
    "CLAHE preprocessing (clip=2.0, grid=8)"
    "HEC preprocessing (CLAHE + Canny edge blend)"
    "Resolution 640"
    "Resolution 768"
    "P2 small-object detection head (stride=4)"
    "No mosaic + no mixup"
    "Combined: HEC + P2 + no mosaic/mixup"
    "Optuna TPE hyperparameter sweep (20 trials)"
)

TOTAL=${#EXPERIMENTS[@]}
PASSED=0
FAILED=0
FAILED_LIST=""

OVERALL_START=$(date +%s)

echo "============================================================"
echo " IMPROVEMENT EXPERIMENTS — Single GPU (5090)"
echo " Started: $(date)"
echo " Total experiments: $TOTAL"
echo "============================================================"
echo ""

for i in "${!EXPERIMENTS[@]}"; do
    EXP_NUM=${EXPERIMENTS[$i]}
    EXP_NAME=${EXP_NAMES[$i]}
    EXP_DESC=${EXP_DESCRIPTIONS[$i]}
    LOG_FILE="${LOG_DIR}/${EXP_NAME}.log"

    echo "------------------------------------------------------------"
    echo " [$((i+1))/$TOTAL] ${EXP_NAME}"
    echo " ${EXP_DESC}"
    echo " Log: ${LOG_FILE}"
    echo " Start: $(date)"
    echo "------------------------------------------------------------"

    EXP_START=$(date +%s)

    if python run_improvement_experiments.py --experiments "$EXP_NUM" 2>&1 | tee "$LOG_FILE"; then
        EXP_END=$(date +%s)
        ELAPSED=$(( (EXP_END - EXP_START) / 60 ))
        echo ""
        echo " [PASS] ${EXP_NAME} completed in ${ELAPSED} min"
        PASSED=$((PASSED + 1))
    else
        EXP_END=$(date +%s)
        ELAPSED=$(( (EXP_END - EXP_START) / 60 ))
        echo ""
        echo " [FAIL] ${EXP_NAME} failed after ${ELAPSED} min (see ${LOG_FILE})"
        FAILED=$((FAILED + 1))
        FAILED_LIST="${FAILED_LIST}  - ${EXP_NAME}\n"
    fi

    echo ""
done

OVERALL_END=$(date +%s)
TOTAL_ELAPSED=$(( (OVERALL_END - OVERALL_START) / 60 ))

echo "============================================================"
echo " SUMMARY"
echo "============================================================"
echo " Finished: $(date)"
echo " Total time: ${TOTAL_ELAPSED} min"
echo " Passed: ${PASSED}/${TOTAL}"
echo " Failed: ${FAILED}/${TOTAL}"

if [ -n "$FAILED_LIST" ]; then
    echo ""
    echo " Failed experiments:"
    echo -e "$FAILED_LIST"
fi

echo "============================================================"
echo " Results saved to: improvement_experiments.json"
echo " Logs directory:   ${LOG_DIR}/"
echo "============================================================"
