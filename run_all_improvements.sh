#!/usr/bin/env bash
# =============================================================================
# Run All Improvement Experiments — Concurrent, Auto-GPU
# =============================================================================
#
# Usage:
#   bash run_all_improvements.sh              # run all 9 experiments
#   bash run_all_improvements.sh 2>&1 | tee improvements.log  # with full log
#
# Each experiment runs as a separate background process. ANGIO_FORCE_AUTO_DEVICE
# causes utils/config_loader.load_config to call gpu_scheduler.acquire_free_gpu
# at startup: each process claims a per-GPU fcntl lock under
# /tmp/angiography_vqa_gpu_locks/, so N processes spread across whatever GPUs
# are free and the surplus blocks until a lock is released. No manual
# concurrency cap is needed — the lockfiles enforce exclusion.
#
# Per-experiment log files still live in runs/logs/<EXP_NAME>.log. Failures
# are aggregated and reported in the summary.
# =============================================================================

set -uo pipefail

export ANGIO_FORCE_AUTO_DEVICE=1

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
OVERALL_START=$(date +%s)

echo "============================================================"
echo " IMPROVEMENT EXPERIMENTS — Concurrent, Auto-GPU"
echo " Started: $(date)"
echo " Total experiments: $TOTAL"
echo " GPU pool: auto (via utils/gpu_scheduler file locks)"
echo "============================================================"
echo ""

PIDS=()
PID_TO_NAME=()
PID_TO_START=()

for i in "${!EXPERIMENTS[@]}"; do
    EXP_NUM=${EXPERIMENTS[$i]}
    EXP_NAME=${EXP_NAMES[$i]}
    EXP_DESC=${EXP_DESCRIPTIONS[$i]}
    LOG_FILE="${LOG_DIR}/${EXP_NAME}.log"

    echo "------------------------------------------------------------"
    echo " [launch $((i+1))/$TOTAL] ${EXP_NAME}"
    echo " ${EXP_DESC}"
    echo " Log: ${LOG_FILE}"
    echo " Start: $(date)"
    echo "------------------------------------------------------------"

    EXP_START=$(date +%s)
    python run_improvement_experiments.py --experiments "$EXP_NUM" \
        > "$LOG_FILE" 2>&1 &
    PID=$!
    PIDS+=("$PID")
    PID_TO_NAME+=("$EXP_NAME")
    PID_TO_START+=("$EXP_START")
    echo " PID: $PID"
    echo ""

    # Tiny stagger so two processes don't race the same lock fd in the same
    # microsecond. acquire_free_gpu is race-free, but spacing keeps the
    # banner output legible.
    sleep 1
done

echo "============================================================"
echo " All $TOTAL experiments launched; waiting for completion…"
echo "============================================================"
echo ""

PASSED=0
FAILED=0
FAILED_LIST=""

for idx in "${!PIDS[@]}"; do
    PID=${PIDS[$idx]}
    EXP_NAME=${PID_TO_NAME[$idx]}
    EXP_START=${PID_TO_START[$idx]}
    LOG_FILE="${LOG_DIR}/${EXP_NAME}.log"

    if wait "$PID"; then
        RC=0
    else
        RC=$?
    fi
    EXP_END=$(date +%s)
    ELAPSED=$(( (EXP_END - EXP_START) / 60 ))

    if [ "$RC" -eq 0 ]; then
        echo " [PASS] ${EXP_NAME} completed in ${ELAPSED} min"
        PASSED=$((PASSED + 1))
    else
        echo " [FAIL] ${EXP_NAME} failed after ${ELAPSED} min (rc=${RC}, see ${LOG_FILE})"
        FAILED=$((FAILED + 1))
        FAILED_LIST="${FAILED_LIST}  - ${EXP_NAME}\n"
    fi
done

OVERALL_END=$(date +%s)
TOTAL_ELAPSED=$(( (OVERALL_END - OVERALL_START) / 60 ))

echo ""
echo "============================================================"
echo " SUMMARY"
echo "============================================================"
echo " Finished: $(date)"
echo " Total wall time: ${TOTAL_ELAPSED} min"
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

if [ "$FAILED" -gt 0 ]; then
    exit 1
fi
