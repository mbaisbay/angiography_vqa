#!/bin/bash
# =============================================================================
# Run all 4 ARCADE YOLO experiments sequentially on a single GPU
# =============================================================================
# Usage:
#   bash run_all.sh                          # full pipeline
#   bash run_all.sh --skip-data-prep         # skip data prep (already done)
#   bash run_all.sh --runs 1,3               # run only run1 and run3
#   bash run_all.sh 2>&1 | tee pipeline.log  # save full log
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

ARCADE_ROOT="../../arcade/submission"
DATA_DIR="../data"
SKIP_DATA_PREP=false
SELECTED_RUNS="1,2,3,4"

for arg in "$@"; do
    case $arg in
        --skip-data-prep) SKIP_DATA_PREP=true ;;
        --runs=*) SELECTED_RUNS="${arg#*=}" ;;
        --runs) shift; SELECTED_RUNS="$2" ;;  # handled below
    esac
done
# Handle --runs N form
while [[ $# -gt 0 ]]; do
    case $1 in
        --runs) SELECTED_RUNS="$2"; shift 2 ;;
        *) shift ;;
    esac
done

TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
echo "============================================"
echo "ARCADE Experiments Pipeline"
echo "Started: $TIMESTAMP"
echo "Runs:    $SELECTED_RUNS"
echo "============================================"

# Step 0: Data preparation
if [ "$SKIP_DATA_PREP" = false ]; then
    echo ""
    echo "============================================"
    echo "[STEP 0] Data Preparation"
    echo "============================================"
    python prepare_data.py --arcade-root "$ARCADE_ROOT" --data-dir "$DATA_DIR"
else
    echo ""
    echo "[SKIP] Data preparation (--skip-data-prep)"
fi

# Run experiments
CONFIGS=(
    "1:../configs/run1_yolo11m.yaml"
    "2:../configs/run2_yolo11l.yaml"
    "3:../configs/run3_yolo26m.yaml"
    "4:../configs/run4_yolo11x.yaml"
)

for entry in "${CONFIGS[@]}"; do
    RUN_NUM="${entry%%:*}"
    CONFIG="${entry#*:}"

    if [[ ! ",$SELECTED_RUNS," == *",$RUN_NUM,"* ]]; then
        continue
    fi

    RUN_START=$(date '+%Y-%m-%d %H:%M:%S')
    echo ""
    echo "============================================"
    echo "[RUN $RUN_NUM] $CONFIG"
    echo "Started: $RUN_START"
    echo "============================================"

    python run_pipeline.py \
        --config "$CONFIG" \
        --arcade-root "$ARCADE_ROOT" \
        --skip-data-prep

    echo "[RUN $RUN_NUM] Finished: $(date '+%Y-%m-%d %H:%M:%S')"
done

# Compare all results
echo ""
echo "============================================"
echo "[COMPARE] All runs"
echo "============================================"
python compare_runs.py --results-root ../results

END_TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
echo ""
echo "============================================"
echo "ALL EXPERIMENTS COMPLETE"
echo "Started:  $TIMESTAMP"
echo "Finished: $END_TIMESTAMP"
echo "Results:  ../results/"
echo "============================================"
