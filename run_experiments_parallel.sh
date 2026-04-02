#!/bin/bash
# =============================================================================
# Run all 5 YOLO experiments in parallel across GPUs 1-5 (skip GPU 0)
# =============================================================================
# Usage:
#   bash run_experiments_parallel.sh                    # full run
#   bash run_experiments_parallel.sh --skip-preprocess  # already preprocessed
#   bash run_experiments_parallel.sh --skip-preprocess --epochs 20  # quick test
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PASS_THROUGH_ARGS=()
SKIP_PREPROCESS=false

for arg in "$@"; do
    if [ "$arg" = "--skip-preprocess" ]; then
        SKIP_PREPROCESS=true
    else
        PASS_THROUGH_ARGS+=("$arg")
    fi
done

TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
echo "============================================"
echo "  Parallel Experiment Runner"
echo "  Started: $TIMESTAMP"
echo "  GPUs: 1 2 3 4 5 (skipping GPU 0)"
echo "============================================"

# Step 1: Preprocess once (sequential) if needed
if [ "$SKIP_PREPROCESS" = false ]; then
    echo ""
    echo "[PREP] Filtering classes..."
    python filter_classes.py --config config.yaml --min-count 300
    echo "[PREP] Preprocessing images..."
    python preprocess_images.py --config config.yaml
    echo "[PREP] Done."
fi

# Step 2: Launch all 5 experiments in parallel, one per GPU
echo ""
echo "============================================"
echo "  Launching 5 experiments in parallel..."
echo "============================================"

PIDS=()
LOGS_DIR="runs/logs"
mkdir -p "$LOGS_DIR"

for i in 1 2 3 4 5; do
    GPU_ID=$i
    LOG_FILE="$LOGS_DIR/exp${i}.log"

    echo "  Exp $i -> GPU $GPU_ID (log: $LOG_FILE)"

    python run_experiments.py \
        --skip-preprocess \
        --experiments "$i" \
        --device "$GPU_ID" \
        "${PASS_THROUGH_ARGS[@]}" \
        > "$LOG_FILE" 2>&1 &

    PIDS+=($!)
done

echo ""
echo "  PIDs: ${PIDS[*]}"
echo "  Waiting for all experiments to finish..."
echo "  Monitor with: tail -f runs/logs/exp*.log"
echo ""

# Step 3: Wait for all and track failures
FAILED=()
for i in 0 1 2 3 4; do
    EXP_NUM=$((i + 1))
    if wait "${PIDS[$i]}"; then
        echo "  [DONE] Exp $EXP_NUM (GPU $((i + 1))) finished successfully"
    else
        echo "  [FAIL] Exp $EXP_NUM (GPU $((i + 1))) failed (see runs/logs/exp${EXP_NUM}.log)"
        FAILED+=($EXP_NUM)
    fi
done

# Step 4: Print comparison across all experiments
echo ""
echo "============================================"
echo "  All experiments finished."
if [ ${#FAILED[@]} -gt 0 ]; then
    echo "  Failed: ${FAILED[*]}"
fi
echo "============================================"

# Gather results by running comparison-only (all experiments already done)
echo ""
echo "  Generating comparison table..."
python -c "
from run_experiments import EXPERIMENTS, load_metrics, load_training_losses, print_comparison
from collections import OrderedDict
from pathlib import Path

results = OrderedDict()
for exp_name, exp_def in EXPERIMENTS.items():
    exp_output = str(Path('runs') / exp_name)
    metrics = load_metrics(exp_output)
    losses = load_training_losses(exp_output)
    results[exp_name] = {
        'description': exp_def.get('description', ''),
        'config': f'runs/configs/config_{exp_name}.yaml',
        'train_ok': bool(metrics),
        'elapsed_min': 0,
        'metrics': metrics,
        'losses': losses,
    }
print_comparison(results)
"

END_TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
echo ""
echo "  Started:  $TIMESTAMP"
echo "  Finished: $END_TIMESTAMP"
