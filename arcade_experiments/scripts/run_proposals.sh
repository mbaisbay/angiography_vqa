#!/bin/bash
# Launch script for running all proposal experiments.
# Intended for a 2x RTX 5090 setup.
#
# Usage:
#   ./run_proposals.sh                  # run all phases
#   ./run_proposals.sh 0                # diagnostics only
#   ./run_proposals.sh 0,1              # phases 0 and 1
#   ./run_proposals.sh 2,3 --devices 0  # phases 2-3 on single GPU
#
#   # Override ARCADE_ROOT if needed:
#   ARCADE_ROOT=/path/to/arcade/submission ./run_proposals.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Auto-detect arcade/submission path (check common locations)
if [ -z "${ARCADE_ROOT:-}" ]; then
    if [ -d "${SCRIPT_DIR}/../../arcade/submission/syntax" ]; then
        ARCADE_ROOT="${SCRIPT_DIR}/../../arcade/submission"
    elif [ -d "${HOME}/Documents/repos/angiography_vqa/arcade/submission/syntax" ]; then
        ARCADE_ROOT="${HOME}/Documents/repos/angiography_vqa/arcade/submission"
    else
        echo "ERROR: Could not find arcade/submission directory."
        echo "Set ARCADE_ROOT=/path/to/arcade/submission and re-run."
        exit 1
    fi
fi

PHASE="${1:-all}"
shift 2>/dev/null || true

# Install required packages
pip install ensemble-boxes scikit-learn --quiet 2>/dev/null || true

echo "========================================"
echo "ARCADE Proposal Experiments Runner"
echo "========================================"
echo "Arcade root:  ${ARCADE_ROOT}"
echo "Phase(s):     ${PHASE}"
echo "Extra args:   $@"
echo ""

cd "${SCRIPT_DIR}"

mkdir -p "${SCRIPT_DIR}/../results/proposal_runs"

python run_all_proposals.py \
    --arcade-root "${ARCADE_ROOT}" \
    --phase "${PHASE}" \
    --devices "0,1" \
    "$@" \
    2>&1 | tee -a "${SCRIPT_DIR}/../results/proposal_runs/run_$(date +%Y%m%d_%H%M%S).log"
