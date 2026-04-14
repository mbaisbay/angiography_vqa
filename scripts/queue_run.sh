#!/usr/bin/env bash
# =============================================================================
# queue_run.sh — launch N commands concurrently, auto-spreading across GPUs
# =============================================================================
#
# Usage:
#   scripts/queue_run.sh \
#       "python train.py --task syntax" \
#       "python train.py --task stenosis" \
#       "python train.py --task combined"
#
# Each argument is one shell command string. Every command is launched in
# the background with ANGIO_FORCE_AUTO_DEVICE=1 in its environment, which
# causes utils/config_loader.load_config to call
# utils.gpu_scheduler.acquire_free_gpu() and grab a free GPU via fcntl lock.
# The fourth and later commands (on a 2-GPU box) block inside
# acquire_free_gpu until a lock is released, giving you a queue.
#
# Per-command stdout+stderr goes to runs/logs/queue_<timestamp>/cmd_<N>.log.
# The script waits on every PID and exits non-zero if any command failed.
# =============================================================================

set -uo pipefail

if [ "$#" -eq 0 ]; then
    echo "usage: $(basename "$0") 'cmd1' 'cmd2' [... 'cmdN']" >&2
    exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"

TS=$(date +%Y%m%d_%H%M%S)
LOG_DIR="runs/logs/queue_${TS}"
mkdir -p "$LOG_DIR"

export ANGIO_FORCE_AUTO_DEVICE=1

TOTAL=$#
echo "============================================================"
echo " queue_run.sh — $TOTAL command(s)"
echo " Started: $(date)"
echo " Log dir: ${LOG_DIR}"
echo " GPU pool: auto (via utils/gpu_scheduler file locks)"
echo "============================================================"

PIDS=()
CMDS=()
STARTS=()
i=0
for CMD in "$@"; do
    i=$((i + 1))
    LOG_FILE="${LOG_DIR}/cmd_$(printf '%02d' "$i").log"
    echo ""
    echo " [launch $i/$TOTAL] $CMD"
    echo " Log: $LOG_FILE"
    START=$(date +%s)
    # Run the user-provided command string through bash -c so shell
    # features like pipes and && work naturally.
    bash -c "$CMD" > "$LOG_FILE" 2>&1 &
    PID=$!
    PIDS+=("$PID")
    CMDS+=("$CMD")
    STARTS+=("$START")
    echo " PID: $PID"
    sleep 1
done

echo ""
echo "============================================================"
echo " All $TOTAL command(s) launched; waiting for completion…"
echo "============================================================"

PASSED=0
FAILED=0
FAILED_LIST=""

for idx in "${!PIDS[@]}"; do
    PID=${PIDS[$idx]}
    CMD=${CMDS[$idx]}
    START=${STARTS[$idx]}
    LOG_FILE="${LOG_DIR}/cmd_$(printf '%02d' "$((idx + 1))").log"

    if wait "$PID"; then
        RC=0
    else
        RC=$?
    fi
    END=$(date +%s)
    ELAPSED=$(( (END - START) / 60 ))

    if [ "$RC" -eq 0 ]; then
        echo " [PASS] cmd_$((idx + 1)) (${ELAPSED} min): $CMD"
        PASSED=$((PASSED + 1))
    else
        echo " [FAIL] cmd_$((idx + 1)) (${ELAPSED} min, rc=${RC}): $CMD"
        echo "        see $LOG_FILE"
        FAILED=$((FAILED + 1))
        FAILED_LIST="${FAILED_LIST}  - cmd_$((idx + 1)): $CMD\n"
    fi
done

echo ""
echo "============================================================"
echo " SUMMARY"
echo "============================================================"
echo " Finished: $(date)"
echo " Passed: ${PASSED}/${TOTAL}"
echo " Failed: ${FAILED}/${TOTAL}"

if [ -n "$FAILED_LIST" ]; then
    echo ""
    echo " Failed commands:"
    echo -e "$FAILED_LIST"
fi

echo "============================================================"

if [ "$FAILED" -gt 0 ]; then
    exit 1
fi
