#!/usr/bin/env bash
#
# Runs every proposal experiment end-to-end in one shot.
#
#   Phase 0/1: zero-retraining diagnostics on the existing S54 weights
#              (weighted F1 report, label audit, conf sweep, tile
#               inference, score calibration, multi-seed WBF, small-CC
#               post-processing)
#
#   Phase 2/3: retraining experiments — 2 GPUs in parallel via the
#              existing scheduler
#                F1_official_split, B1_class_balanced, B4_warm_start_stenosis,
#                C1a/C1b label smoothing, C2a/C2b dropout,
#                C3a/C3b/C3c DFL sweep, C4a/C4b mask-loss sweep,
#                C5 multi-scale, D1 erasing, D2 flipud
#
#   Phase 4:   compound / pipeline-heavy experiments run sequentially
#              after Phase 2/3 finishes (they depend on a fresh S54
#              pass for their data_prep step and would contend for disk
#              with Phase 2/3 if run concurrently)
#                B3_background_stenosis, B5_iter_pl, B6_tail_copy_paste
#
# Logs land in results/stenosis_strategies/proposal_runs.log
# Intermediate results go to the usual per-experiment subdirs.
#
# Usage:
#   cd arcade_experiments/scripts
#   bash run_proposal_all.sh                 # default: gpus 0,1
#   GPUS=0,1 bash run_proposal_all.sh        # override
#   SKIP_STANDALONES=1 bash run_proposal_all.sh   # training only
#   SKIP_TRAINING=1 bash run_proposal_all.sh      # standalones only
set -uo pipefail

# ── Paths ─────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

ARCADE_ROOT="${ARCADE_ROOT:-../../arcade/submission}"
RESULTS_DIR="${RESULTS_DIR:-../results/stenosis_strategies}"
GPUS="${GPUS:-0,1}"
LOG="$RESULTS_DIR/proposal_runs.log"
mkdir -p "$RESULTS_DIR"

# ── Helpers ───────────────────────────────────────────────────────
log() { echo -e "\n[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

section() {
    echo -e "\n========================================================" | tee -a "$LOG"
    echo "  $*" | tee -a "$LOG"
    echo "========================================================"   | tee -a "$LOG"
}

run_step() {
    local name="$1"; shift
    log "STEP: $name"
    log "  cmd: $*"
    if "$@" >> "$LOG" 2>&1; then
        log "  ✓ $name"
    else
        log "  ✗ $name (rc=$?) — continuing"
    fi
}

# Block until every GPU in GPUS has >= MIN_FREE_MB free, or until
# MAX_WAIT seconds elapse. Protects the boundary between phases where
# a previous Python process may still be releasing CUDA memory.
wait_for_gpus_idle() {
    local min_free_mb="${1:-3000}"
    local max_wait="${2:-300}"
    local waited=0
    IFS=',' read -r -a gpu_arr <<< "$GPUS"
    while : ; do
        local all_idle=1
        for g in "${gpu_arr[@]}"; do
            local free
            free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$g" 2>/dev/null | tr -d ' ')
            if [[ -z "$free" ]]; then
                # nvidia-smi unavailable; skip check, assume ok
                all_idle=1
                break
            fi
            if (( free < min_free_mb )); then
                all_idle=0
                break
            fi
        done
        if [[ "$all_idle" == "1" ]]; then
            log "  GPUs [$GPUS] idle (>= ${min_free_mb}MB free)"
            return 0
        fi
        if (( waited >= max_wait )); then
            log "  !! GPU idle timeout after ${waited}s — proceeding anyway"
            return 0
        fi
        sleep 10
        waited=$((waited + 10))
    done
}

# Resolve S54 weight paths (existing run)
S54_DIR="$RESULTS_DIR/S54_s43b_clahe"
S54_SYN="$(ls "$S54_DIR"/syntax_model/*best*.pt 2>/dev/null | head -n1 || true)"
S54_STN="$(ls "$S54_DIR"/stenosis_model/*best*.pt 2>/dev/null | head -n1 || true)"
S54_SYN_YAML="$RESULTS_DIR/data/S54_s43b_clahe/dataset_configs/syntax_only.yaml"
S54_STN_YAML="$RESULTS_DIR/data/S54_s43b_clahe/dataset_configs/stenosis_only.yaml"

log "Using S54 syntax weights:   $S54_SYN"
log "Using S54 stenosis weights: $S54_STN"
log "Using S54 syntax yaml:      $S54_SYN_YAML"
log "Using S54 stenosis yaml:    $S54_STN_YAML"

# ══════════════════════════════════════════════════════════════════
# PHASE 0/1 — zero-retraining diagnostics + inference-time wins
# ══════════════════════════════════════════════════════════════════
if [[ -z "${SKIP_STANDALONES:-}" ]]; then
section "PHASE 0/1 — diagnostics + zero-retraining"

wait_for_gpus_idle 3000 300

# F-4: weighted F1 report over all existing runs
run_step "F-4 compute_weighted_f1" \
    python -u compute_weighted_f1.py \
        --results "$RESULTS_DIR/strategy_results.json" \
        --output "$RESULTS_DIR/weighted_f1_report.json"

# F-2: label audit (class 9 / 9a contamination check)
if [[ -d "$RESULTS_DIR/data/S54_s43b_clahe" ]]; then
    run_step "F-2 audit_labels" \
        python -u audit_labels.py \
            --arcade-root "$ARCADE_ROOT" \
            --data-dir "$RESULTS_DIR/data/S54_s43b_clahe" \
            --output "$RESULTS_DIR/label_audit_S54.json"
else
    log "  SKIP F-2: no data dir at $RESULTS_DIR/data/S54_s43b_clahe"
fi

wait_for_gpus_idle 5000 300

# A-1: per-class confidence sweep (syntax + stenosis)
if [[ -n "$S54_SYN" && -f "$S54_SYN_YAML" ]]; then
    run_step "A-1 conf sweep syntax" \
        python -u sweep_confidence.py \
            --model "$S54_SYN" \
            --data-yaml "$S54_SYN_YAML" \
            --imgsz 768 --device 0 \
            --output "$RESULTS_DIR/A1_conf_sweep_syntax.json"
fi
if [[ -n "$S54_STN" && -f "$S54_STN_YAML" ]]; then
    wait_for_gpus_idle 5000 300
    run_step "A-1 conf sweep stenosis" \
        python -u sweep_confidence.py \
            --model "$S54_STN" \
            --data-yaml "$S54_STN_YAML" \
            --imgsz 768 --device 0 \
            --output "$RESULTS_DIR/A1_conf_sweep_stenosis.json"
fi

# A-2: tile inference for stenosis
if [[ -n "$S54_STN" && -f "$S54_STN_YAML" ]]; then
    wait_for_gpus_idle 5000 300
    run_step "A-2 tile inference" \
        python -u tile_inference_stenosis.py \
            --model "$S54_STN" \
            --data-yaml "$S54_STN_YAML" \
            --split test \
            --upscale 3 --tile 512 --overlap 128 \
            --imgsz 512 --device 0 \
            --output "$RESULTS_DIR/A2_tile_inference_stenosis.json"
fi

# A-4: fixed CPU small-CC eval
if [[ -n "$S54_STN" && -f "$S54_STN_YAML" ]]; then
    wait_for_gpus_idle 5000 300
    run_step "A-4 small-CC post-processing" \
        python -u -c "
import sys; sys.path.insert(0,'.')
import json
from small_cc_postprocess import evaluate_with_filter_cpu
out = evaluate_with_filter_cpu(
    '$S54_STN',
    '$S54_STN_YAML',
    split='test', imgsz=768,
    min_area_px=30, device='0', conf=0.25,
)
print(json.dumps(out, indent=2))
json.dump(out, open('$RESULTS_DIR/A4_small_cc_S54.json','w'), indent=2)
"
fi

# A-5: score calibration (syntax)
if [[ -n "$S54_SYN" && -f "$S54_SYN_YAML" ]]; then
    wait_for_gpus_idle 5000 300
    run_step "A-5 isotonic calibration syntax" \
        python -u calibrate_scores.py \
            --model "$S54_SYN" \
            --data-yaml "$S54_SYN_YAML" \
            --imgsz 768 --device 0 \
            --output "$RESULTS_DIR/A5_calibrators_syntax/"
fi

log "PHASE 0/1 finished"
else
log "SKIP_STANDALONES set — skipping Phase 0/1"
fi

# ══════════════════════════════════════════════════════════════════
# PHASE 2/3 — retraining experiments in parallel on 2 GPUs
# ══════════════════════════════════════════════════════════════════
if [[ -z "${SKIP_TRAINING:-}" ]]; then
section "PHASE 2/3 — parallel retraining experiments (gpus=$GPUS)"

# Ordered so low-variance / highest-ROI runs go first; independent so
# the scheduler can parallelise freely across the 2 GPUs.
PHASE_2_3_EXPERIMENTS=(
    "F1_official_split"
    "B1_class_balanced"
    "B4_warm_start_stenosis"
    "B6_tail_copy_paste"
    "C1a_label_smooth_05"
    "C1b_label_smooth_10"
    "C2a_dropout_05"
    "C2b_dropout_10"
    "C3a_dfl_2"
    "C3b_dfl_25"
    "C3c_dfl_30"
    "C4a_mask_10"
    "C4b_mask_15"
    "C5_multiscale_stenosis"
    "D1_erasing_4"
    "D2_flipud_5"
)

wait_for_gpus_idle 8000 600

# Auto-detect foreign compute processes (rustdesk, X server, etc.) so
# the scheduler doesn't block waiting for them. These PIDs are passed
# to --exclude-pids so they never count as lingering workers.
FOREIGN_PIDS=""
if command -v nvidia-smi >/dev/null 2>&1; then
    FOREIGN_PIDS=$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null \
        | awk -F',' '{gsub(/ /, "", $1); print $1}' | paste -sd, -)
fi
EXCLUDE_PID_ARGS=()
if [[ -n "$FOREIGN_PIDS" ]]; then
    log "  Detected foreign GPU PIDs (will be excluded): $FOREIGN_PIDS"
    EXCLUDE_PID_ARGS=(--exclude-pids "$FOREIGN_PIDS")
fi

run_step "PHASE 2/3 scheduler" \
    python -u run_stenosis_strategies_v2.py \
        --experiments "${PHASE_2_3_EXPERIMENTS[@]}" \
        --arcade-root "$ARCADE_ROOT" \
        --gpus "$GPUS" \
        --min-free-mb 1000 \
        "${EXCLUDE_PID_ARGS[@]}" \
        --skip-splits

# ══════════════════════════════════════════════════════════════════
# PHASE 4 — compound experiments (depend on S54 retrain side-effect,
# run AFTER Phase 2/3 so they don't clobber each other's data dirs)
# ══════════════════════════════════════════════════════════════════
section "PHASE 4 — compound / pipeline-heavy experiments"

PHASE_4_EXPERIMENTS=(
    "B3_background_stenosis"
    "B5_iter_pl"
)

wait_for_gpus_idle 8000 600

# Re-detect foreign PIDs before Phase 4 in case new ones appeared.
FOREIGN_PIDS=""
if command -v nvidia-smi >/dev/null 2>&1; then
    FOREIGN_PIDS=$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null \
        | awk -F',' '{gsub(/ /, "", $1); print $1}' | paste -sd, -)
fi
EXCLUDE_PID_ARGS=()
if [[ -n "$FOREIGN_PIDS" ]]; then
    log "  Detected foreign GPU PIDs (will be excluded): $FOREIGN_PIDS"
    EXCLUDE_PID_ARGS=(--exclude-pids "$FOREIGN_PIDS")
fi

run_step "PHASE 4 scheduler" \
    python -u run_stenosis_strategies_v2.py \
        --experiments "${PHASE_4_EXPERIMENTS[@]}" \
        --arcade-root "$ARCADE_ROOT" \
        --gpus "$GPUS" \
        --min-free-mb 1000 \
        "${EXCLUDE_PID_ARGS[@]}" \
        --skip-splits

log "PHASE 2/3/4 finished"
else
log "SKIP_TRAINING set — skipping Phase 2/3/4"
fi

# ══════════════════════════════════════════════════════════════════
# A-6: Multi-seed WBF ensemble — only makes sense once C1a/C1b/D1/C3*
# finished (we use their syntax models as "seeds" of the same recipe
# neighbourhood). Kicked off last so the .pt files exist.
# ══════════════════════════════════════════════════════════════════
if [[ -z "${SKIP_STANDALONES:-}" ]]; then
section "PHASE 5 — multi-seed WBF over trained variants"

# Collect syntax best.pt files from any runs that finished
ENSEMBLE_SYNTAX=()
for n in S54_s43b_clahe C1a_label_smooth_05 C1b_label_smooth_10 D1_erasing_4; do
    w="$(ls "$RESULTS_DIR/$n"/syntax_model/*best*.pt 2>/dev/null | head -n1 || true)"
    if [[ -n "$w" ]]; then ENSEMBLE_SYNTAX+=("$w"); fi
done
ENSEMBLE_STENOSIS=()
for n in S54_s43b_clahe C3a_dfl_2 C3b_dfl_25 C3c_dfl_30; do
    w="$(ls "$RESULTS_DIR/$n"/stenosis_model/*best*.pt 2>/dev/null | head -n1 || true)"
    if [[ -n "$w" ]]; then ENSEMBLE_STENOSIS+=("$w"); fi
done

if [[ ${#ENSEMBLE_SYNTAX[@]} -ge 2 && -f "$S54_SYN_YAML" ]]; then
    wait_for_gpus_idle 5000 300
    run_step "A-6 WBF syntax ensemble (${#ENSEMBLE_SYNTAX[@]} models)" \
        python -u multiseed_wbf.py \
            --models "${ENSEMBLE_SYNTAX[@]}" \
            --data-yaml "$S54_SYN_YAML" \
            --split test --imgsz 768 --device 0 \
            --output "$RESULTS_DIR/A6_wbf_syntax.json"
else
    log "  SKIP A-6 syntax: only ${#ENSEMBLE_SYNTAX[@]} models available"
fi
if [[ ${#ENSEMBLE_STENOSIS[@]} -ge 2 && -f "$S54_STN_YAML" ]]; then
    wait_for_gpus_idle 5000 300
    run_step "A-6 WBF stenosis ensemble (${#ENSEMBLE_STENOSIS[@]} models)" \
        python -u multiseed_wbf.py \
            --models "${ENSEMBLE_STENOSIS[@]}" \
            --data-yaml "$S54_STN_YAML" \
            --split test --imgsz 768 --device 0 \
            --output "$RESULTS_DIR/A6_wbf_stenosis.json"
else
    log "  SKIP A-6 stenosis: only ${#ENSEMBLE_STENOSIS[@]} models available"
fi
fi

section "ALL PROPOSAL EXPERIMENTS COMPLETE"
log "See $LOG for full output"
log "Per-experiment metrics: $RESULTS_DIR/<exp_name>/all_metrics.json"
log "Standalone reports in: $RESULTS_DIR/{weighted_f1_report,label_audit_S54,A1_*,A2_*,A4_*,A5_*,A6_*}.json"
