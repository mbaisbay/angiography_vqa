#!/bin/bash
# =============================================================================
# Full YOLO Pipeline: Filter classes -> Train -> Cross-inference -> Final dataset
# =============================================================================
# Run with: bash run_pipeline.sh 2>&1 | tee pipeline.log
# =============================================================================

set -e  # Exit on any error

CONFIG="config.yaml"
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

echo "============================================"
echo "YOLO Pipeline started at $TIMESTAMP"
echo "============================================"

# STEP 0a: Filter classes (keep >=300 annotations only)
echo ""
echo "============================================"
echo "[STEP 0a] Filtering classes (>=300 only)..."
echo "============================================"
python filter_classes.py --config $CONFIG --min-count 300

# STEP 0b: Preprocess images (CLAHE + white top-hat, following ARCADE Enhanced recipe)
echo ""
echo "============================================"
echo "[STEP 0b] Preprocessing images (CLAHE + top-hat)..."
echo "============================================"
python preprocess_images.py --config $CONFIG

# STEP 1: Train syntax model (12 filtered vessel classes + stenosis)
echo ""
echo "============================================"
echo "[STEP 1/6] Training SYNTAX model..."
echo "============================================"
python train.py --config $CONFIG --task syntax

# STEP 2: Cross-inference: syntax model on stenosis images
echo ""
echo "============================================"
echo "[STEP 2/6] Cross-inference: syntax on stenosis..."
echo "============================================"
python cross_inference.py --config $CONFIG --direction syntax_on_stenosis

# STEP 3: Build combined dataset + train combined model
echo ""
echo "============================================"
echo "[STEP 3a/6] Building combined dataset..."
echo "============================================"
python build_combined_dataset.py --config $CONFIG --min-confidence 0.5

echo ""
echo "============================================"
echo "[STEP 3b/6] Training COMBINED model..."
echo "============================================"
python train.py --config $CONFIG --task combined

# STEP 4: Run combined model on syntax images (cross-inference)
echo ""
echo "============================================"
echo "[STEP 4/6] Cross-inference: combined on syntax..."
echo "============================================"
python cross_inference.py --config $CONFIG --direction combined_on_syntax

# STEP 5: Build final quality-filtered dataset
echo ""
echo "============================================"
echo "[STEP 5/6] Building final dataset..."
echo "============================================"
python build_final_dataset.py --config $CONFIG

# STEP 6: Validate quality
echo ""
echo "============================================"
echo "[STEP 6/6] Running validation..."
echo "============================================"
python extract_and_validate.py --config $CONFIG

# Done
END_TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
echo ""
echo "============================================"
echo "PIPELINE COMPLETE"
echo "Started:  $TIMESTAMP"
echo "Finished: $END_TIMESTAMP"
echo "============================================"
echo ""
echo "Results:"
echo "  Filtered data:  arcade/submission/syntax_filtered/"
echo "  Filtered data:  arcade/submission/stenosis_filtered/"
echo "  Combined data:  arcade/submission/combined/"
echo "  Final dataset:  arcade/submission/final/"
echo "  Model weights:  runs/syntax/weights/best.pt"
echo "                  runs/combined/weights/best.pt"
echo "  Validation:     runs/validation/"
echo "  Log:            pipeline.log (if piped with tee)"
echo "============================================"
