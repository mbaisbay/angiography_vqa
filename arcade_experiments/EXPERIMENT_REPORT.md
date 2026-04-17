# ARCADE Experiment Report

**Scope:** ~100 experiments across S0-S56, E1-E4, H1-H5, and proposal runs.
**Architecture constraint:** YOLO instance segmentation only.
**Tasks:** Syntax (coronary vessel segment classification, 12 classes) + Stenosis (lesion detection, 1 class).

---

## 1. Data Splits

The original ARCADE dataset provides 1500 images per task split into train/val/test (1000/200/300). However, the official val and test sets have a distribution mismatch: val stenoses are small (mean 3730 px^2, 2.03/image) while test stenoses are large (mean 7647 px^2, 1.29/image). This causes models early-stopped on val to underperform on test.

To address this, we pool all 1500 images and re-stratify into 999/200/301 based on annotation density and object size, producing a balanced evaluation. All primary results below use this stratified split. Official-split results (1000/200/300, no re-stratification) are reported separately for publishable comparison.

Class filtering: only syntax classes with >=300 pooled instances are kept, reducing 25 to 12: {1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 13, 16}.

---

## 2. Best Results

### Overall best: S54_s43b_clahe

| Metric | Value |
|--------|-------|
| Syntax mean F1 | **0.7398** |
| Stenosis F1 | **0.4569** |
| Overall F1 | **0.7181** |
| Weighted F1 | **0.7456** |
| Split | Stratified (999/200/301) |

**Pipeline:** Two separate YOLO11m-seg models (syntax-only + stenosis-only), each with two-stage training (15 epochs frozen backbone, then unfrozen at lr/10).

**Syntax model config:**
- SGD, lr=0.01, wd=0.0005, cosine schedule, 300 epochs, patience 50
- imgsz=768, batch=8, freeze=10 layers
- Augmentation: fliplr=0.5, degrees=20, scale=0.4, translate=0.1, hsv_v=0.3
- No mosaic, no copy-paste, no CLAHE

**Stenosis model config:**
- SGD, lr=0.005, wd=0.0005, cosine schedule, 300 epochs, patience 50
- imgsz=768, batch=8, freeze=10 layers
- Augmentation: mosaic=0.8, copy_paste=0.3, scale=0.5, fliplr=0.5, degrees=20
- CLAHE preprocessing (clip=2.0, tile=8x8)
- box=10.0, cls=1.0

### Best stenosis: S31_sgd_lr005

| Metric | Value |
|--------|-------|
| Stenosis F1 | **0.4613** |
| Syntax mean F1 | 0.7188 |
| Split | Stratified |

Same pipeline as S54 but: 200 epochs, patience 25, copy_paste=0.3 on stenosis, no CLAHE. The 0.4613 is a single-seed result; multi-seed mean for the same recipe is ~0.44 (seed variance is 5 pp).

### Official ARCADE split baseline

| Metric | Value |
|--------|-------|
| Syntax mean F1 | 0.6646 |
| Stenosis F1 | 0.3483 |
| Split | Official (1000/200/300) |

Trained with S54 recipe. Lower scores reflect the harder official test distribution, not a recipe regression.

---

## 3. What Works

- **Separate models** over combined/cross-task pipelines. Syntax-only + stenosis-only consistently outperforms joint training by 10+ pp.
- **SGD optimizer** for stenosis (+2.5 pp multi-seed mean over AdamW). AdamW is marginally better for syntax (+1.4 pp over SGD).
- **Mosaic augmentation (0.8)** on stenosis: +3.5 pp. Single biggest augmentation lever.
- **Copy-paste augmentation (0.3)** on stenosis: +3.3 pp. Additive with mosaic (combined: +6 pp).
- **CLAHE preprocessing** on stenosis images: +2.9 pp. Applied before training, not at inference.
- **Extended training** (300 epochs, patience 50): +0.2 pp syntax over 200/25. Marginal but free.
- **Two-stage freeze/unfreeze:** 15 frozen epochs is the sweet spot. 25 hurts.

---

## 4. What Doesn't Work

### Augmentation & Preprocessing
- **Mosaic on syntax** (0.5): no effect. Syntax F1 0.7317 vs 0.7302 baseline. Vessels are full-image structures, not helped by mosaic tiling.
- **Mixup (0.15)** on stenosis: no effect or slight regression. S13 0.4144 vs S8 0.4152.
- **Top-hat morphological filter + CLAHE**: -2 pp vs CLAHE alone. S52 0.4268 vs S51 0.4466. Over-sharpens vessel edges.
- **Unsharp mask + median blur**: hurts syntax. E4s 0.6950 vs S54 0.7398. Destroys fine texture needed for vessel classification.
- **Random erasing, dropout, label smoothing** (proposal runs): zero effect. 6 experiments produced identical mAP50=0.6506. Training converges to the same checkpoint.
- **Vertical flip (flipud=0.5)**: slight regression. 0.6274 F1 vs 0.6646 baseline. Coronary anatomy is not vertically symmetric.

### Resolution & Model Size
- **imgsz=1024**: hurts syntax (-6 pp). S21 0.6464 vs S8 0.7302. Likely overfits on sparse high-resolution features.
- **imgsz=768 for stenosis**: helps on official split (+5.5 pp) but neutral on stratified. Resolution matters when test objects are larger.
- **Larger models (yolo11l, yolo11x)**: no benefit. S19 yolo11l 0.7235 vs S8 yolo11m 0.7302. Model capacity is not the bottleneck.
- **YOLOv8 vs YOLO11**: YOLO11m matches or beats YOLOv8m on both tasks. No reason to use v8.

### Training Strategies
- **Knowledge distillation** (yolo11x teacher to yolo11m student): student (0.55) < teacher (0.65) < GT-trained (0.67). Soft labels lose information.
- **Pseudo-labeling** (3 rounds at decaying confidence): each round degrades. Round 1 mAP50=0.23, Round 2 0.16, Round 3 0.17. Noisy pseudo-labels poison training.
- **Background images for stenosis** (syntax images as hard negatives): catastrophic, mAP50=0.14. 150 negatives overwhelm the 1000-image training set.
- **WBF ensemble at conf=0.05**: mean F1 drops from 0.62 individual to 0.32 ensemble. Low threshold floods with false positives before merging.
- **Combined/cross-task pipeline** (syntax + stenosis in one model): 0.56 vs 0.73 separate. Joint loss pulls both tasks toward mediocrity.
- **Warm-start stenosis from syntax backbone**: mAP50=0.30 vs 0.25 baseline on official split. Marginal, syntax features don't transfer well to lesion detection.

### Published Method Reproductions
- **SSASS (1st place ARCADE 2023)**: E1/E1s 0.40-0.43 stenosis. Underperforms our S54 (0.46). Their pseudo-labeling doesn't transfer at this scale.
- **YOLO-Angio**: E2s syntax 0.7108 vs S54 0.7398. Their top-hat+CLAHE+3-seed ensemble is worse than our simpler recipe.
- **StenUNet (multi-channel input)**: E3/E3s 0.44 stenosis. Within noise of baseline.
- **Cross-Task PL**: E4s syntax 0.6950. Pseudo-label noise from cross-task transfer hurts more than it helps.

### Hyperparameter Sweeps (Diminishing Returns)
- **SGD lr sweep** (0.005/0.01/0.02): 0.01 best for syntax, 0.005 best for stenosis. 0.02 diverged.
- **Weight decay** (0.0005/0.001/0.005): 0.001 +0.6 pp on SGD syntax. Marginal.
- **Class loss weight** (0.5/0.75/1.0/1.25/2.0): no monotonic trend. All within noise.
- **DFL weight** (1.5/2.0/2.5/3.0): best at 2.5 for stenosis (mAP50=0.31 vs 0.25 baseline on official split). Modest gain.
- **nbs, lrf, warmup epochs**: all within noise of baseline.

---

## 5. Remaining Bottlenecks

**Syntax:** Capped by tail-class imbalance. Classes 9 (F1=0.50), 13 (F1=0.52), 16 (F1=0.67) have 320-360 training instances vs 500-1100 for top classes. No hyperparameter has moved class 9 past 0.55. Closing class 9 to 0.75 alone would add +1.7 pp to mean F1.

**Stenosis:** Capped by data scarcity and object size. 1615 annotations across 1000 images, median area ~5000 px^2 at 512px input (roughly 4x4 output cells). Single-seed variance is 5 pp, making ablation unreliable without multi-seed runs.

---

## 6. Currently Running (4 Untested Levers)

These experiments target the identified bottlenecks and are evaluated on the stratified split:

1. **L1 — Per-class weighted loss:** Oversample tail classes (9/13/16) to equalize instance counts + cls loss weight 1.5. 3 seeds. Expected: +2-3 pp syntax.
2. **L2 — Per-class confidence threshold sweep:** Optimize the decision threshold per class on val, apply on test. Zero retraining. Expected: +1-3 pp.
3. **L3 — Proper WBF ensemble:** 3-seed S54 recipe merged at conf=0.25 (not 0.05). Expected: +1-2 pp.
4. **L4 — CLAHE + 768px for stenosis:** Stack two proven levers that were never combined. 3 seeds. Expected: +3-5 pp stenosis.
