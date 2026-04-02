# ARCADE CAD Diagnostic Pipeline

## Project Overview
YOLOv8x-seg instance segmentation for coronary artery vessel segments (syntax task) and stenosis detection on the ARCADE dataset. The pipeline includes class filtering (26 → 13 classes), optional image preprocessing (CLAHE + top-hat), training, cross-inference, and mask intersection for stenosis-vessel assignment.

## Round 2 Experiment Results (2026-04-02)

Trained on filtered 13-class dataset (12 syntax + stenosis), 512x512 images, SGD optimizer, seed=42, preprocessing DISABLED (raw images).

### Comparison Table

| Experiment | ARCADE F1 | mAP50 | mAP50-95 | Precision | Recall | Epochs | Key Overrides |
|---|---|---|---|---|---|---|---|
| **exp3_original_match** | **0.6383** | 0.7796 | 0.3638 | 0.7825 | 0.7482 | 100 | lrf=0.01, amp=true |
| exp4_original_patience50 | 0.6232 | 0.7924 | 0.3527 | 0.8100 | 0.7435 | 150 | lrf=0.01, amp=true, patience=50, epochs=150 |
| exp2_amp_on | 0.6150 | 0.7831 | 0.3612 | 0.7853 | 0.7160 | 82 | amp=true |
| exp5_original_batch16 | 0.6127 | 0.7894 | 0.3476 | 0.8090 | 0.7189 | 73 | lrf=0.01, amp=true, batch=16 |
| exp1_lrf_high | 0.5991 | 0.7581 | 0.3407 | 0.7455 | 0.7048 | 68 | lrf=0.01 |

### Per-Class ARCADE F1 (Best: exp3_original_match)

| Class | F1 | Std | Images | Notes |
|---|---|---|---|---|
| 11 | 0.7616 | 0.2291 | 118 | Best class |
| 5 | 0.7319 | 0.2905 | 195 | Strong, most samples |
| 3 | 0.7181 | 0.2375 | 100 | Strong |
| 1 | 0.7184 | 0.1887 | 100 | Strong, low variance |
| 2 | 0.7064 | 0.2060 | 100 | Strong |
| 7 | 0.6474 | 0.2722 | 83 | Good |
| 6 | 0.6179 | 0.3403 | 188 | Good but high variance |
| 4 | 0.5708 | 0.3628 | 92 | Moderate |
| 16 | 0.5652 | 0.3380 | 88 | Moderate |
| 8 | 0.5540 | 0.3151 | 80 | Moderate |
| 13 | 0.5440 | 0.3104 | 110 | Weak |
| 9 | 0.2787 | 0.3642 | 66 | Weakest, fewest samples |

### Per-Class mAP50 (exp3_original_match)

| YOLO ID | Class Name | AP50 |
|---|---|---|
| 0 | 1 | 0.9483 |
| 1 | 2 | 0.6835 |
| 2 | 3 | 0.9411 |
| 3 | 4 | 0.6914 |
| 4 | 5 | 0.9241 |
| 5 | 6 | 0.8200 |
| 6 | 7 | 0.8043 |
| 7 | 8 | 0.8954 |
| 8 | 9 | 0.4439 |
| 9 | 11 | 0.9013 |
| 10 | 13 | 0.5497 |
| 11 | 16 | 0.7523 |

### Training Loss Progression

| Experiment | First box_loss | Last box_loss | First seg_loss | Last seg_loss |
|---|---|---|---|---|
| exp3_original_match | 2.028 | 0.699 | 2.043 | 0.583 |
| exp4_original_patience50 | 2.028 | 0.556 | 2.043 | 0.550 |
| exp2_amp_on | 2.028 | 0.866 | 2.043 | 0.777 |
| exp5_original_batch16 | 2.090 | 0.923 | 2.302 | 0.787 |
| exp1_lrf_high | 2.005 | 0.982 | 1.964 | 0.813 |

### Key Findings

1. **Best config: lrf=0.01 + amp=true** (exp3) — F1=0.6383, the only change from base config that matters
2. **Preprocessing (CLAHE + top-hat) is harmful** — enabling it dropped F1 from 0.6+ to 0.13. Must stay disabled.
3. **Longer training doesn't help** — exp4 (150 epochs) scored lower F1 than exp3 (100 epochs) despite lower final loss, suggesting overfitting
4. **Larger batch hurts** — batch=16 (exp5) slightly worse than batch=8 (exp3)
5. **AMP alone helps but lrf matters more** — exp2 (amp only) < exp3 (amp + lrf=0.01)
6. **Weakest classes: 9 (F1=0.28) and 13 (F1=0.54)** — likely due to fewer training samples and morphological similarity to neighboring vessels

### Best Weights
`runs/exp3_original_match/syntax/weights/best.pt`

### Base Config (config.yaml)
- Model: yolov8x-seg
- Batch: 8, ImgSz: 512
- lr0: 0.01, lrf: 0.001 (override to 0.01 for best results)
- SGD, momentum=0.937, weight_decay=0.0005
- amp: false (override to true for best results)
- cos_lr: true, patience: 20, epochs: 100
- preprocessing.enabled: false (DO NOT enable — causes F1 collapse)
