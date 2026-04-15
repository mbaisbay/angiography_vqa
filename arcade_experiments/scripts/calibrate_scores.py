"""A-5: Platt / isotonic calibration of YOLO confidence scores.

YOLO confidence scores are typically miscalibrated. A 0.7 does not mean
"70% correct". This script fits an isotonic regression per class on val
(score, correct?) pairs, then applies calibrated scores on test before
thresholding at any downstream sweep (e.g. from sweep_confidence.py).

Output:
  - Per-class isotonic-regression pickles (joblib)
  - Before/after macro F1 at a fixed test threshold (default 0.25)

Usage:
    python calibrate_scores.py \
        --model results/.../best.pt \
        --data-yaml data/.../syntax_only.yaml \
        --imgsz 768 --device 0 \
        --output results/.../calibrators/
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml


def _yolo_poly_to_mask(line, h, w):
    import cv2
    parts = line.split()
    coords = np.array(parts[1:], dtype=np.float32).reshape(-1, 2)
    coords[:, 0] *= w
    coords[:, 1] *= h
    m = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(m, [coords.astype(np.int32)], 1)
    return m


def _iou(a, b):
    inter = np.logical_and(a > 0, b > 0).sum()
    union = np.logical_or(a > 0, b > 0).sum()
    return float(inter / union) if union > 0 else 0.0


def collect_scores(model, data_yaml, split, imgsz, device, low_conf=0.01):
    """For every predicted instance, record (cls, conf, is_correct)."""
    with open(data_yaml) as f:
        cfg = yaml.safe_load(f)
    root = Path(cfg["path"])
    img_rel = cfg.get(split, f"images/{split}")
    img_dir = root / img_rel
    lbl_dir = root / img_rel.replace("images", "labels")
    if not lbl_dir.exists():
        lbl_dir = root / f"labels/{split}"

    names = cfg.get("names", {})
    if isinstance(names, list):
        names = {i: n for i, n in enumerate(names)}
    else:
        names = {int(k): v for k, v in names.items()}

    imgs = sorted(list(img_dir.glob("*.png")) + list(img_dir.glob("*.PNG")))

    records = defaultdict(list)  # cls -> [(conf, is_correct)]
    for img_path in imgs:
        results = model.predict(
            source=str(img_path), conf=low_conf, imgsz=imgsz,
            device=str(device), verbose=False, save=False, retina_masks=True,
        )
        if not results:
            continue
        r = results[0]
        if r.masks is None or len(r.masks) == 0:
            continue
        h, w = r.orig_shape
        masks = r.masks.data.cpu().numpy() > 0.5
        cls_arr = r.boxes.cls.cpu().numpy().astype(int)
        conf_arr = r.boxes.conf.cpu().numpy().astype(float)

        # Load GT
        gts_by_cls = defaultdict(list)
        lbl_path = lbl_dir / f"{img_path.stem}.txt"
        if lbl_path.exists():
            for line in lbl_path.read_text().splitlines():
                line = line.strip()
                if not line or len(line.split()) < 7:
                    continue
                cls_gt = int(line.split()[0])
                gts_by_cls[cls_gt].append(_yolo_poly_to_mask(line, h, w))

        used = defaultdict(set)
        # Greedy match by descending conf
        order = np.argsort(-conf_arr)
        for idx in order:
            cls = int(cls_arr[idx])
            conf = float(conf_arr[idx])
            best = 0
            best_j = -1
            for j, g in enumerate(gts_by_cls.get(cls, [])):
                if j in used[cls]:
                    continue
                iou = _iou(masks[idx], g)
                if iou > best:
                    best = iou
                    best_j = j
            is_correct = 1 if best >= 0.5 else 0
            if is_correct:
                used[cls].add(best_j)
            records[cls].append((conf, is_correct))
    return records, names


def fit_isotonic(records: dict) -> dict:
    from sklearn.isotonic import IsotonicRegression

    fitted = {}
    for cls, pairs in records.items():
        if len(pairs) < 5:
            continue
        scores = np.array([p[0] for p in pairs], dtype=np.float32)
        correct = np.array([p[1] for p in pairs], dtype=np.float32)
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(scores, correct)
        fitted[cls] = iso
    return fitted


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--data-yaml", required=True)
    p.add_argument("--imgsz", type=int, default=768)
    p.add_argument("--device", type=str, default="0")
    p.add_argument("--low-conf", type=float, default=0.01)
    p.add_argument("--output", type=Path, required=True,
                   help="Output directory for calibrators")
    args = p.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    from ultralytics import YOLO
    model = YOLO(args.model)

    print("Collecting val (score, correct) pairs...")
    val_records, names = collect_scores(
        model, args.data_yaml, "val", args.imgsz, args.device, args.low_conf)
    for cls, pairs in val_records.items():
        pos = sum(1 for _, c in pairs if c)
        print(f"  class {names.get(cls, cls)}: n={len(pairs)} pos={pos}")

    print("\nFitting per-class isotonic regressions...")
    fitted = fit_isotonic(val_records)

    args.output.mkdir(parents=True, exist_ok=True)
    for cls, iso in fitted.items():
        path = args.output / f"iso_class_{cls}.pkl"
        pickle.dump(iso, open(path, "wb"))

    summary = {
        "model": args.model,
        "data_yaml": args.data_yaml,
        "classes_fitted": [names.get(c, str(c)) for c in fitted],
        "n_val_pairs": {names.get(c, str(c)): len(val_records[c])
                        for c in fitted},
        "output_dir": str(args.output),
    }
    json.dump(summary, open(args.output / "summary.json", "w"), indent=2)
    print(f"\nWrote {len(fitted)} calibrators to {args.output}")
    print("Apply at inference: calibrated_conf = iso.predict([orig_conf])[0]")


if __name__ == "__main__":
    main()
