"""Post-hoc confidence-threshold sweep for a trained stenosis model.

Rationale
---------
Ultralytics' default inference confidence is 0.25. For the stenosis
head, this is a recall-bound operating point (S27 test: P=0.469,
R=0.404, F1=0.434). Sweeping `conf` on the *val* split usually shifts
F1 by 1-3pp for free without any retraining, then we re-evaluate the
chosen threshold on the held-out test split.

What this script does
---------------------
  1. Run `model.predict()` on the val split across a grid of conf
     values (default: 0.10, 0.12, ..., 0.40).
  2. For each threshold, compute precision / recall / F1 against the
     YOLO-format ground-truth labels using IoU >= 0.5 matching
     (standard for AP50).
  3. Pick the conf that maximises val F1.
  4. Re-run predictions on the test split at that conf and report the
     test P/R/F1 vs the default-conf baseline.
  5. Save everything to JSON next to the original metrics file.

Usage
-----
    python tune_stenosis_conf.py \\
        --model results/.../S27_sgd_optimizer/stenosis_model/stenosis_768/weights/best.pt \\
        --data-yaml data/.../dataset_configs/stenosis_only.yaml \\
        --output results/.../S27_sgd_optimizer/conf_sweep.json \\
        --imgsz 768

This is post-hoc: it does NOT retrain the model. Safe to re-run as
often as you want.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml
from ultralytics import YOLO


# ── IoU / matching helpers ──────────────────────────────────────────────

def _xywhn_to_xyxy(box: np.ndarray, W: int, H: int) -> np.ndarray:
    """Convert YOLO normalized [cx, cy, w, h] to absolute [x1, y1, x2, y2]."""
    cx, cy, w, h = box
    x1 = (cx - w / 2) * W
    y1 = (cy - h / 2) * H
    x2 = (cx + w / 2) * W
    y2 = (cy + h / 2) * H
    return np.array([x1, y1, x2, y2], dtype=np.float64)


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    """IoU between two [x1, y1, x2, y2] boxes."""
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    iw = max(0.0, x2 - x1)
    ih = max(0.0, y2 - y1)
    inter = iw * ih
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return float(inter / union)


def _load_yolo_gt(label_path: Path, W: int, H: int) -> list:
    """Read a YOLO-format label file and return a list of xyxy GT boxes.

    Supports both bbox format (cls cx cy w h) and segmentation format
    (cls x1 y1 x2 y2 ... xN yN) — for seg we compute the tight
    axis-aligned bbox around the polygon.
    """
    if not label_path.exists():
        return []
    boxes = []
    for line in label_path.read_text().strip().splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        # class id is parts[0]; we only have one class (stenosis) so ignore
        vals = [float(v) for v in parts[1:]]
        if len(vals) == 4:
            # bbox format
            box = _xywhn_to_xyxy(np.array(vals), W, H)
            boxes.append(box)
        else:
            # polygon format — bbox of the polygon
            xs = [vals[i] * W for i in range(0, len(vals), 2)]
            ys = [vals[i] * H for i in range(1, len(vals), 2)]
            if not xs or not ys:
                continue
            boxes.append(np.array([min(xs), min(ys), max(xs), max(ys)],
                                  dtype=np.float64))
    return boxes


# ── Split resolution ────────────────────────────────────────────────────

def _resolve_split_dir(data_yaml: Path, split: str) -> Path:
    """Resolve the image directory for a given split from a YOLO YAML."""
    with open(data_yaml) as f:
        cfg = yaml.safe_load(f)
    root = Path(cfg.get("path", data_yaml.parent)).expanduser()
    rel = cfg.get(split)
    if rel is None:
        raise ValueError(f"split '{split}' not found in {data_yaml}")
    p = (root / rel).resolve()
    if not p.exists():
        # fallback: relative to data_yaml dir
        p = (data_yaml.parent / rel).resolve()
    return p


def _label_path_for(image_path: Path) -> Path:
    """YOLO convention: replace /images/ with /labels/ and .png with .txt."""
    parts = list(image_path.parts)
    try:
        idx = len(parts) - 1 - parts[::-1].index("images")
        parts[idx] = "labels"
    except ValueError:
        pass
    return Path(*parts).with_suffix(".txt")


# ── Core sweep ──────────────────────────────────────────────────────────

def _evaluate_at_conf(model: YOLO, image_dir: Path, conf: float,
                      imgsz: int, iou_match: float = 0.5) -> dict:
    """Predict over all images in `image_dir` at `conf` and compute
    aggregate P/R/F1 against YOLO-format ground truth.
    """
    tp = 0
    fp = 0
    fn = 0
    n_images = 0
    image_exts = {".png", ".jpg", ".jpeg"}

    for img_path in sorted(image_dir.iterdir()):
        if img_path.suffix.lower() not in image_exts:
            continue
        n_images += 1

        # Predict
        results = model.predict(
            source=str(img_path),
            conf=conf,
            imgsz=imgsz,
            verbose=False,
            augment=True,
        )
        if not results:
            continue
        r = results[0]
        H = int(r.orig_shape[0])
        W = int(r.orig_shape[1])

        # Predicted boxes in xyxy absolute coords
        pred_boxes: list[np.ndarray] = []
        pred_scores: list[float] = []
        if r.boxes is not None and len(r.boxes) > 0:
            xyxy = r.boxes.xyxy.detach().cpu().numpy()
            scores = r.boxes.conf.detach().cpu().numpy()
            order = scores.argsort()[::-1]
            for i in order:
                pred_boxes.append(xyxy[i].astype(np.float64))
                pred_scores.append(float(scores[i]))

        # Ground truth
        gt = _load_yolo_gt(_label_path_for(img_path), W, H)

        # Greedy match: sort preds by score desc, each GT used once
        matched_gt = set()
        for pb in pred_boxes:
            best_iou = 0.0
            best_j = -1
            for j, gb in enumerate(gt):
                if j in matched_gt:
                    continue
                i = _iou(pb, gb)
                if i > best_iou:
                    best_iou = i
                    best_j = j
            if best_iou >= iou_match and best_j >= 0:
                tp += 1
                matched_gt.add(best_j)
            else:
                fp += 1
        fn += len(gt) - len(matched_gt)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)

    return {
        "conf": conf,
        "n_images": n_images,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def sweep_conf(model_path: str, data_yaml: str, imgsz: int = 768,
               conf_grid: list | None = None,
               iou_match: float = 0.5) -> dict:
    """Sweep conf on val, pick best F1, re-eval on test."""
    if conf_grid is None:
        conf_grid = [round(0.10 + 0.02 * i, 2) for i in range(16)]  # 0.10..0.40

    data_yaml_p = Path(data_yaml)
    val_dir = _resolve_split_dir(data_yaml_p, "val")
    test_dir = _resolve_split_dir(data_yaml_p, "test")

    print(f"Model:   {model_path}")
    print(f"Val dir: {val_dir}")
    print(f"Test dir:{test_dir}")
    print(f"Conf grid: {conf_grid}")

    model = YOLO(model_path)

    val_rows = []
    for c in conf_grid:
        row = _evaluate_at_conf(model, val_dir, c, imgsz, iou_match)
        val_rows.append(row)
        print(f"  [val] conf={c:.2f}  P={row['precision']:.4f}  "
              f"R={row['recall']:.4f}  F1={row['f1']:.4f}")

    best_val = max(val_rows, key=lambda r: r["f1"])
    print(f"\nBest val conf = {best_val['conf']} (val F1 = {best_val['f1']})")

    # Test at default conf (0.25) and at tuned conf
    test_default = _evaluate_at_conf(model, test_dir, 0.25, imgsz, iou_match)
    test_tuned = _evaluate_at_conf(model, test_dir, best_val["conf"],
                                    imgsz, iou_match)
    print(f"\nTest @ conf=0.25 (default): P={test_default['precision']:.4f}  "
          f"R={test_default['recall']:.4f}  F1={test_default['f1']:.4f}")
    print(f"Test @ conf={best_val['conf']:.2f} (tuned):   "
          f"P={test_tuned['precision']:.4f}  "
          f"R={test_tuned['recall']:.4f}  F1={test_tuned['f1']:.4f}")
    delta = test_tuned["f1"] - test_default["f1"]
    print(f"Δ F1 vs default: {delta:+.4f}")

    return {
        "model": model_path,
        "data_yaml": str(data_yaml),
        "imgsz": imgsz,
        "iou_match": iou_match,
        "conf_grid": conf_grid,
        "val_sweep": val_rows,
        "best_val_conf": best_val["conf"],
        "best_val_f1": best_val["f1"],
        "test_default_conf": test_default,
        "test_tuned_conf": test_tuned,
        "delta_f1_test": round(delta, 4),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Sweep inference confidence threshold on the val "
                    "split, pick best F1, re-evaluate on test."
    )
    parser.add_argument("--model", type=str, required=True,
                        help="Path to trained stenosis .pt weights")
    parser.add_argument("--data-yaml", type=str, required=True,
                        help="Path to stenosis_only.yaml")
    parser.add_argument("--imgsz", type=int, default=768)
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON (default: next to model, "
                             "named conf_sweep.json)")
    parser.add_argument("--conf-min", type=float, default=0.10)
    parser.add_argument("--conf-max", type=float, default=0.40)
    parser.add_argument("--conf-step", type=float, default=0.02)
    parser.add_argument("--iou-match", type=float, default=0.5)
    args = parser.parse_args()

    grid = []
    c = args.conf_min
    while c <= args.conf_max + 1e-9:
        grid.append(round(c, 4))
        c += args.conf_step

    result = sweep_conf(args.model, args.data_yaml,
                        imgsz=args.imgsz, conf_grid=grid,
                        iou_match=args.iou_match)

    out = Path(args.output) if args.output else Path(args.model).parent / "conf_sweep.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, default=str))
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
