"""A-1: Per-class inference-time confidence threshold sweep.

Ultralytics' default `conf=0.25` is a recall-bound operating point for
small/rare classes. Sweeping conf *per class* on val and applying the
per-class optimal thresholds on test typically adds 1-3 pp F1 for free.

This script:
  1. Runs `model.predict()` on val at a very low global conf (0.01) to
     collect all candidate detections.
  2. For each class, sweeps the post-hoc threshold from 0.05 to 0.95 in
     steps of 0.05 and computes P/R/F1 vs GT at IoU>=0.5.
  3. Picks per-class argmax-F1 on val.
  4. Re-runs predictions on test with the same low global conf, applies
     the per-class thresholds, and reports test F1 per class + mean.

Works for both 1-class stenosis models and multi-class syntax models.

Usage:
    python sweep_confidence.py \
        --model results/.../syntax_768_best.pt \
        --data-yaml data/.../syntax_only.yaml \
        --imgsz 768 --device 0 \
        --output results/.../S54_syntax_conf_sweep.json
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml


def _yolo_poly_to_mask(line: str, h: int, w: int) -> np.ndarray:
    import cv2
    parts = line.split()
    coords = np.array(parts[1:], dtype=np.float32).reshape(-1, 2)
    coords[:, 0] *= w
    coords[:, 1] *= h
    m = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(m, [coords.astype(np.int32)], 1)
    return m


def _read_gt(labels_dir: Path, img_stem: str, h: int, w: int):
    p = labels_dir / f"{img_stem}.txt"
    if not p.exists():
        return []
    out = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or len(line.split()) < 7:
            continue
        cls = int(line.split()[0])
        out.append({"cls": cls, "mask": _yolo_poly_to_mask(line, h, w)})
    return out


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a > 0, b > 0).sum()
    union = np.logical_or(a > 0, b > 0).sum()
    return float(inter / union) if union > 0 else 0.0


def collect_predictions(model_path: str, data_yaml: str, split: str,
                         imgsz: int, device: str, low_conf: float = 0.01):
    """Run model once at low conf; return per-image list of predictions."""
    from ultralytics import YOLO

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

    model = YOLO(model_path)
    img_files = sorted(list(img_dir.glob("*.png"))
                        + list(img_dir.glob("*.PNG"))
                        + list(img_dir.glob("*.jpg")))

    all_preds = []   # [{stem, h, w, preds: [{cls, conf, mask}], gts: [{cls, mask}]}]
    for img_path in img_files:
        results = model.predict(
            source=str(img_path),
            conf=low_conf,
            imgsz=imgsz,
            device=str(device),
            verbose=False, save=False, retina_masks=True,
        )
        if not results:
            continue
        r = results[0]
        h, w = r.orig_shape
        preds = []
        if r.masks is not None and len(r.masks) > 0:
            masks = r.masks.data.cpu().numpy()
            cls_arr = r.boxes.cls.cpu().numpy().astype(int)
            conf_arr = r.boxes.conf.cpu().numpy().astype(float)
            for i in range(len(masks)):
                preds.append({
                    "cls": int(cls_arr[i]),
                    "conf": float(conf_arr[i]),
                    "mask": (masks[i] > 0.5).astype(np.uint8),
                })
        gts = _read_gt(lbl_dir, img_path.stem, h, w)
        all_preds.append({
            "stem": img_path.stem, "h": h, "w": w,
            "preds": preds, "gts": gts,
        })
    return all_preds, names


def evaluate_at_thresholds(all_items: list, thresholds: dict) -> dict:
    """Given per-class thresholds {cls: thr}, compute per-class P/R/F1."""
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)

    for item in all_items:
        # Filter preds by per-class threshold
        preds = [p for p in item["preds"]
                 if p["conf"] >= thresholds.get(p["cls"], 0.25)]
        gts_by_cls = defaultdict(list)
        for g in item["gts"]:
            gts_by_cls[g["cls"]].append(g)

        # Greedy per-class match
        matched = defaultdict(set)
        for p in sorted(preds, key=lambda x: -x["conf"]):
            best_iou = 0.0
            best_j = -1
            for j, g in enumerate(gts_by_cls[p["cls"]]):
                if j in matched[p["cls"]]:
                    continue
                iou = _iou(p["mask"], g["mask"])
                if iou > best_iou:
                    best_iou = iou
                    best_j = j
            if best_iou >= 0.5:
                tp[p["cls"]] += 1
                matched[p["cls"]].add(best_j)
            else:
                fp[p["cls"]] += 1
        for cls, gs in gts_by_cls.items():
            fn[cls] += len(gs) - len(matched[cls])

    per_class = {}
    all_cls = set(tp) | set(fp) | set(fn)
    for cls in all_cls:
        _tp, _fp, _fn = tp[cls], fp[cls], fn[cls]
        p = _tp / (_tp + _fp) if (_tp + _fp) else 0
        r = _tp / (_tp + _fn) if (_tp + _fn) else 0
        f1 = 2 * p * r / (p + r) if (p + r) else 0
        per_class[cls] = {"p": round(p, 4), "r": round(r, 4),
                          "f1": round(f1, 4), "tp": _tp, "fp": _fp, "fn": _fn}
    return per_class


def sweep_per_class(val_items: list) -> dict:
    """For each class, pick the threshold in [0.05, 0.95] that maximises F1."""
    classes = set()
    for item in val_items:
        for p in item["preds"]:
            classes.add(p["cls"])
        for g in item["gts"]:
            classes.add(g["cls"])
    thr_grid = [round(x, 2) for x in np.arange(0.05, 0.96, 0.05)]

    chosen = {}
    per_cls_curves = {}
    for cls in classes:
        curve = []
        best = (0, 0.25)
        for thr in thr_grid:
            pc = evaluate_at_thresholds(val_items, {cls: thr})
            f1 = pc.get(cls, {}).get("f1", 0)
            curve.append((thr, f1))
            if f1 > best[0]:
                best = (f1, thr)
        chosen[cls] = best[1]
        per_cls_curves[cls] = curve
    return chosen, per_cls_curves


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--data-yaml", type=str, required=True)
    p.add_argument("--imgsz", type=int, default=768)
    p.add_argument("--device", type=str, default="0")
    p.add_argument("--low-conf", type=float, default=0.01)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)

    print("Collecting VAL predictions...")
    val_items, names = collect_predictions(
        args.model, args.data_yaml, "val", args.imgsz, args.device, args.low_conf)
    print(f"  {len(val_items)} val images")

    print("Sweeping per-class thresholds on val...")
    chosen, curves = sweep_per_class(val_items)
    chosen_named = {names.get(c, str(c)): v for c, v in chosen.items()}
    print(f"  Best thresholds (val-F1 max): {chosen_named}")

    val_metrics = evaluate_at_thresholds(val_items, chosen)

    # Baseline with default 0.25 everywhere
    default = {c: 0.25 for c in chosen}
    val_default = evaluate_at_thresholds(val_items, default)

    print("\nCollecting TEST predictions...")
    test_items, _ = collect_predictions(
        args.model, args.data_yaml, "test", args.imgsz, args.device, args.low_conf)
    print(f"  {len(test_items)} test images")

    test_metrics = evaluate_at_thresholds(test_items, chosen)
    test_default = evaluate_at_thresholds(test_items, default)

    def _mean_f1(m):
        vs = [v["f1"] for v in m.values()]
        return round(sum(vs) / len(vs), 4) if vs else 0

    report = {
        "model": args.model,
        "data_yaml": args.data_yaml,
        "imgsz": args.imgsz,
        "chosen_thresholds": chosen_named,
        "val": {
            "default_conf25": {
                names.get(c, str(c)): v for c, v in val_default.items()},
            "tuned": {
                names.get(c, str(c)): v for c, v in val_metrics.items()},
            "mean_f1_default": _mean_f1(val_default),
            "mean_f1_tuned":   _mean_f1(val_metrics),
        },
        "test": {
            "default_conf25": {
                names.get(c, str(c)): v for c, v in test_default.items()},
            "tuned": {
                names.get(c, str(c)): v for c, v in test_metrics.items()},
            "mean_f1_default": _mean_f1(test_default),
            "mean_f1_tuned":   _mean_f1(test_metrics),
        },
    }
    print("\n=== RESULTS ===")
    print(f"Val  mean F1: default {report['val']['mean_f1_default']:.4f}  tuned {report['val']['mean_f1_tuned']:.4f}")
    print(f"Test mean F1: default {report['test']['mean_f1_default']:.4f}  tuned {report['test']['mean_f1_tuned']:.4f}")
    print(f"Delta test:   {report['test']['mean_f1_tuned'] - report['test']['mean_f1_default']:+.4f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(args.output, "w"), indent=2)
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
