"""Small connected-component removal for YOLO segmentation predictions.

SSASS post-processing: drop any predicted instance whose mask area
(in pixels) falls below a threshold. This cleans the false positives
that small spurious detections produce in the F1 metric.

Two entry points:
  - `filter_results`: mutate Ultralytics Results objects in memory.
  - `evaluate_with_filter_cpu`: CPU-only re-eval that avoids the GPU
    OOM that bit E1/E3 — it streams predict() one image at a time and
    immediately releases the GPU tensors.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np


def filter_results(results, min_area_px: int = 30):
    """Drop instances whose mask area is below `min_area_px`.

    Args:
        results: list of ultralytics.engine.results.Results
        min_area_px: pixel-area threshold

    Returns:
        Same results list, mutated in place.
    """
    import torch

    for r in results:
        if r.masks is None or len(r.masks) == 0:
            continue
        m = r.masks.data  # (N, H, W) bool/float tensor
        areas = m.reshape(m.shape[0], -1).sum(dim=1)
        keep = (areas >= min_area_px).cpu().numpy()
        if keep.all():
            continue
        keep_idx = np.where(keep)[0]
        if len(keep_idx) == 0:
            r.masks = None
            r.boxes = r.boxes[torch.zeros(0, dtype=torch.long)]
            continue
        keep_t = torch.as_tensor(keep_idx, dtype=torch.long, device=m.device)
        r.masks.data = m[keep_t]
        if hasattr(r.masks, "xy"):
            r.masks.xy = [r.masks.xy[i] for i in keep_idx]
        if hasattr(r.masks, "xyn"):
            r.masks.xyn = [r.masks.xyn[i] for i in keep_idx]
        r.boxes = r.boxes[keep_t]
    return results


def predict_and_filter(
    model_path: str,
    image_dir: Path,
    imgsz: int = 768,
    conf: float = 0.25,
    device: str = "0",
    min_area_px: int = 30,
):
    """Run inference over a directory and return filtered Results list."""
    from ultralytics import YOLO

    model = YOLO(model_path)
    image_dir = Path(image_dir)
    img_paths = sorted(
        list(image_dir.glob("*.png"))
        + list(image_dir.glob("*.PNG"))
        + list(image_dir.glob("*.jpg"))
    )
    results = model.predict(
        source=[str(p) for p in img_paths],
        imgsz=imgsz,
        conf=conf,
        device=f"cuda:{device}" if str(device).isdigit() else device,
        verbose=False,
        save=False,
        retina_masks=True,
    )
    return filter_results(list(results), min_area_px=min_area_px)


def evaluate_with_filter_cpu(
    model_path: str,
    data_yaml: str,
    split: str = "test",
    imgsz: int = 768,
    min_area_px: int = 30,
    device: str = "0",
    conf: float = 0.25,
) -> dict:
    """CPU-only streaming eval — fix for the GPU OOM in E1/E3.

    Predict one image at a time, move each result to CPU numpy, release
    the GPU tensors, then compute per-image TP/FP/FN with CC filtering.
    Never holds >1 image worth of tensors on the GPU.
    """
    import yaml as _yaml
    import cv2
    import torch
    from ultralytics import YOLO

    with open(data_yaml) as f:
        cfg = _yaml.safe_load(f)
    root = Path(cfg["path"])
    img_rel = cfg.get(split, f"images/{split}")
    img_dir = root / img_rel
    lbl_dir = root / img_rel.replace("images", "labels")
    if not lbl_dir.exists():
        lbl_dir = root / f"labels/{split}"

    model = YOLO(model_path)
    img_files = sorted(list(img_dir.glob("*.png"))
                        + list(img_dir.glob("*.PNG")))

    tp = fp = fn = 0
    tp_raw = fp_raw = fn_raw = 0
    for img_path in img_files:
        results = model.predict(
            source=str(img_path), conf=conf, imgsz=imgsz,
            device=str(device), verbose=False, save=False, retina_masks=True,
        )
        if not results:
            continue
        r = results[0]
        h, w = r.orig_shape
        gt_path = lbl_dir / f"{img_path.stem}.txt"
        gt_polys = _read_yolo_polys(gt_path) if gt_path.exists() else []
        gt_masks = [_poly_to_mask(p, h, w) for p in gt_polys]

        preds_raw = []
        preds_filt = []
        if r.masks is not None and len(r.masks) > 0:
            masks_np = (r.masks.data.cpu().numpy() > 0.5).astype(np.uint8)
            confs = r.boxes.conf.cpu().numpy().tolist()
            for i, m in enumerate(masks_np):
                preds_raw.append({"conf": confs[i], "mask": m})
                # CC filter (per-instance; drop if its single blob <min area)
                num_labels, _labels, stats, _ = cv2.connectedComponentsWithStats(
                    m.astype(np.uint8), connectivity=8)
                areas = stats[1:, cv2.CC_STAT_AREA] if num_labels > 1 else np.array([])
                if areas.size > 0 and areas.max() >= min_area_px:
                    preds_filt.append({"conf": confs[i], "mask": m})
        # Release GPU memory
        del r, results
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        def _match(preds, gts):
            matched = set()
            ltp = lfp = 0
            for pr in sorted(preds, key=lambda x: -x["conf"]):
                best_iou = 0
                best_j = -1
                for j, g in enumerate(gts):
                    if j in matched:
                        continue
                    inter = np.logical_and(pr["mask"] > 0, g > 0).sum()
                    union = np.logical_or(pr["mask"] > 0, g > 0).sum()
                    iou = inter / union if union > 0 else 0
                    if iou > best_iou:
                        best_iou = iou
                        best_j = j
                if best_iou >= 0.5:
                    ltp += 1
                    matched.add(best_j)
                else:
                    lfp += 1
            lfn = len(gts) - len(matched)
            return ltp, lfp, lfn

        _tp, _fp, _fn = _match(preds_raw, gt_masks)
        tp_raw += _tp; fp_raw += _fp; fn_raw += _fn
        _tp, _fp, _fn = _match(preds_filt, gt_masks)
        tp += _tp; fp += _fp; fn += _fn

    def _prf(tp, fp, fn):
        p = tp / (tp + fp) if (tp + fp) else 0
        r = tp / (tp + fn) if (tp + fn) else 0
        f1 = 2 * p * r / (p + r) if (p + r) else 0
        return {"precision": round(p, 4), "recall": round(r, 4),
                "f1": round(f1, 4), "tp": tp, "fp": fp, "fn": fn}

    return {
        "raw": _prf(tp_raw, fp_raw, fn_raw),
        "filtered": _prf(tp, fp, fn),
        "min_area_px": min_area_px,
    }


def evaluate_with_filter(
    model_path: str,
    data_yaml: str,
    split: str = "test",
    imgsz: int = 768,
    min_area_px: int = 30,
    augment: bool = False,
) -> dict:
    """Re-evaluate a YOLO model and apply small-CC filtering before scoring.

    We use ultralytics' validator twice: once unfiltered (baseline) and
    once with a custom predict loop + manual F1 calc. Since plumbing the
    filter into ultralytics' validator is brittle, the post-filter score
    is computed manually from per-image confusion counts at IoU=0.5.

    Returns dict with keys {raw: {...}, filtered: {...}, min_area_px}.
    """
    import yaml as _yaml
    from ultralytics import YOLO

    with open(data_yaml) as f:
        cfg = _yaml.safe_load(f)
    root = Path(cfg["path"])
    img_dir = root / cfg.get(split, f"images/{split}")
    lbl_dir = root / cfg.get(split, f"images/{split}").replace("images", "labels")
    if not lbl_dir.exists():
        lbl_dir = root / f"labels/{split}"

    model = YOLO(model_path)
    raw_metrics = model.val(
        data=data_yaml, split=split, imgsz=imgsz, augment=augment,
        save_json=False, verbose=False,
    )
    raw = {
        "mAP50": round(float(raw_metrics.seg.map50), 4),
        "precision": round(float(raw_metrics.seg.mp), 4),
        "recall": round(float(raw_metrics.seg.mr), 4),
    }

    filtered_results = predict_and_filter(
        model_path=model_path,
        image_dir=img_dir,
        imgsz=imgsz,
        conf=0.25,
        device="0",
        min_area_px=min_area_px,
    )

    tp = fp = fn = 0
    for r in filtered_results:
        img_stem = Path(r.path).stem
        gt_path = lbl_dir / f"{img_stem}.txt"
        gt_polys = _read_yolo_polys(gt_path) if gt_path.exists() else []
        pred_masks = []
        if r.masks is not None and len(r.masks) > 0:
            pred_masks = [m.cpu().numpy() for m in r.masks.data]

        matched_gt = set()
        for pm in pred_masks:
            best_iou = 0.0
            best_j = -1
            h, w = pm.shape
            for j, poly in enumerate(gt_polys):
                if j in matched_gt:
                    continue
                gm = _poly_to_mask(poly, h, w)
                inter = np.logical_and(pm > 0.5, gm > 0).sum()
                union = np.logical_or(pm > 0.5, gm > 0).sum()
                iou = inter / union if union > 0 else 0
                if iou > best_iou:
                    best_iou = iou
                    best_j = j
            if best_iou >= 0.5:
                tp += 1
                matched_gt.add(best_j)
            else:
                fp += 1
        fn += len(gt_polys) - len(matched_gt)

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

    filtered = {
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "tp": tp, "fp": fp, "fn": fn,
    }

    return {"raw": raw, "filtered": filtered, "min_area_px": min_area_px}


def _read_yolo_polys(label_path: Path) -> List[np.ndarray]:
    polys = []
    for line in label_path.read_text().strip().splitlines():
        parts = line.split()
        if len(parts) < 7:
            continue
        coords = np.array(parts[1:], dtype=np.float32).reshape(-1, 2)
        polys.append(coords)
    return polys


def _poly_to_mask(poly_norm: np.ndarray, h: int, w: int) -> np.ndarray:
    import cv2
    poly_px = poly_norm.copy()
    poly_px[:, 0] *= w
    poly_px[:, 1] *= h
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [poly_px.astype(np.int32)], 1)
    return mask
