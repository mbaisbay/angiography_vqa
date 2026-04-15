"""A-2: Tile / overlap-crop inference for stenosis.

Upscales each test image 3x (to 1536), extracts a 3x3 grid of 512x512
tiles with 128 px overlap, runs YOLO per tile, maps predictions back to
image coordinates, and merges via Weighted Box Fusion.

Output:
  - Per-image predictions at full-image coordinates
  - Overall P/R/F1 vs ground truth at IoU >= 0.5
  - Comparison vs plain single-pass inference on the same model

Needs: `pip install ensemble-boxes`

Usage:
    python tile_inference_stenosis.py \
        --model results/.../S54_s43b_clahe/stenosis_model/stenosis_768_best.pt \
        --data-yaml data/.../dataset_configs/stenosis_only.yaml \
        --split test --upscale 3 --tile 512 --overlap 128 \
        --output results/.../S54_tile_inference.json
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


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a > 0, b > 0).sum()
    union = np.logical_or(a > 0, b > 0).sum()
    return float(inter / union) if union > 0 else 0.0


def tile_predict(model, img_path: Path, upscale: int, tile: int,
                 overlap: int, imgsz: int, device: str, conf: float):
    """Return a list of (box_xyxy_norm, conf, mask_full) for one image."""
    import cv2

    img = cv2.imread(str(img_path))
    if img is None:
        return []
    H, W = img.shape[:2]
    up_h, up_w = H * upscale, W * upscale
    img_up = cv2.resize(img, (up_w, up_h), interpolation=cv2.INTER_CUBIC)

    stride = tile - overlap
    ys = list(range(0, max(up_h - tile, 0) + 1, stride)) or [0]
    xs = list(range(0, max(up_w - tile, 0) + 1, stride)) or [0]
    if ys[-1] + tile < up_h:
        ys.append(up_h - tile)
    if xs[-1] + tile < up_w:
        xs.append(up_w - tile)

    collected = []
    for y0 in ys:
        for x0 in xs:
            crop = img_up[y0:y0 + tile, x0:x0 + tile]
            if crop.shape[0] < tile or crop.shape[1] < tile:
                pad_bottom = tile - crop.shape[0]
                pad_right = tile - crop.shape[1]
                crop = cv2.copyMakeBorder(crop, 0, pad_bottom, 0, pad_right,
                                            cv2.BORDER_CONSTANT, value=0)
            results = model.predict(
                source=crop, conf=conf, imgsz=imgsz,
                device=str(device), verbose=False, save=False,
                retina_masks=True,
            )
            if not results or results[0].masks is None:
                continue
            r = results[0]
            masks = r.masks.data.cpu().numpy()
            boxes = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            for i in range(len(masks)):
                full_mask = np.zeros((H, W), dtype=np.uint8)
                tile_mask = (masks[i] > 0.5).astype(np.uint8)
                th, tw = tile_mask.shape
                # Paste back into upscaled space, then downscale to orig
                up_full = np.zeros((up_h, up_w), dtype=np.uint8)
                up_full[y0:y0 + th, x0:x0 + tw] = np.maximum(
                    up_full[y0:y0 + th, x0:x0 + tw], tile_mask)
                full_mask = cv2.resize(
                    up_full, (W, H), interpolation=cv2.INTER_NEAREST)

                x1, y1, x2, y2 = boxes[i]
                bx1 = (x1 + x0) / up_w
                by1 = (y1 + y0) / up_h
                bx2 = (x2 + x0) / up_w
                by2 = (y2 + y0) / up_h
                collected.append({
                    "bbox": [float(bx1), float(by1), float(bx2), float(by2)],
                    "conf": float(confs[i]),
                    "mask": full_mask,
                })
    return collected


def wbf_merge(detections: list, iou_thr: float = 0.5,
              skip_box_thr: float = 0.01):
    """Weighted Box Fusion over a single image's tile detections."""
    try:
        from ensemble_boxes import weighted_boxes_fusion
    except ImportError:
        raise SystemExit("Install: pip install ensemble-boxes")
    if not detections:
        return []
    boxes = [[d["bbox"] for d in detections]]
    scores = [[d["conf"] for d in detections]]
    labels = [[0 for _ in detections]]   # stenosis = 1 class
    mb, ms, ml = weighted_boxes_fusion(
        boxes, scores, labels, iou_thr=iou_thr, skip_box_thr=skip_box_thr)

    # Map each merged box back to a mask via OR of constituent masks
    merged = []
    for i in range(len(mb)):
        mx1, my1, mx2, my2 = mb[i]
        combined_mask = None
        for d in detections:
            dx1, dy1, dx2, dy2 = d["bbox"]
            overlap_x = max(0, min(mx2, dx2) - max(mx1, dx1))
            overlap_y = max(0, min(my2, dy2) - max(my1, dy1))
            if overlap_x * overlap_y > 0:
                if combined_mask is None:
                    combined_mask = d["mask"].copy()
                else:
                    combined_mask = np.maximum(combined_mask, d["mask"])
        merged.append({
            "bbox": [float(mx1), float(my1), float(mx2), float(my2)],
            "conf": float(ms[i]),
            "mask": combined_mask if combined_mask is not None else np.zeros((1, 1), np.uint8),
        })
    return merged


def evaluate(per_image, lbl_dir: Path) -> dict:
    tp = fp = fn = 0
    for item in per_image:
        img_stem = item["stem"]
        h, w = item["h"], item["w"]
        lbl_path = lbl_dir / f"{img_stem}.txt"
        gts = []
        if lbl_path.exists():
            for line in lbl_path.read_text().splitlines():
                line = line.strip()
                if line and len(line.split()) >= 7:
                    gts.append(_yolo_poly_to_mask(line, h, w))

        matched = set()
        for p in sorted(item["merged"], key=lambda x: -x["conf"]):
            best_iou = 0
            best_j = -1
            for j, g in enumerate(gts):
                if j in matched:
                    continue
                iou = _iou(p["mask"], g)
                if iou > best_iou:
                    best_iou = iou
                    best_j = j
            if best_iou >= 0.5:
                tp += 1
                matched.add(best_j)
            else:
                fp += 1
        fn += len(gts) - len(matched)

    p = tp / (tp + fp) if (tp + fp) else 0
    r = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * p * r / (p + r) if (p + r) else 0
    return {"tp": tp, "fp": fp, "fn": fn,
            "precision": round(p, 4), "recall": round(r, 4),
            "f1": round(f1, 4)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--data-yaml", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--upscale", type=int, default=3)
    p.add_argument("--tile", type=int, default=512)
    p.add_argument("--overlap", type=int, default=128)
    p.add_argument("--imgsz", type=int, default=512)
    p.add_argument("--device", type=str, default="0")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    from ultralytics import YOLO
    model = YOLO(args.model)

    with open(args.data_yaml) as f:
        cfg = yaml.safe_load(f)
    root = Path(cfg["path"])
    img_rel = cfg.get(args.split, f"images/{args.split}")
    img_dir = root / img_rel
    lbl_dir = root / img_rel.replace("images", "labels")
    if not lbl_dir.exists():
        lbl_dir = root / f"labels/{args.split}"

    img_files = sorted(list(img_dir.glob("*.png"))
                        + list(img_dir.glob("*.PNG")))

    per_image = []
    for img_path in img_files:
        import cv2
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        H, W = img.shape[:2]
        dets = tile_predict(
            model, img_path, args.upscale, args.tile, args.overlap,
            args.imgsz, args.device, args.conf)
        merged = wbf_merge(dets)
        per_image.append({
            "stem": img_path.stem, "h": H, "w": W,
            "merged": merged,
        })

    report = {
        "model": args.model,
        "upscale": args.upscale,
        "tile": args.tile,
        "overlap": args.overlap,
        "n_images": len(per_image),
        "metrics": evaluate(per_image, lbl_dir),
    }
    print(f"\n=== TILE INFERENCE RESULTS ===")
    print(f"  Images: {report['n_images']}")
    for k, v in report["metrics"].items():
        print(f"  {k}: {v}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(args.output, "w"), indent=2)
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
