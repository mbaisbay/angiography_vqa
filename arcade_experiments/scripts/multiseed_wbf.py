"""A-6: Multi-seed WBF ensemble — merge predictions, not metrics.

Takes N trained models (same architecture, different seeds) and merges
their predictions on a test set via Weighted Box Fusion, then evaluates
the merged output. This is the *real* ensemble — the S15_multiseed
"summary" rows in strategy_results.json were just averaged metrics,
not merged predictions.

Usage:
    python multiseed_wbf.py \
        --models results/.../S54_seed42/syntax_model/syntax_768_best.pt \
                 results/.../S54_seed7/syntax_model/syntax_768_best.pt \
                 results/.../S54_seed2024/syntax_model/syntax_768_best.pt \
        --data-yaml data/.../syntax_only.yaml \
        --split test --imgsz 768 --device 0 \
        --output results/.../S54_wbf3.json
"""

from __future__ import annotations

import argparse
import json
import os
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


def predict_all(model, imgs, imgsz, device, conf):
    per_image = {}
    for img_path in imgs:
        results = model.predict(
            source=str(img_path), conf=conf, imgsz=imgsz, device=str(device),
            verbose=False, save=False, retina_masks=True,
        )
        if not results or results[0].masks is None:
            per_image[img_path.stem] = {"h": 0, "w": 0, "dets": []}
            continue
        r = results[0]
        h, w = r.orig_shape
        masks = r.masks.data.cpu().numpy() > 0.5
        cls_arr = r.boxes.cls.cpu().numpy().astype(int)
        conf_arr = r.boxes.conf.cpu().numpy().astype(float)
        boxes = r.boxes.xyxyn.cpu().numpy()
        dets = []
        for i in range(len(masks)):
            dets.append({
                "cls": int(cls_arr[i]),
                "conf": float(conf_arr[i]),
                "bbox": boxes[i].tolist(),
                "mask": masks[i].astype(np.uint8),
            })
        per_image[img_path.stem] = {"h": h, "w": w, "dets": dets}
    return per_image


def wbf_merge_image(all_dets_for_image, iou_thr=0.55, skip_box_thr=0.01):
    try:
        from ensemble_boxes import weighted_boxes_fusion
    except ImportError:
        raise SystemExit("pip install ensemble-boxes")

    # Each element is one model's list of dets
    if not any(all_dets_for_image):
        return []

    boxes = [[d["bbox"] for d in model_dets] for model_dets in all_dets_for_image]
    scores = [[d["conf"] for d in model_dets] for model_dets in all_dets_for_image]
    labels = [[d["cls"] for d in model_dets] for model_dets in all_dets_for_image]

    boxes = [b if b else [[0, 0, 0, 0]] for b in boxes]
    scores = [s if s else [0.0] for s in scores]
    labels = [l if l else [0] for l in labels]

    mb, ms, ml = weighted_boxes_fusion(
        boxes, scores, labels, iou_thr=iou_thr, skip_box_thr=skip_box_thr)

    # For each merged box, OR the contributing masks from all models
    merged = []
    flat = [d for model_dets in all_dets_for_image for d in model_dets]
    for i in range(len(mb)):
        mx1, my1, mx2, my2 = mb[i]
        ml_i = int(ml[i])
        combined_mask = None
        for d in flat:
            if d["cls"] != ml_i:
                continue
            dx1, dy1, dx2, dy2 = d["bbox"]
            overlap_x = max(0, min(mx2, dx2) - max(mx1, dx1))
            overlap_y = max(0, min(my2, dy2) - max(my1, dy1))
            if overlap_x * overlap_y > 0:
                if combined_mask is None:
                    combined_mask = d["mask"].copy()
                else:
                    combined_mask = np.maximum(combined_mask, d["mask"])
        merged.append({
            "cls": ml_i,
            "conf": float(ms[i]),
            "bbox": [float(x) for x in mb[i]],
            "mask": combined_mask if combined_mask is not None else
                    np.zeros((1, 1), np.uint8),
        })
    return merged


def evaluate(merged_by_image: dict, lbl_dir: Path) -> dict:
    tp = defaultdict(int); fp = defaultdict(int); fn = defaultdict(int)
    for stem, data in merged_by_image.items():
        h, w = data["h"], data["w"]
        gts_by_cls = defaultdict(list)
        lbl_path = lbl_dir / f"{stem}.txt"
        if lbl_path.exists():
            for line in lbl_path.read_text().splitlines():
                line = line.strip()
                if not line or len(line.split()) < 7:
                    continue
                cls = int(line.split()[0])
                gts_by_cls[cls].append(_yolo_poly_to_mask(line, h, w))

        used = defaultdict(set)
        for p in sorted(data["merged"], key=lambda x: -x["conf"]):
            best = 0
            best_j = -1
            for j, g in enumerate(gts_by_cls.get(p["cls"], [])):
                if j in used[p["cls"]]:
                    continue
                iou = _iou(p["mask"], g)
                if iou > best:
                    best = iou
                    best_j = j
            if best >= 0.5:
                tp[p["cls"]] += 1
                used[p["cls"]].add(best_j)
            else:
                fp[p["cls"]] += 1
        for cls, gs in gts_by_cls.items():
            fn[cls] += len(gs) - len(used[cls])

    per_class = {}
    all_cls = set(tp) | set(fp) | set(fn)
    for cls in all_cls:
        _tp, _fp, _fn = tp[cls], fp[cls], fn[cls]
        p = _tp / (_tp + _fp) if (_tp + _fp) else 0
        r = _tp / (_tp + _fn) if (_tp + _fn) else 0
        f1 = 2 * p * r / (p + r) if (p + r) else 0
        per_class[cls] = {"p": round(p, 4), "r": round(r, 4),
                          "f1": round(f1, 4), "tp": _tp, "fp": _fp, "fn": _fn}

    f1s = [v["f1"] for v in per_class.values()]
    return {"per_class": per_class,
            "mean_f1": round(sum(f1s) / len(f1s), 4) if f1s else 0}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", required=True,
                   help="List of .pt files to ensemble")
    p.add_argument("--data-yaml", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--imgsz", type=int, default=768)
    p.add_argument("--device", type=str, default="0")
    p.add_argument("--conf", type=float, default=0.05)
    p.add_argument("--iou-thr", type=float, default=0.55)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    from ultralytics import YOLO

    with open(args.data_yaml) as f:
        cfg = yaml.safe_load(f)
    root = Path(cfg["path"])
    img_rel = cfg.get(args.split, f"images/{args.split}")
    img_dir = root / img_rel
    lbl_dir = root / img_rel.replace("images", "labels")
    if not lbl_dir.exists():
        lbl_dir = root / f"labels/{args.split}"

    img_files = sorted(list(img_dir.glob("*.png")) + list(img_dir.glob("*.PNG")))
    print(f"{len(img_files)} {args.split} images")

    per_model_preds = []
    for mp in args.models:
        print(f"\nInference with {mp}")
        model = YOLO(mp)
        preds = predict_all(model, img_files, args.imgsz, args.device, args.conf)
        per_model_preds.append(preds)

    merged_by_image = {}
    for img_path in img_files:
        stem = img_path.stem
        hw = next((per_model_preds[k][stem] for k in range(len(per_model_preds))
                   if per_model_preds[k][stem]["h"]), {"h": 512, "w": 512})
        all_dets = [per_model_preds[k][stem]["dets"] for k in range(len(per_model_preds))]
        merged = wbf_merge_image(all_dets, iou_thr=args.iou_thr)
        merged_by_image[stem] = {
            "h": hw["h"], "w": hw["w"], "merged": merged,
        }

    report = {
        "models": args.models,
        "iou_thr": args.iou_thr,
        "conf": args.conf,
        "n_images": len(img_files),
        "metrics": evaluate(merged_by_image, lbl_dir),
    }
    print(f"\n=== MULTI-SEED WBF ENSEMBLE ({len(args.models)} models) ===")
    print(f"  mean F1: {report['metrics']['mean_f1']}")
    for cls, m in sorted(report["metrics"]["per_class"].items()):
        print(f"  class {cls}: f1={m['f1']} p={m['p']} r={m['r']}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(args.output, "w"), indent=2)
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
