"""S14 — Weighted Box Fusion ensemble of stenosis detection models.

Motivation
----------
Different stenosis checkpoints achieved similar F1 via *different*
failure modes (S8 via mosaic, old-S5 via copy_paste, S10a via
filtered data). Weighted Box Fusion (WBF) merges their predictions
at inference time and typically extracts a +1-3pp F1 gain at zero
training cost.

This script is standalone (NOT dispatched from get_experiments()).
Run it manually after the training experiments finish:

    python ensemble_stenosis_wbf.py \\
        --models <path_to_model1.pt> <path_to_model2.pt> ... \\
        --data-yaml <stenosis_only.yaml> \\
        --split test \\
        --output <metrics.json>

Dependencies: ``ensemble-boxes`` (see requirements.txt).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml
from ultralytics import YOLO


def _load_yolo_gt_boxes(label_path: Path) -> np.ndarray:
    """Load YOLO-format ground-truth boxes.

    Supports both detection (cls cx cy w h) and segmentation
    (cls x1 y1 x2 y2 ... ) formats. For segmentation labels, the
    bounding box is computed as the axis-aligned bbox of the polygon.

    Returns
    -------
    np.ndarray of shape (N, 4) with normalised xyxy coordinates.
    """
    if not label_path.exists():
        return np.zeros((0, 4), dtype=np.float32)

    boxes = []
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            # cls_id = int(parts[0])  # unused (stenosis is single-class)
            vals = [float(x) for x in parts[1:]]
            if len(vals) == 4:
                cx, cy, w, h = vals
                x1 = cx - w / 2
                y1 = cy - h / 2
                x2 = cx + w / 2
                y2 = cy + h / 2
            elif len(vals) >= 6 and len(vals) % 2 == 0:
                xs = vals[0::2]
                ys = vals[1::2]
                x1, y1 = min(xs), min(ys)
                x2, y2 = max(xs), max(ys)
            else:
                continue
            boxes.append([
                max(0.0, min(1.0, x1)),
                max(0.0, min(1.0, y1)),
                max(0.0, min(1.0, x2)),
                max(0.0, min(1.0, y2)),
            ])
    return np.asarray(boxes, dtype=np.float32) if boxes else np.zeros((0, 4), dtype=np.float32)


def _iou_xyxy(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """Compute IoU between two xyxy boxes."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _match_predictions(
    pred_boxes: np.ndarray,
    pred_scores: np.ndarray,
    gt_boxes: np.ndarray,
    iou_thr: float = 0.5,
) -> tuple[list[bool], list[bool]]:
    """Greedy IoU matching of predictions to GT.

    Returns
    -------
    (pred_is_tp, gt_matched): two boolean lists, aligned with
    ``pred_boxes`` (sorted descending by score) and ``gt_boxes``.
    """
    order = np.argsort(-pred_scores)
    pred_is_tp = [False] * len(pred_boxes)
    gt_matched = [False] * len(gt_boxes)

    for i in order:
        if len(gt_boxes) == 0:
            break
        best_j = -1
        best_iou = iou_thr
        for j in range(len(gt_boxes)):
            if gt_matched[j]:
                continue
            iou = _iou_xyxy(pred_boxes[i], gt_boxes[j])
            if iou >= best_iou:
                best_iou = iou
                best_j = j
        if best_j >= 0:
            pred_is_tp[i] = True
            gt_matched[best_j] = True

    return pred_is_tp, gt_matched


def _compute_metrics(
    all_preds: list[tuple[np.ndarray, np.ndarray]],
    all_gts: list[np.ndarray],
    iou_thr: float = 0.5,
    score_thr: float = 0.25,
) -> dict:
    """Compute precision, recall, F1 at a fixed score threshold.

    Args:
        all_preds: Per-image list of (pred_boxes_xyxy_norm, pred_scores).
        all_gts: Per-image list of gt_boxes_xyxy_norm.
        iou_thr: IoU threshold for TP matching.
        score_thr: Score threshold below which predictions are dropped.
    """
    tp = 0
    fp = 0
    fn = 0
    total_gt = 0
    total_pred = 0

    for (pred_boxes, pred_scores), gt_boxes in zip(all_preds, all_gts):
        mask = pred_scores >= score_thr
        pb = pred_boxes[mask]
        ps = pred_scores[mask]
        total_gt += len(gt_boxes)
        total_pred += len(pb)
        pred_is_tp, gt_matched = _match_predictions(pb, ps, gt_boxes, iou_thr)
        tp += sum(pred_is_tp)
        fp += len(pb) - sum(pred_is_tp)
        fn += len(gt_boxes) - sum(gt_matched)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "total_gt": int(total_gt),
        "total_pred": int(total_pred),
        "iou_thr": iou_thr,
        "score_thr": score_thr,
    }


def _resolve_split_images(data_yaml: Path, split: str) -> tuple[Path, Path]:
    """Return (images_dir, labels_dir) for the given split."""
    with open(data_yaml) as f:
        data = yaml.safe_load(f)

    root = Path(data.get("path", data_yaml.parent)).resolve()
    split_rel = data.get(split)
    if split_rel is None:
        raise KeyError(f"Split '{split}' not found in {data_yaml}")
    images_dir = (root / split_rel).resolve()
    # Convention: images/<split>  →  labels/<split>
    labels_dir = Path(str(images_dir).replace("/images/", "/labels/"))
    if not labels_dir.exists():
        raise FileNotFoundError(f"Labels dir not found: {labels_dir}")
    return images_dir, labels_dir


def _extract_pred_boxes_norm(result, W: int, H: int) -> tuple[np.ndarray, np.ndarray]:
    """Extract xyxy (normalised) and confidence from an ultralytics result."""
    if result.boxes is None or len(result.boxes) == 0:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    xyxy = result.boxes.xyxy.cpu().numpy().astype(np.float32)
    conf = result.boxes.conf.cpu().numpy().astype(np.float32)
    # Normalise
    xyxy_norm = xyxy.copy()
    xyxy_norm[:, [0, 2]] /= max(W, 1)
    xyxy_norm[:, [1, 3]] /= max(H, 1)
    xyxy_norm = np.clip(xyxy_norm, 0.0, 1.0)
    return xyxy_norm, conf


def ensemble_wbf(
    model_paths: list[str],
    data_yaml: Path,
    split: str,
    imgsz: int,
    weights: list[float] | None,
    iou_thr: float,
    skip_box_thr: float,
    score_thr: float,
    augment: bool,
) -> dict:
    """Run WBF ensemble over ``model_paths`` on ``split``.

    Returns a metrics dict comparable to the per-model baselines and
    including per-component single-model scores for ablation.
    """
    try:
        from ensemble_boxes import weighted_boxes_fusion
    except ImportError as exc:
        raise SystemExit(
            "ensemble-boxes package required. Install via "
            "`pip install ensemble-boxes`."
        ) from exc

    images_dir, labels_dir = _resolve_split_images(data_yaml, split)
    img_files = sorted(
        list(images_dir.glob("*.png")) + list(images_dir.glob("*.PNG"))
        + list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.JPG"))
    )
    if not img_files:
        raise RuntimeError(f"No images found in {images_dir}")
    print(f"  Found {len(img_files)} images in {images_dir}")

    if weights is None:
        weights = [1.0] * len(model_paths)
    elif len(weights) != len(model_paths):
        raise ValueError(
            f"Number of weights ({len(weights)}) must match number of "
            f"models ({len(model_paths)})"
        )

    print(f"  Loading {len(model_paths)} models:")
    models = []
    for i, mp in enumerate(model_paths):
        print(f"    [{i}] w={weights[i]}  {mp}")
        models.append(YOLO(mp))

    # Per-model collectors for ablation
    per_model_preds = [[] for _ in models]
    ensemble_preds = []
    all_gts = []

    import cv2
    for img_idx, img_path in enumerate(img_files):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        H, W = img.shape[:2]

        boxes_list = []
        scores_list = []
        labels_list = []

        for mi, model in enumerate(models):
            res = model.predict(
                str(img_path),
                imgsz=imgsz,
                conf=0.01,
                augment=augment,
                verbose=False,
            )[0]
            pb_norm, pc = _extract_pred_boxes_norm(res, W, H)
            per_model_preds[mi].append((pb_norm, pc))
            boxes_list.append(pb_norm.tolist())
            scores_list.append(pc.tolist())
            labels_list.append([0] * len(pb_norm))

        fused_boxes, fused_scores, _ = weighted_boxes_fusion(
            boxes_list, scores_list, labels_list,
            weights=weights,
            iou_thr=iou_thr,
            skip_box_thr=skip_box_thr,
        )
        fused_boxes = np.asarray(fused_boxes, dtype=np.float32).reshape(-1, 4)
        fused_scores = np.asarray(fused_scores, dtype=np.float32).reshape(-1)
        ensemble_preds.append((fused_boxes, fused_scores))

        # GT from label file
        label_file = labels_dir / f"{img_path.stem}.txt"
        all_gts.append(_load_yolo_gt_boxes(label_file))

        if (img_idx + 1) % 50 == 0:
            print(f"    Processed {img_idx + 1}/{len(img_files)} images")

    ensemble_metrics = _compute_metrics(
        ensemble_preds, all_gts, iou_thr=0.5, score_thr=score_thr,
    )
    per_model_metrics = []
    for mi, preds in enumerate(per_model_preds):
        m = _compute_metrics(preds, all_gts, iou_thr=0.5, score_thr=score_thr)
        m["model"] = model_paths[mi]
        m["weight"] = weights[mi]
        per_model_metrics.append(m)

    return {
        "name": "S14_ensemble_wbf",
        "split": split,
        "imgsz": imgsz,
        "augment": augment,
        "score_thr": score_thr,
        "wbf": {
            "iou_thr": iou_thr,
            "skip_box_thr": skip_box_thr,
        },
        "models": list(model_paths),
        "weights": list(weights),
        "ensemble": ensemble_metrics,
        "per_model": per_model_metrics,
    }


def main():
    parser = argparse.ArgumentParser(
        description="S14: Weighted Box Fusion ensemble of stenosis models"
    )
    parser.add_argument(
        "--models", nargs="+", required=True,
        help="Paths to 2 or more stenosis model checkpoints (.pt)"
    )
    parser.add_argument(
        "--data-yaml", type=str, required=True,
        help="Path to stenosis_only.yaml (single-class stenosis dataset)"
    )
    parser.add_argument(
        "--split", type=str, default="test", choices=["val", "test"],
        help="Which split to evaluate on (default: test)"
    )
    parser.add_argument(
        "--imgsz", type=int, default=768,
        help="Inference image size (default: 768)"
    )
    parser.add_argument(
        "--weights", type=str, default=None,
        help="Comma-separated per-model weights (default: equal weights)"
    )
    parser.add_argument(
        "--iou-thr", type=float, default=0.55,
        help="WBF IoU threshold (default: 0.55)"
    )
    parser.add_argument(
        "--skip-box-thr", type=float, default=0.1,
        help="WBF skip-box confidence threshold (default: 0.1)"
    )
    parser.add_argument(
        "--score-thr", type=float, default=0.25,
        help="Score threshold for P/R/F1 evaluation (default: 0.25)"
    )
    parser.add_argument(
        "--no-augment", action="store_true",
        help="Disable TTA during per-model inference"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Path to save metrics JSON"
    )
    parser.add_argument(
        "--append-strategy-results", type=str, default=None,
        help="Also append the result to strategy_results.json at this path"
    )
    args = parser.parse_args()

    if len(args.models) < 2:
        parser.error("Provide at least 2 models for ensemble")

    weights = None
    if args.weights:
        weights = [float(x) for x in args.weights.split(",")]

    print("=" * 64)
    print("S14 — Stenosis ensemble (Weighted Box Fusion)")
    print("=" * 64)
    print(f"  Models      : {len(args.models)}")
    print(f"  Data        : {args.data_yaml}")
    print(f"  Split       : {args.split}")
    print(f"  imgsz       : {args.imgsz}")
    print(f"  augment     : {not args.no_augment}")
    print(f"  wbf iou_thr : {args.iou_thr}")
    print(f"  skip_box_thr: {args.skip_box_thr}")
    print(f"  score_thr   : {args.score_thr}")

    result = ensemble_wbf(
        model_paths=args.models,
        data_yaml=Path(args.data_yaml),
        split=args.split,
        imgsz=args.imgsz,
        weights=weights,
        iou_thr=args.iou_thr,
        skip_box_thr=args.skip_box_thr,
        score_thr=args.score_thr,
        augment=not args.no_augment,
    )

    # ── Print summary ──
    em = result["ensemble"]
    print("\n" + "=" * 64)
    print("RESULTS")
    print("=" * 64)
    print(f"\n  Ensemble (WBF):")
    print(f"    precision : {em['precision']:.4f}")
    print(f"    recall    : {em['recall']:.4f}")
    print(f"    F1        : {em['f1']:.4f}")
    print(f"    tp/fp/fn  : {em['tp']}/{em['fp']}/{em['fn']}")

    print(f"\n  Per-model (ablation):")
    print(f"    {'F1':>8s} {'Prec':>8s} {'Recall':>8s}  model")
    print(f"    {'-' * 60}")
    for m in result["per_model"]:
        print(f"    {m['f1']:>8.4f} {m['precision']:>8.4f} "
              f"{m['recall']:>8.4f}  {Path(m['model']).name}")

    # Find best single-model F1 for comparison
    best_single = max(m["f1"] for m in result["per_model"])
    ens_delta = em["f1"] - best_single
    print(f"\n  Ensemble vs best single model: "
          f"{em['f1']:.4f} vs {best_single:.4f}  "
          f"(Δ = {ens_delta:+.4f})")

    # ── Save JSON ──
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"\n  Saved: {out_path}")

    # ── Append to strategy_results.json ──
    if args.append_strategy_results:
        sr_path = Path(args.append_strategy_results)
        existing = []
        if sr_path.exists():
            try:
                with open(sr_path) as f:
                    existing = json.load(f)
                if not isinstance(existing, list):
                    existing = [existing]
            except (json.JSONDecodeError, ValueError):
                existing = []

        # Build an entry shaped like other strategy_results entries
        entry = {
            "name": "S14_ensemble_wbf",
            "gpu": -1,
            "description": (
                "Weighted Box Fusion ensemble of stenosis models "
                f"({len(args.models)} components)"
            ),
            "elapsed_hours": 0,
            "status": "success",
            "metrics": {
                "final_test": {
                    "split": args.split,
                    "mAP50": 0,  # not computed here
                    "per_class": {
                        "stenosis": {
                            "precision": em["precision"],
                            "recall": em["recall"],
                            "f1": em["f1"],
                            "ap50": 0,
                        },
                    },
                    "stenosis_AP50": 0,
                    "precision": em["precision"],
                    "recall": em["recall"],
                    "ensemble_details": {
                        "models": list(args.models),
                        "weights": list(weights) if weights else None,
                        "per_model": result["per_model"],
                        "wbf": result["wbf"],
                    },
                },
            },
        }

        by_name = {r["name"]: r for r in existing if isinstance(r, dict)}
        by_name["S14_ensemble_wbf"] = entry
        with open(sr_path, "w") as f:
            json.dump(list(by_name.values()), f, indent=2, default=str)
        print(f"  Appended to: {sr_path}")


if __name__ == "__main__":
    main()
