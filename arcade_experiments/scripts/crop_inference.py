"""Two-stage crop-based stenosis inference and evaluation.

Stage 1: Run syntax (vessel) model on full images to get vessel bboxes.
Stage 2: Crop each vessel bbox (+padding), run dedicated stenosis model
         on each crop, remap predictions back to original image coordinates.
Stage 3: Polygon NMS across all crops, then evaluate using ARCADE F1.

Usage:
    python crop_inference.py \
        --syntax-weights /path/to/syntax_best.pt \
        --stenosis-weights /path/to/crop_stenosis_best.pt \
        --arcade-root ../../arcade/submission \
        --split test
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from shapely.geometry import Polygon
from shapely.validation import make_valid
from ultralytics import YOLO

from crop_utils import (
    bbox_xywh_to_xyxy,
    compute_padded_bbox,
    crop_and_resize,
    remap_crop_to_original,
    polygon_nms,
)


# ---------------------------------------------------------------------------
# ARCADE F1 metric (polygon area-based)
# ---------------------------------------------------------------------------

def shapely_f1(pred_poly, gt_poly):
    """Compute area-based F1 between two Shapely polygons."""
    if pred_poly.is_empty or gt_poly.is_empty:
        return 0.0

    try:
        intersection = pred_poly.intersection(gt_poly).area
    except Exception:
        return 0.0

    pred_area = pred_poly.area
    gt_area = gt_poly.area

    if pred_area == 0 or gt_area == 0:
        return 0.0

    precision = intersection / pred_area
    recall = intersection / gt_area

    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)


def coords_to_shapely(coords, width=512, height=512, normalized=False):
    """Convert coordinate list to Shapely Polygon.

    Args:
        coords: Flat list [x1,y1,x2,y2,...] or list of [x,y] pairs.
        width, height: Image dimensions (for denormalization).
        normalized: If True, coords are in [0,1].

    Returns:
        Shapely Polygon.
    """
    # Handle flat list
    if coords and not isinstance(coords[0], (list, tuple)):
        points = [(coords[i], coords[i + 1]) for i in range(0, len(coords), 2)]
    else:
        points = [(c[0], c[1]) for c in coords]

    if normalized:
        points = [(x * width, y * height) for x, y in points]

    if len(points) < 3:
        return Polygon()

    try:
        p = Polygon(points)
        if not p.is_valid:
            p = make_valid(p)
        return p
    except Exception:
        return Polygon()


# ---------------------------------------------------------------------------
# Two-stage inference
# ---------------------------------------------------------------------------

def run_crop_inference(
    syntax_weights: str,
    stenosis_weights: str,
    images_dir: str,
    padding_frac: float = 0.20,
    crop_size: int = 512,
    min_dim: int = 16,
    vessel_conf: float = 0.15,
    stenosis_conf: float = 0.25,
    nms_iou: float = 0.5,
    vessel_imgsz: int = 768,
    stenosis_imgsz: int = 512,
) -> dict:
    """Run two-stage crop-based stenosis inference.

    Returns:
        {image_name: [(polygon_pixel_coords, confidence), ...]}
    """
    img_size = 512  # ARCADE native

    # Load models
    print(f"  Loading syntax model: {syntax_weights}")
    syntax_model = YOLO(syntax_weights)
    print(f"  Loading stenosis crop model: {stenosis_weights}")
    stenosis_model = YOLO(stenosis_weights)

    image_dir = Path(images_dir)
    image_files = sorted(image_dir.glob("*.png")) + sorted(image_dir.glob("*.PNG"))

    all_predictions = {}
    total_vessels = 0
    total_stenoses = 0

    print(f"\n  Running two-stage inference on {len(image_files)} images...")

    for img_path in image_files:
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        image_name = img_path.name

        # Stage 1: Vessel detection on full image
        vessel_results = syntax_model.predict(
            source=str(img_path),
            conf=vessel_conf,
            imgsz=vessel_imgsz,
            save=False,
            verbose=False,
            retina_masks=True,
        )

        image_stenoses = []

        for result in vessel_results:
            if result.masks is None or len(result.masks) == 0:
                continue

            for i in range(len(result.masks)):
                total_vessels += 1
                bbox_xywh = result.boxes.xywh[i].tolist()
                bbox_xyxy = bbox_xywh_to_xyxy(bbox_xywh)
                padded = compute_padded_bbox(bbox_xyxy, padding_frac, img_size, min_dim)

                if padded is None:
                    continue

                # Crop and resize
                crop_resized = crop_and_resize(img, padded, crop_size)

                # Convert to 3-channel for YOLO
                crop_3ch = cv2.cvtColor(crop_resized, cv2.COLOR_GRAY2BGR)

                # Stage 2: Stenosis detection on crop
                stenosis_results = stenosis_model.predict(
                    source=crop_3ch,
                    conf=stenosis_conf,
                    imgsz=stenosis_imgsz,
                    save=False,
                    verbose=False,
                    retina_masks=True,
                )

                for s_result in stenosis_results:
                    if s_result.masks is None or len(s_result.masks) == 0:
                        continue

                    for j in range(len(s_result.masks)):
                        conf = float(s_result.boxes.conf[j].item())
                        polygon_norm = s_result.masks.xyn[j].tolist()

                        # Remap to original image coords
                        polygon_original = remap_crop_to_original(
                            polygon_norm, padded, img_size
                        )

                        if len(polygon_original) >= 3:
                            image_stenoses.append((polygon_original, conf))
                            total_stenoses += 1

        # NMS across all crops for this image
        before_nms = len(image_stenoses)
        image_stenoses = polygon_nms(image_stenoses, nms_iou)
        all_predictions[image_name] = image_stenoses

    images_with_preds = sum(1 for v in all_predictions.values() if v)
    final_count = sum(len(v) for v in all_predictions.values())
    print(f"  Vessels processed:  {total_vessels}")
    print(f"  Raw stenoses found: {total_stenoses}")
    print(f"  After NMS:          {final_count}")
    print(f"  Images with preds:  {images_with_preds}/{len(image_files)}")

    return all_predictions


# ---------------------------------------------------------------------------
# Evaluation (ARCADE F1)
# ---------------------------------------------------------------------------

def evaluate_crop_predictions(predictions, coco_json_path):
    """Evaluate crop-based predictions using ARCADE polygon F1.

    Args:
        predictions: {image_name: [(polygon_pixel_coords, confidence), ...]}
        coco_json_path: Path to stenosis COCO JSON with GT annotations.

    Returns:
        dict with overall_mean_f1, per_class metrics, and per-image breakdown.
    """
    with open(coco_json_path) as f:
        coco = json.load(f)

    images_by_id = {img["id"]: img for img in coco["images"]}

    # Group GT by image
    gt_by_image = defaultdict(list)
    for ann in coco["annotations"]:
        img_info = images_by_id.get(ann["image_id"])
        if img_info is None:
            continue
        fname = img_info["file_name"]
        w, h = img_info["width"], img_info["height"]

        for seg in ann.get("segmentation", []):
            if len(seg) < 6:
                continue
            gt_poly = coords_to_shapely(seg, w, h, normalized=False)
            if not gt_poly.is_empty:
                gt_by_image[fname].append(gt_poly)

    per_image_f1 = []
    detailed = []

    # Union of all image names
    all_image_names = set(gt_by_image.keys()) | set(predictions.keys())

    for image_name in sorted(all_image_names):
        gt_polys = gt_by_image.get(image_name, [])
        pred_list = predictions.get(image_name, [])

        # Convert predictions to Shapely
        pred_polys = []
        for coords, _conf in pred_list:
            p = coords_to_shapely(coords, normalized=False)
            if not p.is_empty:
                pred_polys.append(p)

        if not gt_polys and not pred_polys:
            continue

        # Compute F1: for each pred, find best GT match
        pred_scores = []
        for pred_p in pred_polys:
            best_f1 = 0.0
            for gt_p in gt_polys:
                f1 = shapely_f1(pred_p, gt_p)
                best_f1 = max(best_f1, f1)
            pred_scores.append(best_f1)

        # Pad with zeros for unmatched GT
        num_unmatched_gt = max(0, len(gt_polys) - len(pred_polys))
        all_scores = pred_scores + [0.0] * num_unmatched_gt

        if all_scores:
            mean_f1 = float(np.mean(all_scores))
            per_image_f1.append(mean_f1)
            detailed.append({
                "image": image_name,
                "f1": round(mean_f1, 4),
                "n_gt": len(gt_polys),
                "n_pred": len(pred_polys),
            })

    overall_f1 = float(np.mean(per_image_f1)) if per_image_f1 else 0.0

    # Build metrics in the same format as evaluate.py's evaluate_model()
    metrics = {
        "split": "test",
        "overall_mean_f1": round(overall_f1, 4),
        "num_images_evaluated": len(per_image_f1),
        "num_images_with_gt": len(gt_by_image),
        "total_predictions": sum(len(v) for v in predictions.values()),
        "per_class": {
            "stenosis": {
                "f1": round(overall_f1, 4),
                "ap50": round(overall_f1, 4),  # approximate — F1 is the primary metric
            }
        },
    }

    return metrics, detailed


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Two-stage crop-based stenosis inference and evaluation"
    )
    parser.add_argument(
        "--syntax-weights", type=str, required=True,
        help="Path to trained syntax model weights (.pt)"
    )
    parser.add_argument(
        "--stenosis-weights", type=str, required=True,
        help="Path to trained crop-stenosis model weights (.pt)"
    )
    parser.add_argument(
        "--arcade-root", type=str, default="../../arcade/submission",
        help="Path to arcade/submission directory"
    )
    parser.add_argument(
        "--data-dir", type=str, default="../data",
        help="Pipeline data directory (for prepared images)"
    )
    parser.add_argument(
        "--split", type=str, default="test", choices=["train", "val", "test"],
        help="Which split to evaluate (default: test)"
    )
    parser.add_argument(
        "--padding", type=float, default=0.20,
        help="Padding around vessel bbox (default: 0.20)"
    )
    parser.add_argument(
        "--crop-size", type=int, default=512,
        help="Crop resize dimension (default: 512)"
    )
    parser.add_argument(
        "--vessel-conf", type=float, default=0.15,
        help="Vessel model confidence threshold (default: 0.15)"
    )
    parser.add_argument(
        "--stenosis-conf", type=float, default=0.25,
        help="Stenosis model confidence threshold (default: 0.25)"
    )
    parser.add_argument(
        "--nms-iou", type=float, default=0.5,
        help="NMS IoU threshold (default: 0.5)"
    )
    parser.add_argument(
        "--vessel-imgsz", type=int, default=768,
        help="Image size for vessel model (default: 768)"
    )
    parser.add_argument(
        "--stenosis-imgsz", type=int, default=512,
        help="Image size for stenosis model (default: 512)"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Path to save evaluation results JSON"
    )
    parser.add_argument(
        "--save-predictions", action="store_true",
        help="Save per-image predictions to JSON"
    )
    args = parser.parse_args()

    arcade_root = Path(args.arcade_root).resolve()
    data_dir = Path(args.data_dir).resolve()

    # Find images
    images_dir = data_dir / "stenosis" / "images" / args.split
    if not images_dir.exists():
        images_dir = arcade_root / "stenosis" / args.split / "images"
    if not images_dir.exists():
        raise FileNotFoundError(f"Images not found in {images_dir}")

    # Find COCO GT annotations
    coco_path = arcade_root / "stenosis" / args.split / "annotations" / f"{args.split}.json"
    if not coco_path.exists():
        raise FileNotFoundError(f"COCO annotations not found: {coco_path}")

    for name, path in [("Syntax", args.syntax_weights),
                       ("Stenosis", args.stenosis_weights)]:
        if not Path(path).exists():
            raise FileNotFoundError(f"{name} weights not found: {path}")

    print("=" * 60)
    print(f"Crop-based stenosis inference ({args.split})")
    print("=" * 60)
    print(f"  Syntax model:   {args.syntax_weights}")
    print(f"  Stenosis model:  {args.stenosis_weights}")
    print(f"  Images:          {images_dir}")
    print(f"  Padding:         {args.padding}")
    print(f"  Crop size:       {args.crop_size}")

    # Run inference
    predictions = run_crop_inference(
        syntax_weights=args.syntax_weights,
        stenosis_weights=args.stenosis_weights,
        images_dir=str(images_dir),
        padding_frac=args.padding,
        crop_size=args.crop_size,
        vessel_conf=args.vessel_conf,
        stenosis_conf=args.stenosis_conf,
        nms_iou=args.nms_iou,
        vessel_imgsz=args.vessel_imgsz,
        stenosis_imgsz=args.stenosis_imgsz,
    )

    # Evaluate
    print(f"\n{'='*60}")
    print(f"Evaluating ({args.split})")
    print(f"{'='*60}")

    metrics, detailed = evaluate_crop_predictions(predictions, str(coco_path))

    print(f"  Overall Mean F1:     {metrics['overall_mean_f1']}")
    print(f"  Images evaluated:    {metrics['num_images_evaluated']}")
    print(f"  Total predictions:   {metrics['total_predictions']}")
    print(f"{'='*60}")

    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        results = {**metrics, "detailed": detailed}
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n  Saved: {output_path}")

    if args.save_predictions:
        preds_path = Path(args.output or "crop_predictions.json").with_suffix(
            ".predictions.json"
        )
        serializable = {}
        for img_name, preds in predictions.items():
            serializable[img_name] = [
                {"polygon": coords, "confidence": round(conf, 4)}
                for coords, conf in preds
            ]
        with open(preds_path, "w") as f:
            json.dump(serializable, f, indent=2)
        print(f"  Predictions saved: {preds_path}")

    return metrics


if __name__ == "__main__":
    main()
