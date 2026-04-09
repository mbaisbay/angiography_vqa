"""Two-stage crop-based stenosis inference and evaluation.

Stage 1: Run syntax (vessel) model on full images to get vessel bboxes.
Stage 2: Crop each vessel bbox, run stenosis model on each crop,
         remap predictions back to original image coordinates.
Evaluate using ARCADE-compatible polygon F1.
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
from shapely.geometry import Polygon
from shapely.validation import make_valid

from ultralytics import YOLO

from utils.config_loader import load_config, get_inference_args
from evaluate import shapely_f1, coords_to_shapely
from build_crop_dataset import bbox_xywh_to_xyxy, pad_and_clip_bbox


# ---------------------------------------------------------------------------
# Coordinate remapping (crop → original)
# ---------------------------------------------------------------------------

def remap_crop_to_original(polygon_normalized, crop_bbox_xyxy, img_size=512):
    """Remap normalized crop-space polygon back to original image pixel coords.

    Args:
        polygon_normalized: List of [x, y] pairs normalized to [0, 1] in crop space.
        crop_bbox_xyxy: [x1, y1, x2, y2] of the crop in original image.
        img_size: Original image size.

    Returns:
        List of [x, y] pairs in original image pixel coords.
    """
    x1, y1, x2, y2 = crop_bbox_xyxy
    crop_w = x2 - x1
    crop_h = y2 - y1

    remapped = []
    for nx, ny in polygon_normalized:
        px = x1 + nx * crop_w
        py = y1 + ny * crop_h
        # Clip to image bounds
        px = max(0, min(img_size, px))
        py = max(0, min(img_size, py))
        remapped.append([px, py])

    return remapped


# ---------------------------------------------------------------------------
# Polygon NMS
# ---------------------------------------------------------------------------

def polygon_nms(predictions, iou_threshold=0.5):
    """Non-maximum suppression for polygon predictions.

    Args:
        predictions: List of (polygon_pixel_coords, confidence) tuples.
        iou_threshold: IoU threshold for suppression.

    Returns:
        Filtered list of (polygon_pixel_coords, confidence) tuples.
    """
    if len(predictions) <= 1:
        return predictions

    # Sort by confidence (descending)
    predictions = sorted(predictions, key=lambda x: x[1], reverse=True)

    # Build Shapely polygons
    polys = []
    for coords, conf in predictions:
        try:
            p = Polygon(coords)
            if not p.is_valid:
                p = p.buffer(0)
            polys.append(p)
        except Exception:
            polys.append(Polygon())

    keep = []
    suppressed = set()

    for i in range(len(predictions)):
        if i in suppressed:
            continue
        keep.append(predictions[i])

        for j in range(i + 1, len(predictions)):
            if j in suppressed:
                continue
            if polys[i].is_empty or polys[j].is_empty:
                continue
            try:
                intersection = polys[i].intersection(polys[j]).area
                union = polys[i].union(polys[j]).area
                if union > 0 and intersection / union >= iou_threshold:
                    suppressed.add(j)
            except Exception:
                continue

    return keep


# ---------------------------------------------------------------------------
# Two-stage inference
# ---------------------------------------------------------------------------

def run_crop_inference(config, split="test"):
    """Run two-stage crop-based stenosis inference.

    Returns:
        dict: {image_name: [(polygon_pixel_coords, confidence), ...]}
    """
    cp = config["crop_pipeline"]
    padding_frac = cp["padding_frac"]
    crop_size = cp["crop_size"]
    min_bbox_px = cp["min_vessel_bbox_px"]
    nms_iou = cp["nms_iou_threshold"]
    syntax_weights = config["cross_inference"]["syntax_weights"]
    stenosis_weights = cp["stenosis_crop_weights"]
    dataset_root = Path(config["dataset_root"])
    img_size = 512

    images_dir = dataset_root / "stenosis" / split / "images"
    if not images_dir.exists():
        raise FileNotFoundError(f"Images not found: {images_dir}")

    # Load models
    print(f"Loading syntax model: {syntax_weights}")
    syntax_model = YOLO(syntax_weights)
    print(f"Loading stenosis crop model: {stenosis_weights}")
    stenosis_model = YOLO(stenosis_weights)

    # Inference args
    vessel_inf_args = get_inference_args(config)
    vessel_inf_args["conf"] = cp["vessel_conf_threshold"]

    stenosis_inf_args = get_inference_args(config)
    stenosis_inf_args["conf"] = cp["stenosis_conf_threshold"]

    image_files = sorted(images_dir.glob("*.png")) + sorted(images_dir.glob("*.PNG"))
    all_predictions = {}

    print(f"\nRunning two-stage inference on {len(image_files)} {split} images...")

    for img_path in image_files:
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        image_name = img_path.name

        # Stage 1: Vessel detection
        vessel_results = syntax_model.predict(
            source=str(img_path), save=False, verbose=False, **vessel_inf_args
        )

        image_stenoses = []

        for result in vessel_results:
            if result.masks is None or len(result.masks) == 0:
                continue

            for i in range(len(result.masks)):
                bbox_xywh = result.boxes.xywh[i].tolist()
                bbox_xyxy = bbox_xywh_to_xyxy(bbox_xywh)
                padded = pad_and_clip_bbox(bbox_xyxy, padding_frac, img_size)
                x1, y1, x2, y2 = padded
                crop_w = x2 - x1
                crop_h = y2 - y1

                if crop_w < min_bbox_px or crop_h < min_bbox_px:
                    continue

                # Crop and resize
                crop = img[y1:y2, x1:x2]
                crop_resized = cv2.resize(crop, (crop_size, crop_size),
                                          interpolation=cv2.INTER_LINEAR)

                # Stage 2: Stenosis detection on crop
                # Convert single-channel to 3-channel for YOLO
                crop_3ch = cv2.cvtColor(crop_resized, cv2.COLOR_GRAY2BGR)
                stenosis_results = stenosis_model.predict(
                    source=crop_3ch, save=False, verbose=False, **stenosis_inf_args
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

        # NMS across all crops
        image_stenoses = polygon_nms(image_stenoses, nms_iou)
        all_predictions[image_name] = image_stenoses

    total_preds = sum(len(v) for v in all_predictions.values())
    images_with_preds = sum(1 for v in all_predictions.values() if v)
    print(f"  Total predictions: {total_preds} across {images_with_preds} images")

    return all_predictions


# ---------------------------------------------------------------------------
# Evaluation (ARCADE F1)
# ---------------------------------------------------------------------------

def evaluate_crop_predictions(predictions, coco_json_path, split="test"):
    """Evaluate crop-based predictions using ARCADE polygon F1.

    Args:
        predictions: {image_name: [(polygon_pixel_coords, confidence), ...]}
        coco_json_path: Path to stenosis COCO JSON with GT annotations.

    Returns:
        dict with overall_mean_f1 and per-image breakdown.
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

    # Evaluate all images that have GT
    all_image_names = set(gt_by_image.keys())
    # Also include images with predictions but no GT (for tracking false positives)
    all_image_names.update(predictions.keys())

    for image_name in sorted(all_image_names):
        gt_polys = gt_by_image.get(image_name, [])
        pred_list = predictions.get(image_name, [])

        # Convert predictions to Shapely
        pred_polys = []
        for coords, conf in pred_list:
            try:
                p = Polygon(coords)
                if not p.is_valid:
                    p = make_valid(p)
                if not p.is_empty:
                    pred_polys.append(p)
            except Exception:
                continue

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

    return {
        "overall_mean_f1": round(overall_f1, 4),
        "num_images_evaluated": len(per_image_f1),
        "num_images_with_gt": len(gt_by_image),
        "total_predictions": sum(len(v) for v in predictions.values()),
        "detailed": detailed,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Two-stage crop-based stenosis inference and evaluation"
    )
    parser.add_argument(
        "--config", type=str, default="config.yaml",
        help="Path to pipeline config YAML (default: config.yaml)"
    )
    parser.add_argument(
        "--split", type=str, default="test", choices=["train", "val", "test"],
        help="Which split to evaluate (default: test)"
    )
    parser.add_argument(
        "--save-predictions", action="store_true",
        help="Save per-image predictions to JSON"
    )
    args = parser.parse_args()

    config = load_config(args.config)
    cp = config["crop_pipeline"]
    dataset_root = Path(config["dataset_root"])

    # Verify weights exist
    syntax_weights = config["cross_inference"]["syntax_weights"]
    stenosis_weights = cp["stenosis_crop_weights"]
    for name, path in [("Syntax", syntax_weights), ("Stenosis crop", stenosis_weights)]:
        if not Path(path).exists():
            raise FileNotFoundError(f"{name} weights not found: {path}")

    # Run inference
    predictions = run_crop_inference(config, args.split)

    # Evaluate
    annotations_path = dataset_root / "stenosis" / args.split / "annotations" / f"{args.split}.json"
    if not annotations_path.exists():
        print(f"[ERROR] Annotations not found: {annotations_path}")
        return

    print(f"\n{'='*60}")
    print(f"Evaluating crop-based stenosis predictions ({args.split})")
    print(f"{'='*60}")

    results = evaluate_crop_predictions(predictions, str(annotations_path), args.split)

    print(f"  Overall Mean F1:     {results['overall_mean_f1']}")
    print(f"  Images evaluated:    {results['num_images_evaluated']}")
    print(f"  Images with GT:      {results['num_images_with_gt']}")
    print(f"  Total predictions:   {results['total_predictions']}")
    print(f"{'='*60}")

    # Save results
    output_dir = Path(config["output_dir"]) / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)

    results_file = output_dir / f"stenosis_crop_{args.split}_metrics.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved: {results_file}")

    # Optionally save raw predictions
    if args.save_predictions:
        preds_file = output_dir / f"stenosis_crop_{args.split}_predictions.json"
        serializable = {}
        for img_name, preds in predictions.items():
            serializable[img_name] = [
                {"polygon": coords, "confidence": round(conf, 4)}
                for coords, conf in preds
            ]
        with open(preds_file, "w") as f:
            json.dump(serializable, f, indent=2)
        print(f"  Predictions saved: {preds_file}")


if __name__ == "__main__":
    main()
