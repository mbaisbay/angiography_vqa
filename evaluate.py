"""Evaluation metrics for syntax, stenosis, and intersection results."""

import argparse
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
from shapely.geometry import Polygon
from shapely.validation import make_valid

from ultralytics import YOLO

from utils.config_loader import load_config, get_inference_args
from utils.coco_to_yolo import load_category_mapping


def shapely_f1(pred_polygon: Polygon, gt_polygon: Polygon) -> float:
    """Compute area-based F1 between two Shapely polygons (ARCADE-compatible)."""
    if not pred_polygon.is_valid:
        pred_polygon = make_valid(pred_polygon)
    if not gt_polygon.is_valid:
        gt_polygon = make_valid(gt_polygon)

    intersection = pred_polygon.intersection(gt_polygon).area
    pred_area = pred_polygon.area
    gt_area = gt_polygon.area

    if pred_area + gt_area == 0:
        return 0.0

    precision = intersection / pred_area if pred_area > 0 else 0.0
    recall = intersection / gt_area if gt_area > 0 else 0.0

    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def coords_to_shapely(polygon_coords: list, width: int = 512, height: int = 512,
                       normalized: bool = True) -> Polygon:
    """Convert polygon coordinates to a Shapely Polygon.

    Args:
        polygon_coords: List of [x, y] pairs (if normalized=True) or flat list [x1,y1,x2,y2,...].
        width, height: Image dimensions for denormalization.
        normalized: Whether coordinates are normalized to [0, 1].
    """
    if not polygon_coords or len(polygon_coords) < 3:
        return Polygon()

    # Handle flat list format
    if isinstance(polygon_coords[0], (int, float)):
        pairs = []
        for i in range(0, len(polygon_coords), 2):
            if i + 1 < len(polygon_coords):
                pairs.append([polygon_coords[i], polygon_coords[i + 1]])
        polygon_coords = pairs

    if len(polygon_coords) < 3:
        return Polygon()

    if normalized:
        points = [(x * width, y * height) for x, y in polygon_coords]
    else:
        points = [(x, y) for x, y in polygon_coords]

    try:
        poly = Polygon(points)
        if not poly.is_valid:
            poly = make_valid(poly)
        return poly
    except Exception:
        return Polygon()


def evaluate_segmentation_arcade(model_path: str, coco_json_path: str,
                                  images_dir: str, category_mapping: dict,
                                  inference_args: dict) -> dict:
    """Evaluate using ARCADE-compatible polygon area F1 metric.

    For each predicted mask, compute F1 against all GT masks of the same class
    for that image, take the maximum. Pad with zeros for unmatched GT masks.
    Mean F1 per image, then overall mean.
    """
    model = YOLO(model_path)

    # Load GT
    with open(coco_json_path, "r") as f:
        coco = json.load(f)

    images_by_id = {img["id"]: img for img in coco["images"]}
    cats_by_id = {cat["id"]: str(cat["name"]) for cat in coco["categories"]}

    # Group GT annotations by image and class
    gt_by_image = defaultdict(lambda: defaultdict(list))
    for ann in coco["annotations"]:
        img_info = images_by_id.get(ann["image_id"])
        if img_info is None:
            continue
        image_name = img_info["file_name"]
        cat_name = cats_by_id.get(ann["category_id"], str(ann["category_id"]))
        w, h = img_info["width"], img_info["height"]

        for polygon in ann.get("segmentation", []):
            if len(polygon) < 6:
                continue
            gt_poly = coords_to_shapely(polygon, w, h, normalized=False)
            if gt_poly.is_empty:
                continue
            gt_by_image[image_name][cat_name].append(gt_poly)

    # Build image dimensions lookup from COCO JSON
    image_dims = {}
    for img in coco["images"]:
        image_dims[img["file_name"]] = (img["width"], img["height"])

    # Run predictions
    yolo_to_name = category_mapping["yolo_to_name"]
    per_image_f1 = []
    per_class_f1 = defaultdict(list)

    print(f"    Running predictions on {images_dir}...")
    for result in model.predict(
        source=images_dir, stream=True, save=False, verbose=False, **inference_args
    ):
        image_name = Path(result.path).name
        gt_classes = gt_by_image.get(image_name, {})

        # Use actual image dimensions from COCO JSON (fallback to 512x512)
        img_w, img_h = image_dims.get(image_name, (512, 512))

        # Group predictions by class
        pred_by_class = defaultdict(list)
        if result.masks is not None and len(result.masks) > 0:
            for i in range(len(result.masks)):
                cls_id = int(result.boxes.cls[i].item())
                cls_name = yolo_to_name.get(cls_id, str(cls_id))
                polygon = result.masks.xyn[i].tolist()
                pred_poly = coords_to_shapely(polygon, img_w, img_h, normalized=True)
                if not pred_poly.is_empty:
                    pred_by_class[cls_name].append(pred_poly)

        # Compute F1 per class for this image
        all_classes = set(list(gt_classes.keys()) + list(pred_by_class.keys()))
        image_scores = []

        for cls_name in all_classes:
            gt_polys = gt_classes.get(cls_name, [])
            pred_polys = pred_by_class.get(cls_name, [])

            if not gt_polys and not pred_polys:
                continue

            # For each predicted mask, find best F1 with any GT of same class
            pred_scores = []
            for pred_p in pred_polys:
                best_f1 = 0.0
                for gt_p in gt_polys:
                    f1 = shapely_f1(pred_p, gt_p)
                    best_f1 = max(best_f1, f1)
                pred_scores.append(best_f1)

            # Pad with zeros for unmatched GT masks
            num_unmatched_gt = max(0, len(gt_polys) - len(pred_polys))
            all_scores = pred_scores + [0.0] * num_unmatched_gt

            if all_scores:
                mean_f1 = np.mean(all_scores)
                image_scores.append(mean_f1)
                per_class_f1[cls_name].append(mean_f1)

        if image_scores:
            per_image_f1.append(np.mean(image_scores))

    # Aggregate
    overall_f1 = float(np.mean(per_image_f1)) if per_image_f1 else 0.0
    class_results = {}
    for cls_name, scores in sorted(per_class_f1.items()):
        class_results[cls_name] = {
            "mean_f1": round(float(np.mean(scores)), 4),
            "std_f1": round(float(np.std(scores)), 4),
            "num_images": len(scores),
        }

    return {
        "overall_mean_f1": round(overall_f1, 4),
        "num_images_evaluated": len(per_image_f1),
        "per_class": class_results,
    }


def evaluate_ultralytics(model_path: str, data_yaml: str) -> dict:
    """Run standard ultralytics validation for mAP metrics."""
    model = YOLO(model_path)
    results = model.val(data=data_yaml, verbose=False)

    metrics = {
        "mAP50": round(float(results.seg.map50), 4),
        "mAP50_95": round(float(results.seg.map), 4),
        "precision": round(float(results.seg.mp), 4),
        "recall": round(float(results.seg.mr), 4),
    }

    # Per-class AP if available
    if hasattr(results.seg, "ap50") and results.seg.ap50 is not None:
        metrics["per_class_ap50"] = {
            str(i): round(float(v), 4) for i, v in enumerate(results.seg.ap50)
        }

    return metrics


def evaluate_intersection(assignments_path: str) -> dict:
    """Compute descriptive statistics for the intersection results."""
    with open(assignments_path, "r") as f:
        assignments = json.load(f)

    total_stenoses = 0
    total_matched = 0
    total_unmatched = 0
    segment_counts = defaultdict(int)
    overlap_scores = []

    for split_key, split_results in assignments.items():
        for result in split_results:
            summary = result.get("summary", {})
            total_stenoses += summary.get("total_stenoses", 0)
            total_matched += summary.get("matched_count", 0)
            total_unmatched += summary.get("unmatched_count", 0)

            for match in result.get("matched", []):
                seg = match.get("matched_vessel_segment", "unknown")
                segment_counts[seg] += 1
                overlap_scores.append(match.get("overlap_score", 0))

    return {
        "total_stenoses": total_stenoses,
        "matched": total_matched,
        "unmatched": total_unmatched,
        "match_rate": round(total_matched / max(total_stenoses, 1), 4),
        "mean_overlap_score": round(float(np.mean(overlap_scores)), 4) if overlap_scores else 0.0,
        "median_overlap_score": round(float(np.median(overlap_scores)), 4) if overlap_scores else 0.0,
        "affected_segment_distribution": dict(sorted(segment_counts.items())),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate ARCADE pipeline results")
    parser.add_argument(
        "--config", type=str, default="config.yaml",
        help="Path to pipeline config YAML (default: config.yaml)"
    )
    parser.add_argument(
        "--task", type=str, required=True,
        choices=["syntax", "stenosis", "intersection", "all"],
        help="What to evaluate"
    )
    parser.add_argument(
        "--method", type=str, default="both",
        choices=["ultralytics", "arcade", "both"],
        help="Evaluation method for syntax/stenosis (default: both)"
    )
    args = parser.parse_args()

    config = load_config(args.config)
    inference_args = get_inference_args(config)
    dataset_root = Path(config["dataset_root"])
    output_dir = Path(config["output_dir"])
    mappings_dir = Path(config["mappings_dir"])
    eval_dir = output_dir / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)

    tasks = ["syntax", "stenosis", "intersection"] if args.task == "all" else [args.task]

    for task in tasks:
        print(f"\n{'='*60}")
        print(f"Evaluating: {task.upper()}")
        print(f"{'='*60}")

        if task in ("syntax", "stenosis"):
            weights = config["cross_inference"][f"{task}_weights"]
            if not Path(weights).exists():
                print(f"  [ERROR] Weights not found: {weights}")
                continue

            mapping = load_category_mapping(str(mappings_dir / f"{task}_categories.json"))
            results = {}

            # Ultralytics standard metrics
            if args.method in ("ultralytics", "both"):
                print(f"\n  [Ultralytics Validation]")
                data_yaml = config[f"{task}_data_yaml"]
                ul_results = evaluate_ultralytics(weights, data_yaml)
                results["ultralytics"] = ul_results
                print(f"    mAP50:    {ul_results['mAP50']}")
                print(f"    mAP50-95: {ul_results['mAP50_95']}")
                print(f"    Precision: {ul_results['precision']}")
                print(f"    Recall:    {ul_results['recall']}")

            # ARCADE-compatible F1
            if args.method in ("arcade", "both"):
                print(f"\n  [ARCADE Polygon F1]")
                # Find test COCO JSON
                test_json = None
                for candidate in [
                    dataset_root / task / "test" / "annotations" / "test.json",
                    dataset_root / task / "test" / "annotations" / "test.JSON",
                ]:
                    if candidate.exists():
                        test_json = candidate
                        break

                if test_json is None:
                    print(f"    [ERROR] Test annotations not found")
                else:
                    images_dir = str(dataset_root / task / "test" / "images")
                    arcade_results = evaluate_segmentation_arcade(
                        weights, str(test_json), images_dir, mapping, inference_args
                    )
                    results["arcade_f1"] = arcade_results
                    print(f"    Overall Mean F1: {arcade_results['overall_mean_f1']}")
                    print(f"    Images evaluated: {arcade_results['num_images_evaluated']}")
                    print(f"\n    Per-class F1:")
                    for cls_name, cls_data in arcade_results["per_class"].items():
                        print(f"      {cls_name:>4s}: F1={cls_data['mean_f1']:.4f} "
                              f"(+/-{cls_data['std_f1']:.4f}, n={cls_data['num_images']})")

            # Save
            output_file = eval_dir / f"{task}_metrics.json"
            with open(output_file, "w") as f:
                json.dump(results, f, indent=2)
            print(f"\n  Saved: {output_file}")

        elif task == "intersection":
            assignments_path = Path(config["intersection"]["results_output_dir"]) / "assignments.json"
            if not assignments_path.exists():
                print(f"  [ERROR] Assignments not found: {assignments_path}")
                print(f"  Run intersect_masks.py first.")
                continue

            results = evaluate_intersection(str(assignments_path))
            print(f"  Total stenoses:  {results['total_stenoses']}")
            print(f"  Matched:         {results['matched']}")
            print(f"  Unmatched:       {results['unmatched']}")
            print(f"  Match rate:      {results['match_rate']:.1%}")
            print(f"  Mean overlap:    {results['mean_overlap_score']}")
            print(f"  Median overlap:  {results['median_overlap_score']}")
            print(f"\n  Affected segments:")
            for seg, count in results["affected_segment_distribution"].items():
                print(f"    Segment {seg:>4s}: {count} stenoses")

            output_file = eval_dir / "intersection_metrics.json"
            with open(output_file, "w") as f:
                json.dump(results, f, indent=2)
            print(f"\n  Saved: {output_file}")

    print(f"\n{'='*60}")
    print("Evaluation complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
