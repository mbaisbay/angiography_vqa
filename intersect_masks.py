"""Overlay vessel segment masks with stenosis masks to determine vessel-stenosis mapping."""

import argparse
import json
from pathlib import Path

import cv2

from utils.config_loader import load_config
from utils.coco_to_yolo import load_category_mapping
from utils.mask_utils import assign_stenosis_to_vessels
from utils.visualization import draw_assignment_visualization


def load_cross_inference_results(results_path: str) -> dict:
    """Load cross-inference JSON results."""
    with open(results_path, "r") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Intersect vessel segment masks with stenosis masks"
    )
    parser.add_argument(
        "--config", type=str, default="config.yaml",
        help="Path to pipeline config YAML (default: config.yaml)"
    )
    args = parser.parse_args()

    config = load_config(args.config)
    inter = config["intersection"]
    ci_dir = Path(config["cross_inference"]["output_dir"])
    dataset_root = Path(config["dataset_root"])

    mask_size = inter["mask_size"]
    threshold = inter["iou_threshold_for_match"]
    metric = inter["metric"]
    save_vis = inter["save_overlay_images"]
    results_dir = Path(inter["results_output_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    if save_vis:
        overlay_dir = Path(inter["overlay_output_dir"])
        overlay_dir.mkdir(parents=True, exist_ok=True)

    # Load cross-inference results
    # syntax_on_stenosis: vessel segment predictions on stenosis images
    # stenosis_on_syntax: stenosis predictions on syntax images
    syntax_on_stenosis_path = ci_dir / "syntax_on_stenosis.json"
    stenosis_on_syntax_path = ci_dir / "stenosis_on_syntax.json"

    # Load category mappings if available (supports filtered 13-class scheme)
    mappings_dir = Path(config["mappings_dir"])
    stenosis_mapping = None
    syntax_mapping = None
    for task, var_name in [("stenosis", "stenosis_mapping"), ("syntax", "syntax_mapping")]:
        mapping_path = mappings_dir / f"{task}_categories.json"
        if mapping_path.exists():
            mapping = load_category_mapping(str(mapping_path))
            if var_name == "stenosis_mapping":
                stenosis_mapping = mapping
            else:
                syntax_mapping = mapping

    print("=" * 60)
    print("Mask Intersection: Vessel-Stenosis Assignment")
    print("=" * 60)
    print(f"  Metric: {metric}")
    print(f"  Threshold: {threshold}")
    print(f"  Mask size: {mask_size}x{mask_size}")

    all_assignments = {}

    # --- Direction 1: Stenosis images with vessel predictions ---
    # Use ground-truth or model stenosis + cross-inferred vessel segments
    if syntax_on_stenosis_path.exists():
        print(f"\n  Processing stenosis images with vessel predictions...")
        vessel_results = load_cross_inference_results(str(syntax_on_stenosis_path))

        # Also need stenosis predictions on these same images
        # If stenosis_on_syntax doesn't cover these, use GT or the stenosis model directly
        # For now, load stenosis GT from COCO JSON for stenosis test images
        for split, split_results in vessel_results.items():
            # Build lookup: image_name -> vessel predictions
            vessel_by_image = {r["image_name"]: r["predictions"] for r in split_results}

            # Load stenosis GT annotations for this split
            stenosis_gt = load_stenosis_gt(dataset_root, split, stenosis_mapping)

            split_assignments = []
            for image_name, vessel_preds in vessel_by_image.items():
                stenosis_preds = stenosis_gt.get(image_name, [])

                if not stenosis_preds or not vessel_preds:
                    continue

                result = assign_stenosis_to_vessels(
                    stenosis_preds, vessel_preds,
                    mask_size=mask_size,
                    threshold=threshold,
                    metric=metric,
                )
                result["image_name"] = image_name
                result["split"] = split
                split_assignments.append(result)

                # Save visualization
                if save_vis and (result["summary"]["matched_count"] > 0):
                    image_path = dataset_root / "stenosis" / split / "images" / image_name
                    if image_path.exists():
                        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
                        vis = draw_assignment_visualization(
                            image, vessel_preds, stenosis_preds, result
                        )
                        vis_path = overlay_dir / f"{split}_{image_name}"
                        cv2.imwrite(str(vis_path), vis)

            all_assignments[f"stenosis_images_{split}"] = split_assignments
            matched = sum(a["summary"]["matched_count"] for a in split_assignments)
            total = sum(a["summary"]["total_stenoses"] for a in split_assignments)
            print(f"    [{split.upper()}] {matched}/{total} stenoses matched to vessels "
                  f"across {len(split_assignments)} images")

    # --- Direction 2: Syntax images with stenosis predictions ---
    if stenosis_on_syntax_path.exists():
        print(f"\n  Processing syntax images with stenosis predictions...")
        stenosis_results = load_cross_inference_results(str(stenosis_on_syntax_path))

        # Load vessel GT from COCO JSON for syntax images
        for split, split_results in stenosis_results.items():
            stenosis_by_image = {r["image_name"]: r["predictions"] for r in split_results}

            vessel_gt = load_vessel_gt(dataset_root, split, syntax_mapping)

            split_assignments = []
            for image_name, stenosis_preds in stenosis_by_image.items():
                vessel_preds = vessel_gt.get(image_name, [])

                if not stenosis_preds or not vessel_preds:
                    continue

                result = assign_stenosis_to_vessels(
                    stenosis_preds, vessel_preds,
                    mask_size=mask_size,
                    threshold=threshold,
                    metric=metric,
                )
                result["image_name"] = image_name
                result["split"] = split
                split_assignments.append(result)

                if save_vis and (result["summary"]["matched_count"] > 0):
                    image_path = dataset_root / "syntax" / split / "images" / image_name
                    if image_path.exists():
                        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
                        vis = draw_assignment_visualization(
                            image, vessel_preds, stenosis_preds, result
                        )
                        vis_path = overlay_dir / f"syntax_{split}_{image_name}"
                        cv2.imwrite(str(vis_path), vis)

            all_assignments[f"syntax_images_{split}"] = split_assignments
            matched = sum(a["summary"]["matched_count"] for a in split_assignments)
            total = sum(a["summary"]["total_stenoses"] for a in split_assignments)
            print(f"    [{split.upper()}] {matched}/{total} stenoses matched to vessels "
                  f"across {len(split_assignments)} images")

    # Save all assignments
    output_file = results_dir / "assignments.json"
    with open(output_file, "w") as f:
        json.dump(all_assignments, f, indent=2)
    print(f"\n  Saved assignments: {output_file}")

    if save_vis:
        print(f"  Saved overlays: {overlay_dir}")

    print(f"\n{'='*60}")
    print("Intersection complete!")
    print(f"{'='*60}")


def load_stenosis_gt(dataset_root: Path, split: str,
                     category_mapping: dict = None) -> dict:
    """Load stenosis ground truth from COCO JSON and convert to prediction format.

    Returns dict: {image_name: [list of prediction-like dicts]}.
    """
    return _load_gt_as_predictions(dataset_root / "stenosis", split, category_mapping)


def load_vessel_gt(dataset_root: Path, split: str,
                   category_mapping: dict = None) -> dict:
    """Load vessel segment ground truth from COCO JSON and convert to prediction format."""
    return _load_gt_as_predictions(dataset_root / "syntax", split, category_mapping)


def _load_gt_as_predictions(task_dir: Path, split: str,
                            category_mapping: dict = None) -> dict:
    """Load COCO GT annotations and format like model predictions for mask intersection.

    Args:
        task_dir: Path to task directory (e.g., arcade/submission/syntax).
        split: Dataset split (train/val/test).
        category_mapping: Optional pre-built mapping with 'coco_to_yolo' dict.
            If provided, uses this instead of building a 26-class mapping from
            the COCO JSON. This supports the filtered 13-class scheme.
    """
    ann_dir = task_dir / split / "annotations"
    json_candidates = list(ann_dir.glob("*.json")) + list(ann_dir.glob("*.JSON"))

    if not json_candidates:
        return {}

    with open(json_candidates[0], "r") as f:
        coco = json.load(f)

    images_by_id = {img["id"]: img for img in coco["images"]}
    cats_by_id = {cat["id"]: str(cat["name"]) for cat in coco["categories"]}

    # Use provided mapping if available, otherwise build from COCO JSON
    if category_mapping is not None:
        coco_id_to_yolo = category_mapping["coco_to_yolo"]
    else:
        sorted_cat_ids = sorted(cats_by_id.keys())
        coco_id_to_yolo = {cid: idx for idx, cid in enumerate(sorted_cat_ids)}

    # Group annotations by image
    preds_by_image = {}
    for ann in coco["annotations"]:
        img_info = images_by_id.get(ann["image_id"])
        if img_info is None:
            continue

        coco_cat_id = ann["category_id"]
        # Skip annotations for classes not in the mapping (filtered out)
        if coco_cat_id not in coco_id_to_yolo:
            continue

        image_name = img_info["file_name"]
        w, h = img_info["width"], img_info["height"]

        for polygon in ann.get("segmentation", []):
            if len(polygon) < 6:
                continue

            # Normalize polygon
            normalized = []
            for i in range(0, len(polygon), 2):
                normalized.append([polygon[i] / w, polygon[i + 1] / h])

            pred = {
                "class_id": coco_id_to_yolo[coco_cat_id],
                "class_name": cats_by_id.get(coco_cat_id, str(coco_cat_id)),
                "confidence": 1.0,
                "bbox_xywh": ann.get("bbox", []),
                "polygon_normalized": normalized,
            }

            if image_name not in preds_by_image:
                preds_by_image[image_name] = []
            preds_by_image[image_name].append(pred)

    return preds_by_image


if __name__ == "__main__":
    main()
