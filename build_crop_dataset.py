"""Build cropped vessel-ROI dataset for stenosis detection.

Runs the trained syntax (vessel) model on stenosis images, crops each
detected vessel bbox (+padding), remaps GT stenosis annotations to crop
coordinates, and outputs a YOLO-format dataset ready for training.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import yaml
from shapely.geometry import Polygon, box as shapely_box

from ultralytics import YOLO

from utils.config_loader import load_config, get_inference_args


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def bbox_xywh_to_xyxy(bbox_xywh):
    """Convert [cx, cy, w, h] to [x1, y1, x2, y2]."""
    cx, cy, w, h = bbox_xywh
    return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]


def pad_and_clip_bbox(bbox_xyxy, padding_frac, img_size):
    """Add fractional padding to bbox and clip to image bounds.

    Args:
        bbox_xyxy: [x1, y1, x2, y2] in pixel coords.
        padding_frac: Fraction of bbox dimension to add as padding.
        img_size: Image dimension (assumes square).

    Returns:
        [x1, y1, x2, y2] as integers, clipped to [0, img_size].
    """
    x1, y1, x2, y2 = bbox_xyxy
    w = x2 - x1
    h = y2 - y1
    pad_x = w * padding_frac
    pad_y = h * padding_frac
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(img_size, x2 + pad_x)
    y2 = min(img_size, y2 + pad_y)
    return [int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))]


def find_overlapping_stenoses(stenosis_anns, crop_bbox_xyxy):
    """Find stenosis annotations whose polygons overlap with the crop bbox.

    Args:
        stenosis_anns: List of dicts with 'segmentation' (pixel coords, flat list).
        crop_bbox_xyxy: [x1, y1, x2, y2] of the crop region.

    Returns:
        List of (annotation, clipped_polygon_pixels) tuples for overlapping stenoses.
    """
    x1, y1, x2, y2 = crop_bbox_xyxy
    crop_box = shapely_box(x1, y1, x2, y2)
    results = []

    for ann in stenosis_anns:
        for seg in ann.get("segmentation", []):
            if len(seg) < 6:
                continue
            # Convert flat list to (x, y) pairs
            points = [(seg[i], seg[i + 1]) for i in range(0, len(seg), 2)]
            if len(points) < 3:
                continue
            try:
                poly = Polygon(points)
                if not poly.is_valid:
                    poly = poly.buffer(0)
                if poly.is_empty:
                    continue
                intersection = poly.intersection(crop_box)
                if not intersection.is_empty and intersection.area > 0:
                    # Use the original polygon (not clipped) — YOLO will handle
                    # points outside [0,1] by clipping during training
                    results.append((ann, points))
            except Exception:
                continue

    return results


def remap_polygon_to_crop(polygon_pixels, crop_bbox_xyxy, crop_size):
    """Remap polygon from original image pixel coords to crop-normalized coords.

    Args:
        polygon_pixels: List of (x, y) tuples in original image pixel coords.
        crop_bbox_xyxy: [x1, y1, x2, y2] of the crop region.
        crop_size: Target crop size (for normalization).

    Returns:
        List of (x_norm, y_norm) tuples normalized to [0, 1] in crop space.
        Points are clipped to [0, 1].
    """
    x1, y1, x2, y2 = crop_bbox_xyxy
    crop_w = x2 - x1
    crop_h = y2 - y1

    if crop_w <= 0 or crop_h <= 0:
        return []

    remapped = []
    for px, py in polygon_pixels:
        # Translate to crop origin, then normalize
        nx = (px - x1) / crop_w
        ny = (py - y1) / crop_h
        # Clip to [0, 1]
        nx = max(0.0, min(1.0, nx))
        ny = max(0.0, min(1.0, ny))
        remapped.append((nx, ny))

    return remapped


def save_yolo_label(label_path, annotations):
    """Write YOLO segmentation label file.

    Args:
        label_path: Output .txt path.
        annotations: List of (class_id, [(x_norm, y_norm), ...]) tuples.
    """
    with open(label_path, "w") as f:
        for cls_id, polygon_norm in annotations:
            if len(polygon_norm) < 3:
                continue
            coords = " ".join(
                f"{x:.6f} {y:.6f}" for x, y in polygon_norm
            )
            f.write(f"{cls_id} {coords}\n")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def load_stenosis_gt(coco_json_path):
    """Load stenosis GT annotations grouped by image filename.

    Returns:
        dict: {image_filename: [annotation_dicts]} where annotations have
              'segmentation' in pixel coords (flat lists).
    """
    with open(coco_json_path) as f:
        coco = json.load(f)

    images_by_id = {img["id"]: img for img in coco["images"]}
    gt_by_image = {}

    for ann in coco["annotations"]:
        img_info = images_by_id.get(ann["image_id"])
        if img_info is None:
            continue
        fname = img_info["file_name"]
        if fname not in gt_by_image:
            gt_by_image[fname] = []
        gt_by_image[fname].append(ann)

    return gt_by_image


def run_vessel_predictions(model_path, images_dir, inference_args):
    """Run syntax model on images and return predictions per image.

    Returns:
        dict: {image_filename: [{'bbox_xywh': [...], 'polygon_normalized': [...], ...}]}
    """
    model = YOLO(model_path)
    results_dict = {}

    for result in model.predict(
        source=str(images_dir),
        stream=True,
        save=False,
        verbose=False,
        **inference_args,
    ):
        image_name = Path(result.path).name
        predictions = []

        if result.masks is not None and len(result.masks) > 0:
            for i in range(len(result.masks)):
                conf = float(result.boxes.conf[i].item())
                bbox = result.boxes.xywh[i].tolist()
                polygon = result.masks.xyn[i].tolist()

                predictions.append({
                    "confidence": conf,
                    "bbox_xywh": bbox,
                    "polygon_normalized": polygon,
                })

        results_dict[image_name] = predictions

    return results_dict


def build_crop_dataset(config):
    """Build the cropped vessel-ROI stenosis dataset."""
    cp = config["crop_pipeline"]
    padding_frac = cp["padding_frac"]
    crop_size = cp["crop_size"]
    min_bbox_px = cp["min_vessel_bbox_px"]
    crop_dataset_dir = Path(cp["crop_dataset_dir"])
    syntax_weights = config["cross_inference"]["syntax_weights"]
    dataset_root = Path(config["dataset_root"])
    img_size = 512  # ARCADE native

    # Inference args — use lower confidence for vessels to catch more
    inference_args = get_inference_args(config)
    inference_args["conf"] = cp["vessel_conf_threshold"]

    stats = {"total_crops": 0, "positive_crops": 0, "negative_crops": 0,
             "skipped_small": 0, "total_stenoses_remapped": 0}

    for split in ["train", "val", "test"]:
        print(f"\n{'='*60}")
        print(f"Building crop dataset: {split}")
        print(f"{'='*60}")

        images_dir = dataset_root / "stenosis" / split / "images"
        annotations_path = dataset_root / "stenosis" / split / "annotations" / f"{split}.json"

        if not images_dir.exists():
            print(f"  [SKIP] Images not found: {images_dir}")
            continue
        if not annotations_path.exists():
            print(f"  [SKIP] Annotations not found: {annotations_path}")
            continue

        # Output directories
        out_images = crop_dataset_dir / split / "images"
        out_labels = crop_dataset_dir / split / "labels"
        out_images.mkdir(parents=True, exist_ok=True)
        out_labels.mkdir(parents=True, exist_ok=True)

        # Load stenosis GT
        stenosis_gt = load_stenosis_gt(str(annotations_path))
        print(f"  Loaded stenosis GT for {len(stenosis_gt)} images")

        # Run vessel model on stenosis images
        print(f"  Running vessel model on {split} images...")
        vessel_preds = run_vessel_predictions(
            syntax_weights, str(images_dir), inference_args
        )
        print(f"  Got vessel predictions for {len(vessel_preds)} images")

        split_crops = 0
        split_positive = 0

        for image_name, vessels in sorted(vessel_preds.items()):
            img_path = images_dir / image_name
            if not img_path.exists():
                continue

            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            stenoses = stenosis_gt.get(image_name, [])
            stem = Path(image_name).stem

            for v_idx, vessel in enumerate(vessels):
                # Convert bbox and add padding
                bbox_xyxy = bbox_xywh_to_xyxy(vessel["bbox_xywh"])
                padded = pad_and_clip_bbox(bbox_xyxy, padding_frac, img_size)
                x1, y1, x2, y2 = padded
                crop_w = x2 - x1
                crop_h = y2 - y1

                # Skip tiny vessels
                if crop_w < min_bbox_px or crop_h < min_bbox_px:
                    stats["skipped_small"] += 1
                    continue

                # Crop and resize
                crop = img[y1:y2, x1:x2]
                crop_resized = cv2.resize(crop, (crop_size, crop_size),
                                          interpolation=cv2.INTER_LINEAR)

                # Find overlapping stenoses
                overlapping = find_overlapping_stenoses(stenoses, padded)

                # Remap polygons to crop space
                yolo_annotations = []
                for _ann, poly_pixels in overlapping:
                    remapped = remap_polygon_to_crop(poly_pixels, padded, crop_size)
                    if len(remapped) >= 3:
                        # Verify the remapped polygon has meaningful area
                        try:
                            rp = Polygon(remapped)
                            if rp.is_valid and rp.area > 1e-6:
                                yolo_annotations.append((0, remapped))  # class 0 = stenosis
                        except Exception:
                            pass

                # Save crop image
                crop_name = f"{stem}_v{v_idx}.png"
                cv2.imwrite(str(out_images / crop_name), crop_resized)

                # Save label (empty file for negatives)
                label_name = f"{stem}_v{v_idx}.txt"
                if yolo_annotations:
                    save_yolo_label(out_labels / label_name, yolo_annotations)
                    split_positive += 1
                    stats["positive_crops"] += 1
                    stats["total_stenoses_remapped"] += len(yolo_annotations)
                else:
                    # Empty label file for negative sample
                    (out_labels / label_name).write_text("")
                    stats["negative_crops"] += 1

                split_crops += 1
                stats["total_crops"] += 1

        print(f"  {split}: {split_crops} crops ({split_positive} positive, "
              f"{split_crops - split_positive} negative)")

    # Generate data.yaml
    data_yaml_content = {
        "path": str(crop_dataset_dir.resolve()),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "nc": 1,
        "names": {0: "stenosis"},
    }
    data_yaml_path = Path(cp["stenosis_crop_data_yaml"])
    with open(data_yaml_path, "w") as f:
        yaml.dump(data_yaml_content, f, default_flow_style=False)
    print(f"\n  Data YAML saved: {data_yaml_path}")

    # Summary
    print(f"\n{'='*60}")
    print("Crop dataset build complete!")
    print(f"  Total crops:     {stats['total_crops']}")
    print(f"  Positive crops:  {stats['positive_crops']} (with stenosis)")
    print(f"  Negative crops:  {stats['negative_crops']} (no stenosis)")
    print(f"  Skipped (small): {stats['skipped_small']}")
    print(f"  Stenoses remapped: {stats['total_stenoses_remapped']}")
    print(f"  Output dir:      {crop_dataset_dir}")
    print(f"{'='*60}")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Build cropped vessel-ROI dataset for stenosis detection"
    )
    parser.add_argument(
        "--config", type=str, default="config.yaml",
        help="Path to pipeline config YAML (default: config.yaml)"
    )
    args = parser.parse_args()

    config = load_config(args.config)

    # Verify syntax model weights exist
    syntax_weights = config["cross_inference"]["syntax_weights"]
    if not Path(syntax_weights).exists():
        raise FileNotFoundError(
            f"Syntax model weights not found: {syntax_weights}\n"
            f"Train the syntax model first: python train.py --task syntax"
        )

    build_crop_dataset(config)


if __name__ == "__main__":
    main()
