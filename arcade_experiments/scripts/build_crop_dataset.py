"""Build cropped vessel-ROI dataset for stenosis detection.

Uses a trained syntax (vessel) model to detect vessels in stenosis images,
then crops each vessel bbox (+padding) and remaps GT stenosis annotations
to the crop coordinate space. Outputs a YOLO-format dataset for training
a dedicated single-class stenosis model on high-resolution vessel crops.

Usage:
    python build_crop_dataset.py \
        --syntax-weights /path/to/syntax_best.pt \
        --arcade-root ../../arcade/submission \
        --data-dir ../data \
        --output-dir ../data/stenosis_crops \
        --padding 0.20 \
        --crop-size 512
"""

import argparse
import json
import os
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import yaml
from ultralytics import YOLO

from crop_utils import (
    bbox_xywh_to_xyxy,
    compute_padded_bbox,
    crop_and_resize,
    find_overlapping_stenoses,
    remap_polygon_to_crop,
    save_yolo_label,
)


# ---------------------------------------------------------------------------
# Annotation loading
# ---------------------------------------------------------------------------

def load_stenosis_coco(coco_json_path):
    """Load stenosis GT annotations from COCO JSON, grouped by image filename.

    Returns:
        images_by_name: {filename: image_info_dict}
        anns_by_image: {filename: [annotation_dicts]}
            Annotations have 'segmentation' in pixel coords (flat lists).
    """
    with open(coco_json_path) as f:
        coco = json.load(f)

    images_by_id = {img["id"]: img for img in coco["images"]}
    images_by_name = {img["file_name"]: img for img in coco["images"]}
    anns_by_image = {}

    for ann in coco["annotations"]:
        img_info = images_by_id.get(ann["image_id"])
        if img_info is None:
            continue
        fname = img_info["file_name"]
        if fname not in anns_by_image:
            anns_by_image[fname] = []
        anns_by_image[fname].append(ann)

    return images_by_name, anns_by_image


# ---------------------------------------------------------------------------
# Main dataset builder
# ---------------------------------------------------------------------------

def build_crop_dataset(
    syntax_weights: str,
    images_dir: str,
    stenosis_coco_path: str,
    output_dir: Path,
    split: str,
    padding_frac: float = 0.20,
    crop_size: int = 512,
    min_dim: int = 16,
    vessel_conf: float = 0.15,
    imgsz: int = 512,
) -> dict:
    """Build crop dataset for one split.

    Args:
        syntax_weights: Path to trained syntax model .pt file.
        images_dir: Directory of stenosis images for this split.
        stenosis_coco_path: Path to COCO JSON with stenosis GT annotations.
        output_dir: Root output directory for crop dataset.
        split: 'train', 'val', or 'test'.
        padding_frac: Padding around vessel bbox as fraction of bbox size.
        crop_size: Resize each crop to this dimension.
        min_dim: Skip vessels with bbox smaller than this.
        vessel_conf: Confidence threshold for vessel predictions.
        imgsz: Image size for vessel model inference.

    Returns:
        Stats dict.
    """
    # Output dirs
    out_images = output_dir / split / "images"
    out_labels = output_dir / split / "labels"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    # Load stenosis GT
    _images_by_name, anns_by_image = load_stenosis_coco(stenosis_coco_path)
    print(f"  Loaded stenosis GT: {len(anns_by_image)} images with annotations")

    # Run syntax model on stenosis images
    print(f"  Running syntax model on {split} images (conf>={vessel_conf})...")
    model = YOLO(syntax_weights)
    vessel_results = model.predict(
        source=images_dir,
        conf=vessel_conf,
        imgsz=imgsz,
        save=False,
        verbose=False,
        stream=True,
        retina_masks=True,
    )

    stats = Counter()
    img_size = 512  # ARCADE native

    for result in vessel_results:
        image_name = Path(result.path).name
        stem = Path(image_name).stem
        stats["images"] += 1

        # Read image
        img = cv2.imread(str(result.path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            stats["read_failures"] += 1
            continue

        stenoses = anns_by_image.get(image_name, [])

        if result.masks is None or len(result.masks) == 0:
            stats["images_no_vessels"] += 1
            continue

        for v_idx in range(len(result.masks)):
            bbox_xywh = result.boxes.xywh[v_idx].tolist()
            bbox_xyxy = bbox_xywh_to_xyxy(bbox_xywh)
            padded = compute_padded_bbox(bbox_xyxy, padding_frac, img_size, min_dim)

            if padded is None:
                stats["skipped_small"] += 1
                continue

            # Crop and resize
            crop_resized = crop_and_resize(img, padded, crop_size)

            # Convert to 3-channel for YOLO compatibility
            crop_3ch = cv2.cvtColor(crop_resized, cv2.COLOR_GRAY2BGR)

            # Find overlapping stenoses
            overlapping = find_overlapping_stenoses(stenoses, padded)

            # Remap polygons to crop space
            yolo_annotations = []
            for _ann, poly_pixels in overlapping:
                remapped = remap_polygon_to_crop(poly_pixels, padded, normalize=True)
                if len(remapped) >= 3:
                    from shapely.geometry import Polygon as ShapelyPolygon
                    try:
                        rp = ShapelyPolygon(remapped)
                        if rp.is_valid and rp.area > 1e-6:
                            yolo_annotations.append((0, remapped))  # class 0 = stenosis
                    except Exception:
                        pass

            # Save crop
            crop_name = f"{stem}_v{v_idx:03d}.png"
            cv2.imwrite(str(out_images / crop_name), crop_3ch)

            # Save label
            label_name = f"{stem}_v{v_idx:03d}.txt"
            if yolo_annotations:
                save_yolo_label(out_labels / label_name, yolo_annotations)
                stats["positive_crops"] += 1
                stats["stenoses_remapped"] += len(yolo_annotations)
            else:
                (out_labels / label_name).write_text("")
                stats["negative_crops"] += 1

            stats["total_crops"] += 1

    return dict(stats)


def generate_crop_yaml(output_dir: Path, yaml_path: Path) -> None:
    """Generate YOLO data.yaml for the crop dataset."""
    data_yaml = {
        "path": str(output_dir.resolve()),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "nc": 1,
        "names": {0: "stenosis"},
    }
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    with open(yaml_path, "w") as f:
        yaml.dump(data_yaml, f, default_flow_style=False, sort_keys=False)
    print(f"  Data YAML saved: {yaml_path}")


def build_all_splits(
    syntax_weights: str,
    arcade_root: Path,
    data_dir: Path,
    output_dir: Path,
    padding_frac: float = 0.20,
    crop_size: int = 512,
    min_dim: int = 16,
    vessel_conf: float = 0.15,
    imgsz: int = 512,
) -> dict:
    """Build crop dataset for all splits.

    Uses the prepared data in data_dir (from prepare_data.py) when available,
    falls back to raw ARCADE data.
    """
    all_stats = {}

    for split in ["train", "val", "test"]:
        print(f"\n{'='*60}")
        print(f"Building crop dataset: {split}")
        print(f"{'='*60}")

        # Find images — prefer pipeline's prepared 3ch images
        images_dir = data_dir / "stenosis" / "images" / split
        if not images_dir.exists():
            images_dir = arcade_root / "stenosis" / split / "images"

        # Find COCO annotations — prefer pipeline's prepared annotations,
        # but need ORIGINAL with pixel coords (not YOLO labels)
        coco_path = arcade_root / "stenosis" / split / "annotations" / f"{split}.json"
        if not coco_path.exists():
            print(f"  [SKIP] COCO annotations not found: {coco_path}")
            continue

        if not images_dir.exists():
            print(f"  [SKIP] Images not found: {images_dir}")
            continue

        print(f"  Images: {images_dir}")
        print(f"  COCO GT: {coco_path}")

        stats = build_crop_dataset(
            syntax_weights=syntax_weights,
            images_dir=str(images_dir),
            stenosis_coco_path=str(coco_path),
            output_dir=output_dir,
            split=split,
            padding_frac=padding_frac,
            crop_size=crop_size,
            min_dim=min_dim,
            vessel_conf=vessel_conf,
            imgsz=imgsz,
        )

        all_stats[split] = stats
        print(f"  {split}: {stats.get('total_crops', 0)} crops "
              f"({stats.get('positive_crops', 0)} positive, "
              f"{stats.get('negative_crops', 0)} negative)")
        if stats.get("skipped_small", 0):
            print(f"  Skipped (too small): {stats['skipped_small']}")

    # Generate YAML
    yaml_path = output_dir.parent / "dataset_configs" / "stenosis_crop.yaml"
    generate_crop_yaml(output_dir, yaml_path)

    # Summary
    total = sum(s.get("total_crops", 0) for s in all_stats.values())
    pos = sum(s.get("positive_crops", 0) for s in all_stats.values())
    neg = sum(s.get("negative_crops", 0) for s in all_stats.values())
    sten = sum(s.get("stenoses_remapped", 0) for s in all_stats.values())
    print(f"\n{'='*60}")
    print(f"Crop dataset complete!")
    print(f"  Total crops:       {total}")
    print(f"  Positive (w/ sten): {pos}")
    print(f"  Negative (no sten): {neg}")
    print(f"  Stenoses remapped: {sten}")
    print(f"  Output: {output_dir}")
    print(f"{'='*60}")

    return all_stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Build cropped vessel-ROI dataset for stenosis detection"
    )
    parser.add_argument(
        "--syntax-weights", type=str, required=True,
        help="Path to trained syntax model weights (.pt)"
    )
    parser.add_argument(
        "--arcade-root", type=str, default="../../arcade/submission",
        help="Path to arcade/submission directory"
    )
    parser.add_argument(
        "--data-dir", type=str, default="../data",
        help="Pipeline data directory (from prepare_data.py)"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory for crop dataset (default: data_dir/stenosis_crops)"
    )
    parser.add_argument(
        "--padding", type=float, default=0.20,
        help="Padding around vessel bbox as fraction (default: 0.20)"
    )
    parser.add_argument(
        "--crop-size", type=int, default=512,
        help="Resize crops to this size (default: 512)"
    )
    parser.add_argument(
        "--min-dim", type=int, default=16,
        help="Skip vessels smaller than this (default: 16)"
    )
    parser.add_argument(
        "--vessel-conf", type=float, default=0.15,
        help="Vessel model confidence threshold (default: 0.15, low to maximize recall)"
    )
    parser.add_argument(
        "--imgsz", type=int, default=512,
        help="Image size for vessel model inference (default: 512)"
    )
    args = parser.parse_args()

    arcade_root = Path(args.arcade_root).resolve()
    data_dir = Path(args.data_dir).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else data_dir / "stenosis_crops"

    if not Path(args.syntax_weights).exists():
        raise FileNotFoundError(
            f"Syntax model weights not found: {args.syntax_weights}\n"
            f"Train the syntax model first."
        )

    build_all_splits(
        syntax_weights=args.syntax_weights,
        arcade_root=arcade_root,
        data_dir=data_dir,
        output_dir=output_dir,
        padding_frac=args.padding,
        crop_size=args.crop_size,
        min_dim=args.min_dim,
        vessel_conf=args.vessel_conf,
        imgsz=args.imgsz,
    )


if __name__ == "__main__":
    main()
