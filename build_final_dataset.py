"""Build the final clean dataset with quality-filtered predictions from both directions.

Takes the outputs of the 4-step pipeline and applies quality filters:
1. Confidence threshold (default: 0.5 for syntax, 0.6 for stenosis)
2. Spatial consistency — stenosis must overlap with vessel masks by at least X%
3. Minimum polygon area — discard tiny noise predictions

Produces:
  arcade/submission/final/
    train/images/  train/labels/
    val/images/    val/labels/
    test/images/   test/labels/

Each label file has all 26 classes (0-24 syntax, 25 stenosis).

Sources:
  - Stenosis-side images: original stenosis GT + syntax cross-inference predictions
  - Syntax-side images: original syntax GT + combined model stenosis predictions
"""

import argparse
import json
import os
import yaml
from pathlib import Path

import cv2
import numpy as np

from utils.config_loader import load_config
from utils.mask_utils import polygon_to_binary_mask


SPLITS = ["train", "val", "test"]
STENOSIS_CLASS = 25


def polygon_to_yolo_line(cls_id: int, polygon: list) -> str:
    """Convert a prediction polygon to YOLO label format."""
    if not polygon or len(polygon) < 3:
        return ""
    coords = []
    for pt in polygon:
        coords.append(f"{pt[0]:.6f}")
        coords.append(f"{pt[1]:.6f}")
    return f"{cls_id} " + " ".join(coords)


def parse_yolo_label_polygons(label_path: Path) -> list:
    """Parse YOLO label file and extract polygons for vessel classes (0-24).

    Returns list of polygon coordinate lists (each is [[x,y], ...]).
    """
    polygons = []
    if not label_path.exists():
        return polygons
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 7:  # class + at least 3 points
                continue
            cls_id = int(parts[0])
            if cls_id >= STENOSIS_CLASS:  # skip stenosis labels
                continue
            coords = list(map(float, parts[1:]))
            poly = [[coords[i], coords[i + 1]] for i in range(0, len(coords), 2)]
            polygons.append(poly)
    return polygons


def build_vessel_mask_from_gt(label_path: Path, mask_size: int) -> np.ndarray:
    """Build vessel union mask from GT YOLO label file."""
    vessel_union = np.zeros((mask_size, mask_size), dtype=np.uint8)
    for poly in parse_yolo_label_polygons(label_path):
        if len(poly) >= 3:
            mask = polygon_to_binary_mask(poly, mask_size)
            vessel_union = np.maximum(vessel_union, mask)
    return vessel_union


def filter_stenosis_predictions(preds: list, vessel_mask: np.ndarray,
                                min_conf: float, min_overlap: float,
                                min_area_frac: float, mask_size: int) -> list:
    """Filter stenosis predictions by confidence, spatial overlap, and area.

    Args:
        preds: Stenosis predictions to filter.
        vessel_mask: Precomputed vessel union mask (from GT labels or predictions).
        min_conf: Minimum confidence threshold.
        min_overlap: Minimum overlap ratio with vessel mask.
        min_area_frac: Minimum polygon area as fraction of image (e.g., 0.0005).
        mask_size: Mask rasterization size.

    Returns:
        Filtered list of stenosis predictions.
    """
    if not preds:
        return []

    total_pixels = mask_size * mask_size
    filtered = []

    for p in preds:
        # 1. Confidence check
        if p["confidence"] < min_conf:
            continue

        poly = p.get("polygon_normalized", [])
        if len(poly) < 3:
            continue

        # 2. Area check
        sten_mask = polygon_to_binary_mask(poly, mask_size)
        area = sten_mask.sum()
        if area / total_pixels < min_area_frac:
            continue

        # 3. Spatial overlap check (stenosis must be ON a vessel)
        if min_overlap > 0 and vessel_mask.sum() > 0:
            overlap = (sten_mask & vessel_mask).sum()
            if area > 0 and (overlap / area) < min_overlap:
                continue

        filtered.append(p)

    return filtered


def process_stenosis_side(config: dict, ci_data: dict, output_dir: Path,
                          min_syntax_conf: float, use_symlinks: bool) -> dict:
    """Process stenosis-side images: stenosis GT + syntax predictions.

    Stenosis labels are from ground truth (reliable).
    Syntax labels are from cross-inference (filtered by confidence).
    """
    dataset_root = Path(config["dataset_root"])
    stats = {"images": 0, "stenosis_labels": 0, "syntax_added": 0, "syntax_filtered": 0}

    for split in SPLITS:
        stenosis_dir = dataset_root / "stenosis" / split
        images_dir = stenosis_dir / "images"
        labels_dir = stenosis_dir / "labels"
        if not images_dir.exists():
            continue

        out_images = output_dir / split / "images"
        out_labels = output_dir / split / "labels"
        out_images.mkdir(parents=True, exist_ok=True)
        out_labels.mkdir(parents=True, exist_ok=True)

        # Index cross-inference syntax predictions
        preds_by_image = {}
        for entry in ci_data.get(split, []):
            preds_by_image[entry["image_name"]] = entry["predictions"]

        image_files = sorted(images_dir.glob("*.png")) + sorted(images_dir.glob("*.PNG"))

        for img_path in image_files:
            stem = img_path.stem
            img_name = img_path.name
            lines = []

            # Original stenosis GT labels (already class 25)
            gt_label = labels_dir / f"{stem}.txt"
            if gt_label.exists():
                with open(gt_label) as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            lines.append(line)
                            stats["stenosis_labels"] += 1

            # Syntax predictions from cross-inference
            for p in preds_by_image.get(img_name, []):
                if p["confidence"] < min_syntax_conf:
                    stats["syntax_filtered"] += 1
                    continue
                yolo_line = polygon_to_yolo_line(p["class_id"], p["polygon_normalized"])
                if yolo_line:
                    lines.append(yolo_line)
                    stats["syntax_added"] += 1

            # Write label
            with open(out_labels / f"{stem}.txt", "w") as f:
                f.write("\n".join(lines) + "\n" if lines else "")

            # Link image
            out_img = out_images / img_name
            if not out_img.exists():
                if use_symlinks:
                    os.symlink(img_path.resolve(), out_img)
                else:
                    import shutil
                    shutil.copy2(img_path, out_img)

            stats["images"] += 1

    return stats


def process_syntax_side(config: dict, ci_data: dict, output_dir: Path,
                        min_stenosis_conf: float, min_overlap: float,
                        min_area_frac: float, mask_size: int,
                        use_symlinks: bool) -> dict:
    """Process syntax-side images: syntax GT + stenosis predictions (quality-filtered).

    Syntax labels are from ground truth (reliable).
    Stenosis labels are from cross-inference (aggressively filtered).
    """
    dataset_root = Path(config["dataset_root"])
    stats = {
        "images": 0, "images_with_stenosis": 0,
        "syntax_labels": 0, "stenosis_added": 0,
        "stenosis_filtered_conf": 0, "stenosis_filtered_spatial": 0,
        "stenosis_total_before_filter": 0,
    }

    for split in SPLITS:
        syntax_dir = dataset_root / "syntax" / split
        images_dir = syntax_dir / "images"
        labels_dir = syntax_dir / "labels"
        if not images_dir.exists():
            continue

        out_images = output_dir / split / "images"
        out_labels = output_dir / split / "labels"
        out_images.mkdir(parents=True, exist_ok=True)
        out_labels.mkdir(parents=True, exist_ok=True)

        # Index combined model predictions
        preds_by_image = {}
        for entry in ci_data.get(split, []):
            preds_by_image[entry["image_name"]] = entry["predictions"]

        image_files = sorted(images_dir.glob("*.png")) + sorted(images_dir.glob("*.PNG"))

        for img_path in image_files:
            stem = img_path.stem
            img_name = img_path.name
            lines = []

            # Original syntax GT labels (classes 0-24)
            gt_label = labels_dir / f"{stem}.txt"
            if gt_label.exists():
                with open(gt_label) as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            lines.append(line)
                            stats["syntax_labels"] += 1

            # Stenosis predictions from combined model (quality-filtered)
            all_preds = preds_by_image.get(img_name, [])
            stenosis_preds = [p for p in all_preds if p["class_id"] == STENOSIS_CLASS]
            stats["stenosis_total_before_filter"] += len(stenosis_preds)

            # Use GT syntax labels for vessel mask (much more accurate than model predictions)
            gt_vessel_mask = build_vessel_mask_from_gt(gt_label, mask_size)

            filtered = filter_stenosis_predictions(
                stenosis_preds, gt_vessel_mask,
                min_conf=min_stenosis_conf,
                min_overlap=min_overlap,
                min_area_frac=min_area_frac,
                mask_size=mask_size,
            )
            stats["stenosis_filtered_conf"] += len(stenosis_preds) - len(filtered)

            has_stenosis = False
            for p in filtered:
                yolo_line = polygon_to_yolo_line(STENOSIS_CLASS, p["polygon_normalized"])
                if yolo_line:
                    lines.append(yolo_line)
                    stats["stenosis_added"] += 1
                    has_stenosis = True

            if has_stenosis:
                stats["images_with_stenosis"] += 1

            # Write label
            with open(out_labels / f"{stem}.txt", "w") as f:
                f.write("\n".join(lines) + "\n" if lines else "")

            # Link image (use different prefix to avoid collisions with stenosis images)
            out_img = out_images / f"syn_{img_name}"
            if not out_img.exists():
                if use_symlinks:
                    os.symlink(img_path.resolve(), out_img)
                else:
                    import shutil
                    shutil.copy2(img_path, out_img)

            # Label file also needs the prefix
            # Rename the label file to match
            old_label = out_labels / f"{stem}.txt"
            new_label = out_labels / f"syn_{stem}.txt"
            if old_label.exists():
                old_label.rename(new_label)

            stats["images"] += 1

    return stats


def generate_data_yaml(output_dir: Path, yaml_path: Path) -> None:
    """Generate data.yaml for the final dataset."""
    class_names = {}
    syntax_cats = {
        1: "1", 2: "2", 3: "3", 4: "4", 5: "5", 6: "6", 7: "7", 8: "8",
        9: "9", 10: "9a", 11: "10", 12: "10a", 13: "11", 14: "12", 15: "12a",
        16: "13", 17: "14", 18: "14a", 19: "15", 20: "16", 21: "16a",
        22: "16b", 23: "16c", 24: "12b", 25: "14b",
    }
    for i, coco_id in enumerate(sorted(syntax_cats.keys())):
        class_names[i] = syntax_cats[coco_id]
    class_names[25] = "stenosis"

    data = {
        "path": str(output_dir.resolve()),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "nc": 26,
        "names": class_names,
    }
    with open(yaml_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    print(f"  Generated: {yaml_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Build final quality-filtered dataset from both pipeline directions"
    )
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--min-syntax-conf", type=float, default=0.5,
                        help="Min confidence for syntax predictions on stenosis images")
    parser.add_argument("--min-stenosis-conf", type=float, default=0.6,
                        help="Min confidence for stenosis predictions on syntax images")
    parser.add_argument("--min-overlap", type=float, default=0.15,
                        help="Min spatial overlap of stenosis with vessel union (0-1)")
    parser.add_argument("--min-area-frac", type=float, default=0.0003,
                        help="Min stenosis polygon area as fraction of image")
    parser.add_argument("--no-symlinks", action="store_true",
                        help="Copy images instead of symlinking")
    args = parser.parse_args()

    config = load_config(args.config)
    dataset_root = Path(config["dataset_root"])
    ci_dir = Path(config["cross_inference"]["output_dir"])
    output_dir = dataset_root / "final"

    print("=" * 60)
    print("Building Final Quality-Filtered Dataset")
    print("=" * 60)
    print(f"  Output: {output_dir}")
    print(f"  Syntax confidence threshold: {args.min_syntax_conf}")
    print(f"  Stenosis confidence threshold: {args.min_stenosis_conf}")
    print(f"  Stenosis-vessel overlap threshold: {args.min_overlap}")
    print(f"  Min area fraction: {args.min_area_frac}")

    # --- Direction 1: Stenosis images (stenosis GT + syntax predictions) ---
    print(f"\n{'='*60}")
    print("Direction 1: Stenosis images (stenosis GT + syntax predictions)")
    print(f"{'='*60}")

    ci_syntax_on_stenosis = ci_dir / "syntax_on_stenosis.json"
    if ci_syntax_on_stenosis.exists():
        with open(ci_syntax_on_stenosis) as f:
            ci_data_1 = json.load(f)

        stats_1 = process_stenosis_side(
            config, ci_data_1, output_dir,
            min_syntax_conf=args.min_syntax_conf,
            use_symlinks=not args.no_symlinks,
        )
        print(f"  Images processed: {stats_1['images']}")
        print(f"  Stenosis labels (GT): {stats_1['stenosis_labels']}")
        print(f"  Syntax predictions added: {stats_1['syntax_added']}")
        print(f"  Syntax predictions filtered: {stats_1['syntax_filtered']}")
    else:
        print(f"  [SKIP] {ci_syntax_on_stenosis} not found")
        stats_1 = {"images": 0}

    # --- Direction 2: Syntax images (syntax GT + stenosis predictions) ---
    print(f"\n{'='*60}")
    print("Direction 2: Syntax images (syntax GT + stenosis predictions)")
    print(f"{'='*60}")

    ci_combined_on_syntax = ci_dir / "combined_on_syntax.json"
    if ci_combined_on_syntax.exists():
        with open(ci_combined_on_syntax) as f:
            ci_data_2 = json.load(f)

        stats_2 = process_syntax_side(
            config, ci_data_2, output_dir,
            min_stenosis_conf=args.min_stenosis_conf,
            min_overlap=args.min_overlap,
            min_area_frac=args.min_area_frac,
            mask_size=512,
            use_symlinks=not args.no_symlinks,
        )
        print(f"  Images processed: {stats_2['images']}")
        print(f"  Syntax labels (GT): {stats_2['syntax_labels']}")
        print(f"  Stenosis before filter: {stats_2['stenosis_total_before_filter']}")
        print(f"  Stenosis after filter: {stats_2['stenosis_added']}")
        print(f"  Stenosis filtered out: {stats_2['stenosis_filtered_conf']}")
        print(f"  Images with stenosis (after filter): {stats_2['images_with_stenosis']}")
    else:
        print(f"  [SKIP] {ci_combined_on_syntax} not found")
        stats_2 = {"images": 0, "images_with_stenosis": 0}

    # Generate data.yaml
    yaml_path = Path(config["_base_dir"]) / "data_final.yaml"
    generate_data_yaml(output_dir, yaml_path)

    # Summary
    total = stats_1.get("images", 0) + stats_2.get("images", 0)
    total_with_both = stats_1.get("images", 0) + stats_2.get("images_with_stenosis", 0)

    print(f"\n{'='*60}")
    print("FINAL DATASET SUMMARY")
    print(f"{'='*60}")
    print(f"  Stenosis-side images: {stats_1.get('images', 0)} (all have stenosis GT)")
    print(f"  Syntax-side images: {stats_2.get('images', 0)} "
          f"({stats_2.get('images_with_stenosis', 0)} with quality-filtered stenosis)")
    print(f"  Total images: {total}")
    print(f"  Images with BOTH syntax + stenosis: ~{total_with_both}")
    print(f"  Output: {output_dir}")
    print(f"  Data YAML: {yaml_path}")
    print(f"{'='*60}")
    print(f"\n  Tip: Adjust thresholds to trade off quality vs quantity:")
    print(f"    --min-stenosis-conf 0.7 --min-overlap 0.2  (stricter)")
    print(f"    --min-stenosis-conf 0.5 --min-overlap 0.1  (more data)")


if __name__ == "__main__":
    main()
