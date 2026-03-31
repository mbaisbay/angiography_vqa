"""Build a combined 26-class YOLO dataset from stenosis GT + syntax cross-inference predictions.

This script is Step 3a of the pipeline:
  1. Train YOLO on syntax data
  2. Cross-inference: syntax model on stenosis images (produces syntax_on_stenosis.json)
  3a. THIS SCRIPT: merge stenosis GT labels + syntax predictions -> combined dataset
  3b. Train YOLO on the combined 26-class dataset
  4. Run combined model on syntax images

The ARCADE dataset uses a shared 26-class scheme:
  0-24: Syntax vessel segments
  25:   Stenosis (already class 25 in existing labels — no remapping needed)
"""

import argparse
import json
import os
import yaml
from pathlib import Path

from utils.config_loader import load_config
from utils.coco_to_yolo import save_category_mapping


SPLITS = ["train", "val", "test"]

# Syntax classes 0-24, stenosis becomes class 25
STENOSIS_COMBINED_CLASS = 25
NUM_SYNTAX_CLASSES = 25


def build_combined_class_names(config: dict) -> dict:
    """Build the 26-class name mapping for the combined dataset."""
    syntax_cats = config["syntax_categories"]
    # syntax_cats is {coco_id: name} with coco_ids 1-25
    # YOLO indices are 0-24, sorted by coco_id
    names = {}
    for i, coco_id in enumerate(sorted(syntax_cats.keys())):
        names[i] = syntax_cats[coco_id]
    names[STENOSIS_COMBINED_CLASS] = "stenosis"
    return names


def remap_stenosis_label_line(line: str) -> str:
    """Pass through stenosis YOLO label lines as-is.

    The existing labels already use the shared 26-class scheme where
    stenosis = class 25. No remapping needed.
    """
    return line.strip()


def prediction_to_yolo_line(pred: dict) -> str:
    """Convert a cross-inference prediction dict to a YOLO label line.

    pred has:
      class_id: int (0-24 for syntax)
      polygon_normalized: list of [x, y] pairs
    """
    cls_id = pred["class_id"]
    polygon = pred["polygon_normalized"]
    if not polygon:
        return ""
    # Flatten [[x1,y1], [x2,y2], ...] to "x1 y1 x2 y2 ..."
    coords = []
    for pt in polygon:
        coords.append(f"{pt[0]:.6f}")
        coords.append(f"{pt[1]:.6f}")
    return f"{cls_id} " + " ".join(coords)


def build_combined_split(
    stenosis_labels_dir: Path,
    stenosis_images_dir: Path,
    cross_inference_results: list,
    combined_labels_dir: Path,
    combined_images_dir: Path,
    min_confidence: float,
    use_symlinks: bool,
) -> dict:
    """Build combined labels for one split.

    Returns stats dict with counts.
    """
    combined_labels_dir.mkdir(parents=True, exist_ok=True)
    combined_images_dir.mkdir(parents=True, exist_ok=True)

    # Index cross-inference predictions by image name
    preds_by_image = {}
    for entry in cross_inference_results:
        preds_by_image[entry["image_name"]] = entry["predictions"]

    stats = {
        "images_processed": 0,
        "stenosis_labels_used": 0,
        "syntax_predictions_added": 0,
        "syntax_predictions_filtered": 0,
        "images_with_both": 0,
    }

    # Get all stenosis images
    image_files = sorted(stenosis_images_dir.glob("*.png")) + sorted(stenosis_images_dir.glob("*.PNG"))

    for img_path in image_files:
        stem = img_path.stem
        img_name = img_path.name

        # --- Stenosis GT labels (already class 25 in shared scheme) ---
        combined_lines = []
        stenosis_label_path = stenosis_labels_dir / f"{stem}.txt"
        has_stenosis = False
        if stenosis_label_path.exists():
            with open(stenosis_label_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        remapped = remap_stenosis_label_line(line)
                        if remapped:
                            combined_lines.append(remapped)
                            has_stenosis = True
                            stats["stenosis_labels_used"] += 1

        # --- Syntax predictions from cross-inference ---
        has_syntax = False
        preds = preds_by_image.get(img_name, [])
        for pred in preds:
            if pred["confidence"] < min_confidence:
                stats["syntax_predictions_filtered"] += 1
                continue
            yolo_line = prediction_to_yolo_line(pred)
            if yolo_line:
                combined_lines.append(yolo_line)
                has_syntax = True
                stats["syntax_predictions_added"] += 1

        # Write combined label file (even if empty — YOLO expects it)
        combined_label_path = combined_labels_dir / f"{stem}.txt"
        with open(combined_label_path, "w") as f:
            f.write("\n".join(combined_lines) + "\n" if combined_lines else "")

        # Link or copy image
        combined_img_path = combined_images_dir / img_name
        if not combined_img_path.exists():
            if use_symlinks:
                os.symlink(img_path.resolve(), combined_img_path)
            else:
                import shutil
                shutil.copy2(img_path, combined_img_path)

        stats["images_processed"] += 1
        if has_stenosis and has_syntax:
            stats["images_with_both"] += 1

    return stats


def generate_combined_data_yaml(config: dict, class_names: dict) -> None:
    """Generate data_combined.yaml for the 26-class combined dataset."""
    combined_dir = Path(config["combined_dataset"]["output_dir"])
    yaml_path = Path(config["combined_data_yaml"])

    data_yaml = {
        "path": str(combined_dir.resolve()),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "nc": len(class_names),
        "names": class_names,
    }

    with open(yaml_path, "w") as f:
        yaml.dump(data_yaml, f, default_flow_style=False, sort_keys=False)

    print(f"\n  Generated: {yaml_path}")
    print(f"    path: {data_yaml['path']}")
    print(f"    nc: {data_yaml['nc']}")


def main():
    parser = argparse.ArgumentParser(
        description="Build combined 26-class dataset from stenosis GT + syntax cross-inference"
    )
    parser.add_argument(
        "--config", type=str, default="config.yaml",
        help="Path to pipeline config YAML"
    )
    parser.add_argument(
        "--min-confidence", type=float, default=None,
        help="Min confidence for syntax predictions (overrides config)"
    )
    parser.add_argument(
        "--cross-inference-json", type=str, default=None,
        help="Path to syntax_on_stenosis.json (default: from config)"
    )
    args = parser.parse_args()

    config = load_config(args.config)
    dataset_root = Path(config["dataset_root"])
    combined_cfg = config["combined_dataset"]
    combined_dir = Path(combined_cfg["output_dir"])

    min_confidence = args.min_confidence or combined_cfg.get("min_confidence", 0.5)
    use_symlinks = combined_cfg.get("use_symlinks", True)

    # Load cross-inference results
    ci_json_path = args.cross_inference_json
    if ci_json_path is None:
        ci_json_path = Path(config["cross_inference"]["output_dir"]) / "syntax_on_stenosis.json"
    ci_json_path = Path(ci_json_path)

    if not ci_json_path.exists():
        raise FileNotFoundError(
            f"Cross-inference results not found: {ci_json_path}\n"
            f"Run cross_inference.py --direction syntax_on_stenosis first."
        )

    print("=" * 60)
    print("Building Combined Dataset (stenosis GT + syntax predictions)")
    print("=" * 60)
    print(f"  Source: {dataset_root / 'stenosis'}")
    print(f"  Cross-inference: {ci_json_path}")
    print(f"  Output: {combined_dir}")
    print(f"  Min confidence: {min_confidence}")
    print(f"  Use symlinks: {use_symlinks}")

    with open(ci_json_path, "r") as f:
        ci_data = json.load(f)

    # Build class names
    class_names = build_combined_class_names(config)
    print(f"\n  Combined classes ({len(class_names)}):")
    for idx in sorted(class_names.keys()):
        print(f"    {idx:>2d}: {class_names[idx]}")

    # Process each split
    total_stats = {
        "images_processed": 0,
        "stenosis_labels_used": 0,
        "syntax_predictions_added": 0,
        "syntax_predictions_filtered": 0,
        "images_with_both": 0,
    }

    for split in SPLITS:
        print(f"\n  [{split.upper()}]")
        stenosis_labels_dir = dataset_root / "stenosis" / split / "labels"
        stenosis_images_dir = dataset_root / "stenosis" / split / "images"

        if not stenosis_images_dir.exists():
            print(f"    Skipping: {stenosis_images_dir} not found")
            continue

        # Get cross-inference results for this split
        ci_results = ci_data.get(split, [])
        if not ci_results:
            print(f"    WARNING: No cross-inference results for split '{split}'")
            print(f"    Make sure to run cross_inference.py with splits including '{split}'")

        combined_labels = combined_dir / split / "labels"
        combined_images = combined_dir / split / "images"

        split_stats = build_combined_split(
            stenosis_labels_dir=stenosis_labels_dir,
            stenosis_images_dir=stenosis_images_dir,
            cross_inference_results=ci_results,
            combined_labels_dir=combined_labels,
            combined_images_dir=combined_images,
            min_confidence=min_confidence,
            use_symlinks=use_symlinks,
        )

        for k in total_stats:
            total_stats[k] += split_stats[k]

        print(f"    Images: {split_stats['images_processed']}")
        print(f"    Stenosis labels: {split_stats['stenosis_labels_used']}")
        print(f"    Syntax predictions added: {split_stats['syntax_predictions_added']}")
        print(f"    Syntax predictions filtered (low conf): {split_stats['syntax_predictions_filtered']}")
        print(f"    Images with BOTH syntax + stenosis: {split_stats['images_with_both']}")

    # Generate data YAML
    generate_combined_data_yaml(config, class_names)

    # Save combined category mapping for cross_inference.py
    mappings_dir = Path(config["mappings_dir"])
    mapping_path = mappings_dir / "combined_categories.json"
    # Build a coco_to_yolo-style mapping (identity for combined)
    coco_to_yolo = {i: i for i in range(len(class_names))}
    class_names_list = [class_names[i] for i in sorted(class_names.keys())]
    save_category_mapping(coco_to_yolo, class_names_list, str(mapping_path))
    print(f"  Saved category mapping: {mapping_path}")

    # Summary
    print(f"\n{'='*60}")
    print("Combined Dataset Summary")
    print(f"{'='*60}")
    for k, v in total_stats.items():
        print(f"  {k}: {v}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
