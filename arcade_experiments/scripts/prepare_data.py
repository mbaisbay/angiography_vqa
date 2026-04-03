"""Master data preparation script for ARCADE experiments.

Orchestrates the full data pipeline:
  1. Filter SYNTAX classes to those with >=300 training instances (10 classes)
  2. Convert filtered COCO JSONs to YOLO segmentation label format
  3. Convert stenosis COCO JSONs to YOLO format (stenosis class only)
  4. Convert grayscale images to 3-channel for COCO pretrained weight compatibility
  5. Generate YOLO dataset YAML config files
"""

import argparse
import json
import os
import yaml
from pathlib import Path

from filter_syntax_classes import (
    load_coco_json,
    count_train_instances,
    build_class_filter,
    filter_coco_json,
)
from convert_coco_to_yolo import convert_coco_to_yolo
from prepare_grayscale import process_directory


SPLITS = ["train", "val", "test"]


def prepare_syntax(arcade_root: Path, data_dir: Path,
                   min_count: int = 300) -> dict:
    """Filter SYNTAX data and convert to YOLO format.

    Returns class mapping dict.
    """
    syntax_dir = arcade_root / "syntax"
    output_dir = data_dir / "syntax_filtered"

    print("=" * 60)
    print("Step 1: Filter SYNTAX classes")
    print("=" * 60)

    # Load train annotations for counting
    train_json = syntax_dir / "train" / "annotations" / "train.json"
    train_data = load_coco_json(train_json)
    train_counts = count_train_instances(train_data)

    # Build filter
    old_to_new, kept_categories = build_class_filter(
        train_counts, train_data["categories"], min_count
    )

    cat_names = {c["id"]: c["name"] for c in train_data["categories"]}
    print(f"\nKept {len(kept_categories)} classes:")
    for cat in kept_categories:
        old_id = [k for k, v in old_to_new.items() if v == cat["id"]][0]
        print(f"  COCO {old_id:>3d} -> {cat['id']:>2d}  "
              f"{cat['name']:>5s}  ({train_counts[old_id]} train instances)")

    # Filter and convert each split
    print("\n" + "=" * 60)
    print("Step 2: Convert SYNTAX to YOLO format")
    print("=" * 60)

    annotations_dir = output_dir / "annotations"
    annotations_dir.mkdir(parents=True, exist_ok=True)

    for split in SPLITS:
        json_path = syntax_dir / split / "annotations" / f"{split}.json"
        if not json_path.exists():
            print(f"  [SKIP] {split}: not found")
            continue

        print(f"\n  [{split.upper()}]")
        coco_data = load_coco_json(json_path)

        # Filter COCO JSON
        filtered = filter_coco_json(coco_data, old_to_new, kept_categories)
        filtered_json = annotations_dir / f"{split}.json"
        with open(filtered_json, "w") as f:
            json.dump(filtered, f)
        print(f"    Filtered: {len(coco_data['annotations'])} -> "
              f"{len(filtered['annotations'])} annotations")

        # Convert to YOLO labels
        coco_to_yolo = {c["id"]: c["id"] for c in kept_categories}
        labels_dir = output_dir / "labels" / split
        stats = convert_coco_to_yolo(str(filtered_json), str(labels_dir),
                                     coco_to_yolo)
        print(f"    YOLO labels: {stats['images_with_labels']} images, "
              f"{stats['total_labels']} labels")

    # Save class mapping
    class_mapping = {
        "old_to_new": {str(k): v for k, v in old_to_new.items()},
        "categories": kept_categories,
        "num_classes": len(kept_categories),
        "class_names": {str(c["id"]): c["name"] for c in kept_categories},
    }
    mapping_path = output_dir / "class_mapping.json"
    with open(mapping_path, "w") as f:
        json.dump(class_mapping, f, indent=2)

    return class_mapping


def prepare_stenosis(arcade_root: Path, data_dir: Path) -> dict:
    """Convert stenosis annotations to YOLO format.

    Stenosis has only 1 relevant class (COCO ID 26 = "stenosis").
    In isolation, it maps to class 0. In merged dataset, it becomes class 10.
    """
    stenosis_dir = arcade_root / "stenosis"
    output_dir = data_dir / "stenosis"

    print("\n" + "=" * 60)
    print("Step 3: Convert STENOSIS to YOLO format")
    print("=" * 60)

    annotations_dir = output_dir / "annotations"
    annotations_dir.mkdir(parents=True, exist_ok=True)

    # Stenosis class mapping: COCO ID 26 -> YOLO ID 0 (in isolation)
    stenosis_coco_id = 26
    coco_to_yolo = {stenosis_coco_id: 0}

    for split in SPLITS:
        json_path = stenosis_dir / split / "annotations" / f"{split}.json"
        if not json_path.exists():
            print(f"  [SKIP] {split}: not found")
            continue

        print(f"\n  [{split.upper()}]")

        # Copy annotation file
        coco_data = load_coco_json(json_path)

        # Filter to only stenosis annotations
        stenosis_anns = [
            a for a in coco_data["annotations"]
            if a["category_id"] == stenosis_coco_id
        ]
        print(f"    Total annotations: {len(coco_data['annotations'])}")
        print(f"    Stenosis annotations: {len(stenosis_anns)}")

        # Save filtered COCO JSON (with only stenosis category)
        filtered = {
            "images": coco_data["images"],
            "categories": [{"id": 0, "name": "stenosis", "supercategory": ""}],
            "annotations": [
                {**a, "category_id": 0, "id": i + 1}
                for i, a in enumerate(stenosis_anns)
            ],
        }
        filtered_json = annotations_dir / f"{split}.json"
        with open(filtered_json, "w") as f:
            json.dump(filtered, f)

        # Convert to YOLO labels
        labels_dir = output_dir / "labels" / split
        yolo_mapping = {0: 0}
        stats = convert_coco_to_yolo(str(filtered_json), str(labels_dir),
                                     yolo_mapping)
        print(f"    YOLO labels: {stats['images_with_labels']} images, "
              f"{stats['total_labels']} labels")

    # Save class mapping
    class_mapping = {
        "original_coco_id": stenosis_coco_id,
        "class_names": {"0": "stenosis"},
        "num_classes": 1,
    }
    mapping_path = output_dir / "class_mapping.json"
    with open(mapping_path, "w") as f:
        json.dump(class_mapping, f, indent=2)

    return class_mapping


def convert_images(arcade_root: Path, data_dir: Path) -> None:
    """Convert grayscale images to 3-channel for all datasets."""
    print("\n" + "=" * 60)
    print("Step 4: Convert grayscale images to 3-channel")
    print("=" * 60)

    conversions = [
        ("syntax", "syntax_filtered"),
        ("stenosis", "stenosis"),
    ]

    for src_task, dst_task in conversions:
        for split in SPLITS:
            src_dir = arcade_root / src_task / split / "images"
            dst_dir = data_dir / dst_task / "images" / split

            if not src_dir.exists():
                print(f"  [SKIP] {src_task}/{split}: not found")
                continue

            print(f"\n  {src_task}/{split}:")
            stats = process_directory(src_dir, dst_dir)
            print(f"    Total: {stats['total']}, Converted: {stats['converted']}, "
                  f"Skipped: {stats['skipped']}")


def generate_dataset_yamls(data_dir: Path, syntax_mapping: dict) -> None:
    """Generate YOLO dataset YAML config files."""
    print("\n" + "=" * 60)
    print("Step 5: Generate dataset YAML configs")
    print("=" * 60)

    configs_dir = data_dir / "dataset_configs"
    configs_dir.mkdir(parents=True, exist_ok=True)

    # Syntax-only dataset (10 classes, IDs 0-9)
    syntax_names = {
        int(k): v for k, v in syntax_mapping["class_names"].items()
    }
    syntax_yaml = {
        "path": str((data_dir / "syntax_filtered").resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": syntax_mapping["num_classes"],
        "names": syntax_names,
    }
    syntax_yaml_path = configs_dir / "syntax_only.yaml"
    with open(syntax_yaml_path, "w") as f:
        yaml.dump(syntax_yaml, f, default_flow_style=False, sort_keys=False)
    print(f"  Generated: {syntax_yaml_path}")
    print(f"    nc={syntax_yaml['nc']}, names={syntax_yaml['names']}")

    # Stenosis-only dataset (1 class)
    stenosis_yaml = {
        "path": str((data_dir / "stenosis").resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": 1,
        "names": {0: "stenosis"},
    }
    stenosis_yaml_path = configs_dir / "stenosis_only.yaml"
    with open(stenosis_yaml_path, "w") as f:
        yaml.dump(stenosis_yaml, f, default_flow_style=False, sort_keys=False)
    print(f"  Generated: {stenosis_yaml_path}")

    # Merged dataset template (11 classes: 10 syntax + 1 stenosis)
    merged_names = dict(syntax_names)
    merged_names[len(syntax_names)] = "stenosis"
    merged_yaml = {
        "path": "TO_BE_SET_BY_PIPELINE",
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(merged_names),
        "names": merged_names,
    }
    merged_yaml_path = configs_dir / "merged_template.yaml"
    with open(merged_yaml_path, "w") as f:
        yaml.dump(merged_yaml, f, default_flow_style=False, sort_keys=False)
    print(f"  Generated: {merged_yaml_path}")
    print(f"    nc={merged_yaml['nc']}, names={merged_yaml['names']}")


def main():
    parser = argparse.ArgumentParser(
        description="Master data preparation for ARCADE experiments"
    )
    parser.add_argument(
        "--arcade-root", type=str, default="../arcade/submission",
        help="Path to arcade/submission directory"
    )
    parser.add_argument(
        "--data-dir", type=str, default="data",
        help="Output data directory (default: data)"
    )
    parser.add_argument(
        "--min-count", type=int, default=300,
        help="Minimum training instances to keep a SYNTAX class (default: 300)"
    )
    parser.add_argument(
        "--skip-images", action="store_true",
        help="Skip grayscale image conversion (if already done)"
    )
    args = parser.parse_args()

    arcade_root = Path(args.arcade_root).resolve()
    data_dir = Path(args.data_dir).resolve()

    if not arcade_root.exists():
        raise FileNotFoundError(f"ARCADE root not found: {arcade_root}")

    print(f"ARCADE root: {arcade_root}")
    print(f"Data output: {data_dir}")
    print()

    # Step 1-2: Filter and convert SYNTAX
    syntax_mapping = prepare_syntax(arcade_root, data_dir, args.min_count)

    # Step 3: Convert stenosis
    prepare_stenosis(arcade_root, data_dir)

    # Step 4: Convert grayscale images
    if not args.skip_images:
        convert_images(arcade_root, data_dir)
    else:
        print("\n[SKIP] Image conversion (--skip-images)")

    # Step 5: Generate dataset YAMLs
    generate_dataset_yamls(data_dir, syntax_mapping)

    print("\n" + "=" * 60)
    print("Data preparation complete!")
    print("=" * 60)
    print(f"  Syntax filtered: {data_dir / 'syntax_filtered'}")
    print(f"  Stenosis:        {data_dir / 'stenosis'}")
    print(f"  Dataset configs: {data_dir / 'dataset_configs'}")


if __name__ == "__main__":
    main()
