"""Convert ARCADE COCO annotations to YOLO segmentation format and generate data YAMLs."""

import argparse
import json
import yaml
from pathlib import Path

from utils.config_loader import load_config
from utils.coco_to_yolo import (
    build_category_mapping,
    convert_coco_to_yolo,
    save_category_mapping,
)


SPLITS = ["train", "val", "test"]


def find_coco_json(task_dir: Path, split: str) -> Path:
    """Locate the COCO JSON annotation file for a given task and split."""
    annotations_dir = task_dir / split / "annotations"

    # Try common naming patterns
    candidates = [
        annotations_dir / f"{split}.json",
        annotations_dir / f"{split}.JSON",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    # Fallback: find any JSON in the annotations directory
    if annotations_dir.exists():
        json_files = list(annotations_dir.glob("*.json")) + list(annotations_dir.glob("*.JSON"))
        if json_files:
            return json_files[0]

    raise FileNotFoundError(
        f"No annotation JSON found for task at {task_dir}, split '{split}'. "
        f"Checked: {[str(c) for c in candidates]}"
    )


def prepare_task(config: dict, task: str) -> None:
    """Prepare YOLO labels and data YAML for a single task (syntax or stenosis)."""
    dataset_root = Path(config["dataset_root"])
    task_dir = dataset_root / task

    if not task_dir.exists():
        raise FileNotFoundError(f"Task directory not found: {task_dir}")

    # Determine which COCO category IDs belong to this task
    if task == "syntax":
        config_categories = config["syntax_categories"]
    else:
        config_categories = config["stenosis_categories"]

    print(f"\n{'='*60}")
    print(f"Preparing {task.upper()} task")
    print(f"{'='*60}")

    coco_to_yolo = None
    class_names = None

    for split in SPLITS:
        split_dir = task_dir / split
        images_dir = split_dir / "images"

        if not images_dir.exists():
            print(f"  [SKIP] {split}: images directory not found at {images_dir}")
            continue

        # Find and load COCO JSON
        coco_json_path = find_coco_json(task_dir, split)
        print(f"\n  [{split.upper()}] Loading: {coco_json_path}")

        with open(coco_json_path, "r") as f:
            coco_data = json.load(f)

        # Build category mapping from the first split (should be consistent)
        if coco_to_yolo is None:
            coco_to_yolo, _, class_names = build_category_mapping(
                coco_data["categories"], task
            )
            print(f"  Category mapping: {len(coco_to_yolo)} classes")
            for coco_id, yolo_idx in sorted(coco_to_yolo.items()):
                print(f"    COCO {coco_id} -> YOLO {yolo_idx} ({class_names[yolo_idx]})")

        # YOLO expects labels at path obtained by replacing 'images' with 'labels'
        labels_dir = split_dir / "labels"

        # Convert annotations
        stats = convert_coco_to_yolo(
            str(coco_json_path), str(labels_dir), coco_to_yolo
        )

        print(f"  Converted: {stats['total_images']} images, "
              f"{stats['images_with_labels']} with labels, "
              f"{stats['total_labels']} total label entries, "
              f"{stats['skipped']} skipped annotations")

    if coco_to_yolo is None:
        raise RuntimeError(f"No valid splits found for task '{task}'")

    # Save category mapping
    mappings_dir = Path(config["mappings_dir"])
    mapping_path = mappings_dir / f"{task}_categories.json"
    save_category_mapping(coco_to_yolo, class_names, str(mapping_path))
    print(f"\n  Saved category mapping: {mapping_path}")

    # Generate YOLO data YAML
    generate_data_yaml(config, task, class_names)


def generate_data_yaml(config: dict, task: str, class_names: list) -> None:
    """Generate the YOLO data YAML file for a task."""
    dataset_root = Path(config["dataset_root"])
    task_dir = dataset_root / task

    if task == "syntax":
        yaml_path = config["syntax_data_yaml"]
    else:
        yaml_path = config["stenosis_data_yaml"]

    # Use absolute path for the 'path' field so YOLO can find images
    data_yaml = {
        "path": str(task_dir.resolve()),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "nc": len(class_names),
        "names": {i: name for i, name in enumerate(class_names)},
    }

    yaml_path = Path(yaml_path)
    with open(yaml_path, "w") as f:
        yaml.dump(data_yaml, f, default_flow_style=False, sort_keys=False)

    print(f"  Generated data YAML: {yaml_path}")
    print(f"    path: {data_yaml['path']}")
    print(f"    nc: {data_yaml['nc']}")
    print(f"    names: {data_yaml['names']}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert ARCADE COCO annotations to YOLO format"
    )
    parser.add_argument(
        "--config", type=str, default="config.yaml",
        help="Path to pipeline config YAML (default: config.yaml)"
    )
    parser.add_argument(
        "--task", type=str, choices=["syntax", "stenosis", "both"], default="both",
        help="Which task to prepare (default: both)"
    )
    args = parser.parse_args()

    config = load_config(args.config)

    tasks = ["syntax", "stenosis"] if args.task == "both" else [args.task]

    for task in tasks:
        prepare_task(config, task)

    print(f"\n{'='*60}")
    print("Data preparation complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
