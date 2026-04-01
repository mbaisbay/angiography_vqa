"""Filter YOLO labels to keep only well-represented classes and remap IDs.

Reads the existing label files, drops classes below a count threshold,
remaps remaining class IDs to be contiguous (0-indexed), and writes
filtered label files + new data.yaml files.

With --min-count 300, the kept classes are:
  12 syntax classes: 1,2,3,4,5,6,7,8,9,11,13,16
  1 stenosis class: stenosis
  = 13 total classes (IDs 0-12)
"""

import argparse
import shutil
import os
import yaml
from collections import Counter
from pathlib import Path

from utils.config_loader import load_config


SPLITS = ["train", "val", "test"]


def count_class_annotations(labels_dir: Path) -> Counter:
    """Count annotations per class across all label files in a directory."""
    counts = Counter()
    if not labels_dir.exists():
        return counts
    for label_file in labels_dir.glob("*.txt"):
        with open(label_file) as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    cls_id = int(parts[0])
                    counts[cls_id] += 1
    return counts


def get_total_class_counts(task_dir: Path) -> Counter:
    """Count annotations per class across all splits."""
    total = Counter()
    for split in SPLITS:
        labels_dir = task_dir / split / "labels"
        total += count_class_annotations(labels_dir)
    return total


def build_class_filter(counts: Counter, min_count: int,
                       class_names_26: dict) -> tuple:
    """Determine which classes to keep and build remapping.

    Args:
        counts: {old_class_id: annotation_count}
        min_count: Minimum annotations to keep a class.
        class_names_26: {old_id: name} for all 26 original classes.

    Returns:
        (old_to_new, new_names)
        old_to_new: {old_id: new_id} for kept classes
        new_names: {new_id: name} for the filtered set
    """
    # Always keep stenosis (class 25) regardless of count
    STENOSIS_OLD_ID = 25

    kept_ids = []
    for cls_id in sorted(counts.keys()):
        if cls_id == STENOSIS_OLD_ID:
            continue  # Handle stenosis separately at the end
        if counts[cls_id] >= min_count:
            kept_ids.append(cls_id)

    # Add stenosis as last class
    kept_ids.append(STENOSIS_OLD_ID)

    # Build remapping
    old_to_new = {}
    new_names = {}
    for new_id, old_id in enumerate(kept_ids):
        old_to_new[old_id] = new_id
        name = class_names_26.get(old_id, str(old_id))
        new_names[new_id] = name

    return old_to_new, new_names


def filter_label_file(input_path: Path, output_path: Path,
                      old_to_new: dict) -> dict:
    """Filter and remap a single YOLO label file.

    Returns stats dict.
    """
    kept = 0
    dropped = 0
    lines_out = []

    with open(input_path) as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            old_cls = int(parts[0])
            if old_cls in old_to_new:
                parts[0] = str(old_to_new[old_cls])
                lines_out.append(" ".join(parts))
                kept += 1
            else:
                dropped += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(lines_out) + "\n" if lines_out else "")

    return {"kept": kept, "dropped": dropped}


def filter_task(task_dir: Path, output_dir: Path, old_to_new: dict,
                use_symlinks: bool = True) -> dict:
    """Filter all label files for a task, symlink images."""
    stats = {"images": 0, "kept": 0, "dropped": 0}

    for split in SPLITS:
        src_images = task_dir / split / "images"
        src_labels = task_dir / split / "labels"
        dst_images = output_dir / split / "images"
        dst_labels = output_dir / split / "labels"

        if not src_images.exists():
            continue

        dst_images.mkdir(parents=True, exist_ok=True)
        dst_labels.mkdir(parents=True, exist_ok=True)

        # Process labels
        for label_file in sorted(src_labels.glob("*.txt")):
            s = filter_label_file(label_file, dst_labels / label_file.name, old_to_new)
            stats["kept"] += s["kept"]
            stats["dropped"] += s["dropped"]

        # Symlink or copy images
        image_files = sorted(src_images.glob("*.png")) + sorted(src_images.glob("*.PNG"))
        for img in image_files:
            dst = dst_images / img.name
            if not dst.exists():
                if use_symlinks:
                    os.symlink(img.resolve(), dst)
                else:
                    shutil.copy2(img, dst)
            stats["images"] += 1

        # Copy annotations (COCO JSON) — needed by evaluate.py
        src_annotations = task_dir / split / "annotations"
        dst_annotations = output_dir / split / "annotations"
        if src_annotations.exists() and not dst_annotations.exists():
            shutil.copytree(str(src_annotations), str(dst_annotations))

    return stats


def generate_data_yaml(output_dir: Path, yaml_path: Path,
                       new_names: dict) -> None:
    """Generate data.yaml for the filtered dataset."""
    data = {
        "path": str(output_dir.resolve()),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "nc": len(new_names),
        "names": new_names,
    }
    with open(yaml_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    # Also write inside the task directory
    local_yaml = output_dir / "data.yaml"
    with open(local_yaml, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    print(f"  Generated: {yaml_path}")
    print(f"  Generated: {local_yaml}")
    print(f"    nc: {data['nc']}")
    print(f"    names: {data['names']}")


def main():
    parser = argparse.ArgumentParser(
        description="Filter YOLO labels to keep only well-represented classes"
    )
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--min-count", type=int, default=300,
                        help="Minimum annotation count to keep a class (default: 300)")
    parser.add_argument("--no-symlinks", action="store_true",
                        help="Copy images instead of symlinking")
    args = parser.parse_args()

    config = load_config(args.config)
    dataset_root = Path(config["dataset_root"])

    # Build the original 26-class name mapping
    class_names_26 = {}
    syntax_cats = config["syntax_categories"]
    for i, coco_id in enumerate(sorted(syntax_cats.keys())):
        class_names_26[i] = syntax_cats[coco_id]
    class_names_26[25] = "stenosis"

    print("=" * 60)
    print(f"Filtering classes with min_count >= {args.min_count}")
    print("=" * 60)

    # Count across BOTH tasks (syntax + stenosis use same 26-class scheme)
    total_counts = Counter()
    for task in ["syntax", "stenosis"]:
        task_dir = dataset_root / task
        if task_dir.exists():
            counts = get_total_class_counts(task_dir)
            total_counts += counts
            print(f"\n  {task.upper()} class counts:")
            for cls_id in sorted(counts.keys()):
                name = class_names_26.get(cls_id, str(cls_id))
                print(f"    class {cls_id:>2d} ({name:>8s}): {counts[cls_id]}")

    # Build filter
    old_to_new, new_names = build_class_filter(total_counts, args.min_count, class_names_26)

    print(f"\n  Kept classes ({len(new_names)}):")
    for new_id in sorted(new_names.keys()):
        old_id = [k for k, v in old_to_new.items() if v == new_id][0]
        count = total_counts[old_id]
        print(f"    {old_id:>2d} -> {new_id:>2d}  {new_names[new_id]:>8s}  ({count} annotations)")

    dropped_classes = [cls_id for cls_id in total_counts if cls_id not in old_to_new]
    print(f"\n  Dropped classes ({len(dropped_classes)}):")
    for cls_id in sorted(dropped_classes):
        name = class_names_26.get(cls_id, str(cls_id))
        print(f"    class {cls_id:>2d} ({name:>8s}): {total_counts[cls_id]}")

    # Filter each task
    for task in ["syntax", "stenosis"]:
        task_dir = dataset_root / task
        if not task_dir.exists():
            continue

        output_dir = dataset_root / f"{task}_filtered"
        print(f"\n{'='*60}")
        print(f"Filtering {task.upper()}: {task_dir} -> {output_dir}")
        print(f"{'='*60}")

        stats = filter_task(task_dir, output_dir, old_to_new,
                            use_symlinks=not args.no_symlinks)
        print(f"  Images: {stats['images']}")
        print(f"  Labels kept: {stats['kept']}")
        print(f"  Labels dropped: {stats['dropped']}")

        # Generate data.yaml
        if task == "syntax":
            yaml_path = Path(config["syntax_data_yaml"])
        else:
            yaml_path = Path(config["stenosis_data_yaml"])
        generate_data_yaml(output_dir, yaml_path, new_names)

    # Update config category definitions for downstream scripts
    print(f"\n{'='*60}")
    print("IMPORTANT: Update config.yaml manually:")
    print(f"{'='*60}")
    print(f"  dataset_root: change to point to filtered dirs, OR")
    print(f"  run the pipeline pointing to the _filtered directories")
    print(f"\n  The data.yaml files have been overwritten to point to filtered dirs.")
    print(f"  Filtered data is at:")
    print(f"    {dataset_root / 'syntax_filtered'}")
    print(f"    {dataset_root / 'stenosis_filtered'}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
