"""Merge ground truth labels with pseudo-labels into a unified dataset.

Unified class scheme:
  - IDs 0-9: 10 filtered SYNTAX vessel segment classes
  - ID 10: stenosis

Merging rules:
  - GT labels take priority over pseudo-labels for the same class
  - val/test splits NEVER receive pseudo-labels (GT only)
  - Images are symlinked to avoid duplication
  - Label files are concatenated (GT lines + pseudo lines)
"""

import argparse
import json
import os
import yaml
from pathlib import Path
from collections import Counter


SPLITS = ["train", "val", "test"]
NUM_SYNTAX_CLASSES = 10
STENOSIS_CLASS_ID = 10  # In unified scheme


def read_label_lines(label_path: Path) -> list:
    """Read YOLO label lines from a file, skipping empty lines."""
    if not label_path.exists():
        return []
    with open(label_path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def get_classes_in_lines(lines: list) -> set:
    """Extract the set of class IDs present in YOLO label lines."""
    classes = set()
    for line in lines:
        parts = line.split()
        if parts:
            classes.add(int(parts[0]))
    return classes


def merge_label_files(gt_lines: list, pseudo_lines: list) -> list:
    """Merge GT and pseudo label lines, GT takes priority per class."""
    gt_classes = get_classes_in_lines(gt_lines)

    merged = list(gt_lines)
    for line in pseudo_lines:
        parts = line.split()
        if parts:
            cls_id = int(parts[0])
            # Only add pseudo predictions for classes NOT in GT
            if cls_id not in gt_classes:
                merged.append(line)

    return merged


def remap_label_lines(lines: list, class_offset: int) -> list:
    """Offset class IDs in YOLO label lines."""
    if class_offset == 0:
        return lines

    remapped = []
    for line in lines:
        parts = line.split()
        if parts:
            old_cls = int(parts[0])
            parts[0] = str(old_cls + class_offset)
            remapped.append(" ".join(parts))
    return remapped


def merge_syntax_images(
    syntax_data_dir: Path,
    pseudo_stenosis_dir: Path,
    output_dir: Path,
    split: str,
    add_pseudo: bool = True,
) -> dict:
    """Merge syntax GT labels with pseudo stenosis labels.

    Syntax GT: classes 0-9 (already in unified scheme)
    Pseudo stenosis: class 10 (from combined model inference)

    Args:
        syntax_data_dir: Path to syntax_filtered data.
        pseudo_stenosis_dir: Path to pseudo stenosis labels for syntax images.
        output_dir: Output merged dataset directory.
        split: train/val/test.
        add_pseudo: If False, only use GT (for val/test).

    Returns:
        Stats dict.
    """
    images_src = syntax_data_dir / "images" / split
    labels_src = syntax_data_dir / "labels" / split
    images_dst = output_dir / "images" / split
    labels_dst = output_dir / "labels" / split

    images_dst.mkdir(parents=True, exist_ok=True)
    labels_dst.mkdir(parents=True, exist_ok=True)

    stats = Counter()
    image_files = sorted(images_src.glob("*.png")) + sorted(images_src.glob("*.PNG"))

    for img_path in image_files:
        stem = img_path.stem
        stats["images"] += 1

        # Symlink image
        dst_img = images_dst / img_path.name
        if not dst_img.exists():
            os.symlink(img_path.resolve(), dst_img)

        # GT syntax labels (already classes 0-9)
        gt_lines = read_label_lines(labels_src / f"{stem}.txt")
        stats["gt_labels"] += len(gt_lines)

        # Pseudo stenosis labels (class 10)
        pseudo_lines = []
        if add_pseudo and pseudo_stenosis_dir:
            pseudo_path = pseudo_stenosis_dir / f"{stem}.txt"
            pseudo_lines = read_label_lines(pseudo_path)
            stats["pseudo_labels"] += len(pseudo_lines)

        # Merge
        merged = merge_label_files(gt_lines, pseudo_lines)
        stats["merged_labels"] += len(merged)

        # Write
        label_path = labels_dst / f"{stem}.txt"
        with open(label_path, "w") as f:
            f.write("\n".join(merged) + "\n" if merged else "")

    return dict(stats)


def merge_stenosis_images(
    stenosis_data_dir: Path,
    pseudo_syntax_dir: Path,
    output_dir: Path,
    split: str,
    add_pseudo: bool = True,
) -> dict:
    """Merge stenosis GT labels with pseudo syntax labels.

    Stenosis GT: class 0 in isolation -> remap to class 10 in unified scheme
    Pseudo syntax: classes 0-9 (from syntax model inference)

    Args:
        stenosis_data_dir: Path to stenosis data.
        pseudo_syntax_dir: Path to pseudo syntax labels for stenosis images.
        output_dir: Output merged dataset directory.
        split: train/val/test.
        add_pseudo: If False, only use GT (for val/test).

    Returns:
        Stats dict.
    """
    images_src = stenosis_data_dir / "images" / split
    labels_src = stenosis_data_dir / "labels" / split
    images_dst = output_dir / "images" / split
    labels_dst = output_dir / "labels" / split

    images_dst.mkdir(parents=True, exist_ok=True)
    labels_dst.mkdir(parents=True, exist_ok=True)

    stats = Counter()
    image_files = sorted(images_src.glob("*.png")) + sorted(images_src.glob("*.PNG"))

    for img_path in image_files:
        stem = img_path.stem
        stats["images"] += 1

        # Symlink image
        dst_img = images_dst / img_path.name
        if not dst_img.exists():
            os.symlink(img_path.resolve(), dst_img)

        # GT stenosis labels: remap from class 0 -> class 10
        gt_lines_raw = read_label_lines(labels_src / f"{stem}.txt")
        gt_lines = remap_label_lines(gt_lines_raw, STENOSIS_CLASS_ID)
        stats["gt_labels"] += len(gt_lines)

        # Pseudo syntax labels (classes 0-9)
        pseudo_lines = []
        if add_pseudo and pseudo_syntax_dir:
            pseudo_path = pseudo_syntax_dir / f"{stem}.txt"
            pseudo_lines = read_label_lines(pseudo_path)
            stats["pseudo_labels"] += len(pseudo_lines)

        # Merge
        merged = merge_label_files(gt_lines, pseudo_lines)
        stats["merged_labels"] += len(merged)

        # Write
        label_path = labels_dst / f"{stem}.txt"
        with open(label_path, "w") as f:
            f.write("\n".join(merged) + "\n" if merged else "")

    return dict(stats)


def generate_merged_yaml(output_dir: Path, class_names_json: str,
                         yaml_path: Path) -> None:
    """Generate YOLO dataset YAML for merged dataset."""
    with open(class_names_json, "r") as f:
        mapping = json.load(f)

    # Build unified class names: syntax (0-9) + stenosis (10)
    syntax_names = {int(k): v for k, v in mapping["class_names"].items()}
    names = dict(syntax_names)
    names[STENOSIS_CLASS_ID] = "stenosis"

    data_yaml = {
        "path": str(output_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(names),
        "names": names,
    }

    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    with open(yaml_path, "w") as f:
        yaml.dump(data_yaml, f, default_flow_style=False, sort_keys=False)

    return data_yaml


def merge_datasets(
    syntax_data_dir: Path,
    stenosis_data_dir: Path,
    pseudo_syntax_dir: Path,
    pseudo_stenosis_dir: Path,
    output_dir: Path,
    class_names_json: str,
    yaml_output: Path,
) -> dict:
    """Full dataset merge: syntax + stenosis with pseudo-labels.

    Args:
        syntax_data_dir: Syntax filtered data directory.
        stenosis_data_dir: Stenosis data directory.
        pseudo_syntax_dir: Pseudo syntax labels for stenosis images (or None).
        pseudo_stenosis_dir: Pseudo stenosis labels for syntax images (or None).
        output_dir: Output merged dataset directory.
        class_names_json: Path to syntax class_mapping.json.
        yaml_output: Path to write YOLO dataset YAML.

    Returns:
        Combined stats dict.
    """
    all_stats = {}

    for split in SPLITS:
        # Only add pseudo labels to training split
        add_pseudo = (split == "train")

        print(f"\n  [{split.upper()}] (pseudo={'YES' if add_pseudo else 'NO - GT only'})")

        # Merge syntax images
        syntax_stats = merge_syntax_images(
            syntax_data_dir, pseudo_stenosis_dir,
            output_dir, split, add_pseudo=add_pseudo,
        )
        print(f"    Syntax:   {syntax_stats.get('images', 0)} images, "
              f"{syntax_stats.get('gt_labels', 0)} GT + "
              f"{syntax_stats.get('pseudo_labels', 0)} pseudo = "
              f"{syntax_stats.get('merged_labels', 0)} merged labels")

        # Merge stenosis images
        stenosis_stats = merge_stenosis_images(
            stenosis_data_dir, pseudo_syntax_dir,
            output_dir, split, add_pseudo=add_pseudo,
        )
        print(f"    Stenosis: {stenosis_stats.get('images', 0)} images, "
              f"{stenosis_stats.get('gt_labels', 0)} GT + "
              f"{stenosis_stats.get('pseudo_labels', 0)} pseudo = "
              f"{stenosis_stats.get('merged_labels', 0)} merged labels")

        all_stats[split] = {
            "syntax": syntax_stats,
            "stenosis": stenosis_stats,
        }

    # Generate dataset YAML
    data_yaml = generate_merged_yaml(output_dir, class_names_json, yaml_output)
    print(f"\n  Dataset YAML: {yaml_output}")
    print(f"    nc={data_yaml['nc']}, path={data_yaml['path']}")

    return all_stats


def main():
    parser = argparse.ArgumentParser(
        description="Merge GT labels with pseudo-labels into unified dataset"
    )
    parser.add_argument(
        "--syntax-dir", type=str, required=True,
        help="Syntax filtered data directory"
    )
    parser.add_argument(
        "--stenosis-dir", type=str, required=True,
        help="Stenosis data directory"
    )
    parser.add_argument(
        "--pseudo-syntax-dir", type=str, default=None,
        help="Pseudo syntax labels for stenosis images"
    )
    parser.add_argument(
        "--pseudo-stenosis-dir", type=str, default=None,
        help="Pseudo stenosis labels for syntax images"
    )
    parser.add_argument(
        "--output-dir", type=str, required=True,
        help="Output merged dataset directory"
    )
    parser.add_argument(
        "--class-mapping", type=str, required=True,
        help="Path to syntax class_mapping.json"
    )
    parser.add_argument(
        "--yaml-output", type=str, required=True,
        help="Path for output YOLO dataset YAML"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Merging datasets")
    print("=" * 60)

    stats = merge_datasets(
        syntax_data_dir=Path(args.syntax_dir),
        stenosis_data_dir=Path(args.stenosis_dir),
        pseudo_syntax_dir=Path(args.pseudo_syntax_dir) if args.pseudo_syntax_dir else None,
        pseudo_stenosis_dir=Path(args.pseudo_stenosis_dir) if args.pseudo_stenosis_dir else None,
        output_dir=Path(args.output_dir),
        class_names_json=args.class_mapping,
        yaml_output=Path(args.yaml_output),
    )

    # Save merge stats
    stats_path = Path(args.output_dir) / "merge_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\n  Stats saved: {stats_path}")


if __name__ == "__main__":
    main()
