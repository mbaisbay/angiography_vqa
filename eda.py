"""Exploratory Data Analysis for ARCADE coronary angiography dataset."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from utils.config_loader import load_config


SPLITS = ["train", "val", "test"]


def find_coco_json(task_dir: Path, split: str) -> Path:
    """Locate the COCO JSON annotation file for a given task and split."""
    annotations_dir = task_dir / split / "annotations"
    candidates = [
        annotations_dir / f"{split}.json",
        annotations_dir / f"{split}.JSON",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    if annotations_dir.exists():
        json_files = list(annotations_dir.glob("*.json")) + list(annotations_dir.glob("*.JSON"))
        if json_files:
            return json_files[0]
    return None


def load_coco_data(task_dir: Path, split: str) -> dict:
    """Load COCO JSON for a task/split. Returns None if not found."""
    json_path = find_coco_json(task_dir, split)
    if json_path is None:
        return None
    with open(json_path, "r") as f:
        return json.load(f)


def analyze_task(task_dir: Path, task_name: str) -> dict:
    """Analyze a single task (syntax or stenosis) across all splits."""
    stats = {
        "task": task_name,
        "splits": {},
        "total_images": 0,
        "total_annotations": 0,
        "class_counts": Counter(),
        "annotations_per_image": [],
        "image_sizes": [],
        "annotation_areas": [],
    }

    # Build category name mapping
    cat_id_to_name = {}

    for split in SPLITS:
        coco = load_coco_data(task_dir, split)
        if coco is None:
            print(f"  [{split.upper()}] No annotations found, skipping")
            continue

        images = coco.get("images", [])
        annotations = coco.get("annotations", [])
        categories = coco.get("categories", [])

        # Build category mapping from first available split
        if not cat_id_to_name and categories:
            cat_id_to_name = {c["id"]: c["name"] for c in categories}

        # Count annotations per image
        img_ann_count = Counter()
        for ann in annotations:
            img_ann_count[ann["image_id"]] += 1

            # Class distribution
            cat_name = cat_id_to_name.get(ann["category_id"], str(ann["category_id"]))
            stats["class_counts"][cat_name] += 1

            # Annotation area
            if "area" in ann:
                stats["annotation_areas"].append(ann["area"])
            elif "bbox" in ann:
                _, _, w, h = ann["bbox"]
                stats["annotation_areas"].append(w * h)

        # Image stats
        for img in images:
            w, h = img.get("width", 0), img.get("height", 0)
            stats["image_sizes"].append((w, h))
            stats["annotations_per_image"].append(img_ann_count.get(img["id"], 0))

        # Count actual image files
        images_dir = task_dir / split / "images"
        actual_files = list(images_dir.glob("*.png")) + list(images_dir.glob("*.PNG")) if images_dir.exists() else []

        # Count label files
        labels_dir = task_dir / split / "labels"
        label_files = list(labels_dir.glob("*.txt")) if labels_dir.exists() else []

        split_stats = {
            "images_in_json": len(images),
            "actual_image_files": len(actual_files),
            "annotations": len(annotations),
            "label_files": len(label_files),
            "categories": len(categories),
            "avg_annotations_per_image": len(annotations) / max(len(images), 1),
        }
        stats["splits"][split] = split_stats
        stats["total_images"] += len(images)
        stats["total_annotations"] += len(annotations)

        print(f"  [{split.upper()}] {len(images)} images, {len(annotations)} annotations, "
              f"{len(actual_files)} image files on disk, {len(label_files)} label files")

    stats["cat_id_to_name"] = cat_id_to_name
    return stats


def print_summary(stats: dict) -> None:
    """Print a summary table for a task."""
    print(f"\n  Total: {stats['total_images']} images, {stats['total_annotations']} annotations")

    if stats["annotations_per_image"]:
        arr = np.array(stats["annotations_per_image"])
        print(f"  Annotations per image: min={arr.min()}, mean={arr.mean():.1f}, "
              f"max={arr.max()}, median={np.median(arr):.0f}")

    if stats["image_sizes"]:
        sizes = np.array(stats["image_sizes"])
        unique_sizes = set(map(tuple, sizes.tolist()))
        print(f"  Image sizes: {len(unique_sizes)} unique — {unique_sizes}")

    if stats["annotation_areas"]:
        areas = np.array(stats["annotation_areas"])
        print(f"  Annotation areas: min={areas.min():.0f}, mean={areas.mean():.0f}, "
              f"max={areas.max():.0f}, median={np.median(areas):.0f}")

    if stats["class_counts"]:
        print(f"\n  Class distribution ({len(stats['class_counts'])} classes):")
        for cls_name, count in stats["class_counts"].most_common():
            print(f"    {cls_name:>6s}: {count}")


def plot_class_distribution(stats: dict, output_dir: Path) -> None:
    """Create bar chart of class distribution."""
    if not stats["class_counts"]:
        return

    task = stats["task"]
    counts = stats["class_counts"]

    # Sort by class name for syntax (numeric), by count for stenosis
    if task == "syntax":
        # Sort by numeric segment number
        def sort_key(name):
            base = name.rstrip("abcd")
            try:
                return (int(base), name)
            except ValueError:
                return (999, name)
        sorted_items = sorted(counts.items(), key=lambda x: sort_key(x[0]))
    else:
        sorted_items = counts.most_common()

    names = [item[0] for item in sorted_items]
    values = [item[1] for item in sorted_items]

    fig, ax = plt.subplots(figsize=(max(12, len(names) * 0.5), 6))
    bars = ax.bar(range(len(names)), values, color="steelblue", edgecolor="black", linewidth=0.5)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Annotation Count")
    ax.set_title(f"{task.upper()} — Class Distribution (total: {sum(values)})")
    ax.grid(axis="y", alpha=0.3)

    # Add count labels on bars
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(values) * 0.01,
                str(val), ha="center", va="bottom", fontsize=7)

    plt.tight_layout()
    fig.savefig(output_dir / f"{task}_class_distribution.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_dir / f'{task}_class_distribution.png'}")


def plot_annotations_per_image(stats: dict, output_dir: Path) -> None:
    """Histogram of annotations per image."""
    if not stats["annotations_per_image"]:
        return

    task = stats["task"]
    arr = np.array(stats["annotations_per_image"])

    fig, ax = plt.subplots(figsize=(10, 5))
    bins = min(int(arr.max()) + 1, 50)
    ax.hist(arr, bins=bins, color="coral", edgecolor="black", linewidth=0.5)
    ax.set_xlabel("Annotations per Image")
    ax.set_ylabel("Number of Images")
    ax.set_title(f"{task.upper()} — Annotations per Image Distribution")
    ax.axvline(arr.mean(), color="red", linestyle="--", label=f"mean={arr.mean():.1f}")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_dir / f"{task}_annotations_per_image.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_dir / f'{task}_annotations_per_image.png'}")


def plot_annotation_areas(stats: dict, output_dir: Path) -> None:
    """Histogram of annotation areas."""
    if not stats["annotation_areas"]:
        return

    task = stats["task"]
    areas = np.array(stats["annotation_areas"])

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(areas, bins=50, color="mediumpurple", edgecolor="black", linewidth=0.5)
    ax.set_xlabel("Annotation Area (pixels)")
    ax.set_ylabel("Count")
    ax.set_title(f"{task.upper()} — Annotation Area Distribution")
    ax.axvline(areas.mean(), color="red", linestyle="--", label=f"mean={areas.mean():.0f}")
    ax.axvline(np.median(areas), color="green", linestyle="--", label=f"median={np.median(areas):.0f}")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_dir / f"{task}_annotation_areas.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_dir / f'{task}_annotation_areas.png'}")


def visualize_samples(task_dir: Path, task_name: str, output_dir: Path,
                      num_samples: int = 5) -> None:
    """Visualize sample images with annotations overlaid."""
    # Find a split with data
    for split in SPLITS:
        coco = load_coco_data(task_dir, split)
        if coco is not None:
            break
    else:
        print(f"  No data found for sample visualization")
        return

    images_dir = task_dir / split / "images"
    if not images_dir.exists():
        return

    image_files = sorted(images_dir.glob("*.png")) + sorted(images_dir.glob("*.PNG"))
    if not image_files:
        return

    # Build image_id -> annotations map
    img_id_to_anns = defaultdict(list)
    for ann in coco.get("annotations", []):
        img_id_to_anns[ann["image_id"]].append(ann)

    # Build image_id -> filename map
    img_id_to_info = {img["id"]: img for img in coco.get("images", [])}

    # Build category mapping
    cat_id_to_name = {c["id"]: c["name"] for c in coco.get("categories", [])}

    # Pick samples that have annotations
    annotated_ids = [img_id for img_id in img_id_to_anns if img_id in img_id_to_info]
    rng = np.random.RandomState(42)
    sample_ids = rng.choice(annotated_ids, size=min(num_samples, len(annotated_ids)), replace=False)

    # Color palette
    colors = plt.cm.tab20(np.linspace(0, 1, 20))

    fig, axes = plt.subplots(1, len(sample_ids), figsize=(5 * len(sample_ids), 5))
    if len(sample_ids) == 1:
        axes = [axes]

    for ax, img_id in zip(axes, sample_ids):
        img_info = img_id_to_info[img_id]
        img_path = images_dir / img_info["file_name"]
        if not img_path.exists():
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        overlay = img.copy()

        anns = img_id_to_anns[img_id]
        for i, ann in enumerate(anns):
            color = (np.array(colors[i % 20][:3]) * 255).astype(int)
            cat_name = cat_id_to_name.get(ann["category_id"], "?")

            if "segmentation" in ann and ann["segmentation"]:
                for seg in ann["segmentation"]:
                    pts = np.array(seg).reshape(-1, 2).astype(np.int32)
                    cv2.fillPoly(overlay, [pts], color.tolist())
                    cv2.polylines(img, [pts], True, color.tolist(), 2)

            # Label
            if "bbox" in ann:
                x, y, w, h = [int(v) for v in ann["bbox"]]
                cv2.putText(img, cat_name, (x, max(y - 5, 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color.tolist(), 1)

        # Blend overlay
        blended = cv2.addWeighted(overlay, 0.3, img, 0.7, 0)
        ax.imshow(blended)
        ax.set_title(f"{img_info['file_name']}\n{len(anns)} anns", fontsize=8)
        ax.axis("off")

    plt.suptitle(f"{task_name.upper()} — Sample Annotations ({split})", fontsize=12)
    plt.tight_layout()
    fig.savefig(output_dir / f"{task_name}_samples.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_dir / f'{task_name}_samples.png'}")


def print_data_yaml(task_dir: Path, task_name: str) -> None:
    """Print contents of existing data.yaml if present."""
    yaml_path = task_dir / "data.yaml"
    if yaml_path.exists():
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)
        print(f"\n  Existing data.yaml for {task_name}:")
        for k, v in data.items():
            if k == "names" and isinstance(v, dict) and len(v) > 10:
                print(f"    {k}: {len(v)} classes")
                for idx in sorted(v.keys()):
                    print(f"      {idx}: {v[idx]}")
            else:
                print(f"    {k}: {v}")


def main():
    parser = argparse.ArgumentParser(description="EDA for ARCADE coronary angiography dataset")
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="Path to pipeline config YAML")
    parser.add_argument("--task", type=str, choices=["syntax", "stenosis", "both"],
                        default="both", help="Which task to analyze")
    parser.add_argument("--samples", type=int, default=5,
                        help="Number of sample images to visualize")
    args = parser.parse_args()

    config = load_config(args.config)
    dataset_root = Path(config["dataset_root"])
    output_dir = Path(config["output_dir"]) / "eda"
    output_dir.mkdir(parents=True, exist_ok=True)

    tasks = ["syntax", "stenosis"] if args.task == "both" else [args.task]

    all_stats = {}
    for task in tasks:
        task_dir = dataset_root / task
        print(f"\n{'='*60}")
        print(f"EDA: {task.upper()}")
        print(f"{'='*60}")
        print(f"  Directory: {task_dir}")

        if not task_dir.exists():
            print(f"  [ERROR] Directory not found: {task_dir}")
            continue

        # Print existing data.yaml
        print_data_yaml(task_dir, task)

        # Analyze
        stats = analyze_task(task_dir, task)
        all_stats[task] = stats
        print_summary(stats)

        # Plots
        plot_class_distribution(stats, output_dir)
        plot_annotations_per_image(stats, output_dir)
        plot_annotation_areas(stats, output_dir)

        # Sample visualizations
        visualize_samples(task_dir, task, output_dir, num_samples=args.samples)

    # Cross-task summary
    if len(all_stats) > 1:
        print(f"\n{'='*60}")
        print("CROSS-TASK SUMMARY")
        print(f"{'='*60}")
        for task, stats in all_stats.items():
            print(f"  {task:>10s}: {stats['total_images']:>5d} images, "
                  f"{stats['total_annotations']:>6d} annotations, "
                  f"{len(stats['class_counts']):>3d} classes")

    print(f"\n  All EDA outputs saved to: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
