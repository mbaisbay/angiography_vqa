"""Pool all ARCADE splits and create stratified train/val/test splits.

Fixes three data distribution issues in the original ARCADE splits:
  1. Stenosis annotation size shift (val=3730px mean, test=7647px mean)
  2. Stenosis density mismatch (val=2.03/img, test=1.29/img)
  3. Class representation gaps across splits

Strategy:
  - Pool all 1500 images per task (train+val+test)
  - Assign each image a stratum based on key features
  - Split proportionally within each stratum -> 1000/200/300
  - Output new COCO JSON annotations + symlinked images

The syntax and stenosis tasks are split independently since they
contain different images (same filenames but different content).
"""

import argparse
import json
import math
import os
import random
from collections import Counter, defaultdict
from pathlib import Path


ORIGINAL_SPLITS = ["train", "val", "test"]
TARGET_SIZES = {"train": 1000, "val": 200, "test": 300}  # Same as ARCADE


def load_and_pool_annotations(task_dir: Path) -> tuple:
    """Pool all split annotations into a single dataset.

    Since filenames collide across splits (1.png exists in train/val/test
    but are different images), we rename to orig_{split}_{filename}.

    Returns:
        (images_list, annotations_list, categories, origin_map)
        origin_map: {new_image_id: (original_split, original_filename, original_image_id)}
    """
    all_images = []
    all_annotations = []
    categories = None
    origin_map = {}
    next_img_id = 1
    next_ann_id = 1
    old_to_new_img = {}  # (split, old_img_id) -> new_img_id

    for split in ORIGINAL_SPLITS:
        ann_path = task_dir / split / "annotations" / f"{split}.json"
        if not ann_path.exists():
            print(f"  WARNING: {ann_path} not found, skipping")
            continue

        with open(ann_path) as f:
            data = json.load(f)

        if categories is None:
            categories = data["categories"]

        for img in data["images"]:
            old_id = img["id"]
            new_id = next_img_id
            next_img_id += 1

            old_fname = img["file_name"]
            new_fname = f"orig_{split}_{old_fname}"

            new_img = dict(img)
            new_img["id"] = new_id
            new_img["file_name"] = new_fname
            new_img["original_file_name"] = old_fname
            new_img["original_split"] = split

            all_images.append(new_img)
            old_to_new_img[(split, old_id)] = new_id
            origin_map[new_id] = (split, old_fname, old_id)

        for ann in data["annotations"]:
            new_ann = dict(ann)
            new_ann["id"] = next_ann_id
            next_ann_id += 1
            new_ann["image_id"] = old_to_new_img[(split, ann["image_id"])]
            all_annotations.append(new_ann)

    return all_images, all_annotations, categories, origin_map


def compute_syntax_strata(images: list, annotations: list,
                          kept_cat_ids: set = None) -> dict:
    """Compute stratification features for syntax images.

    Stratifies by:
      - Number of (kept) annotations per image: low/medium/high
      - Presence of rare classes (those with <100 pooled instances)

    Returns:
        {image_id: stratum_string}
    """
    # Count per-category totals to identify rare classes
    cat_counts = Counter(ann["category_id"] for ann in annotations)
    rare_cats = {cid for cid, cnt in cat_counts.items() if cnt < 100}

    # Per-image features
    img_anns = defaultdict(list)
    for ann in annotations:
        img_anns[ann["image_id"]].append(ann)

    strata = {}
    for img in images:
        img_id = img["id"]
        anns = img_anns.get(img_id, [])

        if kept_cat_ids is not None:
            kept_anns = [a for a in anns if a["category_id"] in kept_cat_ids]
        else:
            kept_anns = anns

        n_anns = len(kept_anns)
        has_rare = any(a["category_id"] in rare_cats for a in anns)

        # Bin annotation count
        if n_anns <= 2:
            density = "low"
        elif n_anns <= 4:
            density = "med"
        else:
            density = "high"

        strata[img_id] = f"d{density}_r{int(has_rare)}"

    return strata


def compute_stenosis_strata(images: list, annotations: list,
                            stenosis_cat_id: int = 26) -> dict:
    """Compute stratification features for stenosis images.

    Stratifies by:
      - Stenosis count per image: 0 / 1 / 2+ (density)
      - Mean stenosis bbox area: small / medium / large (size)

    Returns:
        {image_id: stratum_string}
    """
    img_anns = defaultdict(list)
    for ann in annotations:
        if ann["category_id"] == stenosis_cat_id:
            img_anns[ann["image_id"]].append(ann)

    strata = {}
    for img in images:
        img_id = img["id"]
        anns = img_anns.get(img_id, [])

        n_stenosis = len(anns)

        # Density bucket
        if n_stenosis == 0:
            density = "none"
        elif n_stenosis == 1:
            density = "single"
        else:
            density = "multi"

        # Size bucket (mean bbox area)
        if anns:
            areas = []
            for a in anns:
                if "bbox" in a:
                    _, _, w, h = a["bbox"]
                    areas.append(w * h)
            if areas:
                mean_area = sum(areas) / len(areas)
                if mean_area < 3000:
                    size = "small"
                elif mean_area < 8000:
                    size = "medium"
                else:
                    size = "large"
            else:
                size = "unk"
        else:
            size = "none"

        strata[img_id] = f"d{density}_s{size}"

    return strata


def stratified_split(image_ids: list, strata: dict,
                     target_sizes: dict, seed: int = 42) -> dict:
    """Split image IDs into train/val/test preserving stratum proportions.

    Args:
        image_ids: List of all image IDs.
        strata: {image_id: stratum_string}
        target_sizes: {"train": N, "val": N, "test": N}
        seed: Random seed.

    Returns:
        {"train": [ids], "val": [ids], "test": [ids]}
    """
    rng = random.Random(seed)
    total = sum(target_sizes.values())
    ratios = {s: n / total for s, n in target_sizes.items()}

    # Group by stratum
    stratum_groups = defaultdict(list)
    for img_id in image_ids:
        s = strata.get(img_id, "default")
        stratum_groups[s].append(img_id)

    splits = {"train": [], "val": [], "test": []}

    for stratum, ids in sorted(stratum_groups.items()):
        rng.shuffle(ids)
        n = len(ids)

        # Allocate proportionally, ensuring at least 1 per split if possible
        n_val = max(1, round(n * ratios["val"])) if n >= 3 else 0
        n_test = max(1, round(n * ratios["test"])) if n >= 3 else 0
        n_train = n - n_val - n_test

        # Handle edge cases
        if n_train < 0:
            n_test = max(0, n - n_val)
            n_train = max(0, n - n_val - n_test)

        splits["train"].extend(ids[:n_train])
        splits["val"].extend(ids[n_train:n_train + n_val])
        splits["test"].extend(ids[n_train + n_val:])

    # Final shuffle within each split
    for s in splits:
        rng.shuffle(splits[s])

    return splits


def build_split_coco(images: list, annotations: list, categories: list,
                     split_ids: set) -> dict:
    """Build a COCO JSON dict for a subset of image IDs.

    Re-numbers image IDs to 1..N and annotation IDs to 1..M.
    """
    # Filter and renumber images
    filtered_imgs = []
    old_to_new_id = {}
    for new_idx, img in enumerate(
        sorted((i for i in images if i["id"] in split_ids),
               key=lambda x: x["id"]),
        start=1
    ):
        old_to_new_id[img["id"]] = new_idx
        new_img = dict(img)
        new_img["id"] = new_idx
        filtered_imgs.append(new_img)

    # Filter and renumber annotations
    filtered_anns = []
    ann_id = 1
    for ann in annotations:
        if ann["image_id"] in old_to_new_id:
            new_ann = dict(ann)
            new_ann["id"] = ann_id
            new_ann["image_id"] = old_to_new_id[ann["image_id"]]
            filtered_anns.append(new_ann)
            ann_id += 1

    return {
        "images": filtered_imgs,
        "annotations": filtered_anns,
        "categories": categories,
    }


def create_image_symlinks(images: list, source_dir: Path,
                          output_dir: Path) -> int:
    """Create symlinks for images in the output directory.

    Images have file_name=orig_{split}_{fname} and we need to
    find the source at source_dir/{split}/images/{fname}.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    count = 0

    for img in images:
        new_fname = img["file_name"]
        orig_split = img["original_split"]
        orig_fname = img["original_file_name"]

        src = source_dir / orig_split / "images" / orig_fname
        dst = output_dir / new_fname

        if not dst.exists() and src.exists():
            os.symlink(src.resolve(), dst)
            count += 1

    return count


def print_split_stats(split_name: str, coco_data: dict,
                      task: str, cat_filter: set = None) -> None:
    """Print statistics for a split."""
    n_imgs = len(coco_data["images"])
    n_anns = len(coco_data["annotations"])

    if cat_filter:
        n_kept = sum(1 for a in coco_data["annotations"]
                     if a["category_id"] in cat_filter)
        print(f"    {split_name}: {n_imgs} images, {n_anns} annotations "
              f"({n_kept} in kept classes)")
    else:
        print(f"    {split_name}: {n_imgs} images, {n_anns} annotations")

    # Category distribution
    cat_counts = Counter(a["category_id"] for a in coco_data["annotations"])
    cats = {c["id"]: c["name"] for c in coco_data["categories"]}
    if task == "stenosis":
        for cid in sorted(cat_counts):
            if cats.get(cid) == "stenosis" or cid == 26:
                # Compute mean area
                areas = []
                for a in coco_data["annotations"]:
                    if a["category_id"] == cid and "bbox" in a:
                        _, _, w, h = a["bbox"]
                        areas.append(w * h)
                mean_area = sum(areas) / len(areas) if areas else 0
                density = cat_counts[cid] / n_imgs if n_imgs > 0 else 0
                print(f"      stenosis: {cat_counts[cid]} instances, "
                      f"{density:.2f}/img, mean_area={mean_area:.0f}px")


def create_stratified_splits(arcade_root: Path, output_dir: Path,
                             seed: int = 42) -> None:
    """Main function: pool and re-split both syntax and stenosis."""

    print("=" * 60)
    print("Creating stratified splits")
    print(f"  Source: {arcade_root}")
    print(f"  Output: {output_dir}")
    print(f"  Seed:   {seed}")
    print("=" * 60)

    for task in ["syntax", "stenosis"]:
        task_dir = arcade_root / task
        task_output = output_dir / task

        print(f"\n{'─' * 60}")
        print(f"  Task: {task.upper()}")
        print(f"{'─' * 60}")

        # Pool all splits
        images, annotations, categories, origin_map = \
            load_and_pool_annotations(task_dir)

        print(f"  Pooled: {len(images)} images, {len(annotations)} annotations")

        # Compute strata
        if task == "syntax":
            strata = compute_syntax_strata(images, annotations)
        else:
            strata = compute_stenosis_strata(images, annotations)

        # Show stratum distribution
        stratum_counts = Counter(strata.values())
        print(f"  Strata ({len(stratum_counts)} groups):")
        for s, cnt in sorted(stratum_counts.items()):
            print(f"    {s}: {cnt} images")

        # Split
        all_ids = [img["id"] for img in images]
        splits = stratified_split(all_ids, strata, TARGET_SIZES, seed)

        print(f"\n  Split sizes: "
              f"train={len(splits['train'])}, "
              f"val={len(splits['val'])}, "
              f"test={len(splits['test'])}")

        # Build and save COCO JSONs + symlinks
        for split_name, split_ids in splits.items():
            split_id_set = set(split_ids)

            # Build COCO JSON
            coco_data = build_split_coco(
                images, annotations, categories, split_id_set
            )

            # Save annotation JSON
            ann_dir = task_output / split_name / "annotations"
            ann_dir.mkdir(parents=True, exist_ok=True)
            ann_path = ann_dir / f"{split_name}.json"
            with open(ann_path, "w") as f:
                json.dump(coco_data, f)

            # Create image symlinks
            split_images = [img for img in images if img["id"] in split_id_set]
            img_dir = task_output / split_name / "images"
            n_linked = create_image_symlinks(
                split_images, task_dir, img_dir
            )

            print_split_stats(split_name, coco_data, task)

        # Also copy data.yaml if it exists
        data_yaml_src = task_dir / "data.yaml"
        if data_yaml_src.exists():
            import shutil
            shutil.copy2(data_yaml_src, task_output / "data.yaml")

    # Save metadata
    meta = {
        "seed": seed,
        "target_sizes": TARGET_SIZES,
        "source": str(arcade_root),
        "description": "Stratified re-split of ARCADE dataset",
    }
    with open(output_dir / "split_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"Stratified splits saved to: {output_dir}")
    print(f"{'=' * 60}")


def main():
    parser = argparse.ArgumentParser(
        description="Create stratified train/val/test splits from ARCADE data"
    )
    parser.add_argument(
        "--arcade-root", type=str, required=True,
        help="Path to arcade/submission directory"
    )
    parser.add_argument(
        "--output-dir", type=str, required=True,
        help="Output directory for re-split data"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)"
    )
    args = parser.parse_args()

    create_stratified_splits(
        arcade_root=Path(args.arcade_root),
        output_dir=Path(args.output_dir),
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
