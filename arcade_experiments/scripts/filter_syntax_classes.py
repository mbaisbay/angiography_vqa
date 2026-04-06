"""Filter ARCADE SYNTAX COCO annotations to keep only well-represented classes.

Reads COCO JSON annotations, counts instances per category in the TRAIN split,
keeps only categories with >=min_count instances, remaps IDs to contiguous 0-N,
and saves filtered COCO JSONs for all splits.

With --min-count 300 (default), the kept classes are 10 SYNTAX classes:
  COCO IDs: 1, 2, 3, 4, 5, 6, 7, 8, 13, 16
  Names:    1, 2, 3, 4, 5, 6, 7, 8, 11, 13
  New IDs:  0, 1, 2, 3, 4, 5, 6, 7, 8,  9
"""

import argparse
import json
from collections import Counter
from pathlib import Path


SPLITS = ["train", "val", "test"]


def load_coco_json(json_path: Path) -> dict:
    """Load a COCO-format JSON annotation file."""
    with open(json_path, "r") as f:
        return json.load(f)


def count_train_instances(coco_data: dict) -> Counter:
    """Count annotation instances per category_id."""
    counts = Counter()
    for ann in coco_data["annotations"]:
        counts[ann["category_id"]] += 1
    return counts


def build_class_filter(train_counts: Counter, categories: list,
                       min_count: int) -> tuple:
    """Determine which classes to keep and build ID remapping.

    Args:
        train_counts: {category_id: count} from training split.
        categories: List of category dicts from COCO JSON.
        min_count: Minimum training instances to keep a class.

    Returns:
        (old_to_new, kept_categories):
        old_to_new: {old_cat_id: new_cat_id} for kept classes
        kept_categories: list of new category dicts with remapped IDs
    """
    cat_by_id = {c["id"]: c for c in categories}

    # Find categories meeting threshold, sorted by original ID
    kept_ids = sorted(
        cid for cid, count in train_counts.items()
        if count >= min_count
    )

    # Build remapping
    old_to_new = {}
    kept_categories = []
    for new_id, old_id in enumerate(kept_ids):
        old_to_new[old_id] = new_id
        old_cat = cat_by_id[old_id]
        kept_categories.append({
            "id": new_id,
            "name": old_cat["name"],
            "supercategory": old_cat.get("supercategory", ""),
        })

    return old_to_new, kept_categories


def filter_coco_json(coco_data: dict, old_to_new: dict,
                     kept_categories: list) -> dict:
    """Filter a COCO JSON to keep only the specified categories.

    Remaps category_ids and annotation IDs.
    """
    # Filter annotations
    filtered_anns = []
    new_ann_id = 1
    for ann in coco_data["annotations"]:
        if ann["category_id"] in old_to_new:
            new_ann = dict(ann)
            new_ann["id"] = new_ann_id
            new_ann["category_id"] = old_to_new[ann["category_id"]]
            filtered_anns.append(new_ann)
            new_ann_id += 1

    return {
        "images": coco_data["images"],
        "categories": kept_categories,
        "annotations": filtered_anns,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Filter ARCADE SYNTAX annotations to well-represented classes"
    )
    parser.add_argument(
        "--arcade-root", type=str, required=True,
        help="Path to arcade/submission directory"
    )
    parser.add_argument(
        "--output-dir", type=str, required=True,
        help="Output directory for filtered COCO JSONs"
    )
    parser.add_argument(
        "--min-count", type=int, default=300,
        help="Minimum training instances to keep a class (default: 300)"
    )
    args = parser.parse_args()

    arcade_root = Path(args.arcade_root)
    output_dir = Path(args.output_dir)
    syntax_dir = arcade_root / "syntax"

    # Step 1: Load train annotations and count instances
    train_json_path = syntax_dir / "train" / "annotations" / "train.json"
    print(f"Loading training annotations: {train_json_path}")
    train_data = load_coco_json(train_json_path)
    train_counts = count_train_instances(train_data)

    print(f"\nInstance counts in training split:")
    cat_names = {c["id"]: c["name"] for c in train_data["categories"]}
    for cat_id in sorted(train_counts.keys()):
        name = cat_names.get(cat_id, str(cat_id))
        marker = " <-- KEEP" if train_counts[cat_id] >= args.min_count else ""
        print(f"  Cat {cat_id:>3d} ({name:>5s}): {train_counts[cat_id]:>5d}{marker}")

    # Step 2: Build filter
    old_to_new, kept_categories = build_class_filter(
        train_counts, train_data["categories"], args.min_count
    )

    print(f"\nKept {len(kept_categories)} classes (>={args.min_count} train instances):")
    for cat in kept_categories:
        old_id = [k for k, v in old_to_new.items() if v == cat["id"]][0]
        print(f"  COCO {old_id:>3d} -> New {cat['id']:>2d}  "
              f"name={cat['name']:>5s}  ({train_counts[old_id]} instances)")

    # Step 3: Filter all splits
    annotations_dir = output_dir / "annotations"
    annotations_dir.mkdir(parents=True, exist_ok=True)

    for split in SPLITS:
        json_path = syntax_dir / split / "annotations" / f"{split}.json"
        if not json_path.exists():
            print(f"\n[SKIP] {split}: {json_path} not found")
            continue

        print(f"\n[{split.upper()}] Loading: {json_path}")
        coco_data = load_coco_json(json_path)

        # Count instances in this split before filtering
        split_counts = count_train_instances(coco_data)
        total_before = sum(split_counts.values())

        filtered = filter_coco_json(coco_data, old_to_new, kept_categories)
        total_after = len(filtered["annotations"])

        out_path = annotations_dir / f"{split}.json"
        with open(out_path, "w") as f:
            json.dump(filtered, f)

        print(f"  Images: {len(filtered['images'])}")
        print(f"  Annotations: {total_before} -> {total_after} "
              f"({total_before - total_after} dropped)")
        print(f"  Saved: {out_path}")

    # Save class mapping for reference
    mapping = {
        "old_to_new": {str(k): v for k, v in old_to_new.items()},
        "categories": kept_categories,
        "num_classes": len(kept_categories),
        "class_names": {str(c["id"]): c["name"] for c in kept_categories},
    }
    mapping_path = output_dir / "class_mapping.json"
    with open(mapping_path, "w") as f:
        json.dump(mapping, f, indent=2)
    print(f"\nSaved class mapping: {mapping_path}")


if __name__ == "__main__":
    main()
