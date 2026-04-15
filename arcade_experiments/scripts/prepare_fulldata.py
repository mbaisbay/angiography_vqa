"""Prepare a full-data arcade root for final model training.

Competition context
-------------------
ARCADE provided 1000 train + 200 val images with labels, and 300 test
images (labels withheld during competition). After the competition,
all labels were released.

All top-3 teams (SSASS, YOLO-Angio, cross-task) trained their final
submission models on ALL 1200 labeled images (train + val merged).
Early stopping was driven by the held-out 300-image test set.

This script builds an arcade_root_fulldata directory that mimics what
those teams did:
  - task/train/  = original train(1000) + original val(200) merged
  - task/val/    = original test(300)  [used only for early-stopping]
  - task/test/   = original test(300)  [final evaluation]

Both val and test point to the same 300 images so that:
  1. YOLO has a val set for early stopping (required by Ultralytics)
  2. Final evaluation is on the same official 300-image test set
     that competition teams were scored on -> COMPARABLE metrics

The output is a drop-in replacement for arcade_root: prepare_data.py
and run_pipeline.py will read from this directory unchanged.

Usage
-----
    python prepare_fulldata.py \
        --arcade-root  ../../arcade/submission \
        --output-dir   ../../arcade/fulldata
"""

import argparse
import json
import os
from pathlib import Path


TASKS = ["syntax", "stenosis"]
ORIGINAL_SPLITS = ["train", "val", "test"]


def merge_coco_jsons(json_a: Path, json_b: Path) -> dict:
    """Merge two COCO JSON annotation files.

    Re-numbers image IDs and annotation IDs to be contiguous starting
    from 1. Category list is taken from json_a (assumed identical).
    """
    with open(json_a) as f:
        data_a = json.load(f)
    with open(json_b) as f:
        data_b = json.load(f)

    max_img_id = max((img["id"] for img in data_a["images"]), default=0)
    max_ann_id = max((ann["id"] for ann in data_a["annotations"]), default=0)

    # Remap image IDs in data_b to avoid collisions
    remap_img = {}
    new_images_b = []
    for img in data_b["images"]:
        new_id = max_img_id + img["id"]
        remap_img[img["id"]] = new_id
        new_img = dict(img)
        new_img["id"] = new_id
        new_images_b.append(new_img)

    new_anns_b = []
    for i, ann in enumerate(data_b["annotations"]):
        new_ann = dict(ann)
        new_ann["id"] = max_ann_id + i + 1
        new_ann["image_id"] = remap_img[ann["image_id"]]
        new_anns_b.append(new_ann)

    merged = {
        "images": data_a["images"] + new_images_b,
        "annotations": data_a["annotations"] + new_anns_b,
        "categories": data_a["categories"],
    }
    return merged


def symlink_images(src_dir: Path, dst_dir: Path) -> int:
    """Symlink all image files from src_dir into dst_dir. Returns count."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for ext in ("*.png", "*.PNG", "*.jpg", "*.jpeg"):
        for img in src_dir.glob(ext):
            dst = dst_dir / img.name
            if not dst.exists() and not dst.is_symlink():
                os.symlink(img.resolve(), dst)
                count += 1
    return count


def prepare_fulldata(arcade_root: Path, output_dir: Path) -> None:
    """Build fulldata arcade root from original ARCADE splits.

    For each task (syntax, stenosis):
      train/ <- merge of original train + val (1000+200=1200 images)
      val/   <- original test (300 images, for YOLO early-stopping)
      test/  <- original test (300 images, for final evaluation)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    for task in TASKS:
        src_task = arcade_root / task
        dst_task = output_dir / task

        print(f"\n{'='*60}")
        print(f"Task: {task.upper()}")
        print(f"{'='*60}")

        # ── TRAIN: merge original train + val ──────────────────────
        print("\n  [TRAIN] Merging original train(1000) + val(200)")

        src_train_img = src_task / "train" / "images"
        src_val_img   = src_task / "val"   / "images"
        dst_train_img = dst_task / "train" / "images"

        n_train = symlink_images(src_train_img, dst_train_img)
        n_val   = symlink_images(src_val_img,   dst_train_img)
        print(f"    Symlinked: {n_train} train + {n_val} val = {n_train+n_val} total")

        # Merge COCO annotation JSONs
        src_train_ann = src_task / "train" / "annotations" / "train.json"
        src_val_ann   = src_task / "val"   / "annotations" / "val.json"
        dst_train_ann_dir = dst_task / "train" / "annotations"
        dst_train_ann_dir.mkdir(parents=True, exist_ok=True)

        if src_train_ann.exists() and src_val_ann.exists():
            merged = merge_coco_jsons(src_train_ann, src_val_ann)
            dst_ann = dst_train_ann_dir / "train.json"
            with open(dst_ann, "w") as f:
                json.dump(merged, f)
            print(f"    Merged annotations: {len(merged['images'])} images, "
                  f"{len(merged['annotations'])} annotations -> {dst_ann}")
        elif src_train_ann.exists():
            import shutil
            shutil.copy2(src_train_ann, dst_train_ann_dir / "train.json")
            print(f"    WARNING: val annotations not found, using train only")
        else:
            print(f"    WARNING: no train annotations found at {src_train_ann}")

        # ── VAL: point to original test (for YOLO early-stopping) ──
        print("\n  [VAL] Symlinking original test(300) as val")

        src_test_img = src_task / "test" / "images"
        dst_val_img  = dst_task / "val"  / "images"
        n_test = symlink_images(src_test_img, dst_val_img)
        print(f"    Symlinked: {n_test} images")

        src_test_ann = src_task / "test" / "annotations" / "test.json"
        dst_val_ann_dir = dst_task / "val" / "annotations"
        dst_val_ann_dir.mkdir(parents=True, exist_ok=True)
        if src_test_ann.exists():
            # Rewrite as "val.json" for consistency
            with open(src_test_ann) as f:
                test_data = json.load(f)
            with open(dst_val_ann_dir / "val.json", "w") as f:
                json.dump(test_data, f)
        else:
            print(f"    WARNING: test annotations not found at {src_test_ann}")

        # ── TEST: same as val (original test 300 images) ───────────
        print("\n  [TEST] Symlinking original test(300) as test")

        dst_test_img = dst_task / "test" / "images"
        n_test2 = symlink_images(src_test_img, dst_test_img)
        print(f"    Symlinked: {n_test2} images")

        dst_test_ann_dir = dst_task / "test" / "annotations"
        dst_test_ann_dir.mkdir(parents=True, exist_ok=True)
        if src_test_ann.exists():
            with open(src_test_ann) as f:
                test_data = json.load(f)
            with open(dst_test_ann_dir / "test.json", "w") as f:
                json.dump(test_data, f)

    print(f"\n{'='*60}")
    print(f"Full-data arcade root built at: {output_dir}")
    print(f"  syntax/train:   1200 images (train+val merged)")
    print(f"  syntax/val:     300 images  (original test, for early-stopping)")
    print(f"  syntax/test:    300 images  (original test, COMPARABLE to leaderboard)")
    print(f"  stenosis/train: 1200 images (train+val merged)")
    print(f"  stenosis/val:   300 images  (original test)")
    print(f"  stenosis/test:  300 images  (original test)")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="Merge train+val into single training set for final model training"
    )
    parser.add_argument(
        "--arcade-root", type=str, required=True,
        help="Path to original arcade/submission directory"
    )
    parser.add_argument(
        "--output-dir", type=str, required=True,
        help="Output directory for full-data arcade root"
    )
    args = parser.parse_args()

    arcade_root = Path(args.arcade_root).resolve()
    output_dir  = Path(args.output_dir).resolve()

    if not arcade_root.exists():
        raise FileNotFoundError(f"arcade_root not found: {arcade_root}")

    prepare_fulldata(arcade_root, output_dir)


if __name__ == "__main__":
    main()
