"""Build a vessel-filtered stenosis dataset for the S10 pipeline.

Instead of masking/cropping images (like S6), this script FILTERS stenosis
annotations: only stenoses whose polygon overlaps a predicted vessel region
are kept. Images are unchanged (symlinked), but label files may have fewer
annotations.

Motivation
----------
The stenosis dataset contains annotations on vessels from all 25 syntax
classes. If the syntax model cannot detect a vessel (rare class, bad angle,
etc.), any stenosis on that vessel becomes an "impossible positive" — the
pipeline can never find it because it cannot see the vessel. Training on
these cases introduces noise and penalises the model for correct behaviour.

By filtering to only stenoses inside predicted vessel regions, we create a
cleaner training signal: the model only learns to detect stenoses it could
theoretically find at inference time.

Output layout
-------------
    <output_dir>/
        images/{train,val,test}/*.png   (symlinks to original images)
        labels/{train,val,test}/*.txt   (filtered labels, subset of original)
        data.yaml                       (1-class stenosis YAML)
        filter_stats.json               (per-split filtering statistics)
"""

from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import yaml
from ultralytics import YOLO

from build_vessel_masked_dataset import (
    predict_vessel_mask,
    dilate_mask,
    read_yolo_seg_label,
    write_yolo_seg_label,
)

SPLITS = ["train", "val", "test"]


def filter_stenosis_annotations(
    labels: list[tuple[int, np.ndarray]],
    vessel_mask: np.ndarray,
    H: int,
    W: int,
    overlap_threshold: float = 0.5,
) -> tuple[list[tuple[int, np.ndarray]], dict]:
    """Filter stenosis annotations, keeping only those inside the vessel mask.

    Args:
        labels: List of (class_id, polygon_norm) from YOLO label file.
        vessel_mask: (H, W) binary uint8 mask of predicted vessels (dilated).
        H, W: Image dimensions.
        overlap_threshold: Minimum fraction of stenosis polygon pixels that
                           must overlap vessel mask to be kept.

    Returns:
        (kept_labels, stats): filtered labels and per-annotation stats.
    """
    stats = Counter()
    kept = []

    for cls_id, poly_norm in labels:
        stats["total"] += 1

        # Convert normalized polygon to absolute pixel coords
        poly_abs = poly_norm.copy()
        poly_abs[:, 0] *= W
        poly_abs[:, 1] *= H
        pts = poly_abs.astype(np.int32).reshape(-1, 1, 2)

        # Rasterize the stenosis polygon
        poly_mask = np.zeros((H, W), dtype=np.uint8)
        cv2.fillPoly(poly_mask, [pts], 255)
        poly_pixels = int(np.sum(poly_mask > 0))

        if poly_pixels == 0:
            stats["dropped_empty"] += 1
            continue

        # Compute overlap with dilated vessel mask
        overlap_pixels = int(np.sum((poly_mask > 0) & (vessel_mask > 0)))
        overlap_ratio = overlap_pixels / poly_pixels

        stats["overlap_sum_x1000"] += int(overlap_ratio * 1000)

        if overlap_ratio >= overlap_threshold:
            stats["kept"] += 1
            kept.append((cls_id, poly_norm))
        else:
            stats["dropped_outside"] += 1

    return kept, stats


def filter_stenosis_dataset(
    syntax_weights: str,
    stenosis_data_dir: Path,
    output_dir: Path,
    overlap_threshold: float = 0.5,
    dilate_px: int = 30,
    vessel_conf: float = 0.15,
    vessel_imgsz: int = 768,
) -> dict:
    """Build the filtered stenosis dataset.

    Args:
        syntax_weights: Path to trained syntax YOLO-seg model weights.
        stenosis_data_dir: Path containing images/{train,val,test} and
                           labels/{train,val,test} in YOLO format.
        output_dir: Where to write the filtered dataset.
        overlap_threshold: Min overlap ratio to keep a stenosis annotation.
        dilate_px: Pixels to dilate vessel mask by.
        vessel_conf: Confidence threshold for syntax model predictions.
        vessel_imgsz: Inference resolution for the syntax model.

    Returns:
        Stats dict with per-split filtering results.
    """
    print("=" * 64)
    print("Building vessel-filtered stenosis dataset")
    print("=" * 64)
    print(f"  Syntax weights     : {syntax_weights}")
    print(f"  Stenosis dir       : {stenosis_data_dir}")
    print(f"  Output dir         : {output_dir}")
    print(f"  Overlap threshold  : {overlap_threshold}")
    print(f"  Dilate px          : {dilate_px}")
    print(f"  Vessel conf        : {vessel_conf}")
    print(f"  Vessel imgsz       : {vessel_imgsz}")

    model = YOLO(syntax_weights)

    stats = {
        "overlap_threshold": overlap_threshold,
        "dilate_px": dilate_px,
        "vessel_conf": vessel_conf,
        "splits": {},
    }

    for split in SPLITS:
        img_src = stenosis_data_dir / "images" / split
        lbl_src = stenosis_data_dir / "labels" / split
        if not img_src.exists():
            print(f"\n  [{split}] SKIP — no images dir at {img_src}")
            continue

        img_files = sorted(
            list(img_src.glob("*.png")) + list(img_src.glob("*.PNG"))
        )
        print(f"\n  [{split}] Processing {len(img_files)} images")

        out_img_dir = output_dir / "images" / split
        out_lbl_dir = output_dir / "labels" / split
        out_img_dir.mkdir(parents=True, exist_ok=True)
        out_lbl_dir.mkdir(parents=True, exist_ok=True)

        split_stats = Counter()

        for img_path in img_files:
            stem = img_path.stem
            image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if image is None:
                split_stats["unreadable"] += 1
                continue
            H, W = image.shape[:2]
            split_stats["images"] += 1

            labels = read_yolo_seg_label(lbl_src / f"{stem}.txt")
            split_stats["gt_stenoses"] += len(labels)

            # Predict vessel mask
            mask = predict_vessel_mask(
                model, img_path, conf=vessel_conf, imgsz=vessel_imgsz,
            )

            if mask is None:
                # No vessels found — keep ALL stenoses (fallback)
                split_stats["fallback_no_vessels"] += 1
                mask = np.full((H, W), 255, dtype=np.uint8)

            # Resize to image size if needed
            if mask.shape != (H, W):
                mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)

            # Dilate
            dilated = dilate_mask(mask, dilate_px)

            # Filter annotations
            kept_labels, ann_stats = filter_stenosis_annotations(
                labels, dilated, H, W, overlap_threshold,
            )
            for k, v in ann_stats.items():
                split_stats[k] += v

            # Symlink image (no copy)
            out_img_path = out_img_dir / f"{stem}.png"
            if not out_img_path.exists():
                try:
                    os.symlink(img_path.resolve(), out_img_path)
                except OSError:
                    # Fallback to hard link or copy
                    import shutil
                    shutil.copy2(img_path, out_img_path)

            # Write filtered labels
            write_yolo_seg_label(out_lbl_dir / f"{stem}.txt", kept_labels)

        stats["splits"][split] = dict(split_stats)

        total = split_stats["gt_stenoses"]
        kept = split_stats.get("kept", 0)
        dropped = split_stats.get("dropped_outside", 0)
        pct_kept = (kept / total * 100) if total > 0 else 0
        avg_overlap = 0
        if total > 0:
            avg_overlap = (split_stats.get("overlap_sum_x1000", 0) / total) / 1000.0

        print(f"    Images           : {split_stats['images']}")
        print(f"    GT stenoses      : {total}")
        print(f"    Kept (inside)    : {kept} ({pct_kept:.1f}%)")
        print(f"    Dropped (outside): {dropped}")
        print(f"    No-vessel fallbk : {split_stats.get('fallback_no_vessels', 0)}")
        print(f"    Mean overlap     : {avg_overlap:.3f}")

    # Write dataset YAML
    data_yaml = {
        "path": str(output_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": 1,
        "names": {0: "stenosis"},
    }
    yaml_path = output_dir / "data.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(data_yaml, f, default_flow_style=False, sort_keys=False)
    print(f"\n  Wrote {yaml_path}")

    # Save stats
    stats_path = output_dir / "filter_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  Wrote {stats_path}")

    return stats


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Filter stenosis annotations to vessel-interior only"
    )
    parser.add_argument(
        "--syntax-weights", type=str, required=True,
        help="Path to trained syntax YOLO-seg model weights"
    )
    parser.add_argument(
        "--stenosis-dir", type=str, required=True,
        help="Path to stenosis data directory (images/labels per split)"
    )
    parser.add_argument(
        "--output-dir", type=str, required=True,
        help="Output directory for filtered dataset"
    )
    parser.add_argument(
        "--overlap-threshold", type=float, default=0.5,
        help="Min overlap ratio to keep annotation (default: 0.5)"
    )
    parser.add_argument(
        "--dilate-px", type=int, default=30,
        help="Pixels to dilate vessel mask (default: 30)"
    )
    parser.add_argument(
        "--vessel-conf", type=float, default=0.15,
        help="Confidence threshold for vessel predictions (default: 0.15)"
    )
    parser.add_argument(
        "--vessel-imgsz", type=int, default=768,
        help="Inference image size for syntax model (default: 768)"
    )
    args = parser.parse_args()

    filter_stenosis_dataset(
        syntax_weights=args.syntax_weights,
        stenosis_data_dir=Path(args.stenosis_dir),
        output_dir=Path(args.output_dir),
        overlap_threshold=args.overlap_threshold,
        dilate_px=args.dilate_px,
        vessel_conf=args.vessel_conf,
        vessel_imgsz=args.vessel_imgsz,
    )


if __name__ == "__main__":
    main()
