"""Generate pseudo-stenosis labels for syntax images using a trained stenosis model.

SSASS technique (Medipixel, 1st place ARCADE 2023)
---------------------------------------------------
The core insight: stenosis can only occur inside coronary vessels. The
syntax dataset contains 1200 images of coronary vessels -- many of which
likely contain stenoses that were not annotated (the syntax task only
annotated vessel segments, not lesions). Running the stenosis detector
on these images finds these missed stenoses.

Adding syntax images with pseudo-stenosis labels to the stenosis training
set effectively doubles the training data for stenosis detection from
1200 to ~1600 images (1200 stenosis + ~400 syntax with pseudo-labels),
directly attacking the data scarcity bottleneck.

Pipeline
--------
1. Load trained stenosis model (best from H1 training)
2. Run inference on ALL syntax training images at conf >= conf_threshold
3. Write pseudo-labels as YOLO segmentation format (class 0 = stenosis)
4. Build an extended stenosis training YAML that includes:
   a. Original stenosis train images (1200, with GT labels)
   b. Syntax images that got pseudo-labels (variable, ~200-500)
5. Return path to the extended dataset YAML

This is invoked inside run_fulldata_ssass() in the strategy runner.

Usage (standalone)
------------------
    python generate_stenosis_pseudolabels.py \
        --stenosis-model  /path/to/stenosis_best.pt \
        --syntax-img-dir  /path/to/syntax_filtered/images/train \
        --output-dir      /path/to/pseudo_stenosis_data \
        --conf-threshold  0.40 \
        --device          0
"""

import argparse
import json
import os
import shutil
import yaml
from pathlib import Path


def run_stenosis_on_syntax_images(
    model_path: str,
    syntax_img_dir: Path,
    output_label_dir: Path,
    conf_threshold: float = 0.40,
    imgsz: int = 768,
    device: str = "0",
) -> dict:
    """Run stenosis model on syntax images and write YOLO segmentation labels.

    Args:
        model_path: Path to trained stenosis model (.pt)
        syntax_img_dir: Directory containing syntax training images
        output_label_dir: Where to write pseudo-label .txt files
        conf_threshold: Confidence threshold for accepting predictions
        imgsz: Inference image size
        device: CUDA device id

    Returns:
        stats dict with counts of images processed, images with predictions, etc.
    """
    from ultralytics import YOLO

    output_label_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(model_path)
    model.to(f"cuda:{device}" if device.isdigit() else device)

    img_files = (
        sorted(syntax_img_dir.glob("*.png")) +
        sorted(syntax_img_dir.glob("*.PNG")) +
        sorted(syntax_img_dir.glob("*.jpg"))
    )

    stats = {
        "images_processed": 0,
        "images_with_predictions": 0,
        "total_pseudo_instances": 0,
        "conf_threshold": conf_threshold,
        "imgsz": imgsz,
    }

    for img_path in img_files:
        stats["images_processed"] += 1
        results = model.predict(
            source=str(img_path),
            conf=conf_threshold,
            imgsz=imgsz,
            verbose=False,
            save=False,
            retina_masks=True,
        )

        if not results or results[0].masks is None:
            continue

        r = results[0]
        masks = r.masks
        if masks is None or len(masks) == 0:
            continue

        img_h, img_w = r.orig_shape

        label_lines = []
        for mask_data in masks.xyn:
            # xyn: normalized polygon coordinates [[x1,y1],[x2,y2],...]
            if len(mask_data) < 3:
                continue
            coords = []
            for pt in mask_data:
                coords.extend([f"{pt[0]:.6f}", f"{pt[1]:.6f}"])
            # Class 0 = stenosis
            label_lines.append("0 " + " ".join(coords))

        if label_lines:
            label_path = output_label_dir / (img_path.stem + ".txt")
            with open(label_path, "w") as f:
                f.write("\n".join(label_lines) + "\n")
            stats["images_with_predictions"] += 1
            stats["total_pseudo_instances"] += len(label_lines)

    print(f"  Pseudo-label generation complete:")
    print(f"    Images processed:          {stats['images_processed']}")
    print(f"    Images with predictions:   {stats['images_with_predictions']}")
    print(f"    Total pseudo-instances:    {stats['total_pseudo_instances']}")
    print(f"    Conf threshold:            {conf_threshold}")

    return stats


def build_extended_stenosis_dataset(
    original_stenosis_dir: Path,
    syntax_img_dir: Path,
    pseudo_label_dir: Path,
    output_dir: Path,
    apply_clahe: bool = True,
    clahe_clip_limit: float = 2.0,
    clahe_tile_size: int = 8,
) -> str:
    """Build an extended stenosis dataset combining GT + pseudo-labeled syntax images.

    Layout of output_dir:
        images/train/  GT stenosis images (1200) + syntax images with pseudo-labels
        images/val/    GT stenosis val/test images (unchanged)
        images/test/   GT stenosis test images (unchanged)
        labels/train/  GT stenosis labels + pseudo-stenosis labels for syntax
        labels/val/    GT stenosis labels (unchanged)
        labels/test/   GT stenosis labels (unchanged)

    Returns path to dataset YAML.
    """
    output_dir = Path(output_dir)

    # ── TRAIN: symlink GT stenosis + syntax images with pseudo-labels ──
    dst_train_img = output_dir / "images" / "train"
    dst_train_lbl = output_dir / "labels" / "train"
    dst_train_img.mkdir(parents=True, exist_ok=True)
    dst_train_lbl.mkdir(parents=True, exist_ok=True)

    # 1. GT stenosis training images + labels
    src_sten_train_img = original_stenosis_dir / "images" / "train"
    src_sten_train_lbl = original_stenosis_dir / "labels" / "train"

    n_gt = 0
    for img in list(src_sten_train_img.glob("*.png")) + list(src_sten_train_img.glob("*.PNG")):
        dst = dst_train_img / img.name
        if not dst.exists() and not dst.is_symlink():
            os.symlink(img.resolve(), dst)
        n_gt += 1

    for lbl in src_sten_train_lbl.glob("*.txt"):
        dst = dst_train_lbl / lbl.name
        if not dst.exists() and not dst.is_symlink():
            os.symlink(lbl.resolve(), dst)

    # 2. Syntax images that received pseudo-labels
    n_pseudo = 0
    for lbl_path in pseudo_label_dir.glob("*.txt"):
        stem = lbl_path.stem
        # Find source image
        src_img = None
        for ext in (".png", ".PNG", ".jpg"):
            candidate = syntax_img_dir / (stem + ext)
            if candidate.exists():
                src_img = candidate
                break

        if src_img is None:
            continue

        # Prefix to avoid collision with stenosis images (same filenames possible)
        dst_img_name = f"syn_pseudo_{src_img.name}"
        dst_img = dst_train_img / dst_img_name
        if not dst_img.exists() and not dst_img.is_symlink():
            os.symlink(src_img.resolve(), dst_img)

        dst_lbl = dst_train_lbl / f"syn_pseudo_{stem}.txt"
        if not dst_lbl.exists() and not dst_lbl.is_symlink():
            os.symlink(lbl_path.resolve(), dst_lbl)

        n_pseudo += 1

    print(f"  Extended stenosis train set:")
    print(f"    GT stenosis images:     {n_gt}")
    print(f"    Pseudo-labeled syntax:  {n_pseudo}")
    print(f"    Total train images:     {n_gt + n_pseudo}")

    # ── VAL and TEST: symlink original stenosis val/test unchanged ──
    for split in ("val", "test"):
        src_img_dir = original_stenosis_dir / "images" / split
        src_lbl_dir = original_stenosis_dir / "labels" / split
        dst_img_dir = output_dir / "images" / split
        dst_lbl_dir = output_dir / "labels" / split
        dst_img_dir.mkdir(parents=True, exist_ok=True)
        dst_lbl_dir.mkdir(parents=True, exist_ok=True)

        for img in list(src_img_dir.glob("*.png")) + list(src_img_dir.glob("*.PNG")):
            dst = dst_img_dir / img.name
            if not dst.exists() and not dst.is_symlink():
                os.symlink(img.resolve(), dst)

        if src_lbl_dir.exists():
            for lbl in src_lbl_dir.glob("*.txt"):
                dst = dst_lbl_dir / lbl.name
                if not dst.exists() and not dst.is_symlink():
                    os.symlink(lbl.resolve(), dst)

    # ── Apply CLAHE preprocessing if requested ──────────────────────
    if apply_clahe:
        _apply_clahe_to_dir(
            dst_train_img, clahe_clip_limit, clahe_tile_size,
            tag="extended_stenosis/train"
        )

    # ── Write dataset YAML ──────────────────────────────────────────
    configs_dir = output_dir / "dataset_configs"
    configs_dir.mkdir(parents=True, exist_ok=True)
    yaml_cfg = {
        "path": str(output_dir.resolve()),
        "train": "images/train",
        "val":   "images/val",
        "test":  "images/test",
        "nc":    1,
        "names": {0: "stenosis"},
    }
    yaml_path = configs_dir / "stenosis_only.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(yaml_cfg, f, default_flow_style=False, sort_keys=False)

    return str(yaml_path)


def _apply_clahe_to_dir(
    img_dir: Path,
    clip_limit: float = 2.0,
    tile_size: int = 8,
    tag: str = "",
) -> None:
    """Apply CLAHE in-place to all images in img_dir.

    Skips symlinks pointing to files outside img_dir to avoid corrupting
    the original data — copies them to real files first.
    """
    try:
        import cv2
        import numpy as np
    except ImportError:
        print(f"  [{tag}] WARNING: cv2 not available, skipping CLAHE")
        return

    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=(tile_size, tile_size),
    )

    images = (
        list(img_dir.glob("*.png")) +
        list(img_dir.glob("*.PNG")) +
        list(img_dir.glob("*.jpg"))
    )

    processed = 0
    for p in images:
        # If it is a symlink, materialise it (copy then unlink symlink)
        # so we don't corrupt the source.
        if p.is_symlink():
            real_src = p.resolve()
            p.unlink()
            shutil.copy2(real_src, p)

        img = cv2.imread(str(p))
        if img is None:
            continue

        if len(img.shape) == 2 or img.shape[2] == 1:
            gray = img if len(img.shape) == 2 else img[:, :, 0]
            gray = clahe.apply(gray)
            cv2.imwrite(str(p), gray, [cv2.IMWRITE_JPEG_QUALITY, 95])
        else:
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l_ch, a_ch, b_ch = cv2.split(lab)
            l_ch = clahe.apply(l_ch)
            lab = cv2.merge([l_ch, a_ch, b_ch])
            out = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            cv2.imwrite(str(p), out, [cv2.IMWRITE_JPEG_QUALITY, 95])

        processed += 1

    print(f"  [{tag}] CLAHE applied to {processed}/{len(images)} images")


def main():
    parser = argparse.ArgumentParser(
        description="Generate pseudo-stenosis labels for syntax images (SSASS technique)"
    )
    parser.add_argument("--stenosis-model",  type=str, required=True)
    parser.add_argument("--syntax-img-dir",  type=str, required=True)
    parser.add_argument("--original-stenosis-dir", type=str, required=True)
    parser.add_argument("--output-dir",      type=str, required=True)
    parser.add_argument("--conf-threshold",  type=float, default=0.40)
    parser.add_argument("--imgsz",           type=int, default=768)
    parser.add_argument("--device",          type=str, default="0")
    parser.add_argument("--no-clahe",        action="store_true")
    args = parser.parse_args()

    pseudo_label_dir = Path(args.output_dir) / "pseudo_labels"

    print("=" * 60)
    print("Step 1: Generate pseudo-stenosis labels for syntax images")
    print("=" * 60)
    stats = run_stenosis_on_syntax_images(
        model_path=args.stenosis_model,
        syntax_img_dir=Path(args.syntax_img_dir),
        output_label_dir=pseudo_label_dir,
        conf_threshold=args.conf_threshold,
        imgsz=args.imgsz,
        device=args.device,
    )

    print("\n" + "=" * 60)
    print("Step 2: Build extended stenosis dataset")
    print("=" * 60)
    yaml_path = build_extended_stenosis_dataset(
        original_stenosis_dir=Path(args.original_stenosis_dir),
        syntax_img_dir=Path(args.syntax_img_dir),
        pseudo_label_dir=pseudo_label_dir,
        output_dir=Path(args.output_dir) / "extended_stenosis",
        apply_clahe=not args.no_clahe,
    )

    print(f"\nExtended dataset YAML: {yaml_path}")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
