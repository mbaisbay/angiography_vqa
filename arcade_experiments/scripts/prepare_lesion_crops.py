"""Generate lesion-centric stenosis training crops.

Rationale
---------
Stenoses are tiny objects (~70x70 px in a 512x512 angiogram — roughly
2% of the image area). Training a YOLO detection head on them at
imgsz=768 places each stenosis in ~100 pixels of tensor — below the
receptive field of the stride-8 head, deep below stride-16/32.

This script generates a lesion-centric training dataset: for every
stenosis in the train split, we cut N random 256x256 windows centred
near the lesion. YOLO then resizes those crops to imgsz=768 at train
time, producing a 3x upsampling of the effective lesion resolution.
The lesion occupies ~200-300 tensor pixels — comfortably inside the
receptive field of all three detection heads.

Design choices
--------------
  1. **Crops PLUS full images.** For each train image we keep the
     original full-resolution frame *and* generate N crops. This
     prevents catastrophic distribution shift at test time, where
     the model sees full frames again. The final train set is a
     mix of (native-scale context) + (upsampled lesion views).

  2. **Random jitter** in the crop center (±50 px default). Teaches
     the model to localize, not to rely on the lesion being in the
     middle.

  3. **Multi-lesion crops kept.** When a crop window happens to
     contain a second stenosis from the same image, its polygon is
     remapped into crop coordinates and written to the label file.

  4. **Val and test splits are untouched.** We symlink the original
     val/test images + labels so evaluation metrics are directly
     comparable to every other experiment in the project.

Usage
-----
    python prepare_lesion_crops.py \
        --stenosis-dir /path/to/data/stenosis \
        --output-dir   /path/to/data/stenosis_lesion_crops \
        --n-crops      3 \
        --crop-size    256 \
        --jitter-px    50 \
        --seed         42

Outputs
-------
    stenosis_lesion_crops/
      images/
        train/   lesion_<img_id>_<lesion_idx>_<k>.png  (crop_size x crop_size)
        val/     <orig>.png  (symlinks)
        test/    <orig>.png  (symlinks)
      labels/
        train/   lesion_<img_id>_<lesion_idx>_<k>.txt  (remapped polygons)
        val/     <orig>.txt  (symlinks)
        test/    <orig>.txt  (symlinks)
      dataset_configs/
        stenosis_only.yaml   (points at this directory)
"""

import argparse
import json
import os
import random
from collections import defaultdict
from pathlib import Path

import yaml


SPLITS = ["train", "val", "test"]


def _bbox_center(bbox: list) -> tuple:
    """COCO bbox [x, y, w, h] -> (cx, cy)."""
    x, y, w, h = bbox
    return x + w / 2.0, y + h / 2.0


def _polygon_to_crop_coords(polygon: list, x0: float, y0: float,
                             crop_w: float, crop_h: float) -> list:
    """Shift polygon coords into crop frame. Returns None if polygon
    is fully outside the crop.
    """
    out = []
    any_inside = False
    for i in range(0, len(polygon), 2):
        px = polygon[i] - x0
        py = polygon[i + 1] - y0
        # Clamp to crop bounds; skip polygons with no vertex inside
        if 0 <= px <= crop_w and 0 <= py <= crop_h:
            any_inside = True
        px = max(0.0, min(crop_w, px))
        py = max(0.0, min(crop_h, py))
        out.extend([px, py])
    if not any_inside:
        return None
    return out


def _write_yolo_label(label_path: Path, crop_anns: list,
                      crop_w: int, crop_h: int) -> int:
    """Write YOLO-seg labels for crop annotations. Returns count."""
    lines = []
    for ann in crop_anns:
        for polygon in ann["segmentation"]:
            if len(polygon) < 6:
                continue
            shifted = _polygon_to_crop_coords(
                polygon, ann["_x0"], ann["_y0"], crop_w, crop_h
            )
            if shifted is None or len(shifted) < 6:
                continue
            normalized = []
            for i in range(0, len(shifted), 2):
                nx = max(0.0, min(1.0, shifted[i] / crop_w))
                ny = max(0.0, min(1.0, shifted[i + 1] / crop_h))
                normalized.extend([f"{nx:.6f}", f"{ny:.6f}"])
            lines.append("0 " + " ".join(normalized))
    if lines:
        label_path.write_text("\n".join(lines) + "\n")
    return len(lines)


def _symlink_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        dst.symlink_to(src.resolve())
    except OSError:
        import shutil
        shutil.copy2(src, dst)


def generate_lesion_crops(
    stenosis_dir: Path,
    output_dir: Path,
    n_crops: int = 3,
    crop_size: int = 256,
    jitter_px: int = 50,
    seed: int = 42,
    keep_full_images: bool = True,
) -> dict:
    """Generate a lesion-centric training dataset variant.

    Args:
        stenosis_dir: path to existing stenosis dataset (from prepare_data.py)
                      expected layout:
                        images/{train,val,test}/*.png
                        labels/{train,val,test}/*.txt   (optional)
                        annotations/{train,val,test}.json
        output_dir: destination for new dataset variant
        n_crops: crops per lesion (default 3)
        crop_size: square crop side (default 256)
        jitter_px: random center jitter (default 50)
        seed: RNG seed
        keep_full_images: also include the native full-resolution frame
                          in the train set (default True). Prevents
                          test-time distribution shift.

    Returns:
        stats dict.
    """
    from PIL import Image  # lazy import

    rng = random.Random(seed)

    stenosis_dir = Path(stenosis_dir).resolve()
    output_dir = Path(output_dir).resolve()

    train_ann_json = stenosis_dir / "annotations" / "train.json"
    if not train_ann_json.exists():
        raise FileNotFoundError(
            f"Expected stenosis COCO JSON at {train_ann_json}. "
            "Run prepare_data.py first."
        )

    with open(train_ann_json) as f:
        train_coco = json.load(f)

    images_by_id = {img["id"]: img for img in train_coco["images"]}
    anns_by_image = defaultdict(list)
    for a in train_coco["annotations"]:
        anns_by_image[a["image_id"]].append(a)

    # ── TRAIN: generate crops + optional full images ──
    out_train_img_dir = output_dir / "images" / "train"
    out_train_lbl_dir = output_dir / "labels" / "train"
    out_train_img_dir.mkdir(parents=True, exist_ok=True)
    out_train_lbl_dir.mkdir(parents=True, exist_ok=True)

    stats = {
        "train_full_images_kept": 0,
        "train_crops_written": 0,
        "train_crops_with_labels": 0,
        "lesion_instances_seen": 0,
        "crops_with_multi_lesion": 0,
        "skipped_crops_oob": 0,
    }

    src_train_img_dir = stenosis_dir / "images" / "train"

    for img_id, img_info in images_by_id.items():
        fname = img_info["file_name"]
        src_img_path = src_train_img_dir / fname
        if not src_img_path.exists():
            continue

        H = img_info["height"]
        W = img_info["width"]
        anns = anns_by_image.get(img_id, [])

        # ── Full-image pass-through ──
        if keep_full_images:
            dst_img = out_train_img_dir / fname
            _symlink_or_copy(src_img_path, dst_img)
            # Also symlink YOLO label if present
            src_lbl = stenosis_dir / "labels" / "train" / (Path(fname).stem + ".txt")
            if src_lbl.exists():
                dst_lbl = out_train_lbl_dir / (Path(fname).stem + ".txt")
                _symlink_or_copy(src_lbl, dst_lbl)
            stats["train_full_images_kept"] += 1

        if not anns:
            continue

        # Load image once per source frame if we'll be cropping
        img_pil = None

        for lesion_idx, ann in enumerate(anns):
            stats["lesion_instances_seen"] += 1
            cx, cy = _bbox_center(ann["bbox"])

            for k in range(n_crops):
                # Randomly jittered center
                jx = rng.uniform(-jitter_px, jitter_px)
                jy = rng.uniform(-jitter_px, jitter_px)
                ccx = cx + jx
                ccy = cy + jy
                half = crop_size / 2.0
                x0 = ccx - half
                y0 = ccy - half
                x1 = x0 + crop_size
                y1 = y0 + crop_size

                # Clamp to image; if crop_size > image dim, fall back to full image
                if crop_size > W or crop_size > H:
                    stats["skipped_crops_oob"] += 1
                    continue
                if x0 < 0:
                    x0, x1 = 0, crop_size
                if y0 < 0:
                    y0, y1 = 0, crop_size
                if x1 > W:
                    x0, x1 = W - crop_size, W
                if y1 > H:
                    y0, y1 = H - crop_size, H

                x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)

                # Collect all annotations whose bbox center falls inside the crop
                crop_anns = []
                for other in anns:
                    ocx, ocy = _bbox_center(other["bbox"])
                    if x0 <= ocx < x1 and y0 <= ocy < y1:
                        ca = dict(other)
                        ca["_x0"] = x0
                        ca["_y0"] = y0
                        crop_anns.append(ca)

                if len(crop_anns) > 1:
                    stats["crops_with_multi_lesion"] += 1

                if not crop_anns:
                    continue  # should not happen since primary lesion is inside

                # Lazy-load source image
                if img_pil is None:
                    img_pil = Image.open(src_img_path).convert("RGB")

                crop_img = img_pil.crop((x0, y0, x1, y1))
                crop_name_stem = f"lesion_{img_id}_{lesion_idx}_{k}"
                crop_img_path = out_train_img_dir / f"{crop_name_stem}.png"
                crop_img.save(crop_img_path)

                # Write YOLO label (normalized polygons inside crop)
                crop_lbl_path = out_train_lbl_dir / f"{crop_name_stem}.txt"
                n_polys = _write_yolo_label(
                    crop_lbl_path, crop_anns, crop_size, crop_size
                )
                stats["train_crops_written"] += 1
                if n_polys > 0:
                    stats["train_crops_with_labels"] += 1

    # ── VAL + TEST: mirror original ──
    for split in ("val", "test"):
        src_img_dir = stenosis_dir / "images" / split
        src_lbl_dir = stenosis_dir / "labels" / split
        dst_img_dir = output_dir / "images" / split
        dst_lbl_dir = output_dir / "labels" / split
        dst_img_dir.mkdir(parents=True, exist_ok=True)
        dst_lbl_dir.mkdir(parents=True, exist_ok=True)

        if src_img_dir.exists():
            for img_path in src_img_dir.iterdir():
                if img_path.suffix.lower() in (".png", ".jpg", ".jpeg"):
                    _symlink_or_copy(img_path, dst_img_dir / img_path.name)

        if src_lbl_dir.exists():
            for lbl_path in src_lbl_dir.iterdir():
                if lbl_path.suffix == ".txt":
                    _symlink_or_copy(lbl_path, dst_lbl_dir / lbl_path.name)

    # ── Write dataset YAML ──
    configs_dir = output_dir / "dataset_configs"
    configs_dir.mkdir(parents=True, exist_ok=True)
    yaml_cfg = {
        "path": str(output_dir),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": 1,
        "names": {0: "stenosis"},
    }
    yaml_path = configs_dir / "stenosis_only.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(yaml_cfg, f, default_flow_style=False, sort_keys=False)

    return {**stats, "dataset_yaml": str(yaml_path)}


def main():
    parser = argparse.ArgumentParser(
        description="Generate lesion-centric stenosis training crops"
    )
    parser.add_argument("--stenosis-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--n-crops", type=int, default=3)
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--jitter-px", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-full-images", action="store_true",
                        help="Skip full-image pass-through (crops only)")
    args = parser.parse_args()

    stats = generate_lesion_crops(
        Path(args.stenosis_dir),
        Path(args.output_dir),
        n_crops=args.n_crops,
        crop_size=args.crop_size,
        jitter_px=args.jitter_px,
        seed=args.seed,
        keep_full_images=not args.no_full_images,
    )
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
