"""B-6: Offline synthetic copy-paste augmentation for tail syntax classes.

For each training instance of classes 9, 13, 16, paste its polygon
(with slight rotation/scale jitter) onto a different training image.
The pasted polygon is drawn with the same intensity profile as the
source region so it looks like a vessel branch.

Creates `synth_<src>_<dst>_<i>.png` + YOLO label files in the existing
train split.
"""

from __future__ import annotations

import argparse
import os
import random
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


def _read_poly_labels(p: Path):
    out = []
    if not p.exists():
        return out
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or len(line.split()) < 7:
            continue
        parts = line.split()
        cls = int(parts[0])
        coords = np.array(parts[1:], dtype=np.float32).reshape(-1, 2)
        out.append({"cls": cls, "poly": coords, "raw": line})
    return out


def _poly_to_mask(poly_norm: np.ndarray, h: int, w: int) -> np.ndarray:
    px = poly_norm.copy()
    px[:, 0] *= w
    px[:, 1] *= h
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [px.astype(np.int32)], 1)
    return mask


def _mask_bbox(mask: np.ndarray):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)


def _transform_patch(src_img: np.ndarray, mask: np.ndarray,
                      poly_px: np.ndarray, rng: random.Random):
    angle = rng.uniform(-15, 15)
    scale = rng.uniform(0.85, 1.15)
    h, w = src_img.shape[:2]
    bb = _mask_bbox(mask)
    if bb is None:
        return None, None, None
    x1, y1, x2, y2 = bb
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    M = cv2.getRotationMatrix2D((cx, cy), angle, scale)
    rot_img = cv2.warpAffine(src_img, M, (w, h), flags=cv2.INTER_LINEAR)
    rot_mask = cv2.warpAffine(mask * 255, M, (w, h), flags=cv2.INTER_NEAREST)
    rot_mask = (rot_mask > 127).astype(np.uint8)

    # Transform poly points
    ones = np.ones((poly_px.shape[0], 1))
    stacked = np.hstack([poly_px, ones])
    new_px = (M @ stacked.T).T
    return rot_img, rot_mask, new_px


def synth_one(src_img: np.ndarray, src_poly: np.ndarray,
               dst_img: np.ndarray, rng: random.Random):
    h, w = src_img.shape[:2]
    dh, dw = dst_img.shape[:2]
    # Resize source to dst dims
    if (h, w) != (dh, dw):
        src_img = cv2.resize(src_img, (dw, dh))
        src_poly = src_poly.copy()  # normalized coords, so no rescaling needed
    src_px = src_poly.copy()
    src_px[:, 0] *= dw
    src_px[:, 1] *= dh

    mask = np.zeros((dh, dw), dtype=np.uint8)
    cv2.fillPoly(mask, [src_px.astype(np.int32)], 1)

    rot_img, rot_mask, new_px = _transform_patch(src_img, mask, src_px, rng)
    if rot_mask is None or rot_mask.sum() < 30:
        return None, None

    # Paste at a random translation keeping inside the image
    bb = _mask_bbox(rot_mask)
    if bb is None:
        return None, None
    x1, y1, x2, y2 = bb
    max_dx = max(0, dw - (x2 - x1) - 10)
    max_dy = max(0, dh - (y2 - y1) - 10)
    tx = rng.randint(-x1 + 5, max_dx - x1 + 5) if max_dx > 0 else 0
    ty = rng.randint(-y1 + 5, max_dy - y1 + 5) if max_dy > 0 else 0
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    shifted_img = cv2.warpAffine(rot_img, M, (dw, dh))
    shifted_mask = cv2.warpAffine(rot_mask * 255, M, (dw, dh))
    shifted_mask = (shifted_mask > 127).astype(np.uint8)

    out = dst_img.copy()
    m3 = np.stack([shifted_mask] * 3, axis=-1) if out.ndim == 3 else shifted_mask
    out = np.where(m3 > 0, shifted_img, out)

    # Recover the shifted polygon in normalized coords
    shifted_px = new_px.copy()
    shifted_px[:, 0] += tx
    shifted_px[:, 1] += ty
    shifted_norm = shifted_px.copy()
    shifted_norm[:, 0] /= dw
    shifted_norm[:, 1] /= dh
    return out, shifted_norm


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--syntax-data-dir", type=Path, required=True,
                   help="dir with images/train, labels/train (YOLO)")
    p.add_argument("--tail-classes", type=str, default="9,13,16")
    p.add_argument("--per-instance", type=int, default=2,
                   help="How many synthetic composites per source instance")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    tail = {int(c) for c in args.tail_classes.split(",")}
    rng = random.Random(args.seed)

    img_dir = args.syntax_data_dir / "images" / "train"
    lbl_dir = args.syntax_data_dir / "labels" / "train"

    all_imgs = sorted(list(img_dir.glob("*.png"))
                       + list(img_dir.glob("*.PNG")))

    # Index: (class -> [(img_path, poly)])
    instances = defaultdict(list)
    for img_path in all_imgs:
        labs = _read_poly_labels(lbl_dir / (img_path.stem + ".txt"))
        for l in labs:
            if l["cls"] in tail:
                instances[l["cls"]].append((img_path, l["poly"]))

    print(f"Found tail instances: { {c: len(v) for c, v in instances.items()} }")

    created = 0
    for cls, inst_list in instances.items():
        for src_path, src_poly in inst_list:
            for i in range(args.per_instance):
                dst_path = rng.choice(all_imgs)
                if dst_path == src_path:
                    continue
                src_img = cv2.imread(str(src_path))
                dst_img = cv2.imread(str(dst_path))
                if src_img is None or dst_img is None:
                    continue
                out, new_poly = synth_one(src_img, src_poly, dst_img, rng)
                if out is None or new_poly is None:
                    continue

                out_stem = f"synth_{cls}_{src_path.stem}_{dst_path.stem}_{i}"
                out_img = img_dir / f"{out_stem}.png"
                out_lbl = lbl_dir / f"{out_stem}.txt"
                cv2.imwrite(str(out_img), out)

                # Build label = original labels of dst + new synthetic polygon
                lines = []
                existing = _read_poly_labels(lbl_dir / (dst_path.stem + ".txt"))
                for e in existing:
                    lines.append(e["raw"])
                coord_str = " ".join(f"{v:.6f}" for v in new_poly.flatten())
                lines.append(f"{cls} {coord_str}")
                out_lbl.write_text("\n".join(lines) + "\n")
                created += 1

    print(f"Created {created} synthetic composite images")


if __name__ == "__main__":
    main()
