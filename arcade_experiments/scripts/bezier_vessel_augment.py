"""Bezier-curve synthetic vessel augmentation (SSASS, Medipixel 2023).

Draws thin, curved, branching dark "vessel-like" segments onto coronary
angiogram backgrounds. The intent (per the SSASS paper) is to teach the
detector what generic vessel context looks like, so that the network
focuses its limited capacity on lesion morphology rather than vessel
recognition.

Used offline: builds a shadow copy of a YOLO image directory where a
configurable fraction of the training images receive 1-3 synthetic
curves. Existing labels (real stenoses) are preserved unchanged --
synthetic curves are background distractors, not new positives.
"""

from __future__ import annotations

import argparse
import os
import random
import shutil
from pathlib import Path

import cv2
import numpy as np


def _bezier_points(p0, p1, p2, p3, n: int = 64):
    t = np.linspace(0.0, 1.0, n).reshape(-1, 1)
    one_minus_t = 1.0 - t
    pts = (one_minus_t ** 3) * p0 \
        + 3 * (one_minus_t ** 2) * t * p1 \
        + 3 * one_minus_t * (t ** 2) * p2 \
        + (t ** 3) * p3
    return pts.astype(np.int32)


def _draw_one_curve(img: np.ndarray, rng: random.Random) -> None:
    h, w = img.shape[:2]

    cx, cy = w // 2, h // 2
    radius = min(w, h) // 3
    p0 = np.array([
        cx + rng.randint(-radius, radius),
        cy + rng.randint(-radius, radius),
    ], dtype=np.float32)
    p3 = p0 + np.array([
        rng.randint(-radius, radius),
        rng.randint(-radius, radius),
    ], dtype=np.float32)
    p1 = p0 + (p3 - p0) * 0.33 + np.array([
        rng.randint(-radius // 2, radius // 2),
        rng.randint(-radius // 2, radius // 2),
    ], dtype=np.float32)
    p2 = p0 + (p3 - p0) * 0.66 + np.array([
        rng.randint(-radius // 2, radius // 2),
        rng.randint(-radius // 2, radius // 2),
    ], dtype=np.float32)

    pts = _bezier_points(p0, p1, p2, p3, n=64).reshape(-1, 1, 2)

    if img.ndim == 2:
        bg_mean = float(img.mean())
    else:
        bg_mean = float(img[..., 0].mean())
    intensity = max(0, int(bg_mean * 0.4 + rng.gauss(0, 8)))
    color = intensity if img.ndim == 2 else (intensity, intensity, intensity)

    thickness = rng.randint(2, 4)
    cv2.polylines(img, [pts], isClosed=False, color=color,
                  thickness=thickness, lineType=cv2.LINE_AA)

    if rng.random() < 0.5:
        branch_t = rng.randint(20, 44)
        anchor = pts[branch_t, 0]
        b3 = anchor + np.array([
            rng.randint(-radius // 2, radius // 2),
            rng.randint(-radius // 2, radius // 2),
        ], dtype=np.int32)
        b1 = anchor + (b3 - anchor) * 0 + np.array([
            rng.randint(-20, 20), rng.randint(-20, 20)
        ], dtype=np.int32)
        b2 = anchor + (b3 - anchor) // 2 + np.array([
            rng.randint(-20, 20), rng.randint(-20, 20)
        ], dtype=np.int32)
        bpts = _bezier_points(
            anchor.astype(np.float32),
            b1.astype(np.float32),
            b2.astype(np.float32),
            b3.astype(np.float32),
            n=32,
        ).reshape(-1, 1, 2)
        cv2.polylines(img, [bpts], isClosed=False, color=color,
                      thickness=max(1, thickness - 1), lineType=cv2.LINE_AA)


def augment_image(img: np.ndarray, rng: random.Random,
                  n_curves_min: int = 1, n_curves_max: int = 3) -> np.ndarray:
    out = img.copy()
    n = rng.randint(n_curves_min, n_curves_max)
    for _ in range(n):
        _draw_one_curve(out, rng)
    return out


def augment_dataset(
    images_dir: Path,
    labels_dir: Path,
    output_images_dir: Path,
    output_labels_dir: Path,
    fraction: float = 0.4,
    seed: int = 42,
) -> dict:
    """Symlink originals + write augmented copies for `fraction` of images.

    Output dataset = original 1200 + ~480 augmented = ~1680 images.
    """
    output_images_dir.mkdir(parents=True, exist_ok=True)
    output_labels_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)

    img_paths = sorted(
        list(images_dir.glob("*.png"))
        + list(images_dir.glob("*.PNG"))
        + list(images_dir.glob("*.jpg"))
    )

    n_orig = 0
    n_aug = 0

    for img_path in img_paths:
        dst_img = output_images_dir / img_path.name
        if not dst_img.exists() and not dst_img.is_symlink():
            os.symlink(img_path.resolve(), dst_img)
        n_orig += 1

        lbl_src = labels_dir / (img_path.stem + ".txt")
        if lbl_src.exists():
            dst_lbl = output_labels_dir / lbl_src.name
            if not dst_lbl.exists() and not dst_lbl.is_symlink():
                os.symlink(lbl_src.resolve(), dst_lbl)

        if rng.random() < fraction:
            img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
            if img is None:
                continue
            aug = augment_image(img, rng)
            aug_name = f"{img_path.stem}_bez.png"
            cv2.imwrite(str(output_images_dir / aug_name), aug)

            if lbl_src.exists():
                aug_lbl = output_labels_dir / f"{img_path.stem}_bez.txt"
                shutil.copy2(str(lbl_src), str(aug_lbl))
            else:
                # Empty label = background image (still useful for training)
                (output_labels_dir / f"{img_path.stem}_bez.txt").write_text("")
            n_aug += 1

    return {"original": n_orig, "augmented": n_aug,
            "total": n_orig + n_aug, "fraction": fraction}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--images", type=Path, required=True)
    p.add_argument("--labels", type=Path, required=True)
    p.add_argument("--out-images", type=Path, required=True)
    p.add_argument("--out-labels", type=Path, required=True)
    p.add_argument("--fraction", type=float, default=0.4)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    stats = augment_dataset(
        args.images, args.labels,
        args.out_images, args.out_labels,
        fraction=args.fraction, seed=args.seed,
    )
    print(f"Bezier augmentation: {stats}")


if __name__ == "__main__":
    main()
