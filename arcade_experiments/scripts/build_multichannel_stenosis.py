"""Multi-channel stenosis input (E3, StenUNet -> YOLO port).

For each grayscale angiogram, build a 3-channel BGR PNG where:
    B = raw grayscale
    G = CLAHE-enhanced grayscale (clip=2, tile=8x8)
    R = directional vessel enhancement (Gabor filter bank max response)

This mimics StenUNet's three-modality input but stays inside the
3-channel constraint of standard YOLO weights so we can still init
from COCO. Operates in place on a YOLO-style dataset directory
(images/{train,val,test}).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def _gabor_max_response(gray: np.ndarray, n_orient: int = 8,
                       ksize: int = 21, sigma: float = 3.0,
                       lambd: float = 8.0, gamma: float = 0.5) -> np.ndarray:
    if gray.dtype != np.float32:
        gray32 = gray.astype(np.float32)
    else:
        gray32 = gray
    accum = None
    for i in range(n_orient):
        theta = i * np.pi / n_orient
        kern = cv2.getGaborKernel(
            (ksize, ksize), sigma, theta, lambd, gamma, 0, ktype=cv2.CV_32F)
        kern /= 1.5 * kern.sum() if kern.sum() > 1e-3 else 1.0
        resp = cv2.filter2D(gray32, cv2.CV_32F, kern)
        np.abs(resp, out=resp)
        accum = resp if accum is None else np.maximum(accum, resp)
    if accum is None:
        return gray
    accum -= accum.min()
    if accum.max() > 1e-6:
        accum *= 255.0 / accum.max()
    return accum.astype(np.uint8)


def stack_image(img_path: Path, clip_limit: float = 2.0,
                tile_size: int = 8) -> np.ndarray:
    img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"failed to read {img_path}")
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    clahe = cv2.createCLAHE(clipLimit=clip_limit,
                             tileGridSize=(tile_size, tile_size))
    clahe_ch = clahe.apply(gray)

    gabor_ch = _gabor_max_response(gray)

    return cv2.merge([gray, clahe_ch, gabor_ch])


def stack_directory(img_dir: Path, clip_limit: float = 2.0,
                    tile_size: int = 8) -> int:
    n = 0
    for p in list(img_dir.rglob("*.png")) + list(img_dir.rglob("*.PNG")) \
            + list(img_dir.rglob("*.jpg")):
        try:
            stacked = stack_image(p, clip_limit=clip_limit, tile_size=tile_size)
            cv2.imwrite(str(p), stacked)
            n += 1
        except Exception as exc:
            print(f"  WARN stack failed for {p}: {exc}")
    print(f"  Multichannel stack: processed {n} images under {img_dir}")
    return n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--img-dir", type=Path, required=True,
                   help="Top-level images dir (will recurse into train/val/test)")
    p.add_argument("--clip-limit", type=float, default=2.0)
    p.add_argument("--tile-size", type=int, default=8)
    args = p.parse_args()
    stack_directory(args.img_dir, args.clip_limit, args.tile_size)


if __name__ == "__main__":
    main()
