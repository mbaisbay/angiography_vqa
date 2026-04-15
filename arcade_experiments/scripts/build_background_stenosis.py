"""B-3: Add hard-negative background images to the stenosis training set.

Idea: run the current best stenosis model on the 1000 syntax training
images. Any syntax image where the model predicts NOTHING at a moderate
confidence is a confirmed vessel-only negative → useful as a background
image. Images where the model predicted stenosis at low confidence
(0.1-0.5) are hard negatives → even more useful.

Adds ~10-15% background images to the stenosis train set by creating
empty-label symlinks.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def build(
    stenosis_model: str,
    syntax_img_dir: Path,
    stenosis_data_dir: Path,
    imgsz: int,
    device: str,
    n_hard: int,
    low_conf: float = 0.1,
    high_conf: float = 0.5,
) -> dict:
    from ultralytics import YOLO

    model = YOLO(stenosis_model)
    img_files = sorted(list(syntax_img_dir.glob("*.png"))
                        + list(syntax_img_dir.glob("*.PNG")))

    # Collect max predicted conf per image in [low_conf, high_conf]
    hard = []   # (stem, max_conf)
    clean = []  # no predictions at all above low_conf
    for p in img_files:
        results = model.predict(
            source=str(p), conf=low_conf, imgsz=imgsz,
            device=str(device), verbose=False, save=False, retina_masks=False,
        )
        if not results or results[0].masks is None or len(results[0].masks) == 0:
            clean.append(p)
            continue
        r = results[0]
        confs = r.boxes.conf.cpu().numpy().tolist()
        if confs:
            max_c = max(confs)
            if low_conf <= max_c < high_conf:
                hard.append((p, max_c))
            # If any conf >= high_conf, this is actually a positive candidate → skip
        else:
            clean.append(p)

    # Rank hard negatives by max_conf descending (highest = hardest to reject)
    hard.sort(key=lambda x: -x[1])
    hard_selected = [h[0] for h in hard[:n_hard]]

    # Add hard negatives + some clean negatives to reach target ratio
    needed_clean = max(0, n_hard - len(hard_selected))
    selected = hard_selected + clean[:needed_clean]

    # Symlink into stenosis train with bg_ prefix, write empty label files
    dst_img = stenosis_data_dir / "images" / "train"
    dst_lbl = stenosis_data_dir / "labels" / "train"
    dst_img.mkdir(parents=True, exist_ok=True)
    dst_lbl.mkdir(parents=True, exist_ok=True)

    added = 0
    for src in selected:
        name = f"bg_{src.name}"
        dst_p = dst_img / name
        if dst_p.exists() or dst_p.is_symlink():
            continue
        os.symlink(src.resolve(), dst_p)
        (dst_lbl / f"bg_{src.stem}.txt").write_text("")
        added += 1

    print(f"Added {added} background images to {dst_img}")
    print(f"  hard negatives: {len(hard_selected)}  clean: {added - len(hard_selected)}")
    return {"total_added": added,
            "hard_negatives": len(hard_selected),
            "clean_negatives": added - len(hard_selected)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stenosis-model", required=True)
    p.add_argument("--syntax-img-dir", type=Path, required=True)
    p.add_argument("--stenosis-data-dir", type=Path, required=True)
    p.add_argument("--imgsz", type=int, default=768)
    p.add_argument("--device", type=str, default="0")
    p.add_argument("--n-hard", type=int, default=150)
    args = p.parse_args()

    stats = build(args.stenosis_model, args.syntax_img_dir,
                  args.stenosis_data_dir, args.imgsz, args.device, args.n_hard)
    import json
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
