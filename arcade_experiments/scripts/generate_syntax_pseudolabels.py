"""Cross-Task Pseudo-Label generator: syntax model -> stenosis images.

Inverse of `generate_stenosis_pseudolabels.py`. The intuition (4th place
ARCADE 2023, arXiv 2310.05990): the 1000 stenosis-task images contain
the same coronary anatomy as the syntax images but were never annotated
for vessel class. A syntax model trained on the 1000 syntax images can
be run on those stenosis images to write pseudo vessel-class polygons.
The combined 2000-image dataset doubles the syntax training data.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import yaml


def run_syntax_on_stenosis_images(
    model_path: str,
    stenosis_img_dir: Path,
    output_label_dir: Path,
    conf_threshold: float = 0.5,
    imgsz: int = 768,
    device: str = "0",
) -> dict:
    """Run a multi-class syntax model on stenosis images, write YOLO labels.

    Output label format: standard YOLO segmentation, one polygon per
    instance, class id taken from the model prediction.
    """
    from ultralytics import YOLO

    output_label_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(model_path)
    # NOTE: do NOT call model.to(f"cuda:{device}") here. In worker
    # subprocesses the parent sets CUDA_VISIBLE_DEVICES to the physical
    # GPU id, so from torch's view only cuda:0 exists — a direct
    # .to("cuda:1") raises "invalid device ordinal". Pass device=... to
    # predict() instead; ultralytics' select_device handles the
    # remapping correctly.

    img_files = sorted(
        list(stenosis_img_dir.glob("*.png"))
        + list(stenosis_img_dir.glob("*.PNG"))
        + list(stenosis_img_dir.glob("*.jpg"))
    )

    stats = {
        "images_processed": 0,
        "images_with_predictions": 0,
        "total_pseudo_instances": 0,
        "conf_threshold": conf_threshold,
    }
    by_class = {}

    for img_path in img_files:
        stats["images_processed"] += 1
        results = model.predict(
            source=str(img_path),
            conf=conf_threshold,
            imgsz=imgsz,
            device=str(device),
            verbose=False,
            save=False,
            retina_masks=True,
        )
        if not results or results[0].masks is None:
            continue
        r = results[0]
        masks = r.masks
        cls = r.boxes.cls.cpu().numpy().astype(int)
        if len(masks) == 0:
            continue

        lines = []
        for k, mask_xyn in enumerate(masks.xyn):
            if len(mask_xyn) < 3:
                continue
            cls_id = int(cls[k])
            # Skip the stenosis class (25) -- we only want vessel-class pseudo-labels
            if cls_id == 25:
                continue
            coords = []
            for pt in mask_xyn:
                coords.extend([f"{pt[0]:.6f}", f"{pt[1]:.6f}"])
            lines.append(f"{cls_id} " + " ".join(coords))
            by_class[cls_id] = by_class.get(cls_id, 0) + 1

        if lines:
            (output_label_dir / f"{img_path.stem}.txt").write_text(
                "\n".join(lines) + "\n")
            stats["images_with_predictions"] += 1
            stats["total_pseudo_instances"] += len(lines)

    stats["per_class_counts"] = by_class
    print("  Syntax pseudo-labels:")
    print(f"    Images processed:   {stats['images_processed']}")
    print(f"    With predictions:   {stats['images_with_predictions']}")
    print(f"    Total instances:    {stats['total_pseudo_instances']}")
    print(f"    Per-class:          {by_class}")
    return stats


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--syntax-model", required=True)
    p.add_argument("--stenosis-img-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--conf-threshold", type=float, default=0.5)
    p.add_argument("--imgsz", type=int, default=768)
    p.add_argument("--device", default="0")
    args = p.parse_args()

    stats = run_syntax_on_stenosis_images(
        model_path=args.syntax_model,
        stenosis_img_dir=args.stenosis_img_dir,
        output_label_dir=args.out_dir,
        conf_threshold=args.conf_threshold,
        imgsz=args.imgsz,
        device=args.device,
    )
    import json
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
