#!/usr/bin/env python3
"""Generate cross-task pseudo-labels using the best models.

Syntax model (S54) -> run on all stenosis images -> pseudo-syntax labels
Stenosis model (V2 SSASS) -> run on all syntax images -> pseudo-stenosis labels

Output (YOLO format, one .txt per image):
    arcade_experiments/data/pseudo_labels/
      syntax_on_stenosis/{train,val,test}/  (syntax labels for stenosis images)
      stenosis_on_syntax/{train,val,test}/  (stenosis labels for syntax images)

Usage:
    python generate_cross_pseudo_labels.py \
        --arcade-root ../../arcade/submission \
        --syntax-model ../results/stenosis_strategies/S54_s43b_clahe/syntax_model/syntax_768_best.pt \
        --stenosis-model ../results/ssass_variants/V2_M2/V2_m2_best.pt \
        --device 0
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))


def log(msg, log_file):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(log_file, "a") as f:
        f.write(line + "\n")


def generate_pseudo_labels(model_path, img_dir, output_dir, conf, imgsz, device, log_file):
    """Run model on images in img_dir, write YOLO polygon labels to output_dir."""
    from ultralytics import YOLO

    output_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(model_path)

    imgs = sorted(list(img_dir.glob("*.png")) + list(img_dir.glob("*.PNG"))
                  + list(img_dir.glob("*.jpg")))
    n_total = len(imgs)
    n_pseudo = 0
    n_inst = 0

    for img_path in imgs:
        results = model.predict(
            source=str(img_path), conf=conf, imgsz=imgsz,
            device=str(device), verbose=False, save=False, retina_masks=True)
        if not results or results[0].masks is None or len(results[0].masks) == 0:
            continue
        r = results[0]
        lines = []
        for i, mask_xyn in enumerate(r.masks.xyn):
            cls = int(r.boxes.cls[i].item())
            if len(mask_xyn) < 3:
                continue
            coords = " ".join(f"{pt[0]:.6f} {pt[1]:.6f}" for pt in mask_xyn)
            lines.append(f"{cls} {coords}")
        if lines:
            (output_dir / f"{img_path.stem}.txt").write_text(
                "\n".join(lines) + "\n")
            n_pseudo += 1
            n_inst += len(lines)

    log(f"    {n_pseudo}/{n_total} images pseudo-labeled, {n_inst} instances", log_file)
    return {"total": n_total, "pseudo": n_pseudo, "instances": n_inst}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arcade-root", type=Path, required=True)
    parser.add_argument("--syntax-model", type=Path, required=True)
    parser.add_argument("--stenosis-model", type=Path, required=True)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Default: arcade_experiments/data/pseudo_labels")
    parser.add_argument("--conf", type=float, default=0.3,
                        help="Confidence threshold (default: 0.3, V2 SSASS best)")
    parser.add_argument("--syntax-imgsz", type=int, default=768)
    parser.add_argument("--stenosis-imgsz", type=int, default=640)
    args = parser.parse_args()

    arcade_root = args.arcade_root.resolve()
    output_dir = (args.output_dir or
                  SCRIPT_DIR.parent / "data" / "pseudo_labels").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / "generation_log.txt"

    log(f"ARCADE root:    {arcade_root}", log_file)
    log(f"Syntax model:   {args.syntax_model}", log_file)
    log(f"Stenosis model: {args.stenosis_model}", log_file)
    log(f"Conf threshold: {args.conf}", log_file)
    log(f"Output dir:     {output_dir}", log_file)

    t0 = time.time()
    summary = {"syntax_on_stenosis": {}, "stenosis_on_syntax": {}}

    # ── Syntax model -> stenosis images (cross-task) ──
    log(f"\n{'='*60}", log_file)
    log("Syntax model -> stenosis images (pseudo-syntax labels)", log_file)
    log(f"{'='*60}", log_file)

    for split in ("train", "val", "test"):
        img_dir = arcade_root / "stenosis" / split / "images"
        if not img_dir.exists():
            log(f"  [SKIP] {split}: {img_dir} not found", log_file)
            continue
        out_dir = output_dir / "syntax_on_stenosis" / split
        log(f"\n  [{split}] {img_dir}", log_file)
        stats = generate_pseudo_labels(
            str(args.syntax_model.resolve()), img_dir, out_dir,
            args.conf, args.syntax_imgsz, args.device, log_file)
        summary["syntax_on_stenosis"][split] = stats

    # ── Stenosis model -> syntax images (cross-task) ──
    log(f"\n{'='*60}", log_file)
    log("Stenosis model -> syntax images (pseudo-stenosis labels)", log_file)
    log(f"{'='*60}", log_file)

    for split in ("train", "val", "test"):
        img_dir = arcade_root / "syntax" / split / "images"
        if not img_dir.exists():
            log(f"  [SKIP] {split}: {img_dir} not found", log_file)
            continue
        out_dir = output_dir / "stenosis_on_syntax" / split
        log(f"\n  [{split}] {img_dir}", log_file)
        stats = generate_pseudo_labels(
            str(args.stenosis_model.resolve()), img_dir, out_dir,
            args.conf, args.stenosis_imgsz, args.device, log_file)
        summary["stenosis_on_syntax"][split] = stats

    # Summary
    log(f"\n{'='*60}", log_file)
    log("SUMMARY", log_file)
    log(f"{'='*60}", log_file)
    for task, splits in summary.items():
        total_imgs = sum(s["total"] for s in splits.values())
        total_pseudo = sum(s["pseudo"] for s in splits.values())
        total_inst = sum(s["instances"] for s in splits.values())
        log(f"  {task}: {total_pseudo}/{total_imgs} images, {total_inst} instances", log_file)
        for split, s in splits.items():
            log(f"    {split}: {s['pseudo']}/{s['total']} ({s['instances']} inst)", log_file)

    log(f"\nTotal time: {(time.time()-t0)/60:.1f} min", log_file)
    log(f"Output: {output_dir}", log_file)

    json.dump({
        "conf": args.conf,
        "syntax_model": str(args.syntax_model),
        "stenosis_model": str(args.stenosis_model),
        "summary": summary,
    }, open(output_dir / "generation_summary.json", "w"),
       indent=2, default=str)


if __name__ == "__main__":
    main()
