#!/usr/bin/env python3
"""S54 two-stage recipe + single-round pseudo-labels.

Uses the actual S54 stenosis weights to generate pseudo-labels,
then retrains with S54's own recipe (two-stage, mosaic, copy-paste)
on the expanded dataset.

No Bezier, no CLAHE changes — just adding pseudo-labeled syntax images.

Usage:
    python run_s54_pseudo.py \
        --arcade-root ../../arcade/submission \
        --s54-stenosis ../results/stenosis_strategies/S54_s43b_clahe/stenosis_model/stenosis_768_best.pt \
        --device 0
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))


def log(msg, log_file):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(log_file, "a") as f:
        f.write(line + "\n")


def s54_stenosis_config(device="0"):
    """Exact S54 stenosis recipe — two-stage, mosaic, copy-paste."""
    return {
        "model": "yolo11m-seg.pt",
        "imgsz": 768, "batch": 8,
        "epochs": 300, "patience": 50,
        "optimizer": "SGD", "lr0": 0.005, "lrf": 0.01,
        "weight_decay": 0.0005, "momentum": 0.937,
        "warmup_epochs": 5, "freeze": 10, "freeze_epochs": 15,
        "lr_factor_unfrozen": 0.1,
        "seed": 42, "deterministic": True, "amp": True,
        "cos_lr": True, "device": device, "workers": 4,
        "copy_paste": 0.3, "scale": 0.5,
        "mosaic": 0.8, "close_mosaic": 15,
        "fliplr": 0.5, "flipud": 0.0,
        "degrees": 20.0, "translate": 0.1,
        "hsv_h": 0.0, "hsv_s": 0.0, "hsv_v": 0.3,
        "erasing": 0.0, "shear": 0.0, "perspective": 0.0,
        "mixup": 0.0, "box": 10.0, "cls": 1.0, "dfl": 1.5,
    }


def prepare_data(arcade_root, output_dir):
    splits_dir = output_dir / "stratified_splits"
    data_dir = output_dir / "data"
    if not splits_dir.exists():
        from create_stratified_splits import create_stratified_splits
        create_stratified_splits(arcade_root, splits_dir, 42)
    if not (data_dir / "dataset_configs" / "stenosis_only.yaml").exists():
        from run_pipeline import data_prep
        data_prep(arcade_root, data_dir, min_count=300, splits_dir=splits_dir)
    p = data_dir / "dataset_configs" / "stenosis_only.yaml"
    if p.exists():
        with open(p) as f:
            cfg = yaml.safe_load(f)
        correct = str((data_dir / "stenosis").resolve())
        if cfg.get("path") != correct:
            cfg["path"] = correct
            yaml.dump(cfg, open(p, "w"), default_flow_style=False, sort_keys=False)
    return data_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arcade-root", type=Path, required=True)
    parser.add_argument("--s54-stenosis", type=Path, required=True,
                        help="Path to S54 stenosis_768_best.pt")
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--pseudo-conf", type=float, default=0.3)
    args = parser.parse_args()

    arcade_root = args.arcade_root.resolve()
    s54_weights = str(args.s54_stenosis.resolve())
    output_dir = (args.output_dir or
                  SCRIPT_DIR.parent / "results" / "s54_pseudo").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / "run_log.txt"

    log(f"S54 weights: {s54_weights}", log_file)
    log(f"Pseudo conf: {args.pseudo_conf}", log_file)
    log(f"Device:      {args.device}", log_file)

    from train import train_two_stage
    from evaluate import evaluate_model
    from generate_stenosis_pseudolabels import run_stenosis_on_syntax_images

    t0 = time.time()
    data_dir = prepare_data(arcade_root, output_dir)
    sten_yaml = str(data_dir / "dataset_configs" / "stenosis_only.yaml")
    syntax_train = [data_dir / "syntax_filtered" / "images" / "train"]

    # Evaluate S54 baseline
    log("\nS54 baseline on stratified test:", log_file)
    s54_m = evaluate_model(s54_weights, sten_yaml, split="test",
                            augment=False, imgsz=768)
    s54_f1 = s54_m.get("per_class", {}).get("stenosis", {}).get("f1", 0)
    log(f"  S54 F1: {s54_f1:.4f}", log_file)

    # Generate pseudo-labels using S54 model
    log(f"\nPseudo-labeling syntax train at conf>={args.pseudo_conf}:", log_file)
    pseudo_dir = output_dir / "pseudo_labels"
    if pseudo_dir.exists():
        shutil.rmtree(pseudo_dir)

    total = {"images_processed": 0, "images_with_predictions": 0,
             "total_pseudo_instances": 0}
    for img_dir in syntax_train:
        stats = run_stenosis_on_syntax_images(
            s54_weights, img_dir, pseudo_dir,
            conf_threshold=args.pseudo_conf, imgsz=768, device=args.device)
        for k in total:
            total[k] += stats.get(k, 0)
    log(f"  {total['images_with_predictions']}/{total['images_processed']} images, "
        f"{total['total_pseudo_instances']} instances", log_file)

    # Build combined dataset (GT + pseudo, no other changes)
    log("\nBuilding combined dataset:", log_file)
    combined_dir = output_dir / "combined"
    if combined_dir.exists():
        shutil.rmtree(combined_dir)

    gt_img = data_dir / "stenosis" / "images" / "train"
    gt_lbl = data_dir / "stenosis" / "labels" / "train"
    out_img = combined_dir / "images" / "train"
    out_lbl = combined_dir / "labels" / "train"
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)

    # GT
    n_gt = 0
    for f in sorted(list(gt_img.glob("*.png")) + list(gt_img.glob("*.PNG"))):
        dst = out_img / f.name
        if not dst.exists():
            os.symlink(f.resolve(), dst)
        n_gt += 1
    for f in gt_lbl.glob("*.txt"):
        dst = out_lbl / f.name
        if not dst.exists():
            os.symlink(f.resolve(), dst)

    # Pseudo
    n_pl = 0
    for lbl in pseudo_dir.glob("*.txt"):
        src_img = None
        for img_dir in syntax_train:
            for ext in (".png", ".PNG"):
                cand = img_dir / (lbl.stem + ext)
                if cand.exists():
                    src_img = cand
                    break
            if src_img:
                break
        if not src_img:
            continue
        dst = out_img / f"pl_{src_img.name}"
        dst_l = out_lbl / f"pl_{lbl.stem}.txt"
        if not dst.exists():
            os.symlink(src_img.resolve(), dst)
        if not dst_l.exists():
            os.symlink(lbl.resolve(), dst_l)
        n_pl += 1

    # Val/test
    for split in ("val", "test"):
        for sub in ("images", "labels"):
            src = data_dir / "stenosis" / sub / split
            dst = combined_dir / sub / split
            dst.mkdir(parents=True, exist_ok=True)
            if src.exists():
                for f in src.iterdir():
                    d = dst / f.name
                    if not d.exists():
                        os.symlink(f.resolve(), d)

    log(f"  GT: {n_gt}, pseudo: {n_pl}, total: {n_gt + n_pl}", log_file)

    combined_yaml = combined_dir / "combined.yaml"
    yaml.dump({
        "path": str(combined_dir.resolve()),
        "train": "images/train", "val": "images/val", "test": "images/test",
        "nc": 1, "names": {0: "stenosis"},
    }, open(combined_yaml, "w"), default_flow_style=False, sort_keys=False)

    # Retrain with S54 two-stage recipe on combined data
    log("\nRetraining with S54 recipe (two-stage) on combined data:", log_file)
    cfg = s54_stenosis_config(args.device)
    m2_weights = train_two_stage(
        cfg, str(combined_yaml),
        str(output_dir / "M2"), "s54_pseudo_m2")

    m2_m = evaluate_model(m2_weights, sten_yaml, split="test",
                           augment=False, imgsz=768)
    m2_f1 = m2_m.get("per_class", {}).get("stenosis", {}).get("f1", 0)

    log(f"\n{'='*60}", log_file)
    log(f"S54 baseline:       {s54_f1:.4f}", log_file)
    log(f"S54 + pseudo (M2):  {m2_f1:.4f} (delta: {m2_f1-s54_f1:+.4f})", log_file)
    log(f"V2 SSASS best:      0.4797", log_file)
    log(f"Time: {(time.time()-t0)/3600:.1f}h", log_file)

    json.dump({
        "s54_f1": round(s54_f1, 4),
        "m2_f1": round(m2_f1, 4),
        "delta": round(m2_f1 - s54_f1, 4),
        "pseudo_stats": total,
        "pseudo_conf": args.pseudo_conf,
        "model": m2_weights,
    }, open(output_dir / "results.json", "w"), indent=2, default=str)


if __name__ == "__main__":
    main()
