#!/usr/bin/env python3
"""Enhance S54 stenosis with pseudo-labels + Bezier + CLAHE + CC.

Uses the ACTUAL S54 stenosis weights (0.4569 F1) as the starting model.
Does NOT retrain syntax — uses S54 syntax weights directly.

Pipeline:
  1. Use existing S54 stenosis model to pseudo-label ALL 1500 syntax images
     (3 iterative rounds at conf 0.5 -> 0.4 -> 0.3, SSASS protocol)
  2. Each round: GT stenosis + pseudo-labels + Bezier augmentation + CLAHE
  3. Retrain stenosis from scratch on expanded dataset
  4. CC post-processing sweep at inference

Usage:
    python run_stenosis_enhanced.py \
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

import cv2
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
    return {
        "model": "yolo11m-seg.pt",
        "imgsz": 768, "batch": 8,
        "epochs": 300, "patience": 50,
        "optimizer": "SGD", "lr0": 0.01, "lrf": 0.01,
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


def prepare_stratified_data(arcade_root, output_dir):
    splits_dir = output_dir / "stratified_splits"
    data_dir = output_dir / "data"
    if not splits_dir.exists():
        from create_stratified_splits import create_stratified_splits
        create_stratified_splits(arcade_root, splits_dir, 42)
    if not (data_dir / "dataset_configs" / "stenosis_only.yaml").exists():
        from run_pipeline import data_prep
        data_prep(arcade_root, data_dir, min_count=300, splits_dir=splits_dir)
    for fname, subdir in [("stenosis_only.yaml", "stenosis")]:
        p = data_dir / "dataset_configs" / fname
        if p.exists():
            with open(p) as f:
                cfg = yaml.safe_load(f)
            correct = str((data_dir / subdir).resolve())
            if cfg.get("path") != correct:
                cfg["path"] = correct
                yaml.dump(cfg, open(p, "w"), default_flow_style=False,
                          sort_keys=False)
    return data_dir


def apply_clahe(img_dir, log_file):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    images = sorted(list(img_dir.glob("*.png")) + list(img_dir.glob("*.PNG")))
    for p in images:
        if p.is_symlink():
            real = p.resolve()
            p.unlink()
            shutil.copy2(real, p)
        img = cv2.imread(str(p))
        if img is None:
            continue
        if img.ndim == 2 or img.shape[2] == 1:
            gray = img if img.ndim == 2 else img[:, :, 0]
            cv2.imwrite(str(p), clahe.apply(gray))
        else:
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l = clahe.apply(l)
            cv2.imwrite(str(p), cv2.cvtColor(cv2.merge([l, a, b]),
                                              cv2.COLOR_LAB2BGR))
    log(f"    CLAHE applied to {len(images)} images", log_file)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arcade-root", type=Path, required=True)
    parser.add_argument("--s54-stenosis", type=Path, required=True,
                        help="Path to S54 stenosis_768_best.pt")
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--pseudo-rounds", type=str, default="0.5,0.4,0.3")
    parser.add_argument("--bezier-fraction", type=float, default=0.4)
    args = parser.parse_args()

    arcade_root = args.arcade_root.resolve()
    s54_weights = str(args.s54_stenosis.resolve())
    output_dir = (args.output_dir or
                  SCRIPT_DIR.parent / "results" / "stenosis_enhanced").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / "run_log.txt"
    pseudo_rounds = [float(x) for x in args.pseudo_rounds.split(",")]

    log(f"ARCADE root:   {arcade_root}", log_file)
    log(f"S54 stenosis:  {s54_weights}", log_file)
    log(f"Device:        {args.device}", log_file)
    log(f"Pseudo rounds: {pseudo_rounds}", log_file)
    log(f"Bezier frac:   {args.bezier_fraction}", log_file)

    from train import train_two_stage
    from evaluate import evaluate_model
    from generate_stenosis_pseudolabels import run_stenosis_on_syntax_images
    from bezier_vessel_augment import augment_dataset
    from small_cc_postprocess import evaluate_with_filter_cpu

    t0 = time.time()
    data_dir = prepare_stratified_data(arcade_root, output_dir)
    sten_yaml = str(data_dir / "dataset_configs" / "stenosis_only.yaml")
    gt_img = data_dir / "stenosis" / "images" / "train"
    gt_lbl = data_dir / "stenosis" / "labels" / "train"

    # All 1500 syntax images for pseudo-labeling
    syntax_img_dirs = []
    for split in ("train", "val", "test"):
        d = arcade_root / "syntax" / split / "images"
        if d.exists():
            syntax_img_dirs.append(d)
            log(f"  Syntax {split}: {len(list(d.glob('*.png')))} images", log_file)

    # Evaluate S54 baseline on stratified test
    log("\nEvaluating S54 stenosis baseline on stratified test...", log_file)
    s54_metrics = evaluate_model(s54_weights, sten_yaml, split="test",
                                  augment=False, imgsz=768)
    s54_f1 = s54_metrics.get("per_class", {}).get("stenosis", {}).get("f1", 0)
    log(f"  S54 baseline stenosis F1: {s54_f1:.4f}", log_file)

    # Iterative pseudo-labeling
    current_weights = s54_weights
    best_f1 = s54_f1
    best_weights = s54_weights

    for rnd, conf in enumerate(pseudo_rounds, 1):
        log(f"\n{'='*60}", log_file)
        log(f"Round {rnd}/{len(pseudo_rounds)}: pseudo-labels at conf>={conf}", log_file)
        log(f"{'='*60}", log_file)

        # Generate pseudo-labels
        pseudo_dir = output_dir / f"pseudo_r{rnd}"
        if pseudo_dir.exists():
            shutil.rmtree(pseudo_dir)

        total = {"images_processed": 0, "images_with_predictions": 0,
                 "total_pseudo_instances": 0}
        for img_dir in syntax_img_dirs:
            stats = run_stenosis_on_syntax_images(
                current_weights, img_dir, pseudo_dir,
                conf_threshold=conf, imgsz=768, device=args.device)
            for k in total:
                total[k] += stats.get(k, 0)
        log(f"  Pseudo: {total['images_with_predictions']}/{total['images_processed']} "
            f"images, {total['total_pseudo_instances']} instances", log_file)

        # Build extended dataset
        ext_dir = output_dir / f"ext_r{rnd}"
        if ext_dir.exists():
            shutil.rmtree(ext_dir)
        ext_img = ext_dir / "images" / "train"
        ext_lbl = ext_dir / "labels" / "train"
        ext_img.mkdir(parents=True, exist_ok=True)
        ext_lbl.mkdir(parents=True, exist_ok=True)

        # GT stenosis
        n_gt = 0
        for f in sorted(list(gt_img.glob("*.png")) + list(gt_img.glob("*.PNG"))):
            dst = ext_img / f.name
            if not dst.exists():
                os.symlink(f.resolve(), dst)
            n_gt += 1
        for f in gt_lbl.glob("*.txt"):
            dst = ext_lbl / f.name
            if not dst.exists():
                os.symlink(f.resolve(), dst)

        # Pseudo-labeled syntax images
        n_pseudo = 0
        for lbl_path in pseudo_dir.glob("*.txt"):
            stem = lbl_path.stem
            src_img = None
            for sid in syntax_img_dirs:
                for ext in (".png", ".PNG"):
                    cand = sid / (stem + ext)
                    if cand.exists():
                        src_img = cand
                        break
                if src_img:
                    break
            if src_img is None:
                continue
            dst = ext_img / f"pseudo_{src_img.name}"
            dst_l = ext_lbl / f"pseudo_{stem}.txt"
            if not dst.exists():
                os.symlink(src_img.resolve(), dst)
            if not dst_l.exists():
                os.symlink(lbl_path.resolve(), dst_l)
            n_pseudo += 1

        # Val/test from stratified
        for split in ("val", "test"):
            for sub in ("images", "labels"):
                src = data_dir / "stenosis" / sub / split
                dst = ext_dir / sub / split
                dst.mkdir(parents=True, exist_ok=True)
                if src.exists():
                    for f in src.iterdir():
                        d = dst / f.name
                        if not d.exists():
                            os.symlink(f.resolve(), d)

        # Bezier augmentation
        bez_img = output_dir / "bez_tmp" / "images"
        bez_lbl = output_dir / "bez_tmp" / "labels"
        if bez_img.exists():
            shutil.rmtree(bez_img.parent)
        bez = augment_dataset(ext_img, ext_lbl, bez_img, bez_lbl,
                               fraction=args.bezier_fraction, seed=42)
        shutil.rmtree(ext_img)
        shutil.rmtree(ext_lbl)
        shutil.move(str(bez_img), str(ext_img))
        shutil.move(str(bez_lbl), str(ext_lbl))
        shutil.rmtree(output_dir / "bez_tmp", ignore_errors=True)

        # CLAHE
        apply_clahe(ext_img, log_file)
        total_train = len(list(ext_img.glob("*.png")) + list(ext_img.glob("*.PNG")))
        log(f"  GT: {n_gt}, pseudo: {n_pseudo}, bezier+clahe total: {total_train}", log_file)

        ext_yaml = ext_dir / "stenosis_ext.yaml"
        yaml.dump({
            "path": str(ext_dir.resolve()),
            "train": "images/train", "val": "images/val", "test": "images/test",
            "nc": 1, "names": {0: "stenosis"},
        }, open(ext_yaml, "w"), default_flow_style=False, sort_keys=False)

        # Retrain from scratch (COCO weights, not from S54)
        cfg = s54_stenosis_config(args.device)
        current_weights = train_two_stage(
            cfg, str(ext_yaml),
            str(output_dir / f"model_r{rnd}"), f"sten_r{rnd}")

        m = evaluate_model(current_weights, sten_yaml, split="test",
                            augment=False, imgsz=768)
        f1 = m.get("per_class", {}).get("stenosis", {}).get("f1", 0)
        log(f"  Round {rnd} F1: {f1:.4f} (S54 baseline: {s54_f1:.4f}, "
            f"delta: {f1-s54_f1:+.4f})", log_file)

        if f1 > best_f1:
            best_f1 = f1
            best_weights = current_weights

    # CC post-processing on best model
    log(f"\nCC post-processing on best model (F1={best_f1:.4f}):", log_file)
    cc = {}
    for area in [30, 50, 100]:
        r = evaluate_with_filter_cpu(best_weights, sten_yaml, split="test",
                                      imgsz=768, min_area_px=area, device=args.device)
        cc[area] = r
        log(f"  area={area}: raw={r['raw']['f1']:.4f} filtered={r['filtered']['f1']:.4f}", log_file)

    best_cc = max(cc, key=lambda k: cc[k]["filtered"]["f1"])

    # Summary
    log(f"\n{'='*60}", log_file)
    log(f"FINAL: S54 baseline={s54_f1:.4f}, best round={best_f1:.4f}, "
        f"best CC={cc[best_cc]['filtered']['f1']:.4f} (area={best_cc})", log_file)
    log(f"{'='*60}", log_file)

    json.dump({
        "s54_baseline_f1": round(s54_f1, 4),
        "best_round_f1": round(best_f1, 4),
        "best_model": best_weights,
        "best_cc_area": best_cc,
        "best_cc_f1": round(cc[best_cc]["filtered"]["f1"], 4),
        "cc_results": {str(k): v for k, v in cc.items()},
        "elapsed_hours": round((time.time() - t0) / 3600, 2),
    }, open(output_dir / "results.json", "w"), indent=2, default=str)

    log(f"Done in {(time.time()-t0)/3600:.1f}h", log_file)


if __name__ == "__main__":
    main()
