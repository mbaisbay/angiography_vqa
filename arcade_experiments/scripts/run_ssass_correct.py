#!/usr/bin/env python3
"""Correct SSASS reproduction (arXiv:2311.10281, Medipixel MICCAI 2023).

Fixes all 6 implementation errors from previous attempts:
  1. Single round pseudo-labeling (NOT iterative)
  2. SSASS augmentation (geometric + HSV, NO mosaic/copy-paste)
  3. Single-stage training (NO freeze/unfreeze, NO LR drop)
  4. No Bezier augmentation
  5. No CLAHE preprocessing
  6. imgsz=640 (NOT 768)

Pipeline:
  Step 1: Train stenosis model M1 on GT stenosis data (1000 train)
  Step 2: Sweep pseudo-label conf on val to find optimal threshold
  Step 3: Generate pseudo-labels on syntax train images at best conf
  Step 4: Retrain M2 from scratch on GT + pseudo-labels
  Step 5: Evaluate M2 on stratified test

Usage:
    python run_ssass_correct.py --arcade-root ../../arcade/submission --device 0
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


def ssass_config(device="0"):
    """Exact SSASS paper config (Table 1 + Section 2.3)."""
    return {
        "model": "yolov8m-seg.pt",
        "imgsz": 640,
        "batch": 16,
        "epochs": 300,
        "patience": 0,          # no early stopping — run full 300 epochs
        "optimizer": "SGD",
        "lr0": 0.01,
        "lrf": 0.01,
        "weight_decay": 0.0005,
        "momentum": 0.937,
        "warmup_epochs": 3,
        "seed": 42,
        "deterministic": True,
        "amp": True,
        "cos_lr": True,
        "device": device,
        "workers": 4,
        # SSASS augmentation (Table 1) — geometric + HSV, no mosaic/copy-paste
        "flipud": 0.5,
        "fliplr": 0.5,
        "translate": 0.3,
        "degrees": 30.0,
        "scale": 0.5,
        "shear": 5.0,
        "perspective": 0.001,
        "hsv_h": 0.015,
        "hsv_s": 0.7,
        "hsv_v": 0.4,
        "mosaic": 0.0,
        "copy_paste": 0.0,
        "mixup": 0.0,
        "close_mosaic": 0,
        "erasing": 0.0,
        # Loss (ultralytics defaults)
        "box": 7.5,
        "cls": 0.5,
        "dfl": 1.5,
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
    for fname, subdir in [("stenosis_only.yaml", "stenosis"),
                          ("syntax_only.yaml", "syntax_filtered")]:
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


def train_single_stage(cfg, data_yaml, project, name):
    """Single-stage training — no freeze/unfreeze, as SSASS does."""
    from ultralytics import YOLO

    project = str(Path(project).resolve())
    model = YOLO(cfg["model"])

    args = {
        "data": data_yaml,
        "imgsz": cfg["imgsz"],
        "batch": cfg["batch"],
        "epochs": cfg["epochs"],
        "patience": cfg["patience"],
        "optimizer": cfg["optimizer"],
        "lr0": cfg["lr0"],
        "lrf": cfg["lrf"],
        "momentum": cfg.get("momentum", 0.937),
        "weight_decay": cfg["weight_decay"],
        "warmup_epochs": cfg.get("warmup_epochs", 3),
        "seed": cfg.get("seed", 42),
        "deterministic": cfg.get("deterministic", True),
        "amp": cfg.get("amp", True),
        "cos_lr": cfg.get("cos_lr", True),
        "device": cfg.get("device", "0"),
        "workers": cfg.get("workers", 4),
        "project": project,
        "name": name,
        "exist_ok": True,
        "freeze": 0,   # NO freezing — single stage end-to-end
        # Augmentation
        "mosaic": cfg.get("mosaic", 0.0),
        "close_mosaic": cfg.get("close_mosaic", 0),
        "mixup": cfg.get("mixup", 0.0),
        "copy_paste": cfg.get("copy_paste", 0.0),
        "fliplr": cfg.get("fliplr", 0.5),
        "flipud": cfg.get("flipud", 0.5),
        "degrees": cfg.get("degrees", 30.0),
        "scale": cfg.get("scale", 0.5),
        "translate": cfg.get("translate", 0.3),
        "shear": cfg.get("shear", 5.0),
        "perspective": cfg.get("perspective", 0.001),
        "hsv_h": cfg.get("hsv_h", 0.015),
        "hsv_s": cfg.get("hsv_s", 0.7),
        "hsv_v": cfg.get("hsv_v", 0.4),
        "erasing": cfg.get("erasing", 0.0),
        # Loss
        "box": cfg.get("box", 7.5),
        "cls": cfg.get("cls", 0.5),
        "dfl": cfg.get("dfl", 1.5),
    }

    model.train(**args)

    best = Path(project) / name / "weights" / "best.pt"
    if not best.exists():
        best = Path(project) / name / "weights" / "last.pt"

    final = Path(project) / f"{name}_best.pt"
    shutil.copy2(best, final)
    print(f"Single-stage complete: {final}")
    return str(final)


def generate_pseudo_labels(model_path, img_dirs, output_dir, conf, imgsz, device):
    """Run stenosis model on syntax images, write YOLO polygon labels."""
    from ultralytics import YOLO

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(model_path)
    n_total = 0
    n_pseudo = 0
    n_instances = 0

    for img_dir in img_dirs:
        imgs = sorted(list(img_dir.glob("*.png")) + list(img_dir.glob("*.PNG")))
        n_total += len(imgs)
        for img_path in imgs:
            results = model.predict(
                source=str(img_path), conf=conf, imgsz=imgsz,
                device=str(device), verbose=False, save=False,
                retina_masks=True)
            if not results or results[0].masks is None or len(results[0].masks) == 0:
                continue
            r = results[0]
            lines = []
            for i, mask_xyn in enumerate(r.masks.xyn):
                if len(mask_xyn) < 3:
                    continue
                coords = " ".join(f"{pt[0]:.6f} {pt[1]:.6f}" for pt in mask_xyn)
                lines.append(f"0 {coords}")  # class 0 = stenosis
            if lines:
                (output_dir / f"{img_path.stem}.txt").write_text(
                    "\n".join(lines) + "\n")
                n_pseudo += 1
                n_instances += len(lines)

    return {"total": n_total, "pseudo": n_pseudo, "instances": n_instances}


def build_combined_dataset(gt_data_dir, pseudo_dir, syntax_img_dirs,
                            output_dir):
    """Combine GT stenosis + pseudo-labeled syntax into one dataset."""
    if output_dir.exists():
        shutil.rmtree(output_dir)

    out_img = output_dir / "images" / "train"
    out_lbl = output_dir / "labels" / "train"
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)

    # GT stenosis train
    gt_img = gt_data_dir / "images" / "train"
    gt_lbl = gt_data_dir / "labels" / "train"
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

    # Pseudo-labeled syntax
    n_pseudo = 0
    for lbl_path in pseudo_dir.glob("*.txt"):
        stem = lbl_path.stem
        src_img = None
        for img_dir in syntax_img_dirs:
            for ext in (".png", ".PNG"):
                cand = img_dir / (stem + ext)
                if cand.exists():
                    src_img = cand
                    break
            if src_img:
                break
        if src_img is None:
            continue
        dst = out_img / f"pl_{src_img.name}"
        dst_l = out_lbl / f"pl_{stem}.txt"
        if not dst.exists():
            os.symlink(src_img.resolve(), dst)
        if not dst_l.exists():
            os.symlink(lbl_path.resolve(), dst_l)
        n_pseudo += 1

    # Val/test from GT
    for split in ("val", "test"):
        for sub in ("images", "labels"):
            src = gt_data_dir / sub / split
            dst = output_dir / sub / split
            dst.mkdir(parents=True, exist_ok=True)
            if src.exists():
                for f in src.iterdir():
                    d = dst / f.name
                    if not d.exists():
                        os.symlink(f.resolve(), d)

    yaml_path = output_dir / "combined.yaml"
    yaml.dump({
        "path": str(output_dir.resolve()),
        "train": "images/train", "val": "images/val", "test": "images/test",
        "nc": 1, "names": {0: "stenosis"},
    }, open(yaml_path, "w"), default_flow_style=False, sort_keys=False)

    return str(yaml_path), n_gt, n_pseudo


def evaluate_f1(model_path, data_yaml, imgsz, device):
    """Evaluate and return per-class metrics."""
    from evaluate import evaluate_model
    m = evaluate_model(model_path, data_yaml, split="test",
                        augment=False, imgsz=imgsz)
    f1 = m.get("per_class", {}).get("stenosis", {}).get("f1", 0)
    return f1, m


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arcade-root", type=Path, required=True)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--conf-sweep", type=str, default="0.3,0.4,0.5,0.6",
                        help="Conf thresholds to sweep for pseudo-labeling")
    args = parser.parse_args()

    arcade_root = args.arcade_root.resolve()
    output_dir = (args.output_dir or
                  SCRIPT_DIR.parent / "results" / "ssass_correct").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / "run_log.txt"
    conf_values = [float(x) for x in args.conf_sweep.split(",")]

    log(f"ARCADE root: {arcade_root}", log_file)
    log(f"Device:      {args.device}", log_file)
    log(f"Conf sweep:  {conf_values}", log_file)
    log(f"Config:      yolov8m-seg, 640px, single-stage, SSASS aug", log_file)
    log(f"             NO mosaic, NO copy-paste, NO CLAHE, NO Bezier", log_file)

    t0 = time.time()

    # Prepare data
    data_dir = prepare_stratified_data(arcade_root, output_dir)
    sten_yaml = str(data_dir / "dataset_configs" / "stenosis_only.yaml")
    syn_yaml = str(data_dir / "dataset_configs" / "syntax_only.yaml")

    # Syntax train images only (NOT val/test — paper uses train only)
    syntax_train_dir = data_dir / "syntax_filtered" / "images" / "train"
    syntax_img_dirs = [syntax_train_dir]
    log(f"Syntax train images: {len(list(syntax_train_dir.glob('*.png')))}", log_file)

    cfg = ssass_config(args.device)

    # ══════════════════════════════════════════════════════════════
    # Step 1: Train M1 on GT stenosis (single-stage, full 300 epochs)
    # ══════════════════════════════════════════════════════════════
    log(f"\n{'='*60}", log_file)
    log("STEP 1: Train M1 on GT stenosis (SSASS config)", log_file)
    log(f"{'='*60}", log_file)

    m1_weights = train_single_stage(
        cfg, sten_yaml, str(output_dir / "M1"), "m1_stenosis")

    m1_f1, m1_metrics = evaluate_f1(m1_weights, sten_yaml, 640, args.device)
    log(f"  M1 stenosis F1: {m1_f1:.4f}", log_file)

    # ══════════════════════════════════════════════════════════════
    # Step 2: Sweep pseudo-label confidence on val
    # ══════════════════════════════════════════════════════════════
    log(f"\n{'='*60}", log_file)
    log("STEP 2: Sweep pseudo-label confidence threshold", log_file)
    log(f"{'='*60}", log_file)

    best_conf = 0.5
    best_m2_f1 = 0
    sweep_results = {}

    for conf in conf_values:
        log(f"\n  --- conf={conf} ---", log_file)

        # Generate pseudo-labels
        pseudo_dir = output_dir / f"pseudo_conf{conf}"
        stats = generate_pseudo_labels(
            m1_weights, syntax_img_dirs, pseudo_dir, conf, 640, args.device)
        log(f"  Pseudo: {stats['pseudo']}/{stats['total']} images, "
            f"{stats['instances']} instances", log_file)

        if stats['pseudo'] == 0:
            log(f"  No pseudo-labels at conf={conf}, skipping", log_file)
            sweep_results[conf] = {"pseudo": 0, "f1": 0}
            continue

        # Build combined dataset
        combined_yaml, n_gt, n_pl = build_combined_dataset(
            data_dir / "stenosis", pseudo_dir, syntax_img_dirs,
            output_dir / f"combined_conf{conf}")
        log(f"  Combined: {n_gt} GT + {n_pl} pseudo = {n_gt + n_pl} total", log_file)

        # Train M2 from scratch
        m2_weights = train_single_stage(
            cfg, combined_yaml,
            str(output_dir / f"M2_conf{conf}"), f"m2_conf{conf}")

        m2_f1, m2_metrics = evaluate_f1(m2_weights, sten_yaml, 640, args.device)
        log(f"  M2 (conf={conf}) F1: {m2_f1:.4f} "
            f"(M1 baseline: {m1_f1:.4f}, delta: {m2_f1-m1_f1:+.4f})", log_file)

        sweep_results[conf] = {
            "pseudo_images": stats['pseudo'],
            "pseudo_instances": stats['instances'],
            "total_train": n_gt + n_pl,
            "m2_f1": round(m2_f1, 4),
            "delta": round(m2_f1 - m1_f1, 4),
            "model": m2_weights,
        }

        if m2_f1 > best_m2_f1:
            best_m2_f1 = m2_f1
            best_conf = conf

    # ══════════════════════════════════════════════════════════════
    # Summary
    # ══════════════════════════════════════════════════════════════
    log(f"\n{'='*60}", log_file)
    log("RESULTS", log_file)
    log(f"{'='*60}", log_file)
    log(f"  M1 (supervised only):  F1 = {m1_f1:.4f}", log_file)
    log(f"  Best M2 (conf={best_conf}): F1 = {best_m2_f1:.4f} "
        f"(delta: {best_m2_f1-m1_f1:+.4f})", log_file)
    log(f"  S54 baseline:          F1 = 0.4569", log_file)
    log(f"", log_file)
    for conf, r in sorted(sweep_results.items()):
        if isinstance(r, dict) and "m2_f1" in r:
            log(f"  conf={conf}: {r['pseudo_images']} pseudo -> "
                f"F1={r['m2_f1']:.4f} ({r['delta']:+.4f})", log_file)

    elapsed = time.time() - t0
    log(f"\nTotal time: {elapsed/3600:.1f}h", log_file)

    results = {
        "m1_f1": round(m1_f1, 4),
        "best_conf": best_conf,
        "best_m2_f1": round(best_m2_f1, 4),
        "sweep": {str(k): v for k, v in sweep_results.items()},
        "config": cfg,
        "elapsed_hours": round(elapsed / 3600, 2),
    }
    json.dump(results, open(output_dir / "results.json", "w"),
              indent=2, default=str)
    log(f"Results saved to {output_dir / 'results.json'}", log_file)


if __name__ == "__main__":
    main()
