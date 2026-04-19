#!/usr/bin/env python3
"""Test SSASS pseudo-label pipeline with different model/config variants.

Uses the correct SSASS protocol (single-stage, single round, SSASS aug)
but swaps in configs that worked well for stenosis in prior experiments:

  V1: yolo11m-seg + lr=0.005 (S31 best stenosis config)
  V2: yolo11l-seg + lr=0.01  (S12 larger model)
  V3: yolo11m-seg + lr=0.01 + CLAHE (S54 CLAHE boost)
  V4: yolo11l-seg + lr=0.005 (combine best model + best lr)

All use SSASS augmentation, single-stage, single-round pseudo-labels.
Conf=0.3 (best from ssass_correct sweep).

Usage (2 GPUs):
    python run_ssass_variants.py --arcade-root ../../arcade/submission --exp V1 --device 0 &
    python run_ssass_variants.py --arcade-root ../../arcade/submission --exp V2 --device 1 &
    wait
    python run_ssass_variants.py --arcade-root ../../arcade/submission --exp V3 --device 0 &
    python run_ssass_variants.py --arcade-root ../../arcade/submission --exp V4 --device 1 &
    wait
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


VARIANTS = {
    "V1": {
        "label": "yolo11m + lr=0.005 (S31 best LR)",
        "model": "yolo11m-seg.pt",
        "lr0": 0.005,
        "imgsz": 640,
        "batch": 16,
        "clahe": False,
    },
    "V2": {
        "label": "yolo11l + lr=0.01 (S12 larger model)",
        "model": "yolo11l-seg.pt",
        "lr0": 0.01,
        "imgsz": 640,
        "batch": 8,
        "clahe": False,
    },
    "V3": {
        "label": "yolo11m + lr=0.01 + CLAHE (S54 preprocess)",
        "model": "yolo11m-seg.pt",
        "lr0": 0.01,
        "imgsz": 640,
        "batch": 16,
        "clahe": True,
    },
    "V4": {
        "label": "yolo11l + lr=0.005 (best model + best LR)",
        "model": "yolo11l-seg.pt",
        "lr0": 0.005,
        "imgsz": 640,
        "batch": 8,
        "clahe": False,
    },
}


def ssass_config(device, variant):
    """SSASS augmentation + variant-specific model/lr."""
    v = VARIANTS[variant]
    return {
        "model": v["model"],
        "imgsz": v["imgsz"],
        "batch": v["batch"],
        "epochs": 300,
        "patience": 0,
        "optimizer": "SGD",
        "lr0": v["lr0"],
        "lrf": 0.01,
        "weight_decay": 0.0005,
        "momentum": 0.937,
        "warmup_epochs": 3,
        "seed": 42, "deterministic": True, "amp": True,
        "cos_lr": True, "device": device, "workers": 4,
        # SSASS augmentation
        "flipud": 0.5, "fliplr": 0.5,
        "translate": 0.3, "degrees": 30.0, "scale": 0.5,
        "shear": 5.0, "perspective": 0.001,
        "hsv_h": 0.015, "hsv_s": 0.7, "hsv_v": 0.4,
        "mosaic": 0.0, "copy_paste": 0.0, "mixup": 0.0,
        "close_mosaic": 0, "erasing": 0.0,
        "box": 7.5, "cls": 0.5, "dfl": 1.5,
    }


def train_single_stage(cfg, data_yaml, project, name):
    from ultralytics import YOLO
    project = str(Path(project).resolve())
    model = YOLO(cfg["model"])
    args = {k: v for k, v in cfg.items() if k != "model"}
    args["data"] = data_yaml
    args["project"] = project
    args["name"] = name
    args["exist_ok"] = True
    args["freeze"] = 0
    model.train(**args)
    best = Path(project) / name / "weights" / "best.pt"
    if not best.exists():
        best = Path(project) / name / "weights" / "last.pt"
    final = Path(project) / f"{name}_best.pt"
    shutil.copy2(best, final)
    return str(final)


def generate_pseudo_labels(model_path, img_dirs, output_dir, conf, imgsz, device):
    from ultralytics import YOLO
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(model_path)
    n_total = n_pseudo = n_inst = 0
    for img_dir in img_dirs:
        imgs = sorted(list(img_dir.glob("*.png")) + list(img_dir.glob("*.PNG")))
        n_total += len(imgs)
        for img_path in imgs:
            results = model.predict(
                source=str(img_path), conf=conf, imgsz=imgsz,
                device=str(device), verbose=False, save=False, retina_masks=True)
            if not results or results[0].masks is None or len(results[0].masks) == 0:
                continue
            r = results[0]
            lines = []
            for i, mask_xyn in enumerate(r.masks.xyn):
                if len(mask_xyn) < 3:
                    continue
                coords = " ".join(f"{pt[0]:.6f} {pt[1]:.6f}" for pt in mask_xyn)
                lines.append(f"0 {coords}")
            if lines:
                (output_dir / f"{img_path.stem}.txt").write_text("\n".join(lines) + "\n")
                n_pseudo += 1
                n_inst += len(lines)
    return {"total": n_total, "pseudo": n_pseudo, "instances": n_inst}


def build_combined(gt_dir, pseudo_dir, syntax_img_dirs, output_dir, apply_clahe, log_file):
    if output_dir.exists():
        shutil.rmtree(output_dir)
    out_img = output_dir / "images" / "train"
    out_lbl = output_dir / "labels" / "train"
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)

    # GT
    n_gt = 0
    for f in sorted(list((gt_dir / "images" / "train").glob("*.png")) +
                    list((gt_dir / "images" / "train").glob("*.PNG"))):
        dst = out_img / f.name
        if not dst.exists():
            os.symlink(f.resolve(), dst)
        n_gt += 1
    for f in (gt_dir / "labels" / "train").glob("*.txt"):
        dst = out_lbl / f.name
        if not dst.exists():
            os.symlink(f.resolve(), dst)

    # Pseudo
    n_pl = 0
    for lbl in pseudo_dir.glob("*.txt"):
        src_img = None
        for sid in syntax_img_dirs:
            for ext in (".png", ".PNG"):
                cand = sid / (lbl.stem + ext)
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
            src = gt_dir / sub / split
            dst = output_dir / sub / split
            dst.mkdir(parents=True, exist_ok=True)
            if src.exists():
                for f in src.iterdir():
                    d = dst / f.name
                    if not d.exists():
                        os.symlink(f.resolve(), d)

    # CLAHE on train if requested
    if apply_clahe:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        for p in sorted(list(out_img.glob("*.png")) + list(out_img.glob("*.PNG"))):
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
                cv2.imwrite(str(p), cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR))
        log(f"    CLAHE applied to train images", log_file)

    yaml_path = output_dir / "combined.yaml"
    yaml.dump({
        "path": str(output_dir.resolve()),
        "train": "images/train", "val": "images/val", "test": "images/test",
        "nc": 1, "names": {0: "stenosis"},
    }, open(yaml_path, "w"), default_flow_style=False, sort_keys=False)

    return str(yaml_path), n_gt, n_pl


def prepare_data(arcade_root, output_dir):
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
                c = yaml.safe_load(f)
            correct = str((data_dir / subdir).resolve())
            if c.get("path") != correct:
                c["path"] = correct
                yaml.dump(c, open(p, "w"), default_flow_style=False, sort_keys=False)
    return data_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arcade-root", type=Path, required=True)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--exp", type=str, required=True, choices=list(VARIANTS.keys()))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--pseudo-conf", type=float, default=0.3)
    args = parser.parse_args()

    arcade_root = args.arcade_root.resolve()
    v = VARIANTS[args.exp]
    output_dir = (args.output_dir or
                  SCRIPT_DIR.parent / "results" / "ssass_variants").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / f"run_{args.exp}_log.txt"

    log(f"{args.exp}: {v['label']}", log_file)
    log(f"Device: {args.device}, conf: {args.pseudo_conf}", log_file)

    t0 = time.time()
    data_dir = prepare_data(arcade_root, output_dir)
    sten_yaml = str(data_dir / "dataset_configs" / "stenosis_only.yaml")
    syntax_img_dirs = [data_dir / "syntax_filtered" / "images" / "train"]

    cfg = ssass_config(args.device, args.exp)

    # Step 1: Train M1
    log("\nStep 1: Train M1 (supervised)", log_file)

    # For CLAHE variant, prepare CLAHE data for M1 too
    if v["clahe"]:
        m1_data = output_dir / f"{args.exp}_m1_data"
        if m1_data.exists():
            shutil.rmtree(m1_data)
        shutil.copytree(data_dir / "stenosis", m1_data)
        clahe_obj = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        for p in sorted(list((m1_data / "images" / "train").glob("*.png")) +
                        list((m1_data / "images" / "train").glob("*.PNG"))):
            if p.is_symlink():
                real = p.resolve()
                p.unlink()
                shutil.copy2(real, p)
            img = cv2.imread(str(p))
            if img is None:
                continue
            if img.ndim == 2 or img.shape[2] == 1:
                cv2.imwrite(str(p), clahe_obj.apply(img if img.ndim == 2 else img[:,:,0]))
            else:
                lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                l = clahe_obj.apply(l)
                cv2.imwrite(str(p), cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR))
        log("  CLAHE applied to M1 train data", log_file)
        m1_yaml = m1_data / "stenosis.yaml"
        yaml.dump({
            "path": str(m1_data.resolve()),
            "train": "images/train", "val": "images/val", "test": "images/test",
            "nc": 1, "names": {0: "stenosis"},
        }, open(m1_yaml, "w"), default_flow_style=False, sort_keys=False)
        m1_train_yaml = str(m1_yaml)
    else:
        m1_train_yaml = sten_yaml

    m1 = train_single_stage(cfg, m1_train_yaml,
                             str(output_dir / f"{args.exp}_M1"), f"{args.exp}_m1")

    from evaluate import evaluate_model
    m1_m = evaluate_model(m1, sten_yaml, split="test", augment=False, imgsz=cfg["imgsz"])
    m1_f1 = m1_m.get("per_class", {}).get("stenosis", {}).get("f1", 0)
    log(f"  M1 F1: {m1_f1:.4f}", log_file)

    # Step 2: Pseudo-label syntax train at conf=0.3
    log(f"\nStep 2: Pseudo-label (conf={args.pseudo_conf})", log_file)
    pseudo_dir = output_dir / f"{args.exp}_pseudo"
    stats = generate_pseudo_labels(m1, syntax_img_dirs, pseudo_dir,
                                    args.pseudo_conf, cfg["imgsz"], args.device)
    log(f"  {stats['pseudo']}/{stats['total']} images, {stats['instances']} instances", log_file)

    # Step 3: Train M2 on combined
    log("\nStep 3: Train M2 (GT + pseudo)", log_file)
    combined_yaml, n_gt, n_pl = build_combined(
        data_dir / "stenosis", pseudo_dir, syntax_img_dirs,
        output_dir / f"{args.exp}_combined", v["clahe"], log_file)
    log(f"  Combined: {n_gt} GT + {n_pl} pseudo = {n_gt+n_pl}", log_file)

    m2 = train_single_stage(cfg, combined_yaml,
                             str(output_dir / f"{args.exp}_M2"), f"{args.exp}_m2")

    m2_m = evaluate_model(m2, sten_yaml, split="test", augment=False, imgsz=cfg["imgsz"])
    m2_f1 = m2_m.get("per_class", {}).get("stenosis", {}).get("f1", 0)

    log(f"\n{'='*60}", log_file)
    log(f"{args.exp}: {v['label']}", log_file)
    log(f"  M1 (supervised): {m1_f1:.4f}", log_file)
    log(f"  M2 (+ pseudo):   {m2_f1:.4f} (delta: {m2_f1-m1_f1:+.4f})", log_file)
    log(f"  SSASS baseline:  0.4727 (yolov8m conf=0.3)", log_file)
    log(f"  S54 baseline:    0.4569", log_file)
    log(f"  Time: {(time.time()-t0)/3600:.1f}h", log_file)

    json.dump({
        "exp": args.exp, "label": v["label"],
        "m1_f1": round(m1_f1, 4), "m2_f1": round(m2_f1, 4),
        "delta": round(m2_f1 - m1_f1, 4),
        "pseudo_stats": stats, "model": m2,
        "config": cfg,
    }, open(output_dir / f"results_{args.exp}.json", "w"), indent=2, default=str)


if __name__ == "__main__":
    main()
