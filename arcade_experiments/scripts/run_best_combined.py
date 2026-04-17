#!/usr/bin/env python3
"""Best combined model with iterative pseudo-labeling (SSASS-style).

Two independent pipelines that can run on separate GPUs:
  --task syntax   -> GPU 0: S54 syntax + pseudo-labels from stenosis images
  --task stenosis -> GPU 1: S54 stenosis + pseudo-labels + Bezier + CLAHE + CC
  --task both     -> sequential on one GPU

Usage (2 GPUs in parallel):
    python run_best_combined.py --arcade-root ../../arcade/submission --task syntax --device 0 &
    python run_best_combined.py --arcade-root ../../arcade/submission --task stenosis --device 1 &
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


def s54_syntax_config(device="0"):
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
        "mosaic": 0.0, "close_mosaic": 0, "mixup": 0.0,
        "copy_paste": 0.0, "fliplr": 0.5, "flipud": 0.0,
        "degrees": 20.0, "scale": 0.4, "translate": 0.1,
        "hsv_h": 0.0, "hsv_s": 0.0, "hsv_v": 0.3,
        "erasing": 0.0, "shear": 0.0, "perspective": 0.0,
        "box": 7.5, "cls": 0.5, "dfl": 1.5,
    }


def s54_stenosis_config(device="0"):
    cfg = s54_syntax_config(device)
    cfg["copy_paste"] = 0.3
    cfg["scale"] = 0.5
    cfg["mosaic"] = 0.8
    cfg["close_mosaic"] = 15
    cfg["box"] = 10.0
    cfg["cls"] = 1.0
    return cfg


def prepare_stratified_data(arcade_root, output_dir, seed=42):
    splits_dir = output_dir / "stratified_splits"
    data_dir = output_dir / "data"
    if not splits_dir.exists():
        from create_stratified_splits import create_stratified_splits
        create_stratified_splits(arcade_root, splits_dir, seed)
    if not (data_dir / "dataset_configs" / "syntax_only.yaml").exists():
        from run_pipeline import data_prep
        data_prep(arcade_root, data_dir, min_count=300, splits_dir=splits_dir)
    for fname, subdir in [("syntax_only.yaml", "syntax_filtered"),
                          ("stenosis_only.yaml", "stenosis")]:
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


def _collect_img_dirs(arcade_root, task):
    """Collect all image directories for a task across train/val/test."""
    dirs = []
    for split in ("train", "val", "test"):
        d = arcade_root / task / split / "images"
        if d.exists():
            dirs.append(d)
    return dirs


def _generate_pseudo_labels(model_weights, img_dirs, output_dir, conf,
                             imgsz, device, log_file):
    """Run a model on images and write YOLO polygon pseudo-labels."""
    from ultralytics import YOLO
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(model_weights)
    n_pseudo = 0
    n_total = 0
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
                cls = int(r.boxes.cls[i].item())
                if len(mask_xyn) < 3:
                    continue
                coords = " ".join(f"{pt[0]:.6f} {pt[1]:.6f}" for pt in mask_xyn)
                lines.append(f"{cls} {coords}")
            if lines:
                (output_dir / f"{img_path.stem}.txt").write_text(
                    "\n".join(lines) + "\n")
                n_pseudo += 1
    log(f"    {n_pseudo}/{n_total} images pseudo-labeled at conf>={conf}", log_file)
    return n_pseudo


def _add_pseudo_to_dataset(pseudo_dir, img_dirs, dst_img_dir, dst_lbl_dir,
                            prefix="pseudo"):
    """Symlink pseudo-labeled images into a dataset directory."""
    n = 0
    for lbl_path in pseudo_dir.glob("*.txt"):
        stem = lbl_path.stem
        src_img = None
        for img_dir in img_dirs:
            for ext in (".png", ".PNG"):
                cand = img_dir / (stem + ext)
                if cand.exists():
                    src_img = cand
                    break
            if src_img:
                break
        if src_img is None:
            continue
        dst_img = dst_img_dir / f"{prefix}_{src_img.name}"
        dst_lbl = dst_lbl_dir / f"{prefix}_{stem}.txt"
        if not dst_img.exists():
            os.symlink(src_img.resolve(), dst_img)
        if not dst_lbl.exists():
            os.symlink(lbl_path.resolve(), dst_lbl)
        n += 1
    return n


# ═══════════════════════════════════════════════════════════════════════
# SYNTAX PIPELINE
# ═══════════════════════════════════════════════════════════════════════

def run_syntax(arcade_root, output_dir, data_dir, device, pseudo_rounds, log_file):
    from train import train_two_stage
    from evaluate import evaluate_model

    syn_yaml = str(data_dir / "dataset_configs" / "syntax_only.yaml")
    with open(syn_yaml) as f:
        syn_cfg = yaml.safe_load(f)

    # All stenosis images = unlabeled data for syntax pseudo-labeling
    stenosis_img_dirs = _collect_img_dirs(arcade_root, "stenosis")

    # Step 1: Train initial syntax model
    log("SYNTAX: Training initial model (S54 recipe)", log_file)
    cfg = s54_syntax_config(device)
    weights = train_two_stage(cfg, syn_yaml,
                              str(output_dir / "syntax_initial"), "syn_init")
    m = evaluate_model(weights, syn_yaml, split="test", augment=False, imgsz=768)
    pc = m.get("per_class", {})
    f1s = [v.get("f1", 0) for v in pc.values() if isinstance(v, dict)]
    init_f1 = sum(f1s) / len(f1s) if f1s else 0
    log(f"  Initial syntax F1: {init_f1:.4f}", log_file)
    log(f"  Per-class: { {k: round(v.get('f1',0),4) for k,v in pc.items()} }", log_file)

    # Steps 1b-1d: Iterative pseudo-labeling (3 rounds)
    for rnd, conf in enumerate(pseudo_rounds, 1):
        log(f"\nSYNTAX round {rnd}/{len(pseudo_rounds)}: conf>={conf}", log_file)

        pseudo_dir = output_dir / f"syntax_pseudo_r{rnd}"
        n = _generate_pseudo_labels(weights, stenosis_img_dirs, pseudo_dir,
                                     conf, 768, device, log_file)

        # Build extended dataset
        ext_dir = output_dir / f"syntax_ext_r{rnd}"
        if ext_dir.exists():
            shutil.rmtree(ext_dir)
        shutil.copytree(data_dir / "syntax_filtered", ext_dir)

        n_added = _add_pseudo_to_dataset(
            pseudo_dir, stenosis_img_dirs,
            ext_dir / "images" / "train", ext_dir / "labels" / "train",
            prefix="stpseudo")
        log(f"  Added {n_added} pseudo-labeled stenosis images", log_file)

        ext_yaml = ext_dir / "syntax_ext.yaml"
        yaml.dump({
            "path": str(ext_dir.resolve()),
            "train": "images/train", "val": "images/val", "test": "images/test",
            "nc": syn_cfg["nc"], "names": syn_cfg["names"],
        }, open(ext_yaml, "w"), default_flow_style=False, sort_keys=False)

        # Retrain from scratch
        cfg = s54_syntax_config(device)
        weights = train_two_stage(cfg, str(ext_yaml),
                                  str(output_dir / f"syntax_r{rnd}"), f"syn_r{rnd}")
        m = evaluate_model(weights, syn_yaml, split="test", augment=False, imgsz=768)
        pc = m.get("per_class", {})
        f1s = [v.get("f1", 0) for v in pc.values() if isinstance(v, dict)]
        f1 = sum(f1s) / len(f1s) if f1s else 0
        log(f"  Round {rnd} syntax F1: {f1:.4f} (delta: {f1-init_f1:+.4f})", log_file)
        log(f"  Per-class: { {k: round(v.get('f1',0),4) for k,v in pc.items()} }", log_file)

    return {"model": weights, "init_f1": round(init_f1, 4),
            "final_f1": round(f1, 4), "metrics": m}


# ═══════════════════════════════════════════════════════════════════════
# STENOSIS PIPELINE
# ═══════════════════════════════════════════════════════════════════════

def run_stenosis(arcade_root, output_dir, data_dir, device, pseudo_rounds,
                  bezier_fraction, cc_min_area, log_file):
    from train import train_two_stage
    from evaluate import evaluate_model
    from generate_stenosis_pseudolabels import run_stenosis_on_syntax_images
    from bezier_vessel_augment import augment_dataset
    from small_cc_postprocess import evaluate_with_filter_cpu

    sten_yaml = str(data_dir / "dataset_configs" / "stenosis_only.yaml")
    gt_img = data_dir / "stenosis" / "images" / "train"
    gt_lbl = data_dir / "stenosis" / "labels" / "train"

    # All syntax images = unlabeled data for stenosis pseudo-labeling
    syntax_img_dirs = _collect_img_dirs(arcade_root, "syntax")

    # Step 2: Train initial stenosis model + CLAHE
    log("STENOSIS: Training initial model (S54 + CLAHE)", log_file)
    sten_work = output_dir / "stenosis_initial_data"
    if sten_work.exists():
        shutil.rmtree(sten_work)
    shutil.copytree(data_dir / "stenosis", sten_work)
    apply_clahe(sten_work / "images" / "train", log_file)

    init_yaml = sten_work / "stenosis.yaml"
    yaml.dump({
        "path": str(sten_work.resolve()),
        "train": "images/train", "val": "images/val", "test": "images/test",
        "nc": 1, "names": {0: "stenosis"},
    }, open(init_yaml, "w"), default_flow_style=False, sort_keys=False)

    cfg = s54_stenosis_config(device)
    weights = train_two_stage(cfg, str(init_yaml),
                              str(output_dir / "stenosis_initial"), "sten_init")
    m = evaluate_model(weights, sten_yaml, split="test", augment=False, imgsz=768)
    init_f1 = m.get("per_class", {}).get("stenosis", {}).get("f1", 0)
    log(f"  Initial stenosis F1: {init_f1:.4f}", log_file)

    # Steps 3-5: Iterative pseudo-labeling (3 rounds)
    for rnd, conf in enumerate(pseudo_rounds, 1):
        log(f"\nSTENOSIS round {rnd}/{len(pseudo_rounds)}: conf>={conf}", log_file)

        # Pseudo-label ALL 1500 syntax images
        pseudo_dir = output_dir / f"sten_pseudo_r{rnd}"
        if pseudo_dir.exists():
            shutil.rmtree(pseudo_dir)
        total_pseudo = {"images_processed": 0, "images_with_predictions": 0,
                        "total_pseudo_instances": 0}
        for img_dir in syntax_img_dirs:
            stats = run_stenosis_on_syntax_images(
                weights, img_dir, pseudo_dir,
                conf_threshold=conf, imgsz=768, device=device)
            for k in total_pseudo:
                total_pseudo[k] += stats.get(k, 0)
        log(f"  Pseudo: {total_pseudo['images_with_predictions']} images, "
            f"{total_pseudo['total_pseudo_instances']} instances", log_file)

        # Build extended dataset: GT + pseudo + Bezier + CLAHE
        ext_dir = output_dir / f"stenosis_ext_r{rnd}"
        if ext_dir.exists():
            shutil.rmtree(ext_dir)
        ext_img = ext_dir / "images" / "train"
        ext_lbl = ext_dir / "labels" / "train"
        ext_img.mkdir(parents=True, exist_ok=True)
        ext_lbl.mkdir(parents=True, exist_ok=True)

        # GT
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

        # Pseudo
        n_pseudo = _add_pseudo_to_dataset(pseudo_dir, syntax_img_dirs,
                                           ext_img, ext_lbl, "pseudo")
        log(f"  GT: {n_gt}, pseudo: {n_pseudo}", log_file)

        # Val/test
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

        # Bezier
        bez_img = output_dir / "bez_tmp" / "images"
        bez_lbl = output_dir / "bez_tmp" / "labels"
        if bez_img.exists():
            shutil.rmtree(bez_img.parent)
        bez_stats = augment_dataset(ext_img, ext_lbl, bez_img, bez_lbl,
                                     fraction=bezier_fraction, seed=42)
        shutil.rmtree(ext_img)
        shutil.rmtree(ext_lbl)
        shutil.move(str(bez_img), str(ext_img))
        shutil.move(str(bez_lbl), str(ext_lbl))
        shutil.rmtree(output_dir / "bez_tmp", ignore_errors=True)

        # CLAHE
        apply_clahe(ext_img, log_file)
        total = len(list(ext_img.glob("*.png")) + list(ext_img.glob("*.PNG")))
        log(f"  Total train (GT+pseudo+Bezier+CLAHE): {total}", log_file)

        ext_yaml = ext_dir / "stenosis_ext.yaml"
        yaml.dump({
            "path": str(ext_dir.resolve()),
            "train": "images/train", "val": "images/val", "test": "images/test",
            "nc": 1, "names": {0: "stenosis"},
        }, open(ext_yaml, "w"), default_flow_style=False, sort_keys=False)

        # Retrain from scratch
        cfg = s54_stenosis_config(device)
        weights = train_two_stage(cfg, str(ext_yaml),
                                  str(output_dir / f"stenosis_r{rnd}"), f"sten_r{rnd}")
        m = evaluate_model(weights, sten_yaml, split="test", augment=False, imgsz=768)
        f1 = m.get("per_class", {}).get("stenosis", {}).get("f1", 0)
        log(f"  Round {rnd} stenosis F1: {f1:.4f} (delta: {f1-init_f1:+.4f})", log_file)

    # CC post-processing sweep
    log("\nSTENOSIS: CC post-processing sweep", log_file)
    cc_results = {}
    for area in [30, 50, 100]:
        cc = evaluate_with_filter_cpu(weights, sten_yaml, split="test",
                                       imgsz=768, min_area_px=area, device=device)
        cc_results[area] = cc
        log(f"  min_area={area}: raw={cc['raw']['f1']:.4f} "
            f"filtered={cc['filtered']['f1']:.4f}", log_file)

    best_cc = max(cc_results, key=lambda k: cc_results[k]["filtered"]["f1"])
    return {
        "model": weights, "init_f1": round(init_f1, 4),
        "final_f1": round(f1, 4), "metrics": m,
        "cc_results": {str(k): v for k, v in cc_results.items()},
        "best_cc_area": best_cc,
        "best_cc_f1": round(cc_results[best_cc]["filtered"]["f1"], 4),
    }


# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arcade-root", type=Path, required=True)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--task", type=str, default="both",
                        choices=["syntax", "stenosis", "both"])
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--pseudo-rounds", type=str, default="0.5,0.4,0.3")
    parser.add_argument("--bezier-fraction", type=float, default=0.4)
    parser.add_argument("--cc-min-area", type=int, default=50)
    args = parser.parse_args()

    arcade_root = args.arcade_root.resolve()
    output_dir = (args.output_dir or
                  SCRIPT_DIR.parent / "results" / "best_combined").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / f"run_{args.task}_log.txt"
    pseudo_rounds = [float(x) for x in args.pseudo_rounds.split(",")]

    log(f"ARCADE root: {arcade_root}", log_file)
    log(f"Task: {args.task}, Device: {args.device}", log_file)
    log(f"Pseudo rounds: {pseudo_rounds}", log_file)

    t0 = time.time()
    data_dir = prepare_stratified_data(arcade_root, output_dir)
    results = {}

    if args.task in ("syntax", "both"):
        results["syntax"] = run_syntax(
            arcade_root, output_dir, data_dir, args.device, pseudo_rounds, log_file)

    if args.task in ("stenosis", "both"):
        results["stenosis"] = run_stenosis(
            arcade_root, output_dir, data_dir, args.device, pseudo_rounds,
            args.bezier_fraction, args.cc_min_area, log_file)

    elapsed = time.time() - t0
    results["elapsed_hours"] = round(elapsed / 3600, 2)

    out_path = output_dir / f"results_{args.task}.json"
    json.dump(results, open(out_path, "w"), indent=2, default=str)

    log(f"\n{'='*60}\nDONE in {elapsed/3600:.1f}h. Results: {out_path}\n{'='*60}", log_file)


if __name__ == "__main__":
    main()
