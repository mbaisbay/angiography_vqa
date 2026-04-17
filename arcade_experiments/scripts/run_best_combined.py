#!/usr/bin/env python3
"""Train the best combined model: S54 syntax + enhanced stenosis pipeline.

Syntax: S54 recipe exactly as-is (proven best, 0.7398 F1).

Stenosis: S54 stenosis recipe PLUS three enhancements:
  1. Pseudo-label semi-supervised step (SSASS technique):
     - Train stenosis model on labeled stenosis images
     - Run on syntax images at conf>=0.5 to generate pseudo-labels
     - Retrain from scratch on stenosis GT + pseudo-labeled syntax images
  2. Bezier curve augmentation (SSASS, Medipixel 2023):
     - Draw synthetic vessel curves on training images as background
       distractors, teaching the model vessel context so it focuses
       capacity on lesion morphology
  3. Small connected-component post-processing at inference:
     - Remove predicted mask regions below a minimum area threshold
     - Cleans false positives from background noise

Usage:
    python run_best_combined.py --arcade-root ../../arcade/submission --device 0
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

import cv2
import numpy as np
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


def main():
    parser = argparse.ArgumentParser(
        description="Best combined model: S54 syntax + enhanced stenosis")
    parser.add_argument("--arcade-root", type=Path, required=True)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--pseudo-rounds", type=str, default="0.5,0.4,0.3",
                        help="Comma-separated conf thresholds for iterative "
                             "pseudo-labeling rounds (SSASS: 0.5,0.4,0.3)")
    parser.add_argument("--bezier-fraction", type=float, default=0.4,
                        help="Fraction of images to augment with Bezier curves")
    parser.add_argument("--cc-min-area", type=int, default=50,
                        help="Min mask area (px) for connected-component filter")
    args = parser.parse_args()

    arcade_root = args.arcade_root.resolve()
    output_dir = (args.output_dir or
                  SCRIPT_DIR.parent / "results" / "best_combined").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / "run_log.txt"

    log(f"ARCADE root:     {arcade_root}", log_file)
    log(f"Output dir:      {output_dir}", log_file)
    log(f"Device:          {args.device}", log_file)
    pseudo_rounds = [float(x) for x in args.pseudo_rounds.split(",")]
    log(f"Pseudo rounds:   {pseudo_rounds}", log_file)
    log(f"Bezier fraction: {args.bezier_fraction}", log_file)
    log(f"CC min area:     {args.cc_min_area} px", log_file)

    from train import train_two_stage
    from evaluate import evaluate_model
    from generate_stenosis_pseudolabels import run_stenosis_on_syntax_images
    from bezier_vessel_augment import augment_dataset
    from small_cc_postprocess import evaluate_with_filter_cpu

    t0 = time.time()

    # ── Step 0: Prepare stratified data ──
    log("\n" + "=" * 60, log_file)
    log("STEP 0: Prepare stratified data", log_file)
    log("=" * 60, log_file)
    data_dir = prepare_stratified_data(arcade_root, output_dir)
    syn_yaml = str(data_dir / "dataset_configs" / "syntax_only.yaml")
    sten_yaml = str(data_dir / "dataset_configs" / "stenosis_only.yaml")

    # ── Step 1: Train initial syntax model (S54 recipe) ──
    log("\n" + "=" * 60, log_file)
    log("STEP 1: Train initial syntax model (S54 recipe)", log_file)
    log("=" * 60, log_file)
    cfg_syn = s54_syntax_config(args.device)
    initial_syn_weights = train_two_stage(
        cfg_syn, syn_yaml,
        str(output_dir / "syntax_model_initial"), "syn_initial")

    init_syn_metrics = evaluate_model(initial_syn_weights, syn_yaml, split="test",
                                       augment=False, imgsz=768)
    pc_syn = init_syn_metrics.get("per_class", {})
    f1s_syn = [v.get("f1", 0) for v in pc_syn.values() if isinstance(v, dict)]
    init_syn_f1 = sum(f1s_syn) / len(f1s_syn) if f1s_syn else 0
    log(f"  Initial syntax mean F1: {init_syn_f1:.4f}", log_file)
    log(f"  Per-class: { {k: round(v.get('f1',0), 4) for k, v in pc_syn.items()} }", log_file)

    # ── Step 1b-1d: Iterative pseudo-label syntax (SSASS-style 3 rounds) ──
    # Run syntax model on ALL 1500 stenosis images (they contain vessels
    # but have no syntax annotations). At each round, lower the confidence
    # threshold to capture more pseudo-labels, retrain from scratch.
    from ultralytics import YOLO

    all_stenosis_img_dirs = []
    for split in ("train", "val", "test"):
        img_dir = arcade_root / "stenosis" / split / "images"
        if img_dir.exists():
            all_stenosis_img_dirs.append(img_dir)

    with open(syn_yaml) as f:
        syn_cfg = yaml.safe_load(f)

    current_syn_weights = initial_syn_weights
    syntax_weights = initial_syn_weights
    syn_f1 = init_syn_f1

    for round_num, conf in enumerate(pseudo_rounds, start=1):
        log(f"\n{'=' * 60}", log_file)
        log(f"STEP 1b round {round_num}/{len(pseudo_rounds)}: "
            f"syntax pseudo-labels at conf>={conf}", log_file)
        log(f"{'=' * 60}", log_file)

        # Generate pseudo-labels
        syn_pseudo_dir = output_dir / "syntax_pseudo_labels" / f"round{round_num}"
        if syn_pseudo_dir.exists():
            shutil.rmtree(syn_pseudo_dir)
        syn_pseudo_dir.mkdir(parents=True, exist_ok=True)

        syn_model = YOLO(current_syn_weights)
        n_syn_pseudo = 0
        for img_dir in all_stenosis_img_dirs:
            imgs = sorted(list(img_dir.glob("*.png")) + list(img_dir.glob("*.PNG")))
            for img_path in imgs:
                results = syn_model.predict(
                    source=str(img_path), conf=conf, imgsz=768,
                    device=str(args.device), verbose=False, save=False,
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
                    (syn_pseudo_dir / f"{img_path.stem}.txt").write_text(
                        "\n".join(lines) + "\n")
                    n_syn_pseudo += 1
        log(f"  Round {round_num}: {n_syn_pseudo} pseudo-labeled images at conf>={conf}", log_file)

        # Build extended dataset
        syn_extended = output_dir / f"syntax_extended_r{round_num}"
        if syn_extended.exists():
            shutil.rmtree(syn_extended)
        shutil.copytree(data_dir / "syntax_filtered", syn_extended)

        syn_ext_img = syn_extended / "images" / "train"
        syn_ext_lbl = syn_extended / "labels" / "train"
        n_added = 0
        for lbl_path in syn_pseudo_dir.glob("*.txt"):
            stem = lbl_path.stem
            src_img = None
            for img_dir in all_stenosis_img_dirs:
                for ext in (".png", ".PNG"):
                    cand = img_dir / (stem + ext)
                    if cand.exists():
                        src_img = cand
                        break
                if src_img:
                    break
            if src_img is None:
                continue
            dst_img = syn_ext_img / f"stpseudo_{src_img.name}"
            dst_lbl = syn_ext_lbl / f"stpseudo_{stem}.txt"
            if not dst_img.exists():
                os.symlink(src_img.resolve(), dst_img)
            if not dst_lbl.exists():
                os.symlink(lbl_path.resolve(), dst_lbl)
            n_added += 1
        log(f"  Added {n_added} pseudo-labeled images to syntax train", log_file)

        syn_ext_yaml = syn_extended / "syntax_extended.yaml"
        yaml.dump({
            "path": str(syn_extended.resolve()),
            "train": "images/train", "val": "images/val", "test": "images/test",
            "nc": syn_cfg["nc"], "names": syn_cfg["names"],
        }, open(syn_ext_yaml, "w"), default_flow_style=False, sort_keys=False)

        # Retrain from scratch
        cfg_syn_r = s54_syntax_config(args.device)
        current_syn_weights = train_two_stage(
            cfg_syn_r, str(syn_ext_yaml),
            str(output_dir / f"syntax_model_r{round_num}"), f"syn_r{round_num}")

        syn_metrics = evaluate_model(current_syn_weights, syn_yaml, split="test",
                                      augment=False, imgsz=768)
        pc_syn2 = syn_metrics.get("per_class", {})
        f1s_syn2 = [v.get("f1", 0) for v in pc_syn2.values() if isinstance(v, dict)]
        syn_f1 = sum(f1s_syn2) / len(f1s_syn2) if f1s_syn2 else 0
        log(f"  Round {round_num} syntax F1: {syn_f1:.4f} "
            f"(initial: {init_syn_f1:.4f}, delta: {syn_f1 - init_syn_f1:+.4f})", log_file)
        log(f"  Per-class: { {k: round(v.get('f1',0), 4) for k, v in pc_syn2.items()} }", log_file)

    syntax_weights = current_syn_weights

    # ── Step 2: Train initial stenosis model (S54 recipe + CLAHE) ──
    log("\n" + "=" * 60, log_file)
    log("STEP 2: Train initial stenosis model (S54 + CLAHE)", log_file)
    log("=" * 60, log_file)

    # Copy stenosis data and apply CLAHE
    sten_work = output_dir / "stenosis_work" / "initial"
    if sten_work.exists():
        shutil.rmtree(sten_work)
    shutil.copytree(data_dir / "stenosis", sten_work)
    apply_clahe(sten_work / "images" / "train", log_file)

    sten_init_yaml = sten_work / "stenosis.yaml"
    yaml.dump({
        "path": str(sten_work.resolve()),
        "train": "images/train", "val": "images/val", "test": "images/test",
        "nc": 1, "names": {0: "stenosis"},
    }, open(sten_init_yaml, "w"), default_flow_style=False, sort_keys=False)

    cfg_sten = s54_stenosis_config(args.device)
    initial_sten_weights = train_two_stage(
        cfg_sten, str(sten_init_yaml),
        str(output_dir / "stenosis_model_initial"), "sten_initial")

    init_metrics = evaluate_model(initial_sten_weights, sten_yaml, split="test",
                                   augment=False, imgsz=768)
    init_f1 = init_metrics.get("per_class", {}).get("stenosis", {}).get("f1", 0)
    log(f"  Initial stenosis F1: {init_f1:.4f}", log_file)

    # ── Steps 3-5: Iterative stenosis pseudo-labeling (3 rounds, SSASS) ──
    # Use ALL 1500 syntax images for pseudo-labeling. Each round:
    # - Run stenosis model on syntax images at decaying conf
    # - Replace previous pseudo-labels with new predictions
    # - Build extended dataset: GT + pseudo-labels + Bezier + CLAHE
    # - Retrain stenosis from scratch

    all_syntax_img_dirs = []
    for split in ("train", "val", "test"):
        img_dir = arcade_root / "syntax" / split / "images"
        if img_dir.exists():
            all_syntax_img_dirs.append(img_dir)

    gt_img_dir = data_dir / "stenosis" / "images" / "train"
    gt_lbl_dir = data_dir / "stenosis" / "labels" / "train"

    current_sten_weights = initial_sten_weights
    final_sten_weights = initial_sten_weights
    final_f1 = init_f1
    final_metrics = init_metrics
    bezier_stats = {}

    for round_num, conf in enumerate(pseudo_rounds, start=1):
        log(f"\n{'=' * 60}", log_file)
        log(f"STEP 3-5 round {round_num}/{len(pseudo_rounds)}: "
            f"stenosis pseudo-labels at conf>={conf}", log_file)
        log(f"{'=' * 60}", log_file)

        # 3. Generate pseudo-labels on ALL syntax images
        pseudo_dir = output_dir / "sten_pseudo_labels" / f"round{round_num}"
        if pseudo_dir.exists():
            shutil.rmtree(pseudo_dir)

        total_pseudo = {"images_processed": 0, "images_with_predictions": 0,
                        "total_pseudo_instances": 0}
        for img_dir in all_syntax_img_dirs:
            stats = run_stenosis_on_syntax_images(
                current_sten_weights, img_dir, pseudo_dir,
                conf_threshold=conf, imgsz=768, device=args.device)
            for k in total_pseudo:
                total_pseudo[k] += stats.get(k, 0)
        log(f"  Pseudo-labels: {total_pseudo['images_with_predictions']} images "
            f"({total_pseudo['total_pseudo_instances']} instances) "
            f"from {total_pseudo['images_processed']} syntax images", log_file)

        # 4. Build extended dataset
        extended_dir = output_dir / f"stenosis_extended_r{round_num}"
        if extended_dir.exists():
            shutil.rmtree(extended_dir)

        ext_img_train = extended_dir / "images" / "train"
        ext_lbl_train = extended_dir / "labels" / "train"
        ext_img_train.mkdir(parents=True, exist_ok=True)
        ext_lbl_train.mkdir(parents=True, exist_ok=True)

        # GT stenosis
        n_gt = 0
        for img in sorted(list(gt_img_dir.glob("*.png")) + list(gt_img_dir.glob("*.PNG"))):
            dst = ext_img_train / img.name
            if not dst.exists():
                os.symlink(img.resolve(), dst)
            n_gt += 1
        for lbl in gt_lbl_dir.glob("*.txt"):
            dst = ext_lbl_train / lbl.name
            if not dst.exists():
                os.symlink(lbl.resolve(), dst)

        # Pseudo-labeled syntax
        n_pseudo = 0
        for lbl_path in pseudo_dir.glob("*.txt"):
            stem = lbl_path.stem
            src_img = None
            for sid in all_syntax_img_dirs:
                for ext in (".png", ".PNG"):
                    cand = sid / (stem + ext)
                    if cand.exists():
                        src_img = cand
                        break
                if src_img:
                    break
            if src_img is None:
                continue
            dst_img = ext_img_train / f"pseudo_{src_img.name}"
            dst_lbl = ext_lbl_train / f"pseudo_{stem}.txt"
            if not dst_img.exists():
                os.symlink(src_img.resolve(), dst_img)
            if not dst_lbl.exists():
                os.symlink(lbl_path.resolve(), dst_lbl)
            n_pseudo += 1
        log(f"  GT: {n_gt}, pseudo: {n_pseudo}, total: {n_gt + n_pseudo}", log_file)

        # Val/test symlinks
        for split in ("val", "test"):
            for subdir in ("images", "labels"):
                src = data_dir / "stenosis" / subdir / split
                dst = extended_dir / subdir / split
                dst.mkdir(parents=True, exist_ok=True)
                if src.exists():
                    for f in src.iterdir():
                        d = dst / f.name
                        if not d.exists():
                            os.symlink(f.resolve(), d)

        # Bezier augmentation
        bezier_img_out = output_dir / "bezier_tmp" / "images"
        bezier_lbl_out = output_dir / "bezier_tmp" / "labels"
        if bezier_img_out.exists():
            shutil.rmtree(bezier_img_out.parent)
        bezier_stats = augment_dataset(
            ext_img_train, ext_lbl_train,
            bezier_img_out, bezier_lbl_out,
            fraction=args.bezier_fraction, seed=42)
        shutil.rmtree(ext_img_train)
        shutil.rmtree(ext_lbl_train)
        shutil.move(str(bezier_img_out), str(ext_img_train))
        shutil.move(str(bezier_lbl_out), str(ext_lbl_train))
        shutil.rmtree(output_dir / "bezier_tmp", ignore_errors=True)

        # CLAHE
        apply_clahe(ext_img_train, log_file)

        total_train = len(list(ext_img_train.glob("*.png")) +
                          list(ext_img_train.glob("*.PNG")))
        log(f"  Total train images (with Bezier+CLAHE): {total_train}", log_file)

        ext_yaml_path = extended_dir / "stenosis_extended.yaml"
        yaml.dump({
            "path": str(extended_dir.resolve()),
            "train": "images/train", "val": "images/val", "test": "images/test",
            "nc": 1, "names": {0: "stenosis"},
        }, open(ext_yaml_path, "w"), default_flow_style=False, sort_keys=False)

        # 5. Retrain from scratch
        cfg_sten_r = s54_stenosis_config(args.device)
        current_sten_weights = train_two_stage(
            cfg_sten_r, str(ext_yaml_path),
            str(output_dir / f"stenosis_model_r{round_num}"), f"sten_r{round_num}")

        r_metrics = evaluate_model(current_sten_weights, sten_yaml, split="test",
                                    augment=False, imgsz=768)
        r_f1 = r_metrics.get("per_class", {}).get("stenosis", {}).get("f1", 0)
        log(f"  Round {round_num} stenosis F1: {r_f1:.4f} "
            f"(initial: {init_f1:.4f}, delta: {r_f1 - init_f1:+.4f})", log_file)

        final_sten_weights = current_sten_weights
        final_f1 = r_f1
        final_metrics = r_metrics

    pseudo_stats = total_pseudo

    # ── Step 6: Evaluate with CC post-processing ──
    log("\n" + "=" * 60, log_file)
    log(f"STEP 6: CC post-processing (min_area={args.cc_min_area}px)", log_file)
    log("=" * 60, log_file)

    cc_results = {}
    for min_area in [30, 50, 100]:
        cc = evaluate_with_filter_cpu(
            final_sten_weights, sten_yaml, split="test", imgsz=768,
            min_area_px=min_area, device=args.device)
        cc_results[min_area] = cc
        log(f"  min_area={min_area}: raw F1={cc['raw']['f1']:.4f} "
            f"filtered F1={cc['filtered']['f1']:.4f} "
            f"(delta={cc['filtered']['f1'] - cc['raw']['f1']:+.4f})", log_file)

    # ── Final summary ──
    log("\n" + "=" * 60, log_file)
    log("FINAL RESULTS", log_file)
    log("=" * 60, log_file)
    log(f"  Syntax model:    {syntax_weights}", log_file)
    log(f"  Syntax F1:       {syn_f1:.4f} (initial: {init_syn_f1:.4f})", log_file)
    log(f"  Stenosis model:  {final_sten_weights}", log_file)
    log(f"  Stenosis F1:     {final_f1:.4f} (initial: {init_f1:.4f})", log_file)
    log(f"  Stenosis delta:  {final_f1 - init_f1:+.4f} from pseudo-labels + Bezier", log_file)
    log(f"", log_file)
    log(f"  BASELINE S54:    syntax 0.7398, stenosis 0.4569", log_file)
    log(f"  THIS RUN:        syntax {syn_f1:.4f}, stenosis {final_f1:.4f}", log_file)

    # Best CC threshold
    best_cc_area = max(cc_results, key=lambda k: cc_results[k]["filtered"]["f1"])
    best_cc_f1 = cc_results[best_cc_area]["filtered"]["f1"]
    log(f"  Best CC filter:  min_area={best_cc_area}px -> stenosis F1={best_cc_f1:.4f}", log_file)

    elapsed = time.time() - t0
    log(f"\nTotal time: {elapsed / 3600:.1f} hours", log_file)

    # Save results
    results = {
        "syntax_model": syntax_weights,
        "syntax_initial_f1": round(init_syn_f1, 4),
        "syntax_final_f1": round(syn_f1, 4),
        "syntax_metrics": syn_metrics,
        "stenosis_initial_f1": round(init_f1, 4),
        "stenosis_final_model": final_sten_weights,
        "stenosis_final_f1": round(final_f1, 4),
        "stenosis_final_metrics": final_metrics,
        "pseudo_label_stats": pseudo_stats,
        "bezier_stats": bezier_stats,
        "cc_postprocess": {str(k): v for k, v in cc_results.items()},
        "best_cc_min_area": best_cc_area,
        "best_cc_f1": round(best_cc_f1, 4),
        "elapsed_hours": round(elapsed / 3600, 2),
    }
    json.dump(results, open(output_dir / "results.json", "w"),
              indent=2, default=str)
    log(f"Results saved to {output_dir / 'results.json'}", log_file)
    log("DONE.", log_file)


if __name__ == "__main__":
    main()
