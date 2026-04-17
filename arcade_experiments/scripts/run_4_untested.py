#!/usr/bin/env python3
"""Run the 4 untested high-ROI experiments on stratified splits.

Experiments:
  L1: Per-class weighted loss (α ∝ 1/√count) for syntax tail classes
  L2: Per-class confidence threshold sweep (syntax + stenosis)
  L3: Proper WBF ensemble at conf=0.25 with 3-seed S54 recipe
  L4: CLAHE + 768px combined for stenosis

All use STRATIFIED splits (999/200/301) for evaluation — the same
protocol that produced the 0.74 baseline. Training uses the best
known recipe (S54: SGD lr=0.01, mosaic=0.8, copy_paste=0.3, 300ep).

Usage:
    python run_4_untested.py --arcade-root ../../arcade/submission --devices 0,1

    # Run a single experiment:
    python run_4_untested.py --arcade-root ../../arcade/submission --only L1
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))


# ═══════════════════════════════════════════════════════════════════════
# S54 base configs (the proven best recipe)
# ═══════════════════════════════════════════════════════════════════════

def s54_syntax_config(device: str = "0") -> dict:
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


def s54_stenosis_config(device: str = "0") -> dict:
    cfg = s54_syntax_config(device)
    cfg["copy_paste"] = 0.3
    cfg["scale"] = 0.5
    cfg["mosaic"] = 0.8
    cfg["close_mosaic"] = 15
    cfg["box"] = 10.0
    cfg["cls"] = 1.0
    return cfg


# ═══════════════════════════════════════════════════════════════════════
# Data preparation (stratified splits)
# ═══════════════════════════════════════════════════════════════════════

def prepare_stratified_data(arcade_root: Path, output_dir: Path,
                             seed: int = 42, min_count: int = 300):
    """Create stratified splits and prepare YOLO data."""
    splits_dir = output_dir / "stratified_splits"
    data_dir = output_dir / "data"

    if not splits_dir.exists():
        print("Creating stratified splits...")
        from create_stratified_splits import create_stratified_splits
        create_stratified_splits(arcade_root, splits_dir, seed)

    if not (data_dir / "dataset_configs" / "syntax_only.yaml").exists():
        print("Preparing YOLO data from stratified splits...")
        from run_pipeline import data_prep
        data_prep(arcade_root, data_dir, min_count=min_count,
                  splits_dir=splits_dir)

    # Fix paths in YAMLs
    configs_dir = data_dir / "dataset_configs"
    for fname, subdir in [("syntax_only.yaml", "syntax_filtered"),
                          ("stenosis_only.yaml", "stenosis")]:
        p = configs_dir / fname
        if p.exists():
            with open(p) as f:
                cfg = yaml.safe_load(f)
            correct = str((data_dir / subdir).resolve())
            if cfg.get("path") != correct:
                cfg["path"] = correct
                yaml.dump(cfg, open(p, "w"), default_flow_style=False,
                          sort_keys=False)

    return data_dir, splits_dir


def log(msg: str, log_file: Path = None):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    if log_file:
        with open(log_file, "a") as f:
            f.write(line + "\n")


# ═══════════════════════════════════════════════════════════════════════
# L1: Per-class weighted loss for tail classes
# ═══════════════════════════════════════════════════════════════════════

def run_L1_weighted_loss(arcade_root: Path, output_dir: Path,
                          device: str, log_file: Path):
    """Train syntax with inverse-sqrt class weights via cls loss scaling.

    The idea: classes 9/13/16 have ~320-360 instances vs ~900+ for top
    classes. Ultralytics doesn't support per-class loss weights directly,
    but we can approximate by:
    1. Oversampling tail classes (B-1 style) to equalize counts
    2. Increasing cls loss weight to make classification matter more
    3. Training with higher box weight for better localization on tails

    We also train 3 seeds to get reliable multi-seed mean.
    """
    log("L1: Per-class weighted loss (oversampling + cls emphasis)", log_file)

    data_dir, _ = prepare_stratified_data(arcade_root, output_dir)
    syn_yaml = str(data_dir / "dataset_configs" / "syntax_only.yaml")

    from train import train_two_stage
    from evaluate import evaluate_model
    from oversample_tail_classes import oversample, instance_counts

    results = {}
    for seed in [42, 7, 2024]:
        log(f"  L1 seed={seed}: preparing data...", log_file)

        # Fresh copy for each seed (oversampling modifies in-place)
        seed_data = output_dir / "L1_data" / f"seed{seed}"
        syn_src = data_dir / "syntax_filtered"
        syn_dst = seed_data / "syntax_filtered"
        if syn_dst.exists():
            shutil.rmtree(syn_dst)
        shutil.copytree(syn_src, syn_dst)

        # Count and oversample tail classes to match median
        counts = instance_counts(syn_dst / "labels" / "train")
        sorted_cls = sorted(counts.items(), key=lambda x: x[1])
        tail = [c for c, _ in sorted_cls[:3]]
        target = sorted_cls[len(sorted_cls) // 2][1]
        log(f"    Tail classes: {tail}, target: {target}, counts: {dict(counts)}", log_file)
        oversample(syn_dst, tail, target)

        # Write dataset YAML for oversampled data
        with open(syn_yaml) as f:
            syn_cfg = yaml.safe_load(f)
        os_yaml_path = seed_data / "syntax_oversampled.yaml"
        os_yaml = {
            "path": str(syn_dst.resolve()),
            "train": "images/train", "val": "images/val", "test": "images/test",
            "nc": syn_cfg["nc"], "names": syn_cfg["names"],
        }
        yaml.dump(os_yaml, open(os_yaml_path, "w"),
                  default_flow_style=False, sort_keys=False)

        # Train with higher cls weight
        cfg = s54_syntax_config(device)
        cfg["cls"] = 1.5  # 3x default — emphasize classification
        cfg["seed"] = seed
        project = str(output_dir / "L1_models" / f"seed{seed}")

        log(f"    Training seed={seed} with cls=1.5...", log_file)
        weights = train_two_stage(cfg, str(os_yaml_path), project,
                                  f"L1_seed{seed}")

        # Evaluate on original (non-oversampled) stratified test
        metrics = evaluate_model(weights, syn_yaml, split="test",
                                 augment=True, imgsz=768)
        pc = metrics.get("per_class", {})
        f1s = [v.get("f1", 0) for v in pc.values() if isinstance(v, dict)]
        mean_f1 = sum(f1s) / len(f1s) if f1s else 0

        results[f"seed_{seed}"] = {
            "model": weights, "mean_f1": round(mean_f1, 4),
            "mAP50": metrics.get("mAP50", 0),
            "per_class": pc,
        }
        log(f"    seed={seed}: mean_F1={mean_f1:.4f} mAP50={metrics.get('mAP50',0):.4f}", log_file)

    # Multi-seed summary
    all_f1 = [v["mean_f1"] for v in results.values()]
    results["multi_seed_mean_f1"] = round(sum(all_f1) / len(all_f1), 4)
    log(f"  L1 multi-seed mean F1: {results['multi_seed_mean_f1']:.4f}", log_file)
    return results


# ═══════════════════════════════════════════════════════════════════════
# L2: Per-class confidence threshold sweep
# ═══════════════════════════════════════════════════════════════════════

def run_L2_conf_sweep(arcade_root: Path, output_dir: Path,
                       device: str, log_file: Path):
    """Sweep per-class confidence thresholds on val, apply on test.

    Uses the best existing models (S54 syntax, S31 stenosis equivalent).
    This is a zero-retraining experiment — pure inference optimization.
    """
    log("L2: Per-class confidence threshold sweep", log_file)

    data_dir, _ = prepare_stratified_data(arcade_root, output_dir)
    syn_yaml = str(data_dir / "dataset_configs" / "syntax_only.yaml")
    sten_yaml = str(data_dir / "dataset_configs" / "stenosis_only.yaml")

    from sweep_confidence import (collect_predictions, sweep_per_class,
                                   evaluate_at_thresholds)

    # Find best models — train if not available
    syn_model = _find_or_train_syntax(arcade_root, output_dir, data_dir, device, log_file)
    sten_model = _find_or_train_stenosis(arcade_root, output_dir, data_dir, device, log_file)

    results = {}

    # Syntax sweep
    log("  Sweeping syntax thresholds on val...", log_file)
    val_items, names = collect_predictions(syn_model, syn_yaml, "val", 768, device)
    chosen, _ = sweep_per_class(val_items)
    chosen_named = {names.get(c, str(c)): v for c, v in chosen.items()}
    log(f"    Chosen thresholds: {chosen_named}", log_file)

    test_items, _ = collect_predictions(syn_model, syn_yaml, "test", 768, device)
    test_tuned = evaluate_at_thresholds(test_items, chosen)
    test_default = evaluate_at_thresholds(test_items, {c: 0.25 for c in chosen})

    def _mean_f1(m):
        vs = [v["f1"] for v in m.values()]
        return round(sum(vs) / len(vs), 4) if vs else 0

    results["syntax"] = {
        "thresholds": chosen_named,
        "test_default_f1": _mean_f1(test_default),
        "test_tuned_f1": _mean_f1(test_tuned),
        "delta": round(_mean_f1(test_tuned) - _mean_f1(test_default), 4),
        "per_class_default": {names.get(c, str(c)): v for c, v in test_default.items()},
        "per_class_tuned": {names.get(c, str(c)): v for c, v in test_tuned.items()},
    }
    log(f"  Syntax: default={results['syntax']['test_default_f1']:.4f} "
        f"tuned={results['syntax']['test_tuned_f1']:.4f} "
        f"delta={results['syntax']['delta']:+.4f}", log_file)

    # Stenosis sweep
    log("  Sweeping stenosis thresholds on val...", log_file)
    val_st, names_st = collect_predictions(sten_model, sten_yaml, "val", 768, device)
    chosen_st, _ = sweep_per_class(val_st)

    test_st, _ = collect_predictions(sten_model, sten_yaml, "test", 768, device)
    test_tuned_st = evaluate_at_thresholds(test_st, chosen_st)
    test_default_st = evaluate_at_thresholds(test_st, {c: 0.25 for c in chosen_st})

    results["stenosis"] = {
        "thresholds": {names_st.get(c, str(c)): v for c, v in chosen_st.items()},
        "test_default_f1": _mean_f1(test_default_st),
        "test_tuned_f1": _mean_f1(test_tuned_st),
        "delta": round(_mean_f1(test_tuned_st) - _mean_f1(test_default_st), 4),
    }
    log(f"  Stenosis: default={results['stenosis']['test_default_f1']:.4f} "
        f"tuned={results['stenosis']['test_tuned_f1']:.4f} "
        f"delta={results['stenosis']['delta']:+.4f}", log_file)

    return results


# ═══════════════════════════════════════════════════════════════════════
# L3: Proper WBF ensemble (3 seeds, conf=0.25)
# ═══════════════════════════════════════════════════════════════════════

def run_L3_wbf_ensemble(arcade_root: Path, output_dir: Path,
                         device: str, log_file: Path):
    """3-seed WBF ensemble at conf=0.25 (not 0.05 which flooded with FPs).

    Trains S54 recipe with 3 seeds, merges predictions via WBF.
    """
    log("L3: 3-seed WBF ensemble at conf=0.25", log_file)

    data_dir, _ = prepare_stratified_data(arcade_root, output_dir)
    syn_yaml = str(data_dir / "dataset_configs" / "syntax_only.yaml")
    sten_yaml = str(data_dir / "dataset_configs" / "stenosis_only.yaml")

    from train import train_two_stage
    from evaluate import evaluate_model

    # Train 3 seeds for syntax
    syntax_models = []
    stenosis_models = []
    for seed in [42, 7, 2024]:
        log(f"  Training syntax seed={seed}...", log_file)
        cfg = s54_syntax_config(device)
        cfg["seed"] = seed
        project = str(output_dir / "L3_syntax" / f"seed{seed}")
        w = train_two_stage(cfg, syn_yaml, project, f"L3_syn_s{seed}")
        syntax_models.append(w)

        # Single-seed eval for comparison
        m = evaluate_model(w, syn_yaml, split="test", augment=True, imgsz=768)
        pc = m.get("per_class", {})
        f1s = [v.get("f1", 0) for v in pc.values() if isinstance(v, dict)]
        mf1 = sum(f1s) / len(f1s) if f1s else 0
        log(f"    syntax seed={seed}: mean_F1={mf1:.4f}", log_file)

        log(f"  Training stenosis seed={seed}...", log_file)
        cfg_st = s54_stenosis_config(device)
        cfg_st["seed"] = seed
        project_st = str(output_dir / "L3_stenosis" / f"seed{seed}")
        w_st = train_two_stage(cfg_st, sten_yaml, project_st, f"L3_sten_s{seed}")
        stenosis_models.append(w_st)

        m_st = evaluate_model(w_st, sten_yaml, split="test", augment=True, imgsz=768)
        sten_f1 = m_st.get("per_class", {}).get("stenosis", {}).get("f1", 0)
        log(f"    stenosis seed={seed}: F1={sten_f1:.4f}", log_file)

    # WBF ensemble for syntax
    log("  Running WBF ensemble for syntax (conf=0.25)...", log_file)
    import subprocess
    syn_wbf_out = output_dir / "L3_syntax_wbf.json"
    subprocess.run([
        sys.executable, str(SCRIPT_DIR / "multiseed_wbf.py"),
        "--models", *syntax_models,
        "--data-yaml", syn_yaml,
        "--split", "test",
        "--imgsz", "768",
        "--device", device,
        "--conf", "0.25",  # KEY FIX: 0.25 not 0.05
        "--iou-thr", "0.55",
        "--output", str(syn_wbf_out),
    ], check=True)

    # WBF ensemble for stenosis
    log("  Running WBF ensemble for stenosis (conf=0.25)...", log_file)
    sten_wbf_out = output_dir / "L3_stenosis_wbf.json"
    subprocess.run([
        sys.executable, str(SCRIPT_DIR / "multiseed_wbf.py"),
        "--models", *stenosis_models,
        "--data-yaml", sten_yaml,
        "--split", "test",
        "--imgsz", "768",
        "--device", device,
        "--conf", "0.25",
        "--iou-thr", "0.55",
        "--output", str(sten_wbf_out),
    ], check=True)

    results = {
        "syntax_models": syntax_models,
        "stenosis_models": stenosis_models,
    }
    if syn_wbf_out.exists():
        results["syntax_wbf"] = json.load(open(syn_wbf_out))
        mf1 = results["syntax_wbf"].get("metrics", {}).get("mean_f1", 0)
        log(f"  Syntax WBF ensemble mean_F1: {mf1:.4f}", log_file)
    if sten_wbf_out.exists():
        results["stenosis_wbf"] = json.load(open(sten_wbf_out))
        mf1 = results["stenosis_wbf"].get("metrics", {}).get("mean_f1", 0)
        log(f"  Stenosis WBF ensemble F1: {mf1:.4f}", log_file)

    return results


# ═══════════════════════════════════════════════════════════════════════
# L4: CLAHE + 768px combined for stenosis
# ═══════════════════════════════════════════════════════════════════════

def run_L4_clahe_768(arcade_root: Path, output_dir: Path,
                      device: str, log_file: Path):
    """CLAHE preprocessing + 768px training for stenosis.

    Each helps independently (+2.9pp CLAHE, +5.5pp 768px). Never stacked.
    Also trains 3 seeds for reliable mean.
    """
    log("L4: CLAHE + 768px combined for stenosis (3 seeds)", log_file)

    data_dir, _ = prepare_stratified_data(arcade_root, output_dir)
    sten_yaml_src = str(data_dir / "dataset_configs" / "stenosis_only.yaml")

    from train import train_two_stage
    from evaluate import evaluate_model
    import cv2

    results = {}
    for seed in [42, 7, 2024]:
        log(f"  L4 seed={seed}: preparing CLAHE data...", log_file)

        # Copy stenosis data and apply CLAHE
        seed_data = output_dir / "L4_data" / f"seed{seed}"
        sten_src = data_dir / "stenosis"
        sten_dst = seed_data / "stenosis"
        if sten_dst.exists():
            shutil.rmtree(sten_dst)
        shutil.copytree(sten_src, sten_dst)

        # Apply CLAHE to training images
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        train_img_dir = sten_dst / "images" / "train"
        processed = 0
        for img_path in sorted(train_img_dir.glob("*.png")) + \
                        sorted(train_img_dir.glob("*.PNG")):
            # Materialize symlinks
            if img_path.is_symlink():
                real = img_path.resolve()
                img_path.unlink()
                shutil.copy2(real, img_path)

            img = cv2.imread(str(img_path))
            if img is None:
                continue
            if img.ndim == 2 or img.shape[2] == 1:
                gray = img if img.ndim == 2 else img[:, :, 0]
                out = clahe.apply(gray)
                cv2.imwrite(str(img_path), out)
            else:
                lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                l = clahe.apply(l)
                out = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
                cv2.imwrite(str(img_path), out)
            processed += 1
        log(f"    CLAHE applied to {processed} train images", log_file)

        # Write dataset YAML
        clahe_yaml_path = seed_data / "stenosis_clahe.yaml"
        clahe_yaml = {
            "path": str(sten_dst.resolve()),
            "train": "images/train", "val": "images/val", "test": "images/test",
            "nc": 1, "names": {0: "stenosis"},
        }
        yaml.dump(clahe_yaml, open(clahe_yaml_path, "w"),
                  default_flow_style=False, sort_keys=False)

        # Train at 768px with CLAHE
        cfg = s54_stenosis_config(device)
        cfg["seed"] = seed
        project = str(output_dir / "L4_models" / f"seed{seed}")

        log(f"    Training seed={seed} at 768px with CLAHE...", log_file)
        weights = train_two_stage(cfg, str(clahe_yaml_path), project,
                                  f"L4_seed{seed}")

        # Evaluate on original (non-CLAHE) stratified test
        metrics = evaluate_model(weights, sten_yaml_src, split="test",
                                 augment=True, imgsz=768)
        sten_f1 = metrics.get("per_class", {}).get("stenosis", {}).get("f1", 0)
        results[f"seed_{seed}"] = {
            "model": weights, "f1": round(sten_f1, 4),
            "mAP50": metrics.get("mAP50", 0),
        }
        log(f"    seed={seed}: stenosis F1={sten_f1:.4f} mAP50={metrics.get('mAP50',0):.4f}", log_file)

    all_f1 = [v["f1"] for v in results.values() if isinstance(v, dict) and "f1" in v]
    results["multi_seed_mean_f1"] = round(sum(all_f1) / len(all_f1), 4) if all_f1 else 0
    log(f"  L4 multi-seed mean stenosis F1: {results['multi_seed_mean_f1']:.4f}", log_file)
    return results


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════

def _find_or_train_syntax(arcade_root, output_dir, data_dir, device, log_file):
    """Find existing syntax model or train one."""
    # Check for L3 models first, then L1
    for pattern in ["L3_syntax/seed42/*best.pt", "L1_models/seed42/*best.pt"]:
        candidates = list(output_dir.rglob(pattern))
        if candidates:
            log(f"  Using existing syntax model: {candidates[0]}", log_file)
            return str(candidates[0])

    # Train fresh
    log("  Training baseline syntax model (S54)...", log_file)
    from train import train_two_stage
    cfg = s54_syntax_config(device)
    syn_yaml = str(data_dir / "dataset_configs" / "syntax_only.yaml")
    return train_two_stage(cfg, syn_yaml,
                           str(output_dir / "baseline_syntax"), "baseline_syn")


def _find_or_train_stenosis(arcade_root, output_dir, data_dir, device, log_file):
    """Find existing stenosis model or train one."""
    for pattern in ["L3_stenosis/seed42/*best.pt", "L4_models/seed42/*best.pt"]:
        candidates = list(output_dir.rglob(pattern))
        if candidates:
            log(f"  Using existing stenosis model: {candidates[0]}", log_file)
            return str(candidates[0])

    log("  Training baseline stenosis model...", log_file)
    from train import train_two_stage
    cfg = s54_stenosis_config(device)
    sten_yaml = str(data_dir / "dataset_configs" / "stenosis_only.yaml")
    return train_two_stage(cfg, sten_yaml,
                           str(output_dir / "baseline_stenosis"), "baseline_sten")


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="4 untested high-ROI experiments on stratified splits")
    parser.add_argument("--arcade-root", type=Path, required=True)
    parser.add_argument("--devices", type=str, default="0,1")
    parser.add_argument("--only", type=str, default=None,
                        help="Run only L1, L2, L3, or L4")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    arcade_root = args.arcade_root.resolve()
    output_dir = (args.output_dir or
                  SCRIPT_DIR.parent / "results" / "4_untested").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / "run_log.txt"

    devices = [d.strip() for d in args.devices.split(",")]
    d0 = devices[0]

    log(f"ARCADE root: {arcade_root}", log_file)
    log(f"Output dir:  {output_dir}", log_file)
    log(f"Devices:     {devices}", log_file)

    # Prepare stratified data once
    prepare_stratified_data(arcade_root, output_dir)

    all_results = {}
    experiments = {
        "L1": ("Per-class weighted loss", run_L1_weighted_loss),
        "L2": ("Confidence threshold sweep", run_L2_conf_sweep),
        "L3": ("3-seed WBF ensemble", run_L3_wbf_ensemble),
        "L4": ("CLAHE + 768px stenosis", run_L4_clahe_768),
    }

    to_run = [args.only] if args.only else ["L1", "L2", "L3", "L4"]

    t0 = time.time()
    for exp_id in to_run:
        if exp_id not in experiments:
            log(f"Unknown experiment: {exp_id}", log_file)
            continue
        name, func = experiments[exp_id]
        log(f"\n{'=' * 60}", log_file)
        log(f"{exp_id}: {name}", log_file)
        log(f"{'=' * 60}", log_file)
        try:
            result = func(arcade_root, output_dir, d0, log_file)
            all_results[exp_id] = result
        except Exception as e:
            log(f"ERROR in {exp_id}: {e}", log_file)
            traceback.print_exc()
            all_results[exp_id] = {"status": "failed", "error": str(e)}

    # Save results
    results_path = output_dir / "results.json"
    json.dump(all_results, open(results_path, "w"), indent=2, default=str)

    elapsed = time.time() - t0
    log(f"\nTotal time: {elapsed / 3600:.1f} hours", log_file)
    log(f"Results: {results_path}", log_file)

    # Print summary
    log("\n" + "=" * 60, log_file)
    log("SUMMARY", log_file)
    log("=" * 60, log_file)
    for exp_id, r in all_results.items():
        if isinstance(r, dict) and "status" in r:
            log(f"  {exp_id}: FAILED — {r.get('error', '')}", log_file)
        elif exp_id == "L1":
            log(f"  L1 weighted loss: multi-seed mean F1 = {r.get('multi_seed_mean_f1', 0):.4f}", log_file)
        elif exp_id == "L2":
            syn_delta = r.get("syntax", {}).get("delta", 0)
            sten_delta = r.get("stenosis", {}).get("delta", 0)
            log(f"  L2 conf sweep: syntax delta={syn_delta:+.4f}, stenosis delta={sten_delta:+.4f}", log_file)
        elif exp_id == "L3":
            syn_wbf = r.get("syntax_wbf", {}).get("metrics", {}).get("mean_f1", 0)
            sten_wbf = r.get("stenosis_wbf", {}).get("metrics", {}).get("mean_f1", 0)
            log(f"  L3 WBF ensemble: syntax F1={syn_wbf:.4f}, stenosis F1={sten_wbf:.4f}", log_file)
        elif exp_id == "L4":
            log(f"  L4 CLAHE+768: multi-seed mean stenosis F1 = {r.get('multi_seed_mean_f1', 0):.4f}", log_file)

    log("DONE.", log_file)


if __name__ == "__main__":
    main()
