#!/usr/bin/env python3
"""Final push: conf tuning + stronger backbones with S54 recipe.

Experiments:
  F1: Apply per-class conf tuning on the best S54 syntax+stenosis models
      (from L3 run which reproduced 0.7398 syntax F1). Zero retraining.

  F2: S54 recipe on yolo11x-seg backbone (larger model, same recipe).
      Previous S56 tried yolo11x but at different settings. This uses
      the exact S54 recipe (SGD lr=0.01, mosaic=0.8 stenosis, CLAHE,
      768px, 300ep) — the only change is the backbone.

  F3: S54 recipe on yolov8x-seg backbone (different architecture family).
      Tests if YOLOv8's architecture is better suited to this data.

  F4: S54 recipe on yolo26m-seg (end-to-end, no NMS architecture).
      Fundamentally different detection paradigm.

All on stratified splits. Each backbone trains syntax + stenosis.

Usage:
    python run_final_push.py --arcade-root ../../arcade/submission --device 0

    # Single experiment:
    python run_final_push.py --arcade-root ../../arcade/submission --only F1
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


def log(msg, log_file):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(log_file, "a") as f:
        f.write(line + "\n")


def prepare_stratified_data(arcade_root, output_dir, seed=42, min_count=300):
    splits_dir = output_dir / "stratified_splits"
    data_dir = output_dir / "data"
    if not splits_dir.exists():
        from create_stratified_splits import create_stratified_splits
        create_stratified_splits(arcade_root, splits_dir, seed)
    if not (data_dir / "dataset_configs" / "syntax_only.yaml").exists():
        from run_pipeline import data_prep
        data_prep(arcade_root, data_dir, min_count=min_count,
                  splits_dir=splits_dir)
    # Fix paths
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


def apply_clahe_to_dir(img_dir: Path, log_file):
    """Apply CLAHE in-place. Materializes symlinks first."""
    import cv2
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


def train_and_eval_syntax(cfg, data_yaml, project, name, eval_yaml, log_file):
    """Train syntax model, evaluate, return (weights, mean_f1, metrics)."""
    from train import train_two_stage
    from evaluate import evaluate_model
    weights = train_two_stage(cfg, data_yaml, project, name)
    metrics = evaluate_model(weights, eval_yaml, split="test",
                             augment=True, imgsz=cfg.get("imgsz", 768))
    pc = metrics.get("per_class", {})
    f1s = [v.get("f1", 0) for v in pc.values() if isinstance(v, dict)]
    mf1 = sum(f1s) / len(f1s) if f1s else 0
    log(f"    {name}: syntax mean_F1={mf1:.4f} mAP50={metrics.get('mAP50',0):.4f}", log_file)
    return weights, round(mf1, 4), metrics


def train_and_eval_stenosis(cfg, data_yaml, project, name, eval_yaml, log_file):
    """Train stenosis model, evaluate, return (weights, f1, metrics)."""
    from train import train_two_stage
    from evaluate import evaluate_model
    weights = train_two_stage(cfg, data_yaml, project, name)
    metrics = evaluate_model(weights, eval_yaml, split="test",
                             augment=True, imgsz=cfg.get("imgsz", 768))
    f1 = metrics.get("per_class", {}).get("stenosis", {}).get("f1", 0)
    log(f"    {name}: stenosis F1={f1:.4f} mAP50={metrics.get('mAP50',0):.4f}", log_file)
    return weights, round(f1, 4), metrics


# ═══════════════════════════════════════════════════════════════════════
# F1: Conf tuning on best S54 models
# ═══════════════════════════════════════════════════════════════════════

def run_F1_conf_on_s54(output_dir, data_dir, device, log_file):
    """Apply per-class conf tuning on the best existing S54 models."""
    log("F1: Conf tuning on S54 models", log_file)
    from sweep_confidence import (collect_predictions, sweep_per_class,
                                   evaluate_at_thresholds)

    syn_yaml = str(data_dir / "dataset_configs" / "syntax_only.yaml")
    sten_yaml = str(data_dir / "dataset_configs" / "stenosis_only.yaml")

    # Find the best syntax model (from L3 or train fresh)
    syn_candidates = list(output_dir.rglob("L3_syntax/seed42/*best.pt"))
    if not syn_candidates:
        syn_candidates = list(output_dir.rglob("*syntax*best.pt"))
    if syn_candidates:
        syn_model = str(syn_candidates[0])
        log(f"  Using syntax model: {syn_model}", log_file)
    else:
        log("  No syntax model found — training S54 baseline...", log_file)
        cfg = s54_syntax_config(device)
        syn_model, _, _ = train_and_eval_syntax(
            cfg, syn_yaml, str(output_dir / "F1_syntax"), "s54_baseline",
            syn_yaml, log_file)

    # Find stenosis model
    sten_candidates = list(output_dir.rglob("L3_stenosis/seed42/*best.pt"))
    if not sten_candidates:
        sten_candidates = list(output_dir.rglob("*stenosis*best.pt"))
    if sten_candidates:
        sten_model = str(sten_candidates[0])
        log(f"  Using stenosis model: {sten_model}", log_file)
    else:
        log("  No stenosis model found — training baseline...", log_file)
        cfg = s54_stenosis_config(device)
        sten_model, _, _ = train_and_eval_stenosis(
            cfg, sten_yaml, str(output_dir / "F1_stenosis"), "s54_sten",
            sten_yaml, log_file)

    def _mean_f1(m):
        vs = [v["f1"] for v in m.values()]
        return round(sum(vs) / len(vs), 4) if vs else 0

    results = {}

    # Syntax conf sweep
    log("  Sweeping syntax conf on val...", log_file)
    val_items, names = collect_predictions(syn_model, syn_yaml, "val", 768, device)
    chosen, _ = sweep_per_class(val_items)
    log(f"    Thresholds: { {names.get(c,str(c)): round(v,2) for c,v in chosen.items()} }", log_file)

    test_items, _ = collect_predictions(syn_model, syn_yaml, "test", 768, device)
    test_tuned = evaluate_at_thresholds(test_items, chosen)
    test_default = evaluate_at_thresholds(test_items, {c: 0.25 for c in chosen})

    results["syntax"] = {
        "default_f1": _mean_f1(test_default),
        "tuned_f1": _mean_f1(test_tuned),
        "delta": round(_mean_f1(test_tuned) - _mean_f1(test_default), 4),
        "thresholds": {names.get(c, str(c)): round(v, 2) for c, v in chosen.items()},
        "per_class_tuned": {names.get(c, str(c)): v for c, v in test_tuned.items()},
    }
    log(f"  Syntax: {results['syntax']['default_f1']:.4f} -> {results['syntax']['tuned_f1']:.4f} "
        f"(delta={results['syntax']['delta']:+.4f})", log_file)

    # Stenosis conf sweep
    log("  Sweeping stenosis conf on val...", log_file)
    val_st, names_st = collect_predictions(sten_model, sten_yaml, "val", 768, device)
    chosen_st, _ = sweep_per_class(val_st)

    test_st, _ = collect_predictions(sten_model, sten_yaml, "test", 768, device)
    test_tuned_st = evaluate_at_thresholds(test_st, chosen_st)
    test_default_st = evaluate_at_thresholds(test_st, {c: 0.25 for c in chosen_st})

    results["stenosis"] = {
        "default_f1": _mean_f1(test_default_st),
        "tuned_f1": _mean_f1(test_tuned_st),
        "delta": round(_mean_f1(test_tuned_st) - _mean_f1(test_default_st), 4),
        "thresholds": {names_st.get(c, str(c)): round(v, 2) for c, v in chosen_st.items()},
    }
    log(f"  Stenosis: {results['stenosis']['default_f1']:.4f} -> {results['stenosis']['tuned_f1']:.4f} "
        f"(delta={results['stenosis']['delta']:+.4f})", log_file)

    return results


# ═══════════════════════════════════════════════════════════════════════
# F2/F3/F4: Stronger backbones with S54 recipe
# ═══════════════════════════════════════════════════════════════════════

def run_backbone_experiment(backbone: str, exp_name: str,
                            output_dir: Path, data_dir: Path,
                            device: str, log_file: Path):
    """Run S54 recipe with a different backbone. Train syntax + stenosis."""
    log(f"{exp_name}: S54 recipe on {backbone}", log_file)

    syn_yaml = str(data_dir / "dataset_configs" / "syntax_only.yaml")
    sten_yaml = str(data_dir / "dataset_configs" / "stenosis_only.yaml")

    # Syntax
    log(f"  Training syntax with {backbone}...", log_file)
    cfg_syn = s54_syntax_config(device)
    cfg_syn["model"] = backbone
    # yolo11x and yolov8x need smaller batch at 768
    if "x" in backbone or "l" in backbone:
        cfg_syn["batch"] = 4
    syn_w, syn_f1, syn_m = train_and_eval_syntax(
        cfg_syn, syn_yaml, str(output_dir / f"{exp_name}_syntax"),
        f"{exp_name}_syn", syn_yaml, log_file)

    # Stenosis with CLAHE
    log(f"  Preparing CLAHE stenosis data...", log_file)
    sten_clahe_dir = output_dir / f"{exp_name}_sten_data" / "stenosis"
    if sten_clahe_dir.exists():
        shutil.rmtree(sten_clahe_dir)
    shutil.copytree(data_dir / "stenosis", sten_clahe_dir)
    apply_clahe_to_dir(sten_clahe_dir / "images" / "train", log_file)

    clahe_yaml_path = output_dir / f"{exp_name}_sten_data" / "stenosis_clahe.yaml"
    yaml.dump({
        "path": str(sten_clahe_dir.resolve()),
        "train": "images/train", "val": "images/val", "test": "images/test",
        "nc": 1, "names": {0: "stenosis"},
    }, open(clahe_yaml_path, "w"), default_flow_style=False, sort_keys=False)

    log(f"  Training stenosis with {backbone} + CLAHE...", log_file)
    cfg_st = s54_stenosis_config(device)
    cfg_st["model"] = backbone
    if "x" in backbone or "l" in backbone:
        cfg_st["batch"] = 4
    sten_w, sten_f1, sten_m = train_and_eval_stenosis(
        cfg_st, str(clahe_yaml_path), str(output_dir / f"{exp_name}_stenosis"),
        f"{exp_name}_sten", sten_yaml, log_file)

    return {
        "backbone": backbone,
        "syntax_model": syn_w, "syntax_f1": syn_f1, "syntax_metrics": syn_m,
        "stenosis_model": sten_w, "stenosis_f1": sten_f1, "stenosis_metrics": sten_m,
    }


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Final push experiments")
    parser.add_argument("--arcade-root", type=Path, required=True)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--only", type=str, default=None,
                        help="Run only F1, F2, F3, or F4")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    arcade_root = args.arcade_root.resolve()
    output_dir = (args.output_dir or
                  SCRIPT_DIR.parent / "results" / "final_push").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / "run_log.txt"

    log(f"ARCADE root: {arcade_root}", log_file)
    log(f"Output dir:  {output_dir}", log_file)
    log(f"Device:      {args.device}", log_file)

    data_dir = prepare_stratified_data(arcade_root, output_dir)

    experiments = {
        "F1": ("Conf tuning on S54", lambda: run_F1_conf_on_s54(
            output_dir, data_dir, args.device, log_file)),
        "F2": ("S54 + yolo11x-seg", lambda: run_backbone_experiment(
            "yolo11x-seg.pt", "F2", output_dir, data_dir, args.device, log_file)),
        "F3": ("S54 + yolov8x-seg", lambda: run_backbone_experiment(
            "yolov8x-seg.pt", "F3", output_dir, data_dir, args.device, log_file)),
        "F4": ("S54 + yolo26m-seg", lambda: run_backbone_experiment(
            "yolo26m-seg.pt", "F4", output_dir, data_dir, args.device, log_file)),
    }

    to_run = [args.only] if args.only else ["F1", "F2", "F3", "F4"]
    all_results = {}

    t0 = time.time()
    for exp_id in to_run:
        if exp_id not in experiments:
            log(f"Unknown: {exp_id}", log_file)
            continue
        name, func = experiments[exp_id]
        log(f"\n{'='*60}\n{exp_id}: {name}\n{'='*60}", log_file)
        try:
            all_results[exp_id] = func()
        except Exception as e:
            log(f"ERROR in {exp_id}: {e}", log_file)
            traceback.print_exc()
            all_results[exp_id] = {"status": "failed", "error": str(e)}

    json.dump(all_results, open(output_dir / "results.json", "w"),
              indent=2, default=str)

    log(f"\n{'='*60}\nSUMMARY\n{'='*60}", log_file)
    for exp_id, r in all_results.items():
        if r.get("status") == "failed":
            log(f"  {exp_id}: FAILED — {r.get('error','')[:80]}", log_file)
        elif exp_id == "F1":
            log(f"  F1 conf tune: syntax {r['syntax']['default_f1']:.4f}->{r['syntax']['tuned_f1']:.4f} "
                f"| stenosis {r['stenosis']['default_f1']:.4f}->{r['stenosis']['tuned_f1']:.4f}", log_file)
        else:
            log(f"  {exp_id} ({r.get('backbone','')}): syntax F1={r.get('syntax_f1',0):.4f} "
                f"stenosis F1={r.get('stenosis_f1',0):.4f}", log_file)

    # Compare to baseline
    log(f"\n  BASELINE (S54): syntax 0.7398, stenosis 0.4569", log_file)
    log(f"\nTotal time: {(time.time()-t0)/3600:.1f} hours", log_file)
    log("DONE.", log_file)


if __name__ == "__main__":
    main()
