#!/usr/bin/env python3
"""Test single-stage training for syntax with and without SSASS augmentation.

Experiments (run in parallel on 2 GPUs):
  A: S54 syntax recipe but SINGLE-STAGE (no freeze/unfreeze, no LR drop)
  B: Same + SSASS augmentation (flipud, shear, perspective, HSV)

Both use yolo11m-seg at 768px, SGD lr=0.01, 300 epochs.
Evaluated on stratified test split.

Usage:
    # Both in parallel:
    python run_syntax_singlestage.py --arcade-root ../../arcade/submission --exp A --device 0 &
    python run_syntax_singlestage.py --arcade-root ../../arcade/submission --exp B --device 1 &
    wait
"""

from __future__ import annotations

import argparse
import json
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


def config_A(device):
    """S54 augmentation, single-stage."""
    return {
        "model": "yolo11m-seg.pt",
        "imgsz": 768, "batch": 8,
        "epochs": 300, "patience": 50,
        "optimizer": "SGD", "lr0": 0.01, "lrf": 0.01,
        "weight_decay": 0.0005, "momentum": 0.937,
        "warmup_epochs": 5,
        "seed": 42, "deterministic": True, "amp": True,
        "cos_lr": True, "device": device, "workers": 4,
        # S54 augmentation (same as before, just no freeze)
        "mosaic": 0.0, "close_mosaic": 0, "mixup": 0.0,
        "copy_paste": 0.0, "fliplr": 0.5, "flipud": 0.0,
        "degrees": 20.0, "scale": 0.4, "translate": 0.1,
        "shear": 0.0, "perspective": 0.0,
        "hsv_h": 0.0, "hsv_s": 0.0, "hsv_v": 0.3,
        "erasing": 0.0,
        "box": 7.5, "cls": 0.5, "dfl": 1.5,
    }


def config_B(device):
    """SSASS augmentation, single-stage."""
    return {
        "model": "yolo11m-seg.pt",
        "imgsz": 768, "batch": 8,
        "epochs": 300, "patience": 50,
        "optimizer": "SGD", "lr0": 0.01, "lrf": 0.01,
        "weight_decay": 0.0005, "momentum": 0.937,
        "warmup_epochs": 3,
        "seed": 42, "deterministic": True, "amp": True,
        "cos_lr": True, "device": device, "workers": 4,
        # SSASS augmentation
        "mosaic": 0.0, "close_mosaic": 0, "mixup": 0.0,
        "copy_paste": 0.0, "fliplr": 0.5, "flipud": 0.5,
        "degrees": 30.0, "scale": 0.5, "translate": 0.3,
        "shear": 5.0, "perspective": 0.001,
        "hsv_h": 0.015, "hsv_s": 0.7, "hsv_v": 0.4,
        "erasing": 0.0,
        "box": 7.5, "cls": 0.5, "dfl": 1.5,
    }


def train_single_stage(cfg, data_yaml, project, name):
    from ultralytics import YOLO

    project = str(Path(project).resolve())
    model = YOLO(cfg["model"])
    train_args = {k: v for k, v in cfg.items()
                  if k not in ("model",)}
    train_args["data"] = data_yaml
    train_args["project"] = project
    train_args["name"] = name
    train_args["exist_ok"] = True
    train_args["freeze"] = 0

    model.train(**train_args)

    best = Path(project) / name / "weights" / "best.pt"
    if not best.exists():
        best = Path(project) / name / "weights" / "last.pt"
    final = Path(project) / f"{name}_best.pt"
    shutil.copy2(best, final)
    return str(final)


def prepare_data(arcade_root, output_dir):
    splits_dir = output_dir / "stratified_splits"
    data_dir = output_dir / "data"
    if not splits_dir.exists():
        from create_stratified_splits import create_stratified_splits
        create_stratified_splits(arcade_root, splits_dir, 42)
    if not (data_dir / "dataset_configs" / "syntax_only.yaml").exists():
        from run_pipeline import data_prep
        data_prep(arcade_root, data_dir, min_count=300, splits_dir=splits_dir)
    for fname, subdir in [("syntax_only.yaml", "syntax_filtered")]:
        p = data_dir / "dataset_configs" / fname
        if p.exists():
            with open(p) as f:
                c = yaml.safe_load(f)
            correct = str((data_dir / subdir).resolve())
            if c.get("path") != correct:
                c["path"] = correct
                yaml.dump(c, open(p, "w"), default_flow_style=False,
                          sort_keys=False)
    return data_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arcade-root", type=Path, required=True)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--exp", type=str, required=True, choices=["A", "B"],
                        help="A = S54 aug single-stage, B = SSASS aug single-stage")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    arcade_root = args.arcade_root.resolve()
    output_dir = (args.output_dir or
                  SCRIPT_DIR.parent / "results" / "syntax_singlestage").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / f"run_{args.exp}_log.txt"

    configs = {"A": config_A, "B": config_B}
    labels = {"A": "S54 aug + single-stage", "B": "SSASS aug + single-stage"}
    cfg = configs[args.exp](args.device)

    log(f"Experiment {args.exp}: {labels[args.exp]}", log_file)
    log(f"Device: {args.device}", log_file)
    log(f"Key aug: flipud={cfg['flipud']} shear={cfg['shear']} "
        f"perspective={cfg['perspective']} hsv_s={cfg['hsv_s']}", log_file)

    t0 = time.time()
    data_dir = prepare_data(arcade_root, output_dir)
    syn_yaml = str(data_dir / "dataset_configs" / "syntax_only.yaml")

    log(f"\nTraining...", log_file)
    weights = train_single_stage(
        cfg, syn_yaml, str(output_dir / f"exp_{args.exp}"), f"syntax_{args.exp}")

    from evaluate import evaluate_model
    m = evaluate_model(weights, syn_yaml, split="test", augment=False,
                        imgsz=cfg["imgsz"])
    pc = m.get("per_class", {})
    f1s = [v.get("f1", 0) for v in pc.values() if isinstance(v, dict)]
    mean_f1 = sum(f1s) / len(f1s) if f1s else 0

    log(f"\n{'='*60}", log_file)
    log(f"Exp {args.exp} ({labels[args.exp]}): mean F1 = {mean_f1:.4f}", log_file)
    log(f"S54 two-stage baseline:               mean F1 = 0.7398", log_file)
    log(f"Delta: {mean_f1 - 0.7398:+.4f}", log_file)
    log(f"Per-class: { {k: round(v.get('f1',0),4) for k,v in pc.items()} }", log_file)
    log(f"Time: {(time.time()-t0)/3600:.1f}h", log_file)

    json.dump({
        "exp": args.exp,
        "label": labels[args.exp],
        "mean_f1": round(mean_f1, 4),
        "mAP50": m.get("mAP50", 0),
        "per_class": pc,
        "config": cfg,
        "model": weights,
    }, open(output_dir / f"results_{args.exp}.json", "w"), indent=2, default=str)


if __name__ == "__main__":
    main()
