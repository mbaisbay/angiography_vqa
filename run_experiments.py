#!/usr/bin/env python3
"""Run 5 YOLO training experiments sequentially and compare results.

Each experiment trains a syntax model with different hyperparameters,
then evaluates with ARCADE polygon F1. A comparison table is printed
at the end so you can pick the best config.

Usage:
    python run_experiments.py                              # run all 5
    python run_experiments.py --skip-preprocess             # already preprocessed
    python run_experiments.py --experiments 2,4             # run only exp 2 and 4
    python run_experiments.py --skip-preprocess --experiments 4  # re-run one
"""

import argparse
import copy
import csv
import json
import subprocess
import sys
import time
from collections import OrderedDict
from pathlib import Path

import yaml


# ── Experiment Definitions ─────────────────────────────────────────
# Each entry specifies ONLY the overrides vs base config.yaml.
# Nested dicts (training, augmentation) are deep-merged.

EXPERIMENTS = OrderedDict([
    ("exp1_baseline", {
        "description": "Baseline (current paper-aligned config)",
        # No overrides — uses config.yaml as-is
    }),
    ("exp2_yolo11x", {
        "description": "YOLO11x-seg architecture",
        "model_variant": "yolo11x-seg",
        "pretrained_weights": "yolo11x-seg.pt",
    }),
    ("exp3_sgd_cosine", {
        "description": "SGD optimizer + higher momentum",
        "training": {
            "optimizer": "SGD",
            "momentum": 0.965,
            "lrf": 0.01,
            "warmup_epochs": 5,
        },
    }),
    ("exp4_heavy_aug", {
        "description": "Heavy augmentation + dropout regularization",
        "training": {
            "epochs": 150,
            "patience": 40,
            "dropout": 0.15,
        },
        "augmentation": {
            "mosaic": 1.0,
            "mixup": 0.3,
            "copy_paste": 0.3,
            "degrees": 30.0,
            "scale": 0.5,
            "translate": 0.2,
        },
    }),
    ("exp5_low_lr_multiscale", {
        "description": "Low LR + multi-scale + strong weight decay",
        "training": {
            "epochs": 150,
            "lr0": 0.005,
            "lrf": 0.0005,
            "weight_decay": 0.001,
            "warmup_epochs": 5,
            "multi_scale": True,
        },
    }),
])


# ── Config Generation ──────────────────────────────────────────────

def generate_experiment_config(base_config_path: str, exp_name: str,
                               overrides: dict, configs_dir: str) -> str:
    """Load base config, apply overrides, write per-experiment YAML.

    Returns path to the generated config file.
    """
    with open(base_config_path) as f:
        config = yaml.safe_load(f)

    # Deep merge nested dicts (training, augmentation)
    for section in ("training", "augmentation", "preprocessing", "inference"):
        if section in overrides:
            config[section].update(overrides[section])

    # Top-level overrides
    for key in ("model_variant", "pretrained_weights"):
        if key in overrides:
            config[key] = overrides[key]

    # Isolate output to runs/<exp_name>
    exp_output = f"./runs/{exp_name}"
    config["output_dir"] = exp_output
    config["cross_inference"]["syntax_weights"] = f"{exp_output}/syntax/weights/best.pt"
    config["cross_inference"]["output_dir"] = f"{exp_output}/cross_inference"

    # Write config
    configs_dir = Path(configs_dir)
    configs_dir.mkdir(parents=True, exist_ok=True)
    config_path = configs_dir / f"config_{exp_name}.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    return str(config_path)


# ── Run Helpers ────────────────────────────────────────────────────

def run_command(cmd: list, description: str) -> tuple:
    """Run a subprocess, return (success, elapsed_seconds)."""
    print(f"\n{'='*70}")
    print(f"  {description}")
    print(f"  CMD: {' '.join(cmd)}")
    print(f"{'='*70}\n")

    start = time.time()
    result = subprocess.run(cmd)
    elapsed = time.time() - start

    status = "OK" if result.returncode == 0 else f"FAILED (exit {result.returncode})"
    print(f"\n  [{status}] {description} ({elapsed/60:.1f} min)")
    return result.returncode == 0, elapsed


def load_metrics(exp_name: str) -> dict:
    """Load evaluation metrics for an experiment."""
    metrics_path = Path(f"runs/{exp_name}/evaluation/syntax_metrics.json")
    if metrics_path.exists():
        with open(metrics_path) as f:
            return json.load(f)
    return {}


def load_training_losses(exp_name: str) -> dict:
    """Read first/last epoch losses from results.csv."""
    csv_path = Path(f"runs/{exp_name}/syntax/results.csv")
    if not csv_path.exists():
        return {}
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    if len(rows) < 2:
        return {}
    first = {k.strip(): v.strip() for k, v in rows[0].items()}
    last = {k.strip(): v.strip() for k, v in rows[-1].items()}
    return {
        "first_box_loss": float(first.get("train/box_loss", 0)),
        "last_box_loss": float(last.get("train/box_loss", 0)),
        "first_seg_loss": float(first.get("train/seg_loss", 0)),
        "last_seg_loss": float(last.get("train/seg_loss", 0)),
        "epochs_completed": len(rows),
    }


# ── Comparison Table ───────────────────────────────────────────────

def print_comparison(results: dict):
    """Print formatted comparison of all experiments."""
    print(f"\n{'='*80}")
    print(f"  EXPERIMENT COMPARISON")
    print(f"{'='*80}")

    header = (f"  {'Experiment':<25s} {'F1':>8s} {'mAP50':>8s} "
              f"{'mAP50-95':>10s} {'Epochs':>7s} {'Time':>8s}")
    print(f"\n{header}")
    print(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*10} {'-'*7} {'-'*8}")

    best_exp = None
    best_f1 = -1.0

    for exp_name, data in results.items():
        m = data.get("metrics", {})
        f1 = m.get("arcade_f1", {}).get("overall_mean_f1", None)
        ul = m.get("ultralytics", {})
        map50 = ul.get("mAP50", None)
        map50_95 = ul.get("mAP50_95", None)
        losses = data.get("losses", {})
        epochs = losses.get("epochs_completed", "?")
        elapsed = data.get("elapsed_min", 0)

        f1_s = f"{f1:.4f}" if f1 is not None else "FAIL"
        map50_s = f"{map50:.4f}" if map50 is not None else "N/A"
        map95_s = f"{map50_95:.4f}" if map50_95 is not None else "N/A"

        print(f"  {exp_name:<25s} {f1_s:>8s} {map50_s:>8s} "
              f"{map95_s:>10s} {str(epochs):>7s} {elapsed:>7.1f}m")

        if f1 is not None and f1 > best_f1:
            best_f1 = f1
            best_exp = exp_name

    if best_exp:
        print(f"\n  BEST: {best_exp} (ARCADE F1 = {best_f1:.4f})")
        print(f"  Weights: runs/{best_exp}/syntax/weights/best.pt")

    # Per-class breakdown for best
    if best_exp:
        per_class = results[best_exp]["metrics"].get("arcade_f1", {}).get("per_class", {})
        if per_class:
            print(f"\n  Per-class F1 for {best_exp}:")
            for cls_name, cls_data in sorted(per_class.items()):
                print(f"    {cls_name:>6s}: {cls_data['mean_f1']:.4f}")

    print(f"\n{'='*80}")

    # Save full comparison JSON
    summary_path = Path("runs/experiment_comparison.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Full results saved to: {summary_path}")


# ── Main ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run 5 YOLO experiments and compare results"
    )
    parser.add_argument("--config", default="config.yaml",
                        help="Base config file (default: config.yaml)")
    parser.add_argument("--skip-preprocess", action="store_true",
                        help="Skip filter + preprocess (if already done)")
    parser.add_argument("--experiments", type=str, default=None,
                        help="Comma-separated experiment numbers to run, e.g. '1,3,5'")
    args = parser.parse_args()

    exp_names = list(EXPERIMENTS.keys())

    # Filter to requested experiments
    if args.experiments:
        indices = [int(x) for x in args.experiments.split(",")]
        exp_names = [exp_names[i - 1] for i in indices if 1 <= i <= len(exp_names)]

    total_start = time.time()

    print("=" * 70)
    print("  ARCADE YOLO Experiment Runner")
    print("=" * 70)
    print(f"  Base config:  {args.config}")
    print(f"  Experiments:  {len(exp_names)}")
    for i, name in enumerate(exp_names, 1):
        desc = EXPERIMENTS[name].get("description", "")
        print(f"    {i}. {name}: {desc}")
    print()

    # Step 1: Preprocess once
    if not args.skip_preprocess:
        ok, _ = run_command(
            [sys.executable, "filter_classes.py", "--config", args.config, "--min-count", "300"],
            "Filter classes (>= 300 annotations)",
        )
        if not ok:
            print("  [ABORT] Filtering failed.")
            sys.exit(1)

        ok, _ = run_command(
            [sys.executable, "preprocess_images.py", "--config", args.config],
            "Preprocess images (CLAHE + top-hat)",
        )
        if not ok:
            print("  [ABORT] Preprocessing failed.")
            sys.exit(1)

    # Step 2: Generate all configs upfront
    configs = {}
    for exp_name in exp_names:
        overrides = {k: v for k, v in EXPERIMENTS[exp_name].items()
                     if k != "description"}
        config_path = generate_experiment_config(
            args.config, exp_name, overrides, "runs/configs"
        )
        configs[exp_name] = config_path
        print(f"  Generated config: {config_path}")

    # Step 3: Run experiments sequentially
    results = OrderedDict()

    for i, exp_name in enumerate(exp_names, 1):
        config_path = configs[exp_name]
        desc = EXPERIMENTS[exp_name].get("description", "")

        print(f"\n{'#'*70}")
        print(f"  EXPERIMENT {i}/{len(exp_names)}: {exp_name}")
        print(f"  {desc}")
        print(f"{'#'*70}")

        # Train
        train_ok, elapsed = run_command(
            [sys.executable, "train.py", "--config", config_path, "--task", "syntax"],
            f"Train syntax: {exp_name}",
        )

        # Evaluate (both ultralytics mAP + ARCADE F1)
        metrics = {}
        if train_ok:
            eval_ok, _ = run_command(
                [sys.executable, "evaluate.py", "--config", config_path,
                 "--task", "syntax", "--method", "both"],
                f"Evaluate: {exp_name}",
            )
            if eval_ok:
                metrics = load_metrics(exp_name)

        losses = load_training_losses(exp_name)

        results[exp_name] = {
            "description": desc,
            "config": config_path,
            "train_ok": train_ok,
            "elapsed_min": round(elapsed / 60, 1),
            "metrics": metrics,
            "losses": losses,
        }

        # Progress update
        remaining = len(exp_names) - i
        if remaining > 0:
            avg_time = sum(r["elapsed_min"] for r in results.values()) / len(results)
            eta = avg_time * remaining
            print(f"\n  Progress: {i}/{len(exp_names)} done, ~{eta:.0f} min remaining")

    # Step 4: Comparison
    total_elapsed = (time.time() - total_start) / 60
    print_comparison(results)
    print(f"\n  Total time: {total_elapsed:.1f} min")


if __name__ == "__main__":
    main()
