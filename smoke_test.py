#!/usr/bin/env python3
"""Smoke test: run a minimal pipeline (filter -> preprocess -> train 5 epochs -> evaluate).

Validates that the full pipeline works end-to-end and produces F1 > 0.0
for the 13 filtered classes. Uses a separate output directory (runs_smoke/)
to avoid clobbering real training runs.

Usage:
    python smoke_test.py --config config.yaml
    python smoke_test.py --config config.yaml --epochs 3 --skip-preprocess
"""

import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path

import yaml

from utils.config_loader import load_config


def run_step(description: str, cmd: list) -> bool:
    """Run a pipeline step, return True on success."""
    print(f"\n{'='*60}")
    print(f"  SMOKE TEST: {description}")
    print(f"  CMD: {' '.join(cmd)}")
    print(f"{'='*60}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"  [FAIL] {description} exited with code {result.returncode}")
        return False
    print(f"  [OK] {description}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Smoke test the ARCADE YOLO pipeline")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--epochs", type=int, default=5,
                        help="Number of training epochs (default: 5)")
    parser.add_argument("--patience", type=int, default=5,
                        help="Early stopping patience (default: 5)")
    parser.add_argument("--skip-preprocess", action="store_true",
                        help="Skip preprocessing step")
    parser.add_argument("--task", type=str, default="syntax",
                        choices=["syntax", "stenosis"],
                        help="Which task to smoke-test (default: syntax)")
    args = parser.parse_args()

    print("=" * 60)
    print("ARCADE YOLO Pipeline — Smoke Test")
    print("=" * 60)
    print(f"  Config: {args.config}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Task:   {args.task}")

    failures = []

    # Step 1: Filter classes
    ok = run_step(
        "Filter classes (>=300 annotations)",
        [sys.executable, "filter_classes.py", "--config", args.config, "--min-count", "300"],
    )
    if not ok:
        failures.append("filter_classes")
        print("\n  [ABORT] Cannot continue without filtered data.")
        _print_summary(failures)
        sys.exit(1)

    # Step 2: Preprocess (optional)
    if not args.skip_preprocess:
        ok = run_step(
            "Preprocess images (CLAHE + top-hat)",
            [sys.executable, "preprocess_images.py", "--config", args.config],
        )
        if not ok:
            failures.append("preprocess_images")
            print("\n  [ABORT] Preprocessing failed.")
            _print_summary(failures)
            sys.exit(1)

    # Step 3: Build a temporary smoke config with overridden training params
    config_path = Path(args.config).resolve()
    with open(config_path) as f:
        raw_config = yaml.safe_load(f)

    smoke_config = copy.deepcopy(raw_config)
    smoke_config["training"]["epochs"] = args.epochs
    smoke_config["training"]["patience"] = args.patience
    smoke_config["output_dir"] = "./runs_smoke"
    smoke_config["cross_inference"][f"{args.task}_weights"] = (
        f"./runs_smoke/{args.task}/weights/best.pt"
    )
    smoke_config["cross_inference"]["output_dir"] = "./runs_smoke/cross_inference"
    # Lower confidence threshold for underfitted smoke models
    smoke_config["inference"]["confidence_threshold"] = 0.05

    smoke_config_path = config_path.parent / "config_smoke.yaml"
    with open(smoke_config_path, "w") as f:
        yaml.dump(smoke_config, f, default_flow_style=False, sort_keys=False)
    print(f"\n  Wrote smoke config: {smoke_config_path}")

    try:
        # Step 4: Train for a few epochs
        ok = run_step(
            f"Train {args.task} model ({args.epochs} epochs)",
            [sys.executable, "train.py", "--config", str(smoke_config_path), "--task", args.task],
        )
        if not ok:
            failures.append("train")
            _print_summary(failures)
            sys.exit(1)

        # Step 5: Evaluate with ARCADE F1
        ok = run_step(
            f"Evaluate {args.task} (ARCADE F1)",
            [sys.executable, "evaluate.py", "--config", str(smoke_config_path),
             "--task", args.task, "--method", "arcade"],
        )
        if not ok:
            failures.append("evaluate")

        # Step 6: Check F1 results
        config = load_config(str(smoke_config_path))
        eval_dir = Path(config["output_dir"]) / "evaluation"
        metrics_file = eval_dir / f"{args.task}_metrics.json"

        if metrics_file.exists():
            with open(metrics_file) as f:
                metrics = json.load(f)
            arcade = metrics.get("arcade_f1", {})
            overall_f1 = arcade.get("overall_mean_f1", 0.0)
            per_class = arcade.get("per_class", {})
            num_classes_with_f1 = sum(
                1 for v in per_class.values() if v["mean_f1"] > 0.0
            )

            print(f"\n{'='*60}")
            print(f"  SMOKE TEST RESULTS")
            print(f"{'='*60}")
            print(f"  Overall F1:           {overall_f1:.4f}")
            print(f"  Classes with F1 > 0:  {num_classes_with_f1}/{len(per_class)}")
            for cls_name, cls_data in sorted(per_class.items()):
                status = "OK" if cls_data["mean_f1"] > 0.0 else "--"
                print(f"    {cls_name:>10s}: F1={cls_data['mean_f1']:.4f}  [{status}]")

            if overall_f1 > 0.0 and num_classes_with_f1 > 0:
                print(f"\n  F1 CHECK: PASS (F1 > 0.0 for {num_classes_with_f1} classes)")
            else:
                print(f"\n  F1 CHECK: FAIL (F1 is 0.0 for all classes)")
                failures.append("f1_check")
        else:
            print(f"\n  [WARN] Metrics file not found: {metrics_file}")
            failures.append("metrics_missing")

    finally:
        # Cleanup smoke config
        if smoke_config_path.exists():
            smoke_config_path.unlink()
            print(f"  Cleaned up: {smoke_config_path}")

    _print_summary(failures)
    sys.exit(1 if failures else 0)


def _print_summary(failures: list) -> None:
    """Print final pass/fail summary."""
    print(f"\n{'='*60}")
    if failures:
        print(f"  SMOKE TEST: FAIL ({len(failures)} failures: {', '.join(failures)})")
    else:
        print(f"  SMOKE TEST: PASS")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
