#!/usr/bin/env python3
"""Run improvement experiments sequentially on the syntax task.

Tests architectural and preprocessing changes to improve F1:
  1. Baseline (current best: exp3 config)
  2. CLAHE preprocessing (offline, applied before training)
  3. HEC preprocessing (CLAHE + Canny edge blend)
  4. Resolution 640
  5. Resolution 768
  6. P2 small-object detection head (custom model YAML)
  7. No mosaic + no mixup
  8. Combined: HEC + P2 + no mosaic
  9. Optuna TPE hyperparameter sweep (20 trials)

Usage:
    python run_improvement_experiments.py                    # all experiments
    python run_improvement_experiments.py --experiments 2,3  # specific ones
    python run_improvement_experiments.py --dry-run           # print plan
    python run_improvement_experiments.py --skip-baseline     # skip exp1
"""

import argparse
import copy
import json
import shutil
import subprocess
import sys
import time
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
import yaml

from utils.config_loader import load_config as _load_config_resolved


# ── Configuration ─────────────────────────────────────────────────

BASE_CONFIG = "config.yaml"
RESULTS_FILE = "improvement_experiments.json"

# P2 head model definition — YOLOv8x-seg with added P2 (stride=4) detection head
YOLOV8X_P2_YAML = """
# YOLOv8x-seg with P2 small-object detection head
# Adds stride=4 feature map for detecting small stenoses

nc: 13
scales:
  x: [1.00, 1.00, 512]

backbone:
  - [-1, 1, Conv, [64, 3, 2]]      # 0  P1/2
  - [-1, 1, Conv, [128, 3, 2]]     # 1  P2/4
  - [-1, 3, C2f, [128, True]]      # 2
  - [-1, 1, Conv, [256, 3, 2]]     # 3  P3/8
  - [-1, 6, C2f, [256, True]]      # 4
  - [-1, 1, Conv, [512, 3, 2]]     # 5  P4/16
  - [-1, 6, C2f, [512, True]]      # 6
  - [-1, 1, Conv, [512, 3, 2]]     # 7  P5/32
  - [-1, 3, C2f, [512, True]]      # 8
  - [-1, 1, SPPF, [512, 5]]        # 9

head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]  # 10
  - [[-1, 6], 1, Concat, [1]]                    # 11  cat P4
  - [-1, 3, C2f, [512]]                          # 12

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]  # 13
  - [[-1, 4], 1, Concat, [1]]                    # 14  cat P3
  - [-1, 3, C2f, [256]]                          # 15  P3 output

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]  # 16
  - [[-1, 2], 1, Concat, [1]]                    # 17  cat P2
  - [-1, 3, C2f, [128]]                          # 18  P2 output  <-- NEW

  - [-1, 1, Conv, [128, 3, 2]]                   # 19
  - [[-1, 15], 1, Concat, [1]]                   # 20  cat P3
  - [-1, 3, C2f, [256]]                          # 21

  - [-1, 1, Conv, [256, 3, 2]]                   # 22
  - [[-1, 12], 1, Concat, [1]]                   # 23  cat P4
  - [-1, 3, C2f, [512]]                          # 24

  - [-1, 1, Conv, [512, 3, 2]]                   # 25
  - [[-1, 9], 1, Concat, [1]]                    # 26  cat P5
  - [-1, 3, C2f, [512]]                          # 27

  - [[18, 21, 24, 27], 1, Segment, [nc, 32, 256]]  # 4 heads: P2, P3, P4, P5
"""


# ── Experiment Definitions ────────────────────────────────────────

EXPERIMENTS = OrderedDict([
    ("exp01_baseline", {
        "description": "Baseline (exp3 config: amp=true, lrf=0.01)",
        "config_overrides": {},
        "preprocessing": None,
        "custom_model": None,
    }),
    ("exp02_clahe", {
        "description": "CLAHE preprocessing (clip=2.0, grid=8)",
        "config_overrides": {},
        "preprocessing": "clahe",
        "custom_model": None,
    }),
    ("exp03_hec", {
        "description": "HEC preprocessing (CLAHE + Canny edge blend)",
        "config_overrides": {},
        "preprocessing": "hec",
        "custom_model": None,
    }),
    ("exp04_res640", {
        "description": "Resolution 640 (upscale from 512)",
        "config_overrides": {
            "training": {"image_size": 640},
        },
        "preprocessing": None,
        "custom_model": None,
    }),
    ("exp05_res768", {
        "description": "Resolution 768",
        "config_overrides": {
            "training": {"image_size": 768},
        },
        "preprocessing": None,
        "custom_model": None,
    }),
    ("exp06_p2_head", {
        "description": "P2 small-object detection head (stride=4)",
        "config_overrides": {},
        "preprocessing": None,
        "custom_model": "yolov8x-seg-p2",
    }),
    ("exp07_no_mosaic_mixup", {
        "description": "No mosaic + no mixup (anatomically faithful augmentation)",
        "config_overrides": {
            "augmentation": {"mosaic": 0.0, "mixup": 0.0},
        },
        "preprocessing": None,
        "custom_model": None,
    }),
    ("exp08_combined_best", {
        "description": "Combined: HEC + P2 + no mosaic/mixup",
        "config_overrides": {
            "augmentation": {"mosaic": 0.0, "mixup": 0.0},
        },
        "preprocessing": "hec",
        "custom_model": "yolov8x-seg-p2",
    }),
    ("exp09_optuna_tpe", {
        "description": "Optuna TPE hyperparameter sweep (20 trials)",
        "config_overrides": {},
        "preprocessing": None,
        "custom_model": None,
        "is_hpo": True,
    }),
])


# ── Preprocessing Functions ───────────────────────────────────────

def apply_clahe(img: np.ndarray, clip_limit: float = 2.0,
                grid_size: int = 8) -> np.ndarray:
    """Apply CLAHE to a grayscale or BGR image."""
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    clahe = cv2.createCLAHE(clipLimit=clip_limit,
                            tileGridSize=(grid_size, grid_size))
    enhanced = clahe.apply(gray)
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)


def apply_hec(img: np.ndarray, clip_limit: float = 2.0,
              grid_size: int = 8, canny_low: int = 50,
              canny_high: int = 150, edge_weight: float = 0.3) -> np.ndarray:
    """Apply HEC preprocessing: CLAHE + Canny edge blend.

    Following DCA-YOLOv8 (Duan et al., 2024):
    1. CLAHE enhances vessel-background contrast
    2. Canny extracts vessel edge structure
    3. Weighted blend preserves both intensity and edge info
    """
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    # Step 1: CLAHE
    clahe = cv2.createCLAHE(clipLimit=clip_limit,
                            tileGridSize=(grid_size, grid_size))
    enhanced = clahe.apply(gray)

    # Step 2: Canny edge detection
    edges = cv2.Canny(enhanced, canny_low, canny_high)

    # Step 3: Weighted blend
    blended = cv2.addWeighted(enhanced, 1.0 - edge_weight,
                              edges, edge_weight, 0)
    return cv2.cvtColor(blended, cv2.COLOR_GRAY2BGR)


def preprocess_dataset(src_root: Path, dst_root: Path,
                       method: str) -> None:
    """Apply preprocessing to all images, preserving directory structure.

    Copies label files and symlinks annotation dirs.
    """
    if dst_root.exists():
        print(f"    Preprocessed dir exists, skipping: {dst_root}")
        return

    func = apply_clahe if method == "clahe" else apply_hec
    splits = ["train", "val", "test"]

    for split in splits:
        src_imgs = src_root / split / "images"
        dst_imgs = dst_root / split / "images"
        src_lbls = src_root / split / "labels"
        dst_lbls = dst_root / split / "labels"

        if not src_imgs.exists():
            continue

        dst_imgs.mkdir(parents=True, exist_ok=True)

        # Process images
        for img_path in sorted(src_imgs.glob("*.png")) + \
                         sorted(src_imgs.glob("*.PNG")):
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            processed = func(img)
            cv2.imwrite(str(dst_imgs / img_path.name), processed)

        # Copy labels (unchanged)
        if src_lbls.exists():
            if dst_lbls.exists():
                shutil.rmtree(dst_lbls)
            shutil.copytree(str(src_lbls), str(dst_lbls))

        # Copy annotations if present
        src_ann = src_root / split / "annotations"
        dst_ann = dst_root / split / "annotations"
        if src_ann.exists() and not dst_ann.exists():
            shutil.copytree(str(src_ann), str(dst_ann))

    print(f"    Preprocessed {method}: {dst_root}")


# ── Helper Functions ──────────────────────────────────────────────

def deep_merge(base: dict, overrides: dict) -> dict:
    """Deep-merge overrides into a copy of base."""
    result = copy.deepcopy(base)
    for key, value in overrides.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: str) -> dict:
    """Load config with all paths resolved to absolute."""
    return _load_config_resolved(path)


def write_config(config: dict, path: str) -> None:
    with open(path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def write_p2_model_yaml(output_path: str) -> None:
    """Write the P2 detection head model YAML."""
    with open(output_path, "w") as f:
        f.write(YOLOV8X_P2_YAML)
    print(f"    Wrote P2 model YAML: {output_path}")


def create_data_yaml(preprocessed_root: Path, original_yaml_path: str,
                     output_yaml_path: str) -> None:
    """Create a data.yaml pointing to preprocessed images."""
    with open(original_yaml_path) as f:
        data = yaml.safe_load(f)
    data["path"] = str(preprocessed_root.resolve())
    with open(output_yaml_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


# ── Optuna HPO ────────────────────────────────────────────────────

def run_optuna_hpo(base_config: dict, n_trials: int = 20,
                   exp_name: str = "exp09_optuna_tpe") -> dict:
    """Run Optuna TPE hyperparameter sweep.

    Returns best trial params and score.
    """
    try:
        import optuna
    except ImportError:
        print("    [ERROR] optuna not installed. Run: pip install optuna")
        return {"error": "optuna not installed"}

    from ultralytics import YOLO

    output_dir = Path(base_config["output_dir"]) / exp_name

    def objective(trial):
        # Sample hyperparameters
        lr0 = trial.suggest_float("lr0", 1e-4, 5e-2, log=True)
        lrf = trial.suggest_float("lrf", 1e-3, 1e-1, log=True)
        momentum = trial.suggest_float("momentum", 0.85, 0.97)
        weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)
        mosaic = trial.suggest_float("mosaic", 0.0, 1.0)
        mixup = trial.suggest_float("mixup", 0.0, 0.3)
        degrees = trial.suggest_float("degrees", 0.0, 30.0)
        scale = trial.suggest_float("scale", 0.1, 0.5)
        hsv_v = trial.suggest_float("hsv_v", 0.0, 0.4)
        flipud = trial.suggest_float("flipud", 0.0, 0.5)

        trial_name = f"trial_{trial.number:03d}"

        model = YOLO(base_config["pretrained_weights"])
        results = model.train(
            data=base_config["syntax_data_yaml"],
            epochs=50,  # Shorter for HPO
            batch=base_config["training"]["batch_size"],
            imgsz=base_config["training"]["image_size"],
            device=base_config["training"]["device"],
            workers=base_config["training"]["workers"],
            seed=base_config["training"]["seed"],
            amp=base_config["training"]["amp"],
            cos_lr=base_config["training"]["cos_lr"],
            optimizer=base_config["training"]["optimizer"],
            patience=15,
            lr0=lr0,
            lrf=lrf,
            momentum=momentum,
            weight_decay=weight_decay,
            mosaic=mosaic,
            mixup=mixup,
            degrees=degrees,
            scale=scale,
            hsv_h=0.0,
            hsv_s=0.0,
            hsv_v=hsv_v,
            flipud=flipud,
            fliplr=0.5,
            copy_paste=0.0,
            project=str(output_dir),
            name=trial_name,
            verbose=False,
        )

        # Use mAP50-95 as objective (correlates better with F1 than mAP50)
        metrics = results.results_dict
        score = metrics.get("metrics/mAP50-95(M)", 0.0)
        return score

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        study_name=exp_name,
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_trial
    result = {
        "best_value": best.value,
        "best_params": best.params,
        "n_trials": n_trials,
        "best_trial_number": best.number,
    }

    # Save study results
    results_path = output_dir / "optuna_results.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"    Optuna results: {results_path}")

    return result


# ── Main Experiment Runner ────────────────────────────────────────

def run_single_experiment(exp_name: str, exp_def: dict,
                          base_config: dict) -> dict:
    """Run a single experiment and return results."""

    print(f"\n{'='*70}")
    print(f"  {exp_name}: {exp_def['description']}")
    print(f"{'='*70}")

    start = time.time()
    config = copy.deepcopy(base_config)

    # Apply config overrides
    overrides = exp_def.get("config_overrides", {})
    if overrides:
        config = deep_merge(config, overrides)

    # Set output paths for this experiment (absolute to avoid path issues)
    config["output_dir"] = str(Path("./runs/improvements", exp_name).resolve())

    # Handle HPO separately
    if exp_def.get("is_hpo"):
        hpo_result = run_optuna_hpo(config, n_trials=20, exp_name=exp_name)
        elapsed = time.time() - start
        return {
            "experiment": exp_name,
            "description": exp_def["description"],
            "elapsed_min": round(elapsed / 60, 1),
            "hpo_results": hpo_result,
        }

    # Preprocessing
    preprocessing = exp_def.get("preprocessing")
    dataset_root = Path(config["dataset_root"])
    syntax_src = dataset_root / "syntax_filtered"

    if preprocessing:
        preprocessed_dir = dataset_root / f"syntax_{preprocessing}"
        preprocess_dataset(syntax_src, preprocessed_dir, preprocessing)

        # Create data YAML pointing to preprocessed images
        exp_data_yaml = str(Path(f"./data_syntax_{exp_name}.yaml").resolve())
        create_data_yaml(preprocessed_dir, config["syntax_data_yaml"],
                         exp_data_yaml)
        config["syntax_data_yaml"] = exp_data_yaml

    # Custom model (P2 head)
    custom_model = exp_def.get("custom_model")
    model_weights = config["pretrained_weights"]
    if custom_model == "yolov8x-seg-p2":
        p2_yaml_path = str(Path(f"./models/{custom_model}.yaml").resolve())
        Path(p2_yaml_path).parent.mkdir(exist_ok=True)
        write_p2_model_yaml(p2_yaml_path)
        # For custom architecture, load from YAML instead of .pt
        # but initialize with pretrained backbone weights
        model_weights = p2_yaml_path

    # Write experiment config (use absolute path)
    exp_dir = Path("./runs/improvements", exp_name).resolve()
    exp_config_path = str(exp_dir / "config.yaml")
    exp_dir.mkdir(parents=True, exist_ok=True)
    write_config(config, exp_config_path)

    # Build training command
    # We call train.py directly to reuse its logic
    cmd = [
        sys.executable, "train.py",
        "--task", "syntax",
        "--config", exp_config_path,
    ]

    print(f"    CMD: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    train_ok = result.returncode == 0

    # Run evaluation if training succeeded
    eval_result = {}
    if train_ok:
        eval_cmd = [
            sys.executable, "evaluate.py",
            "--task", "syntax",
            "--method", "arcade",
            "--config", exp_config_path,
        ]
        print(f"    Evaluating...")
        eval_proc = subprocess.run(eval_cmd, capture_output=True, text=True)
        if eval_proc.returncode == 0:
            # Try to parse F1 from output
            for line in eval_proc.stdout.split("\n"):
                if "overall_mean_f1" in line or "mean_f1" in line:
                    eval_result["eval_output"] = line.strip()

    elapsed = time.time() - start

    return {
        "experiment": exp_name,
        "description": exp_def["description"],
        "train_ok": train_ok,
        "elapsed_min": round(elapsed / 60, 1),
        "config_overrides": overrides,
        "preprocessing": preprocessing,
        "custom_model": custom_model,
        "eval_result": eval_result,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run improvement experiments for ARCADE pipeline"
    )
    parser.add_argument("--experiments", type=str, default=None,
                        help="Comma-separated experiment numbers (e.g., '2,3,7')")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print plan without executing")
    parser.add_argument("--skip-baseline", action="store_true",
                        help="Skip experiment 1 (baseline)")
    parser.add_argument("--config", type=str, default=BASE_CONFIG,
                        help="Base config file")
    args = parser.parse_args()

    base_config = load_config(args.config)
    exp_names = list(EXPERIMENTS.keys())

    # Filter experiments
    if args.experiments:
        indices = [int(x) for x in args.experiments.split(",")]
        selected = [exp_names[i - 1] for i in indices if 1 <= i <= len(exp_names)]
    else:
        selected = exp_names

    if args.skip_baseline and "exp01_baseline" in selected:
        selected.remove("exp01_baseline")

    # Print plan
    print("=" * 70)
    print("  IMPROVEMENT EXPERIMENTS")
    print("=" * 70)
    for i, name in enumerate(selected, 1):
        exp = EXPERIMENTS[name]
        print(f"  {i}. {name}: {exp['description']}")
    print()

    if args.dry_run:
        print("  Dry run — no execution.")
        return

    # Execute
    all_results = []
    for name in selected:
        exp_def = EXPERIMENTS[name]
        result = run_single_experiment(name, exp_def, base_config)
        all_results.append(result)

        # Save intermediate results
        with open(RESULTS_FILE, "w") as f:
            json.dump(all_results, f, indent=2)

        if not result.get("train_ok", True) and not result.get("hpo_results"):
            print(f"\n  [ABORT] {name} failed. Stopping.")
            break

    # Print summary
    print(f"\n{'='*70}")
    print("  SUMMARY")
    print(f"{'='*70}")
    for r in all_results:
        status = "OK" if r.get("train_ok", True) else "FAIL"
        print(f"  [{status}] {r['experiment']:<25s} {r['elapsed_min']:>6.1f} min  "
              f"{r.get('eval_result', {}).get('eval_output', '')}")

    print(f"\n  Results saved to: {RESULTS_FILE}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()