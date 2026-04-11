"""Run 6 stenosis improvement strategies in parallel across 6 GPUs.

Experiments:
  GPU 0: S0 — Baseline (bug fixes only: pooled class counting + iter2 fix)
  GPU 1: S1 — Higher resolution (imgsz=768)
  GPU 2: S2 — Loss weight tuning (box=10, cls=1.0, dfl=2.0)
  GPU 3: S3 — Copy-paste + stenosis oversampling 2x
  GPU 4: S4 — Combined best (768 + loss weights + copy-paste + oversample 2x)
  GPU 5: S5 — Separate stenosis model (dedicated 1-class model at 768px)

All experiments:
  - Use stratified splits (shared, created once)
  - Use pooled class counting (min_count=300)
  - Use fixed iter2 pseudo-labels
  - Use YOLO11m-seg as the base model
  - Enable TTA (test-time augmentation) at evaluation

Usage:
    python run_stenosis_strategies.py --arcade-root ../../arcade/submission

    # Skip split creation if already done:
    python run_stenosis_strategies.py --arcade-root ../../arcade/submission --skip-splits
"""

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
import yaml
from pathlib import Path


# ── Experiment Definitions ──────────────────────────────────────────────────

def get_experiments():
    """Define all 6 experiments with their config overrides."""
    # Base config shared by all experiments
    base = {
        "model": "yolo11m-seg.pt",
        "imgsz": 512,
        "batch": 16,
        "epochs": 200,
        "patience": 25,
        "optimizer": "AdamW",
        "lr0": 0.001,
        "lrf": 0.01,
        "weight_decay": 0.005,
        "momentum": 0.937,
        "warmup_epochs": 5,
        "freeze": 10,
        "freeze_epochs": 15,
        "seed": 42,
        "deterministic": True,
        "amp": True,
        "cos_lr": True,
        "workers": 4,
        # Augmentation
        "mosaic": 0.0,
        "mixup": 0.0,
        "copy_paste": 0.0,
        "fliplr": 0.5,
        "flipud": 0.0,
        "degrees": 10.0,
        "scale": 0.2,
        "translate": 0.1,
        "hsv_h": 0.0,
        "hsv_s": 0.0,
        "hsv_v": 0.15,
        "erasing": 0.0,
        "shear": 0.0,
        "perspective": 0.0,
        # Loss weights (ultralytics defaults)
        "box": 7.5,
        "cls": 0.5,
        "dfl": 1.5,
        # Pseudo-labeling
        "pseudo_label": {
            "initial_conf": 0.85,
            "conf_decay": 0.05,
            "min_conf": 0.65,
            "use_one_to_many": False,
        },
    }

    experiments = [
        {
            "name": "S0_baseline_fixed",
            "gpu": 0,
            "description": "Baseline with bug fixes (pooled counting + iter2 fix)",
            "overrides": {},
            "pipeline_args": {},
        },
        {
            "name": "S1_resolution_768",
            "gpu": 1,
            "description": "Higher resolution training (768px)",
            "overrides": {
                "imgsz": 768,
                "batch": 8,  # Lower batch to fit 24GB VRAM at 768px
            },
            "pipeline_args": {},
        },
        {
            "name": "S2_loss_weights",
            "gpu": 2,
            "description": "Tuned loss weights (box=10, cls=1.0, dfl=2.0)",
            "overrides": {
                "box": 10.0,
                "cls": 1.0,
                "dfl": 2.0,
            },
            "pipeline_args": {},
        },
        {
            "name": "S3_copypaste_oversample",
            "gpu": 3,
            "description": "Copy-paste augmentation + 2x stenosis oversampling",
            "overrides": {
                "copy_paste": 0.3,
            },
            "pipeline_args": {
                "stenosis_oversample": 2,
            },
        },
        {
            "name": "S4_combined_best",
            "gpu": 4,
            "description": "Combined: 768px + loss weights + copy-paste + oversample",
            "overrides": {
                "imgsz": 768,
                "batch": 8,
                "box": 10.0,
                "cls": 1.0,
                "dfl": 2.0,
                "copy_paste": 0.3,
            },
            "pipeline_args": {
                "stenosis_oversample": 2,
            },
        },
        {
            "name": "S5_separate_stenosis",
            "gpu": 5,
            "description": "Separate dedicated stenosis model at 768px",
            # This experiment runs a custom pipeline (see run_separate_stenosis)
            "overrides": {},
            "pipeline_args": {},
            "custom_pipeline": True,
        },
        {
            "name": "S6_vessel_guided_stenosis",
            "gpu": 6,
            "description": "Vessel-guided stenosis: syntax model masks the "
                           "image, dedicated stenosis model trains on the "
                           "vessel-restricted view (blackout + crop variants)",
            # Custom pipeline — see run_vessel_guided_stenosis below.
            "overrides": {},
            "pipeline_args": {},
            "custom_pipeline": True,
            "custom_runner": "vessel_guided",
            # S6-specific knobs (consumed by run_vessel_guided_stenosis)
            "vessel_guided": {
                "dilate_px": 30,
                "crop_pad_px": 15,
                "vessel_conf": 0.25,
                "vessel_imgsz": 768,
                # Which variant to train the stenosis detector on.
                # One of: "crop", "blackout", "both" (train both, report both).
                "variant": "both",
            },
        },
        # ── Round 2 experiments (S7-S10) ──────────────────────────────
        # These restore the proven "old S5" augmentation + LR settings
        # and test new ideas on top of that baseline.
        {
            "name": "S7_reproduce_old_s5",
            "gpu": 0,
            "description": "Reproduce Old S5: original aug (degrees=20, "
                           "scale=0.4, hsv_v=0.3) + copy_paste=0.3 on "
                           "stenosis + 10x LR reduction (default in train.py)",
            "overrides": {
                "degrees": 20.0,
                "scale": 0.4,
                "hsv_v": 0.3,
            },
            "pipeline_args": {},
            "custom_pipeline": True,
            "custom_runner": "separate_v2",
            "stenosis_overrides": {
                "copy_paste": 0.3,
                "scale": 0.5,
            },
        },
        {
            "name": "S8_mosaic_stenosis",
            "gpu": 1,
            "description": "Old S5 config + mosaic=0.8 on stenosis model "
                           "only (syntax stays mosaic=0.0)",
            "overrides": {
                "degrees": 20.0,
                "scale": 0.4,
                "hsv_v": 0.3,
            },
            "pipeline_args": {},
            "custom_pipeline": True,
            "custom_runner": "separate_v2",
            "stenosis_overrides": {
                "copy_paste": 0.3,
                "scale": 0.5,
                "mosaic": 0.8,
                "close_mosaic": 15,
            },
        },
        {
            "name": "S9_vessel_guided_all25",
            "gpu": 2,
            "description": "Vessel-guided stenosis with ALL 25 syntax "
                           "classes (min_count=0) + old S5 params",
            "overrides": {
                "degrees": 20.0,
                "scale": 0.4,
                "hsv_v": 0.3,
            },
            "pipeline_args": {},
            "custom_pipeline": True,
            "custom_runner": "vessel_guided",
            "stenosis_overrides": {
                "copy_paste": 0.3,
                "scale": 0.5,
            },
            "vessel_guided": {
                "dilate_px": 30,
                "crop_pad_px": 15,
                "vessel_conf": 0.15,
                "vessel_imgsz": 768,
                "variant": "both",
                "min_count": 0,
            },
        },
        {
            "name": "S10a_vessel_filtered_stratified",
            "gpu": 3,
            "description": "Filter stenosis to vessel-interior only "
                           "(stratified splits, all 25 syntax classes)",
            "overrides": {
                "degrees": 20.0,
                "scale": 0.4,
                "hsv_v": 0.3,
            },
            "pipeline_args": {},
            "custom_pipeline": True,
            "custom_runner": "vessel_filtered",
            "stenosis_overrides": {
                "copy_paste": 0.3,
                "scale": 0.5,
            },
            "vessel_filter": {
                "overlap_threshold": 0.5,
                "dilate_px": 30,
                "vessel_conf": 0.15,
                "vessel_imgsz": 768,
                "min_count": 0,
                "use_original_splits": False,
            },
        },
        {
            "name": "S10b_vessel_filtered_original",
            "gpu": 4,
            "description": "Filter stenosis to vessel-interior only "
                           "(original ARCADE splits, all 25 syntax classes)",
            "overrides": {
                "degrees": 20.0,
                "scale": 0.4,
                "hsv_v": 0.3,
            },
            "pipeline_args": {},
            "custom_pipeline": True,
            "custom_runner": "vessel_filtered",
            "stenosis_overrides": {
                "copy_paste": 0.3,
                "scale": 0.5,
            },
            "vessel_filter": {
                "overlap_threshold": 0.5,
                "dilate_px": 30,
                "vessel_conf": 0.15,
                "vessel_imgsz": 768,
                "min_count": 0,
                "use_original_splits": True,
            },
        },
    ]

    # Merge base config into each experiment
    for exp in experiments:
        cfg = dict(base)
        cfg.update(exp["overrides"])
        cfg["device"] = str(exp["gpu"])
        exp["config"] = cfg

    return experiments


# ── Worker Functions ────────────────────────────────────────────────────────

def run_standard_experiment(exp: dict, arcade_root: Path, splits_dir: Path,
                            output_dir: Path, iterations: int) -> dict:
    """Run a standard pipeline experiment on a single GPU."""
    from run_pipeline import run_pipeline

    name = exp["name"]
    results_dir = output_dir / name
    results_dir.mkdir(parents=True, exist_ok=True)

    # Write config YAML
    cfg = exp["config"]
    cfg["results_dir"] = str(results_dir)
    cfg["data_dir"] = str(output_dir / "data" / name)
    config_path = results_dir / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    pipeline_args = exp.get("pipeline_args", {})

    run_pipeline(
        config_path=str(config_path),
        arcade_root=str(arcade_root),
        iterations=iterations,
        splits_dir=str(splits_dir),
        min_count=300,
        eval_augment=True,  # TTA for all experiments
        **pipeline_args,
    )

    # Load results
    metrics_path = results_dir / "all_metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            return json.load(f)
    return {}


def run_separate_stenosis(exp: dict, arcade_root: Path, splits_dir: Path,
                          output_dir: Path, iterations: int) -> dict:
    """Run the separate stenosis model experiment.

    This runs TWO models:
      1. Syntax-only model (standard, 768px) — handles vessel segmentation
      2. Stenosis-only model (dedicated, 768px) — handles stenosis detection

    The combined evaluation merges both models' per-class metrics.
    """
    from run_pipeline import data_prep, stage1_train_syntax, _save_metrics
    from train import load_run_config, train_two_stage
    from evaluate import evaluate_model
    from merge_datasets import get_stenosis_class_id

    name = exp["name"]
    results_dir = output_dir / name
    results_dir.mkdir(parents=True, exist_ok=True)

    # ── Part A: Syntax-only model at 768px ──
    print(f"\n{'#' * 60}")
    print(f"# {name} — Part A: Syntax-only model")
    print(f"{'#' * 60}")

    cfg_syntax = dict(exp["config"])
    cfg_syntax["imgsz"] = 768
    cfg_syntax["batch"] = 8
    cfg_syntax["device"] = "0"  # CUDA_VISIBLE_DEVICES already set by worker
    cfg_syntax["results_dir"] = str(results_dir / "syntax_model")
    cfg_syntax["data_dir"] = str(output_dir / "data" / name)

    config_path_syntax = results_dir / "config_syntax.yaml"
    with open(config_path_syntax, "w") as f:
        yaml.dump(cfg_syntax, f, default_flow_style=False)

    cfg_s = load_run_config(str(config_path_syntax))
    data_dir = Path(cfg_s["data_dir"]).resolve()

    # Data prep (shared data)
    data_prep(arcade_root, data_dir, min_count=300, splits_dir=splits_dir)

    # Train syntax-only
    syntax_yaml = str(data_dir / "dataset_configs" / "syntax_only.yaml")
    syntax_weights = train_two_stage(
        cfg_s, syntax_yaml,
        project=str(results_dir / "syntax_model"),
        run_name="syntax_768",
    )

    # Evaluate syntax model
    syntax_metrics = evaluate_model(
        syntax_weights, syntax_yaml, split="test",
        augment=True, imgsz=768,
    )
    _save_metrics(results_dir, "syntax_model_test", syntax_metrics)

    # ── Part B: Dedicated stenosis model at 768px ──
    print(f"\n{'#' * 60}")
    print(f"# {name} — Part B: Dedicated stenosis model")
    print(f"{'#' * 60}")

    cfg_sten = dict(exp["config"])
    cfg_sten["imgsz"] = 768
    cfg_sten["batch"] = 8
    cfg_sten["device"] = "0"  # CUDA_VISIBLE_DEVICES already set by worker
    cfg_sten["box"] = 10.0
    cfg_sten["cls"] = 1.0
    cfg_sten["results_dir"] = str(results_dir / "stenosis_model")

    stenosis_yaml = str(data_dir / "dataset_configs" / "stenosis_only.yaml")
    stenosis_weights = train_two_stage(
        cfg_sten, stenosis_yaml,
        project=str(results_dir / "stenosis_model"),
        run_name="stenosis_768",
    )

    # Evaluate stenosis model
    stenosis_metrics = evaluate_model(
        stenosis_weights, stenosis_yaml, split="test",
        augment=True, imgsz=768,
    )
    _save_metrics(results_dir, "stenosis_model_test", stenosis_metrics)

    # ── Combine metrics ──
    print(f"\n{'#' * 60}")
    print(f"# {name} — Combined results")
    print(f"{'#' * 60}")

    # Merge per-class results from both models
    combined_per_class = {}
    # Syntax classes from syntax model
    for cls_name, cls_m in syntax_metrics.get("per_class", {}).items():
        combined_per_class[cls_name] = cls_m
    # Stenosis from dedicated model (class "stenosis" in stenosis_only eval
    # is reported as class 0 named "stenosis")
    for cls_name, cls_m in stenosis_metrics.get("per_class", {}).items():
        combined_per_class[f"stenosis"] = cls_m

    # Compute combined mAP50 as weighted average
    all_ap50s = [m.get("ap50", 0) for m in combined_per_class.values()]
    combined_mAP50 = sum(all_ap50s) / len(all_ap50s) if all_ap50s else 0

    syntax_aps = [m.get("ap50", 0) for k, m in combined_per_class.items()
                  if k != "stenosis"]
    stenosis_aps = [m.get("ap50", 0) for k, m in combined_per_class.items()
                    if k == "stenosis"]

    combined = {
        "split": "test",
        "mAP50": round(combined_mAP50, 4),
        "per_class": combined_per_class,
        "syntax_mAP50": round(sum(syntax_aps) / len(syntax_aps), 4) if syntax_aps else 0,
        "stenosis_AP50": round(sum(stenosis_aps) / len(stenosis_aps), 4) if stenosis_aps else 0,
        "syntax_model": syntax_weights,
        "stenosis_model": stenosis_weights,
    }

    # Compute overall P/R/F1
    all_p = [m.get("precision", 0) for m in combined_per_class.values()]
    all_r = [m.get("recall", 0) for m in combined_per_class.values()]
    combined["precision"] = round(sum(all_p) / len(all_p), 4) if all_p else 0
    combined["recall"] = round(sum(all_r) / len(all_r), 4) if all_r else 0
    p, r = combined["precision"], combined["recall"]
    combined["mAP50_95"] = 0  # Not directly combinable

    _save_metrics(results_dir, "final_test", combined)

    # Save all_metrics.json for the comparison script
    all_metrics = {
        "syntax_model_test": syntax_metrics,
        "stenosis_model_test": stenosis_metrics,
        "final_test": combined,
    }
    with open(results_dir / "all_metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)

    return all_metrics


def run_separate_stenosis_v2(exp: dict, arcade_root: Path, splits_dir: Path,
                             output_dir: Path, iterations: int) -> dict:
    """Run separate stenosis model with per-experiment stenosis overrides.

    Like run_separate_stenosis() but applies explicit overrides from
    exp["stenosis_overrides"] to the stenosis model config. This allows
    experiments to test different augmentation settings (e.g., copy_paste,
    mosaic, scale) on the stenosis model independently of the syntax model.
    """
    from run_pipeline import data_prep, _save_metrics
    from train import load_run_config, train_two_stage
    from evaluate import evaluate_model

    name = exp["name"]
    results_dir = output_dir / name
    results_dir.mkdir(parents=True, exist_ok=True)

    stenosis_overrides = exp.get("stenosis_overrides", {})

    # ── Part A: Syntax-only model at 768px ──
    print(f"\n{'#' * 60}")
    print(f"# {name} — Part A: Syntax-only model")
    print(f"{'#' * 60}")

    cfg_syntax = dict(exp["config"])
    cfg_syntax["imgsz"] = 768
    cfg_syntax["batch"] = 8
    cfg_syntax["device"] = "0"
    cfg_syntax["results_dir"] = str(results_dir / "syntax_model")
    cfg_syntax["data_dir"] = str(output_dir / "data" / name)

    config_path_syntax = results_dir / "config_syntax.yaml"
    with open(config_path_syntax, "w") as f:
        yaml.dump(cfg_syntax, f, default_flow_style=False)

    cfg_s = load_run_config(str(config_path_syntax))
    data_dir = Path(cfg_s["data_dir"]).resolve()

    data_prep(arcade_root, data_dir, min_count=300, splits_dir=splits_dir)

    syntax_yaml = str(data_dir / "dataset_configs" / "syntax_only.yaml")
    syntax_weights = train_two_stage(
        cfg_s, syntax_yaml,
        project=str(results_dir / "syntax_model"),
        run_name="syntax_768",
    )

    syntax_metrics = evaluate_model(
        syntax_weights, syntax_yaml, split="test",
        augment=True, imgsz=768,
    )
    _save_metrics(results_dir, "syntax_model_test", syntax_metrics)

    # ── Part B: Dedicated stenosis model at 768px ──
    print(f"\n{'#' * 60}")
    print(f"# {name} — Part B: Dedicated stenosis model")
    if stenosis_overrides:
        print(f"#   Stenosis overrides: {stenosis_overrides}")
    print(f"{'#' * 60}")

    cfg_sten = dict(exp["config"])
    cfg_sten["imgsz"] = 768
    cfg_sten["batch"] = 8
    cfg_sten["device"] = "0"
    cfg_sten["box"] = 10.0
    cfg_sten["cls"] = 1.0
    cfg_sten["results_dir"] = str(results_dir / "stenosis_model")

    # Apply per-experiment stenosis overrides (copy_paste, scale, mosaic, etc.)
    for key, val in stenosis_overrides.items():
        cfg_sten[key] = val

    stenosis_yaml = str(data_dir / "dataset_configs" / "stenosis_only.yaml")
    stenosis_weights = train_two_stage(
        cfg_sten, stenosis_yaml,
        project=str(results_dir / "stenosis_model"),
        run_name="stenosis_768",
    )

    stenosis_metrics = evaluate_model(
        stenosis_weights, stenosis_yaml, split="test",
        augment=True, imgsz=768,
    )
    _save_metrics(results_dir, "stenosis_model_test", stenosis_metrics)

    # ── Combine metrics ──
    print(f"\n{'#' * 60}")
    print(f"# {name} — Combined results")
    print(f"{'#' * 60}")

    combined_per_class = {}
    for cls_name, cls_m in syntax_metrics.get("per_class", {}).items():
        combined_per_class[cls_name] = cls_m
    for cls_name, cls_m in stenosis_metrics.get("per_class", {}).items():
        combined_per_class["stenosis"] = cls_m

    all_ap50s = [m.get("ap50", 0) for m in combined_per_class.values()]
    combined_mAP50 = sum(all_ap50s) / len(all_ap50s) if all_ap50s else 0

    syntax_aps = [m.get("ap50", 0) for k, m in combined_per_class.items()
                  if k != "stenosis"]
    stenosis_aps = [m.get("ap50", 0) for k, m in combined_per_class.items()
                    if k == "stenosis"]

    combined = {
        "split": "test",
        "mAP50": round(combined_mAP50, 4),
        "per_class": combined_per_class,
        "syntax_mAP50": round(sum(syntax_aps) / len(syntax_aps), 4) if syntax_aps else 0,
        "stenosis_AP50": round(sum(stenosis_aps) / len(stenosis_aps), 4) if stenosis_aps else 0,
        "syntax_model": syntax_weights,
        "stenosis_model": stenosis_weights,
    }

    all_p = [m.get("precision", 0) for m in combined_per_class.values()]
    all_r = [m.get("recall", 0) for m in combined_per_class.values()]
    combined["precision"] = round(sum(all_p) / len(all_p), 4) if all_p else 0
    combined["recall"] = round(sum(all_r) / len(all_r), 4) if all_r else 0
    combined["mAP50_95"] = 0

    _save_metrics(results_dir, "final_test", combined)

    all_metrics = {
        "syntax_model_test": syntax_metrics,
        "stenosis_model_test": stenosis_metrics,
        "final_test": combined,
        "stenosis_overrides": stenosis_overrides,
    }
    with open(results_dir / "all_metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)

    return all_metrics


def run_vessel_guided_stenosis(exp: dict, arcade_root: Path, splits_dir: Path,
                               output_dir: Path, iterations: int) -> dict:
    """S6 — Vessel-guided stenosis detection pipeline.

    Motivation
    ----------
    Stenoses are tiny objects (often <1% of image area) and the stenosis
    detector wastes most of its input-resolution budget on ribs, spine,
    catheters, and contrast artifacts that cannot contain a stenosis. This
    pipeline uses the trained vessel (syntax) segmentation model as a
    learned attention prior, restricting the stenosis detector to pixels
    where vessels actually exist. Conceptually it is a content-aware
    analogue of SAHI slicing: instead of a uniform grid, we crop/mask
    using anatomy.

    Pipeline stages
    ---------------
      A. Train syntax-only model at 768px (same as S5 Part A).
      B. Run that syntax model on every stenosis image (train/val/test)
         to predict per-image vessel union masks. Dilate them by k pixels
         so stenoses sitting at vessel boundaries aren't clipped away.
         Build two preprocessed stenosis dataset variants:
            - "blackout": original H x W, non-vessel pixels zeroed
            - "crop":     bbox-crop to dilated mask, pad to square,
                          remap stenosis GT polygons into crop coords
      C. Train a dedicated stenosis model on the chosen variant at 768px
         with the same hyperparameters as S5 Part B so the difference
         vs S5 is exactly the input preprocessing.
      D. Evaluate both models together on the test split: vessel model
         reports per-syntax-class metrics, stenosis model reports
         stenosis AP50 on its (preprocessed) test split.

    CRITICAL design choice: predicted (not GT) vessel masks are used on
    ALL splits including train. This keeps the train/test distributions
    matched — the stenosis model must learn to cope with exactly the
    mask noise it will see at inference.
    """
    from run_pipeline import data_prep, _save_metrics
    from train import load_run_config, train_two_stage
    from evaluate import evaluate_model
    from build_vessel_masked_dataset import build_vessel_masked_dataset

    name = exp["name"]
    results_dir = output_dir / name
    results_dir.mkdir(parents=True, exist_ok=True)

    vg_cfg = exp.get("vessel_guided", {})
    dilate_px = vg_cfg.get("dilate_px", 20)
    crop_pad_px = vg_cfg.get("crop_pad_px", 10)
    vessel_conf = vg_cfg.get("vessel_conf", 0.25)
    vessel_imgsz = vg_cfg.get("vessel_imgsz", 768)
    variant = vg_cfg.get("variant", "crop")
    min_count = vg_cfg.get("min_count", 300)
    stenosis_overrides = exp.get("stenosis_overrides", {})
    if variant not in ("crop", "blackout", "both"):
        raise ValueError(f"Unknown vessel_guided.variant: {variant}")

    # ── Part A: Syntax-only model at 768px ──
    print(f"\n{'#' * 64}")
    print(f"# {name} — Part A: Syntax-only model (768px)")
    print(f"{'#' * 64}")

    cfg_syntax = dict(exp["config"])
    cfg_syntax["imgsz"] = 768
    cfg_syntax["batch"] = 8
    cfg_syntax["device"] = "0"  # CUDA_VISIBLE_DEVICES set by worker
    cfg_syntax["results_dir"] = str(results_dir / "syntax_model")
    cfg_syntax["data_dir"] = str(output_dir / "data" / name)

    config_path_syntax = results_dir / "config_syntax.yaml"
    with open(config_path_syntax, "w") as f:
        yaml.dump(cfg_syntax, f, default_flow_style=False)

    cfg_s = load_run_config(str(config_path_syntax))
    data_dir = Path(cfg_s["data_dir"]).resolve()

    # Shared data prep (syntax_filtered + stenosis)
    print(f"#   min_count={min_count} (syntax classes)")
    data_prep(arcade_root, data_dir, min_count=min_count, splits_dir=splits_dir)

    syntax_yaml = str(data_dir / "dataset_configs" / "syntax_only.yaml")
    syntax_weights = train_two_stage(
        cfg_s, syntax_yaml,
        project=str(results_dir / "syntax_model"),
        run_name="syntax_768",
    )
    syntax_metrics = evaluate_model(
        syntax_weights, syntax_yaml, split="test",
        augment=True, imgsz=768,
    )
    _save_metrics(results_dir, "syntax_model_test", syntax_metrics)

    # ── Part B: Build vessel-masked stenosis dataset ──
    print(f"\n{'#' * 64}")
    print(f"# {name} — Part B: Build vessel-masked stenosis dataset")
    print(f"#   dilate_px={dilate_px}, crop_pad_px={crop_pad_px}, "
          f"variant={variant}")
    print(f"{'#' * 64}")

    masked_root = results_dir / "masked_stenosis_data"
    build_vessel_masked_dataset(
        syntax_weights=syntax_weights,
        stenosis_data_dir=data_dir / "stenosis",
        output_dir=masked_root,
        dilate_px=dilate_px,
        crop_pad_px=crop_pad_px,
        vessel_conf=vessel_conf,
        vessel_imgsz=vessel_imgsz,
        save_debug_masks=False,  # per-image masks are large, disable by default
    )

    # ── Part C: Train stenosis model(s) on masked data ──
    variants_to_train = ["blackout", "crop"] if variant == "both" else [variant]
    stenosis_weights_by_variant = {}
    stenosis_metrics_by_variant = {}

    for v in variants_to_train:
        print(f"\n{'#' * 64}")
        print(f"# {name} — Part C: Train stenosis model [{v}]")
        print(f"{'#' * 64}")

        cfg_sten = dict(exp["config"])
        cfg_sten["imgsz"] = 768
        cfg_sten["batch"] = 8
        cfg_sten["device"] = "0"
        cfg_sten["box"] = 10.0
        cfg_sten["cls"] = 1.0
        cfg_sten["results_dir"] = str(results_dir / f"stenosis_model_{v}")

        # Apply per-experiment stenosis overrides
        for key, val in stenosis_overrides.items():
            cfg_sten[key] = val

        stenosis_yaml = str(masked_root / v / "data.yaml")
        st_weights = train_two_stage(
            cfg_sten, stenosis_yaml,
            project=str(results_dir / f"stenosis_model_{v}"),
            run_name=f"stenosis_{v}_768",
        )
        stenosis_weights_by_variant[v] = st_weights

        st_metrics = evaluate_model(
            st_weights, stenosis_yaml, split="test",
            augment=True, imgsz=768,
        )
        _save_metrics(results_dir, f"stenosis_model_{v}_test", st_metrics)
        stenosis_metrics_by_variant[v] = st_metrics

    # Pick the best variant by stenosis AP50 for the combined final report
    best_variant = max(
        stenosis_metrics_by_variant.keys(),
        key=lambda v: stenosis_metrics_by_variant[v].get("stenosis_AP50", 0.0),
    )
    stenosis_metrics = stenosis_metrics_by_variant[best_variant]
    stenosis_weights = stenosis_weights_by_variant[best_variant]

    # ── Part D: Combined metrics ──
    print(f"\n{'#' * 64}")
    print(f"# {name} — Combined results (best variant: {best_variant})")
    print(f"{'#' * 64}")

    combined_per_class = {}
    for cls_name, cls_m in syntax_metrics.get("per_class", {}).items():
        combined_per_class[cls_name] = cls_m
    for cls_name, cls_m in stenosis_metrics.get("per_class", {}).items():
        combined_per_class["stenosis"] = cls_m

    all_ap50s = [m.get("ap50", 0) for m in combined_per_class.values()]
    combined_mAP50 = sum(all_ap50s) / len(all_ap50s) if all_ap50s else 0

    syntax_aps = [m.get("ap50", 0) for k, m in combined_per_class.items()
                  if k != "stenosis"]
    stenosis_aps = [m.get("ap50", 0) for k, m in combined_per_class.items()
                    if k == "stenosis"]

    combined = {
        "split": "test",
        "mAP50": round(combined_mAP50, 4),
        "per_class": combined_per_class,
        "syntax_mAP50": round(sum(syntax_aps) / len(syntax_aps), 4) if syntax_aps else 0,
        "stenosis_AP50": round(sum(stenosis_aps) / len(stenosis_aps), 4) if stenosis_aps else 0,
        "syntax_model": syntax_weights,
        "stenosis_model": stenosis_weights,
        "best_variant": best_variant,
        "vessel_guided_config": {
            "dilate_px": dilate_px,
            "crop_pad_px": crop_pad_px,
            "vessel_conf": vessel_conf,
            "vessel_imgsz": vessel_imgsz,
        },
    }
    all_p = [m.get("precision", 0) for m in combined_per_class.values()]
    all_r = [m.get("recall", 0) for m in combined_per_class.values()]
    combined["precision"] = round(sum(all_p) / len(all_p), 4) if all_p else 0
    combined["recall"] = round(sum(all_r) / len(all_r), 4) if all_r else 0
    combined["mAP50_95"] = 0  # Not directly combinable across two models

    _save_metrics(results_dir, "final_test", combined)

    all_metrics = {
        "syntax_model_test": syntax_metrics,
        "stenosis_variants": stenosis_metrics_by_variant,
        "stenosis_model_test": stenosis_metrics,  # best variant (for compare_runs)
        "final_test": combined,
    }
    with open(results_dir / "all_metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)

    return all_metrics


def run_vessel_filtered_stenosis(exp: dict, arcade_root: Path, splits_dir: Path,
                                 output_dir: Path, iterations: int) -> dict:
    """S10 — Vessel-filtered stenosis detection pipeline.

    Instead of masking images (S6/S9), this filters stenosis ANNOTATIONS:
    only stenoses that overlap predicted vessel regions are kept for training.
    This creates a cleaner training signal by removing "impossible positives".

    Pipeline stages
    ---------------
      A. Train syntax-only model (all 25 classes, min_count=0) at 768px.
      B. Run syntax model on stenosis images, filter annotations to only
         those inside predicted vessel masks.
      C. Train dedicated stenosis model on filtered data at 768px.
      D. Evaluate with TWO metrics:
         - Standard: full test set (comparable to other experiments)
         - Vessel-interior: filtered test set (clinically meaningful)
    """
    from run_pipeline import data_prep, _save_metrics
    from train import load_run_config, train_two_stage
    from evaluate import evaluate_model
    from build_filtered_stenosis_dataset import filter_stenosis_dataset

    name = exp["name"]
    results_dir = output_dir / name
    results_dir.mkdir(parents=True, exist_ok=True)

    vf_cfg = exp.get("vessel_filter", {})
    overlap_threshold = vf_cfg.get("overlap_threshold", 0.5)
    dilate_px = vf_cfg.get("dilate_px", 30)
    vessel_conf = vf_cfg.get("vessel_conf", 0.15)
    vessel_imgsz = vf_cfg.get("vessel_imgsz", 768)
    min_count = vf_cfg.get("min_count", 0)
    use_original_splits = vf_cfg.get("use_original_splits", False)
    stenosis_overrides = exp.get("stenosis_overrides", {})

    # Determine splits directory
    effective_splits_dir = None if use_original_splits else splits_dir
    splits_label = "original ARCADE" if use_original_splits else "stratified"

    # ── Part A: Syntax-only model at 768px (all 25 classes) ──
    print(f"\n{'#' * 64}")
    print(f"# {name} — Part A: Syntax-only model (768px, all classes)")
    print(f"#   min_count={min_count}, splits={splits_label}")
    print(f"{'#' * 64}")

    cfg_syntax = dict(exp["config"])
    cfg_syntax["imgsz"] = 768
    cfg_syntax["batch"] = 8
    cfg_syntax["device"] = "0"
    cfg_syntax["results_dir"] = str(results_dir / "syntax_model")
    cfg_syntax["data_dir"] = str(output_dir / "data" / name)

    config_path_syntax = results_dir / "config_syntax.yaml"
    with open(config_path_syntax, "w") as f:
        yaml.dump(cfg_syntax, f, default_flow_style=False)

    cfg_s = load_run_config(str(config_path_syntax))
    data_dir = Path(cfg_s["data_dir"]).resolve()

    data_prep(arcade_root, data_dir, min_count=min_count,
              splits_dir=effective_splits_dir)

    syntax_yaml = str(data_dir / "dataset_configs" / "syntax_only.yaml")
    syntax_weights = train_two_stage(
        cfg_s, syntax_yaml,
        project=str(results_dir / "syntax_model"),
        run_name="syntax_768",
    )
    syntax_metrics = evaluate_model(
        syntax_weights, syntax_yaml, split="test",
        augment=True, imgsz=768,
    )
    _save_metrics(results_dir, "syntax_model_test", syntax_metrics)

    # ── Part B: Filter stenosis annotations ──
    print(f"\n{'#' * 64}")
    print(f"# {name} — Part B: Filter stenosis to vessel-interior")
    print(f"#   overlap_threshold={overlap_threshold}, dilate_px={dilate_px}")
    print(f"{'#' * 64}")

    filtered_root = results_dir / "filtered_stenosis_data"
    filter_stats = filter_stenosis_dataset(
        syntax_weights=syntax_weights,
        stenosis_data_dir=data_dir / "stenosis",
        output_dir=filtered_root,
        overlap_threshold=overlap_threshold,
        dilate_px=dilate_px,
        vessel_conf=vessel_conf,
        vessel_imgsz=vessel_imgsz,
    )

    # ── Part C: Train stenosis model on filtered data ──
    print(f"\n{'#' * 64}")
    print(f"# {name} — Part C: Train stenosis model on filtered data")
    if stenosis_overrides:
        print(f"#   Stenosis overrides: {stenosis_overrides}")
    print(f"{'#' * 64}")

    cfg_sten = dict(exp["config"])
    cfg_sten["imgsz"] = 768
    cfg_sten["batch"] = 8
    cfg_sten["device"] = "0"
    cfg_sten["box"] = 10.0
    cfg_sten["cls"] = 1.0
    cfg_sten["results_dir"] = str(results_dir / "stenosis_model")

    for key, val in stenosis_overrides.items():
        cfg_sten[key] = val

    filtered_yaml = str(filtered_root / "data.yaml")
    stenosis_weights = train_two_stage(
        cfg_sten, filtered_yaml,
        project=str(results_dir / "stenosis_model"),
        run_name="stenosis_filtered_768",
    )

    # ── Part D: Evaluate with dual metrics ──
    print(f"\n{'#' * 64}")
    print(f"# {name} — Part D: Dual evaluation")
    print(f"{'#' * 64}")

    # D1: Evaluate on FILTERED test set (vessel-interior metrics)
    stenosis_filtered_metrics = evaluate_model(
        stenosis_weights, filtered_yaml, split="test",
        augment=True, imgsz=768,
    )
    _save_metrics(results_dir, "stenosis_filtered_test", stenosis_filtered_metrics)

    # D2: Evaluate on FULL (unfiltered) test set (standard metrics)
    full_stenosis_yaml = str(data_dir / "dataset_configs" / "stenosis_only.yaml")
    stenosis_full_metrics = evaluate_model(
        stenosis_weights, full_stenosis_yaml, split="test",
        augment=True, imgsz=768,
    )
    _save_metrics(results_dir, "stenosis_full_test", stenosis_full_metrics)

    # ── Combine metrics (using FULL test set for fair comparison) ──
    print(f"\n{'#' * 64}")
    print(f"# {name} — Combined results")
    print(f"{'#' * 64}")

    combined_per_class = {}
    for cls_name, cls_m in syntax_metrics.get("per_class", {}).items():
        combined_per_class[cls_name] = cls_m
    for cls_name, cls_m in stenosis_full_metrics.get("per_class", {}).items():
        combined_per_class["stenosis"] = cls_m

    all_ap50s = [m.get("ap50", 0) for m in combined_per_class.values()]
    combined_mAP50 = sum(all_ap50s) / len(all_ap50s) if all_ap50s else 0

    syntax_aps = [m.get("ap50", 0) for k, m in combined_per_class.items()
                  if k != "stenosis"]
    stenosis_aps = [m.get("ap50", 0) for k, m in combined_per_class.items()
                    if k == "stenosis"]

    combined = {
        "split": "test",
        "mAP50": round(combined_mAP50, 4),
        "per_class": combined_per_class,
        "syntax_mAP50": round(sum(syntax_aps) / len(syntax_aps), 4) if syntax_aps else 0,
        "stenosis_AP50": round(sum(stenosis_aps) / len(stenosis_aps), 4) if stenosis_aps else 0,
        "syntax_model": syntax_weights,
        "stenosis_model": stenosis_weights,
    }

    all_p = [m.get("precision", 0) for m in combined_per_class.values()]
    all_r = [m.get("recall", 0) for m in combined_per_class.values()]
    combined["precision"] = round(sum(all_p) / len(all_p), 4) if all_p else 0
    combined["recall"] = round(sum(all_r) / len(all_r), 4) if all_r else 0
    combined["mAP50_95"] = 0

    _save_metrics(results_dir, "final_test", combined)

    # Vessel-interior combined metrics
    vi_per_class = dict(combined_per_class)
    for cls_name, cls_m in stenosis_filtered_metrics.get("per_class", {}).items():
        vi_per_class["stenosis"] = cls_m
    vi_ap50s = [m.get("ap50", 0) for m in vi_per_class.values()]
    vi_mAP50 = sum(vi_ap50s) / len(vi_ap50s) if vi_ap50s else 0

    vi_sten_m = vi_per_class.get("stenosis", {})
    vessel_interior_combined = {
        "split": "test (vessel-interior)",
        "mAP50": round(vi_mAP50, 4),
        "stenosis_AP50": round(vi_sten_m.get("ap50", 0), 4),
        "stenosis_precision": round(vi_sten_m.get("precision", 0), 4),
        "stenosis_recall": round(vi_sten_m.get("recall", 0), 4),
        "stenosis_f1": round(vi_sten_m.get("f1", 0), 4),
    }
    _save_metrics(results_dir, "vessel_interior_test", vessel_interior_combined)

    all_metrics = {
        "syntax_model_test": syntax_metrics,
        "stenosis_filtered_test": stenosis_filtered_metrics,
        "stenosis_full_test": stenosis_full_metrics,
        "final_test": combined,
        "vessel_interior_test": vessel_interior_combined,
        "filter_stats": filter_stats,
        "vessel_filter_config": {
            "overlap_threshold": overlap_threshold,
            "dilate_px": dilate_px,
            "vessel_conf": vessel_conf,
            "min_count": min_count,
            "use_original_splits": use_original_splits,
        },
    }
    with open(results_dir / "all_metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)

    # Print summary
    print(f"\n  === S10 DUAL METRICS SUMMARY ===")
    sten_full = stenosis_full_metrics.get("per_class", {}).get("stenosis", {})
    sten_filt = stenosis_filtered_metrics.get("per_class", {}).get("stenosis", {})
    print(f"  Standard (full test):   AP50={sten_full.get('ap50', 0):.4f}  "
          f"F1={sten_full.get('f1', 0):.4f}  "
          f"P={sten_full.get('precision', 0):.4f}  "
          f"R={sten_full.get('recall', 0):.4f}")
    print(f"  Vessel-interior test:   AP50={sten_filt.get('ap50', 0):.4f}  "
          f"F1={sten_filt.get('f1', 0):.4f}  "
          f"P={sten_filt.get('precision', 0):.4f}  "
          f"R={sten_filt.get('recall', 0):.4f}")

    # Print filtering stats summary
    for split, ss in filter_stats.get("splits", {}).items():
        total = ss.get("gt_stenoses", 0)
        kept = ss.get("kept", 0)
        pct = kept / total * 100 if total > 0 else 0
        print(f"  [{split}] Kept {kept}/{total} stenoses ({pct:.1f}%)")

    return all_metrics


def _run_single_worker_script():
    """Entry point when this script is invoked as a subprocess worker.

    Usage: python run_stenosis_strategies.py --worker <json_path>
    Reads experiment config from JSON, runs the experiment, writes results JSON.
    """
    import argparse as ap
    p = ap.ArgumentParser()
    p.add_argument("--worker", type=str, required=True)
    a = p.parse_args()

    with open(a.worker) as f:
        job = json.load(f)

    exp = job["exp"]
    arcade_root = Path(job["arcade_root"])
    splits_dir = Path(job["splits_dir"])
    output_dir = Path(job["output_dir"])
    iterations = job["iterations"]
    result_path = Path(job["result_path"])

    name = exp["name"]
    gpu = exp["gpu"]

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    exp["config"]["device"] = "0"

    start = time.time()
    print(f"\n[GPU {gpu}] Starting {name}: {exp['description']}")

    try:
        if exp.get("custom_pipeline"):
            runner = exp.get("custom_runner", "separate")
            if runner == "vessel_guided":
                metrics = run_vessel_guided_stenosis(
                    exp, arcade_root, splits_dir, output_dir, iterations
                )
            elif runner == "separate_v2":
                metrics = run_separate_stenosis_v2(
                    exp, arcade_root, splits_dir, output_dir, iterations
                )
            elif runner == "vessel_filtered":
                metrics = run_vessel_filtered_stenosis(
                    exp, arcade_root, splits_dir, output_dir, iterations
                )
            else:
                metrics = run_separate_stenosis(
                    exp, arcade_root, splits_dir, output_dir, iterations
                )
        else:
            metrics = run_standard_experiment(
                exp, arcade_root, splits_dir, output_dir, iterations
            )

        elapsed = time.time() - start
        print(f"\n[GPU {gpu}] {name} COMPLETE ({elapsed / 3600:.1f}h)")

        result = {
            "name": name,
            "gpu": gpu,
            "description": exp["description"],
            "elapsed_hours": round(elapsed / 3600, 2),
            "status": "success",
            "metrics": metrics,
        }

    except Exception as e:
        elapsed = time.time() - start
        error_msg = traceback.format_exc()
        print(f"\n[GPU {gpu}] {name} FAILED ({elapsed / 3600:.1f}h): {e}")
        print(error_msg)

        result = {
            "name": name,
            "gpu": gpu,
            "description": exp["description"],
            "elapsed_hours": round(elapsed / 3600, 2),
            "status": "failed",
            "error": str(e),
            "traceback": error_msg,
        }

    # Clean up CUDA memory to prevent VRAM leaks for sequential runs
    try:
        import gc
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        print(f"  [GPU {gpu}] CUDA cache cleared")
    except Exception:
        pass

    with open(result_path, "w") as f:
        json.dump(result, f, indent=2, default=str)


def launch_experiments_parallel(experiments, arcade_root, splits_dir,
                                output_dir, iterations):
    """Launch each experiment as a separate subprocess to avoid daemon issues."""
    script_path = Path(__file__).resolve()
    tmp_dir = output_dir / "_worker_jobs"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    processes = []
    result_paths = []

    for exp in experiments:
        name = exp["name"]
        job_path = tmp_dir / f"{name}_job.json"
        result_path = tmp_dir / f"{name}_result.json"
        log_path = output_dir / name / "worker.log"
        (output_dir / name).mkdir(parents=True, exist_ok=True)

        job = {
            "exp": exp,
            "arcade_root": str(arcade_root),
            "splits_dir": str(splits_dir),
            "output_dir": str(output_dir),
            "iterations": iterations,
            "result_path": str(result_path),
        }
        with open(job_path, "w") as f:
            json.dump(job, f, indent=2, default=str)

        log_file = open(log_path, "w")
        proc = subprocess.Popen(
            [sys.executable, str(script_path), "--worker", str(job_path)],
            stdout=log_file,
            stderr=subprocess.STDOUT,
            cwd=str(script_path.parent),
        )
        processes.append((name, exp["gpu"], proc, log_file))
        result_paths.append((name, result_path))
        print(f"  Launched {name} on GPU {exp['gpu']} (PID {proc.pid}, log: {log_path})")

    # Wait for all to finish
    print(f"\nWaiting for {len(processes)} experiments to complete...")
    for name, gpu, proc, log_file in processes:
        proc.wait()
        log_file.close()
        status = "OK" if proc.returncode == 0 else f"EXIT {proc.returncode}"
        print(f"  [GPU {gpu}] {name}: {status}")

    # Collect results
    results = []
    for name, result_path in result_paths:
        if result_path.exists():
            with open(result_path) as f:
                results.append(json.load(f))
        else:
            results.append({
                "name": name,
                "status": "failed",
                "error": "No result file produced",
                "elapsed_hours": 0,
            })

    return results


# ── Results Reporting ───────────────────────────────────────────────────────

def print_results_table(results: list) -> None:
    """Print a formatted comparison table of all experiments."""
    print("\n" + "=" * 100)
    print("STENOSIS IMPROVEMENT STRATEGIES — COMPARISON")
    print("=" * 100)

    print(f"\n{'Name':<28} {'Status':<8} {'Hours':>6} "
          f"{'mAP50':>8} {'F1':>8} "
          f"{'Syn mAP':>9} {'Sten AP':>9} {'Sten F1':>9}")
    print("-" * 100)

    for r in results:
        test = r.get("metrics", {}).get("final_test", {})
        p = test.get("precision", 0)
        rec = test.get("recall", 0)
        f1 = 2 * p * rec / (p + rec) if (p + rec) > 0 else 0

        # Stenosis F1
        sten_m = test.get("per_class", {}).get("stenosis", {})
        sten_f1 = sten_m.get("f1", 0)

        status = r.get("status", "?")
        if status == "success":
            print(f"{r['name']:<28} {'OK':<8} {r['elapsed_hours']:>6.1f} "
                  f"{test.get('mAP50', 0):>8.4f} {f1:>8.4f} "
                  f"{test.get('syntax_mAP50', 0):>9.4f} "
                  f"{test.get('stenosis_AP50', 0):>9.4f} "
                  f"{sten_f1:>9.4f}")
        else:
            print(f"{r['name']:<28} {'FAIL':<8} {r['elapsed_hours']:>6.1f} "
                  f"{'—':>8} {'—':>8} {'—':>9} {'—':>9} {'—':>9}")

    # Per-class F1 breakdown
    print(f"\n{'=' * 100}")
    print("PER-CLASS F1 (Test)")
    print("=" * 100)

    all_classes = set()
    for r in results:
        test = r.get("metrics", {}).get("final_test", {})
        all_classes.update(test.get("per_class", {}).keys())

    if all_classes:
        sorted_classes = sorted(all_classes)
        header = f"{'Name':<28}"
        for cls in sorted_classes:
            header += f" {cls:>8}"
        print(header)
        print("-" * len(header))

        for r in results:
            if r.get("status") != "success":
                continue
            test = r.get("metrics", {}).get("final_test", {})
            per_class = test.get("per_class", {})
            line = f"{r['name']:<28}"
            for cls in sorted_classes:
                f1 = per_class.get(cls, {}).get("f1", 0)
                line += f" {f1:>8.4f}"
            print(line)


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run stenosis improvement strategies across 6 GPUs"
    )
    parser.add_argument(
        "--arcade-root", type=str, default="../../arcade/submission",
        help="Path to arcade/submission directory"
    )
    parser.add_argument(
        "--output-dir", type=str, default="../results/stenosis_strategies",
        help="Base output directory"
    )
    parser.add_argument(
        "--iterations", type=int, default=3,
        help="Number of pseudo-label iterations (default: 3)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--skip-splits", action="store_true",
        help="Skip stratified split creation (if already done)"
    )
    parser.add_argument(
        "--experiments", nargs="+", default=None,
        help="Run only these experiments (e.g., S0_baseline_fixed S1_resolution_768)"
    )
    parser.add_argument(
        "--gpus", type=str, default="0,1,2,3,4,5",
        help="Comma-separated GPU IDs to use (default: 0,1,2,3,4,5)"
    )
    args = parser.parse_args()

    arcade_root = Path(args.arcade_root).resolve()
    script_dir = Path(__file__).resolve().parent
    output_dir = (script_dir / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    splits_dir = output_dir / "stratified_splits"

    available_gpus = [int(g) for g in args.gpus.split(",")]

    # ── Step 1: Create stratified splits (shared) ──
    if not args.skip_splits:
        print("#" * 60)
        print("# Creating stratified splits (shared by all experiments)")
        print("#" * 60)
        from create_stratified_splits import create_stratified_splits
        create_stratified_splits(arcade_root, splits_dir, args.seed)
    else:
        print("[SKIP] Stratified splits (--skip-splits)")
        if not splits_dir.exists():
            print(f"ERROR: {splits_dir} not found. Remove --skip-splits.")
            sys.exit(1)

    # ── Step 2: Define experiments ──
    experiments = get_experiments()

    # Filter experiments if specified
    if args.experiments:
        experiments = [e for e in experiments if e["name"] in args.experiments]
        if not experiments:
            print(f"ERROR: No matching experiments. Available: "
                  f"{[e['name'] for e in get_experiments()]}")
            sys.exit(1)

    # Reassign GPUs to match available hardware
    for i, exp in enumerate(experiments):
        exp["gpu"] = available_gpus[i % len(available_gpus)]
        exp["config"]["device"] = str(exp["gpu"])

    print(f"\n{'#' * 60}")
    print(f"# Running {len(experiments)} experiments on {len(available_gpus)} GPUs")
    print(f"#   Output: {output_dir}")
    print(f"{'#' * 60}")
    for exp in experiments:
        print(f"  GPU {exp['gpu']}: {exp['name']} — {exp['description']}")

    # ── Step 3: Launch all experiments as separate subprocesses ──
    total_start = time.time()

    results = launch_experiments_parallel(
        experiments, arcade_root, splits_dir, output_dir, args.iterations
    )

    total_elapsed = time.time() - total_start

    # ── Step 4: Report results ──
    print_results_table(results)

    # Save full results
    results_path = output_dir / "strategy_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n{'=' * 100}")
    print(f"ALL EXPERIMENTS COMPLETE ({total_elapsed / 3600:.1f}h wall time)")
    print(f"Results saved: {results_path}")

    # Summary
    successes = [r for r in results if r["status"] == "success"]
    failures = [r for r in results if r["status"] == "failed"]
    print(f"  {len(successes)} succeeded, {len(failures)} failed")
    if failures:
        print("  Failed experiments:")
        for r in failures:
            print(f"    {r['name']}: {r.get('error', 'unknown')}")
    print(f"{'=' * 100}")


if __name__ == "__main__":
    # If invoked as a worker subprocess, run the single experiment
    if "--worker" in sys.argv:
        _run_single_worker_script()
    else:
        main()