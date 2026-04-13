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
        # ── Round 3 experiments (S11-S17) ─────────────────────────────
        # Goal: push stenosis F1 > 0.43 (beat S8's 0.4152 and Old S5's
        # 0.3951). All stack on S8's proven recipe (old S5 aug + mosaic
        # on the stenosis model) and target the stenosis bottleneck.
        {
            "name": "S11_stenosis_1024",
            "gpu": 0,
            "description": "S8 recipe + stenosis model at 1024px "
                           "(bigger objects → highest-priority gain)",
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
                "stenosis_imgsz": 1024,
                "stenosis_batch": 4,
            },
        },
        {
            "name": "S12_stenosis_yolo11l",
            "gpu": 1,
            "description": "S8 recipe + yolo11l-seg (larger model) "
                           "for stenosis; syntax stays yolo11m",
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
                "stenosis_model_weights": "yolo11l-seg.pt",
                "stenosis_batch": 4,
            },
        },
        {
            "name": "S13_mixup_mosaic_stack",
            "gpu": 2,
            "description": "S8 recipe + mixup=0.15 on stenosis "
                           "(mixup + mosaic complementary regularisation)",
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
                "mixup": 0.15,
            },
        },
        # S15 — multi-seed S8 rerun for variance estimation (3 seeds).
        # Aggregator at the end of main() reads all three and appends a
        # computed S15_multiseed_summary entry with mean/std.
        {
            "name": "S15_s8_seed42",
            "gpu": 3,
            "description": "S8 recipe with seed=42 (variance study)",
            "overrides": {
                "degrees": 20.0,
                "scale": 0.4,
                "hsv_v": 0.3,
                "seed": 42,
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
            "name": "S15_s8_seed7",
            "gpu": 4,
            "description": "S8 recipe with seed=7 (variance study)",
            "overrides": {
                "degrees": 20.0,
                "scale": 0.4,
                "hsv_v": 0.3,
                "seed": 7,
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
            "name": "S15_s8_seed2024",
            "gpu": 5,
            "description": "S8 recipe with seed=2024 (variance study)",
            "overrides": {
                "degrees": 20.0,
                "scale": 0.4,
                "hsv_v": 0.3,
                "seed": 2024,
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
            "name": "S16_mosaic_both",
            "gpu": 5,
            "description": "S8 + mosaic=0.5 on syntax too (stretch: "
                           "test whether mosaic helps syntax too)",
            "overrides": {
                "degrees": 20.0,
                "scale": 0.4,
                "hsv_v": 0.3,
                "mosaic": 0.5,
                "close_mosaic": 15,
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
        # ── Round 4 experiments (S18+) ────────────────────────────────
        # Round 3 finding: S12 (yolo11l-seg) produced the highest
        # stenosis F1 ever seen (0.4548, +4pp over S8), BUT S15's
        # 2-of-3 seeds showed ~5pp seed-to-seed variance on the S8
        # recipe — so S12 vs S8 (+4pp) is not yet >1σ above noise.
        # S18 reruns the S12 recipe with 3 seeds to confirm the gain
        # is real. Aggregator in main() produces S18_multiseed_summary.
        {
            "name": "S18_s12_seed42",
            "gpu": 0,
            "description": "S12 (yolo11l-seg) recipe with seed=42 "
                           "(Round 4 variance confirmation)",
            "overrides": {
                "degrees": 20.0,
                "scale": 0.4,
                "hsv_v": 0.3,
                "seed": 42,
            },
            "pipeline_args": {},
            "custom_pipeline": True,
            "custom_runner": "separate_v2",
            "stenosis_overrides": {
                "copy_paste": 0.3,
                "scale": 0.5,
                "mosaic": 0.8,
                "close_mosaic": 15,
                "stenosis_model_weights": "yolo11l-seg.pt",
                "stenosis_batch": 4,
            },
        },
        {
            "name": "S18_s12_seed7",
            "gpu": 1,
            "description": "S12 (yolo11l-seg) recipe with seed=7 "
                           "(Round 4 variance confirmation)",
            "overrides": {
                "degrees": 20.0,
                "scale": 0.4,
                "hsv_v": 0.3,
                "seed": 7,
            },
            "pipeline_args": {},
            "custom_pipeline": True,
            "custom_runner": "separate_v2",
            "stenosis_overrides": {
                "copy_paste": 0.3,
                "scale": 0.5,
                "mosaic": 0.8,
                "close_mosaic": 15,
                "stenosis_model_weights": "yolo11l-seg.pt",
                "stenosis_batch": 4,
            },
        },
        {
            "name": "S18_s12_seed2024",
            "gpu": 0,
            "description": "S12 (yolo11l-seg) recipe with seed=2024 "
                           "(Round 4 variance confirmation)",
            "overrides": {
                "degrees": 20.0,
                "scale": 0.4,
                "hsv_v": 0.3,
                "seed": 2024,
            },
            "pipeline_args": {},
            "custom_pipeline": True,
            "custom_runner": "separate_v2",
            "stenosis_overrides": {
                "copy_paste": 0.3,
                "scale": 0.5,
                "mosaic": 0.8,
                "close_mosaic": 15,
                "stenosis_model_weights": "yolo11l-seg.pt",
                "stenosis_batch": 4,
            },
        },
        {
            "name": "S17_mosaic_crop_1024",
            "gpu": 1,
            "description": "S8 mosaic + vessel-guided CROP + stenosis "
                           "at 1024px (combines the two orthogonal "
                           "Round 2 wins; highest-ceiling experiment)",
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
                "mosaic": 0.8,
                "close_mosaic": 15,
                "stenosis_imgsz": 1024,
                "stenosis_batch": 4,
            },
            "vessel_guided": {
                "dilate_px": 30,
                "crop_pad_px": 15,
                "vessel_conf": 0.15,
                "vessel_imgsz": 768,
                "variant": "crop",
                "min_count": 0,
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

    # Special keys in stenosis_overrides redirect to dedicated cfg slots
    # (so experiments can raise resolution, shrink batch, or swap the
    # backbone for the stenosis model without affecting the syntax model).
    special = {
        "stenosis_imgsz": "imgsz",
        "stenosis_batch": "batch",
        "stenosis_model_weights": "model",
    }
    filtered_overrides = dict(stenosis_overrides)
    for src_key, dst_key in special.items():
        if src_key in filtered_overrides:
            cfg_sten[dst_key] = filtered_overrides.pop(src_key)

    # Apply remaining per-experiment stenosis overrides
    # (copy_paste, scale, mosaic, mixup, close_mosaic, etc.)
    for key, val in filtered_overrides.items():
        cfg_sten[key] = val

    sten_imgsz = cfg_sten["imgsz"]
    run_name = f"stenosis_{sten_imgsz}"

    stenosis_yaml = str(data_dir / "dataset_configs" / "stenosis_only.yaml")
    stenosis_weights = train_two_stage(
        cfg_sten, stenosis_yaml,
        project=str(results_dir / "stenosis_model"),
        run_name=run_name,
    )

    stenosis_metrics = evaluate_model(
        stenosis_weights, stenosis_yaml, split="test",
        augment=True, imgsz=sten_imgsz,
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

        # Special keys in stenosis_overrides redirect to dedicated cfg slots
        # (mirrors the handling in run_separate_stenosis_v2).
        special = {
            "stenosis_imgsz": "imgsz",
            "stenosis_batch": "batch",
            "stenosis_model_weights": "model",
        }
        filtered_overrides = dict(stenosis_overrides)
        for src_key, dst_key in special.items():
            if src_key in filtered_overrides:
                cfg_sten[dst_key] = filtered_overrides.pop(src_key)

        # Apply remaining per-experiment stenosis overrides
        for key, val in filtered_overrides.items():
            cfg_sten[key] = val

        sten_imgsz = cfg_sten["imgsz"]
        stenosis_yaml = str(masked_root / v / "data.yaml")
        st_weights = train_two_stage(
            cfg_sten, stenosis_yaml,
            project=str(results_dir / f"stenosis_model_{v}"),
            run_name=f"stenosis_{v}_{sten_imgsz}",
        )
        stenosis_weights_by_variant[v] = st_weights

        st_metrics = evaluate_model(
            st_weights, stenosis_yaml, split="test",
            augment=True, imgsz=sten_imgsz,
        )
        _save_metrics(results_dir, f"stenosis_model_{v}_test", st_metrics)
        stenosis_metrics_by_variant[v] = st_metrics

    # Pick the best variant by stenosis F1 (not AP50) for the combined
    # final report. F1 better reflects the deployed detection quality —
    # the S9 post-mortem showed crop beat blackout on F1 (0.4098 vs
    # 0.3906) even though blackout had marginally higher AP50.
    def _stenosis_f1(m):
        cls = m.get("per_class", {}).get("stenosis", {})
        p = cls.get("precision", 0.0)
        r = cls.get("recall", 0.0)
        return (2 * p * r / (p + r)) if (p + r) > 0 else 0.0

    best_variant = max(
        stenosis_metrics_by_variant.keys(),
        key=lambda v: _stenosis_f1(stenosis_metrics_by_variant[v]),
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

    # CUDA_VISIBLE_DEVICES should already be set by the parent via Popen(env=...)
    # — this assignment is a belt-and-suspenders no-op in normal operation.
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

    # ── Aggressive CUDA cleanup before exit ──
    # The OS will tear down the CUDA context anyway when this process
    # exits, but we explicitly flush + release to minimise the window
    # where nvidia-smi still reports allocation against this PID and
    # to avoid leaked shared-memory segments from dataloaders.
    try:
        import gc
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()   # flush pending work
            torch.cuda.empty_cache()    # return cached blocks to driver
            try:
                torch.cuda.ipc_collect()  # release IPC handles
            except Exception:
                pass
        gc.collect()
        print(f"  [GPU {gpu}] CUDA flushed, cache cleared, gc collected")
    except Exception as _e:
        print(f"  [GPU {gpu}] CUDA cleanup skipped: {_e}")

    with open(result_path, "w") as f:
        json.dump(result, f, indent=2, default=str)


def _gpu_free_mb(gpu_id: int) -> int:
    """Return free VRAM in MB on ``gpu_id`` via nvidia-smi.

    Returns -1 if nvidia-smi is unavailable or parsing fails so that
    callers can treat an unknown result as "don't block".
    """
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.free",
                "--format=csv,noheader,nounits",
                "-i", str(gpu_id),
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return int(out.strip().split("\n")[0])
    except Exception:
        return -1


def _gpu_total_mb(gpu_id: int) -> int:
    """Return total VRAM in MB on ``gpu_id`` via nvidia-smi (-1 if unknown)."""
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.total",
                "--format=csv,noheader,nounits",
                "-i", str(gpu_id),
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return int(out.strip().split("\n")[0])
    except Exception:
        return -1


def _gpu_compute_pids(gpu_id: int, exclude_pids: set[int] | None = None) -> list[int]:
    """Return the list of compute-mode PIDs currently running on ``gpu_id``.

    Used to detect lingering workers that haven't fully released their
    CUDA context. Returns an empty list if nvidia-smi isn't available
    (so callers don't block on systems without it).
    """
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=pid",
                "--format=csv,noheader,nounits",
                "-i", str(gpu_id),
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        pids = []
        for line in out.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            try:
                pid = int(line)
                if exclude_pids is None or pid not in exclude_pids:
                    pids.append(pid)
            except ValueError:
                continue
        return pids
    except Exception:
        return []


def _pid_cmdline(pid: int) -> str:
    """Return the cmdline of ``pid`` as a space-joined string, or ''.

    Reads ``/proc/<pid>/cmdline`` directly so we don't depend on ps
    or psutil. Returns empty string if the process has disappeared
    or the platform doesn't expose /proc.
    """
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            raw = f.read()
        return raw.replace(b"\x00", b" ").decode("utf-8", "replace").strip()
    except Exception:
        return ""


def _reap_orphan_compute_pids(
    gpu_pool: list[int],
    parent_pid: int,
    project_marker: str = "run_stenosis_strategies",
    kill: bool = False,
) -> list[dict]:
    """Detect (and optionally SIGTERM) stale compute PIDs on ``gpu_pool``.

    Round 3 taught us that a single zombie from an earlier run can
    silently block multiple experiments via the pre-flight idle check.
    At scheduler startup we now enumerate every compute PID on every
    GPU in the pool (excluding our own parent_pid), and for each one:

    - If its cmdline contains ``project_marker`` it's clearly a stale
      worker from an earlier session. Log it, and if ``kill=True``
      send SIGTERM (then SIGKILL after a short grace window).
    - Otherwise it's an unrelated process (someone else's training,
      a Jupyter kernel, etc.). Log it as a warning but DO NOT touch
      it — we don't know whose work we'd be destroying.

    Returns a list of dicts describing each orphan found, so callers
    can decide whether to abort or proceed.
    """
    import signal

    orphans = []
    for gpu_id in gpu_pool:
        for pid in _gpu_compute_pids(gpu_id, exclude_pids={parent_pid}):
            cmdline = _pid_cmdline(pid)
            is_ours = project_marker in cmdline
            entry = {
                "gpu": gpu_id,
                "pid": pid,
                "cmdline": cmdline or "<unknown>",
                "is_project": is_ours,
                "action": "none",
            }
            if is_ours and kill:
                try:
                    os.kill(pid, signal.SIGTERM)
                    entry["action"] = "SIGTERM"
                except ProcessLookupError:
                    entry["action"] = "gone"
                except PermissionError:
                    entry["action"] = "permission_denied"
                except Exception as e:
                    entry["action"] = f"error: {e}"
            orphans.append(entry)

    # If we SIGTERM'd anything, give it a grace window then follow up
    # with SIGKILL on any that are still around.
    if kill and any(o["action"] == "SIGTERM" for o in orphans):
        time.sleep(5)
        for entry in orphans:
            if entry["action"] != "SIGTERM":
                continue
            pid = entry["pid"]
            try:
                os.kill(pid, 0)  # probe — raises if gone
            except ProcessLookupError:
                entry["action"] = "terminated"
                continue
            except Exception:
                continue
            # Still alive after SIGTERM → SIGKILL
            try:
                os.kill(pid, signal.SIGKILL)
                entry["action"] = "SIGKILL"
            except Exception as e:
                entry["action"] = f"kill_failed: {e}"
        # Another short wait so nvidia-smi reflects the kills before
        # the scheduler captures its per-GPU baseline.
        time.sleep(3)

    return orphans


def _wait_for_gpu_free(
    gpu_id: int,
    min_free_mb: int,
    baseline_free_mb: int = -1,
    tolerance_mb: int = 500,
    max_wait_s: int = 600,
    poll_interval_s: int = 5,
    exclude_pids: set[int] | None = None,
) -> tuple[bool, str]:
    """Block until ``gpu_id`` is proven idle.

    The GPU is considered idle when ALL of the following hold:

    1. No compute processes report against the GPU in nvidia-smi
       (other than any pids in ``exclude_pids``).
    2. Free VRAM is at least ``min_free_mb`` MB.
    3. If ``baseline_free_mb`` > 0, free VRAM is also at least
       ``baseline_free_mb - tolerance_mb`` (i.e., back near the
       no-workers-running baseline captured at scheduler startup).

    On systems where nvidia-smi is unavailable, the first check is
    skipped and we rely on the memory check (or proceed blindly if
    the memory query also fails).

    Returns
    -------
    (ok, reason)
        ok: True if the GPU is idle, False on timeout.
        reason: Human-readable reason string for the outcome.
    """
    if min_free_mb <= 0 and baseline_free_mb <= 0:
        return True, "checks disabled"

    waited = 0
    last_reason = ""
    while waited <= max_wait_s:
        free = _gpu_free_mb(gpu_id)
        pids = _gpu_compute_pids(gpu_id, exclude_pids=exclude_pids)

        # Unknown state (nvidia-smi missing) → don't block.
        if free < 0 and not pids:
            return True, "nvidia-smi unavailable — proceeding blind"

        reasons = []
        if pids:
            reasons.append(f"{len(pids)} lingering compute PID(s): {pids[:4]}")
        if free >= 0 and min_free_mb > 0 and free < min_free_mb:
            reasons.append(f"free={free}MB < min_free_mb={min_free_mb}")
        if (free >= 0 and baseline_free_mb > 0
                and free < baseline_free_mb - tolerance_mb):
            reasons.append(
                f"free={free}MB < baseline({baseline_free_mb})-"
                f"tol({tolerance_mb})={baseline_free_mb - tolerance_mb}"
            )

        if not reasons:
            return True, f"idle (free={free}MB)"

        last_reason = "; ".join(reasons)
        print(f"    [GPU {gpu_id}] waiting for idle — {last_reason} "
              f"(waited {waited}s / {max_wait_s}s)")
        time.sleep(poll_interval_s)
        waited += poll_interval_s

    return False, f"timeout after {max_wait_s}s — {last_reason}"


def launch_experiments_parallel(experiments, arcade_root, splits_dir,
                                output_dir, iterations,
                                max_concurrent=None,
                                stagger_seconds=15,
                                min_free_mb=2000,
                                gpu_pool=None,
                                reap_orphans=False):
    """Launch experiments as subprocesses with hardened concurrency.

    Four defences against the "first 3 succeed, rest OOM" failure mode:

    1. **Env set at spawn time.** CUDA_VISIBLE_DEVICES is placed in the
       subprocess's initial environment via ``Popen(env=...)`` — BEFORE
       any Python or torch imports run. Setting it inside the worker
       would be too late: torch's CUDA init would have already touched
       GPU 0 with whatever the default visibility was.

    2. **Dynamic GPU pool scheduler.** ``gpu_pool`` is a list of free
       GPU IDs. Each spawn pops a free GPU; each completion returns
       one. Experiments are NOT locked to their advisory ``exp["gpu"]``
       value — whatever is free wins. This correctly handles the case
       where #experiments > #GPUs, so experiments queue on the pool
       without ever double-booking a GPU.

    3. **Staggered launch.** ``stagger_seconds`` sleep is inserted
       between consecutive spawns so their CUDA context creations
       don't collide. This is the actual root cause of the original
       "exactly 3 succeed" symptom: 6 simultaneous torch.cuda inits
       race on driver resources and 3 of them time out.

    4. **Pre-flight memory check.** Before each spawn we poll
       nvidia-smi on the target GPU and wait until it has at least
       ``min_free_mb`` MB free (default 2000). Catches the case where
       a prior subprocess hasn't fully released its context yet.
    """
    script_path = Path(__file__).resolve()
    tmp_dir = output_dir / "_worker_jobs"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # Determine the free-GPU pool
    if gpu_pool is None:
        gpu_pool = sorted({int(e["gpu"]) for e in experiments})
    else:
        gpu_pool = list(gpu_pool)

    if max_concurrent is None or max_concurrent <= 0:
        max_concurrent = len(gpu_pool)
    max_concurrent = min(max_concurrent, len(gpu_pool))

    free_gpus = list(gpu_pool)  # pool of GPUs currently free
    parent_pid = os.getpid()

    # ── Orphan-process sweep (before baseline capture) ──
    # Round 3 post-mortem: a single zombie process from an earlier
    # session blocked TWO Round 3 experiments via the pre-flight idle
    # check. Sweep each pool GPU for compute PIDs we didn't spawn and,
    # if they look like our own stale workers (cmdline contains
    # ``run_stenosis_strategies``), kill them BEFORE we capture the
    # idle baseline. Unrelated processes are logged but never killed.
    orphans = _reap_orphan_compute_pids(
        gpu_pool, parent_pid,
        project_marker="run_stenosis_strategies",
        kill=reap_orphans,
    )
    if orphans:
        print(f"\n  Orphan sweep ({'kill' if reap_orphans else 'dry-run'}):")
        for o in orphans:
            tag = "OURS" if o["is_project"] else "foreign"
            action = o["action"]
            cmdline = (o["cmdline"][:80] + "…") if len(o["cmdline"]) > 80 else o["cmdline"]
            print(f"    [GPU {o['gpu']}] PID {o['pid']} ({tag}, {action}): {cmdline}")
        stale_still_alive = [
            o for o in orphans
            if o["is_project"] and o["action"] not in ("SIGKILL", "terminated", "gone")
        ]
        if stale_still_alive and not reap_orphans:
            print(f"  NOTE: {len(stale_still_alive)} stale worker(s) still "
                  f"holding memory. Pass --reap-orphans to SIGTERM/SIGKILL "
                  f"them, otherwise the scheduler will wait and may skip "
                  f"experiments whose GPU is blocked.")

    # ── Capture per-GPU idle baseline AFTER orphan sweep ──
    # These baselines let the scheduler prove "GPU is back to idle" by
    # waiting until free memory returns to near the captured level,
    # rather than accepting any arbitrary ``min_free_mb``.
    gpu_baseline_free_mb = {}
    for g in gpu_pool:
        gpu_baseline_free_mb[g] = _gpu_free_mb(g)

    def _prepare_job(exp):
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
        return name, job_path, result_path, log_path

    def _spawn(exp, gpu_id):
        """Spawn a worker on ``gpu_id``.

        Returns either a running-job dict, or a synthetic failure result
        dict if the pre-flight idle check times out (so the experiment
        is cleanly skipped instead of blocking the scheduler forever).
        """
        # Update the exp's GPU assignment to reflect the actually-assigned
        # GPU (differs from the advisory pre-assignment when pool-scheduled).
        exp = dict(exp)  # shallow copy so we don't mutate the caller's list
        exp["gpu"] = gpu_id
        cfg = dict(exp.get("config", {}))
        cfg["device"] = str(gpu_id)
        exp["config"] = cfg

        name, job_path, result_path, log_path = _prepare_job(exp)

        # ── Pre-flight idle check ──
        # Wait until the GPU is proven idle: no compute PIDs running and
        # free memory at least ``min_free_mb`` AND within tolerance of
        # the startup baseline. On timeout, SKIP this experiment (don't
        # spawn) so the rest of the queue can proceed.
        ok, reason = _wait_for_gpu_free(
            gpu_id,
            min_free_mb=min_free_mb,
            baseline_free_mb=gpu_baseline_free_mb.get(gpu_id, -1),
            tolerance_mb=500,
            max_wait_s=600,
            poll_interval_s=5,
            exclude_pids={parent_pid},
        )
        if not ok:
            print(f"  [GPU {gpu_id}] SKIPPING {name}: {reason}")
            return {
                "skipped": True,
                "name": name,
                "gpu": gpu_id,
                "result": {
                    "name": name,
                    "gpu": gpu_id,
                    "status": "failed",
                    "error": f"Pre-flight GPU idle check failed: {reason}",
                    "elapsed_hours": 0,
                },
            }

        # CRITICAL: set CUDA_VISIBLE_DEVICES in the subprocess's initial env,
        # before any Python imports. Setting it inside the worker after torch
        # imports would be too late — torch already ran its CUDA init on the
        # default GPU.
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        # Also discourage over-threaded CPU work that competes for system RAM
        # when multiple workers run concurrently.
        env.setdefault("OMP_NUM_THREADS", "4")
        env.setdefault("MKL_NUM_THREADS", "4")

        log_file = open(log_path, "w")
        proc = subprocess.Popen(
            [sys.executable, str(script_path), "--worker", str(job_path)],
            stdout=log_file,
            stderr=subprocess.STDOUT,
            cwd=str(script_path.parent),
            env=env,
        )
        print(f"  Launched {name} on GPU {gpu_id} "
              f"(PID {proc.pid}, free={_gpu_free_mb(gpu_id)}MB, "
              f"baseline={gpu_baseline_free_mb.get(gpu_id, -1)}MB, "
              f"log: {log_path})")
        return {
            "skipped": False,
            "name": name,
            "gpu": gpu_id,
            "proc": proc,
            "log_file": log_file,
            "result_path": result_path,
        }

    pending = list(experiments)
    running = []
    results = []
    total = len(pending)
    done_count = 0

    print(f"\nScheduling {total} experiments")
    print(f"  GPU pool            : {gpu_pool}")
    print(f"  per-GPU baseline    : "
          f"{ {g: f'{gpu_baseline_free_mb[g]}MB' for g in gpu_pool} }")
    print(f"  max_concurrent      : {max_concurrent}")
    print(f"  stagger_seconds     : {stagger_seconds}")
    print(f"  min_free_mb         : {min_free_mb}")
    print(f"  idle tolerance (MB) : 500")
    print(f"  idle timeout (s)    : 600")

    while pending or running:
        # ── Fill open slots from the free-GPU pool ──
        spawned_this_round = 0
        while (pending and len(running) < max_concurrent and free_gpus):
            exp = pending.pop(0)
            gpu_id = free_gpus.pop(0)
            # Stagger: don't launch two subprocesses back-to-back; give
            # each one time to finish CUDA init + model load before the
            # next one starts torching the driver.
            if spawned_this_round > 0 and stagger_seconds > 0:
                print(f"  (staggering {stagger_seconds}s before next launch)")
                time.sleep(stagger_seconds)
            job = _spawn(exp, gpu_id)
            if job.get("skipped"):
                # Pre-flight failed: record the failure, return GPU to
                # pool (it's still free by definition — we never spawned),
                # and continue with the queue.
                done_count += 1
                results.append(job["result"])
                free_gpus.append(gpu_id)
                print(f"  [{done_count}/{total}] [GPU {gpu_id}] "
                      f"{job['name']}: SKIPPED (pre-flight)")
                continue
            running.append(job)
            spawned_this_round += 1

        if not running:
            # Nothing running and nothing spawned this round — either
            # we're done or all remaining experiments got skipped.
            if not pending:
                break
            continue

        # ── Poll for any finished subprocess ──
        finished_idx = None
        while finished_idx is None:
            for i, job in enumerate(running):
                if job["proc"].poll() is not None:
                    finished_idx = i
                    break
            if finished_idx is None:
                time.sleep(5)

        job = running.pop(finished_idx)
        # Reap zombie and ensure log is flushed/closed
        try:
            job["proc"].wait(timeout=30)
        except Exception:
            pass
        job["log_file"].close()
        rc = job["proc"].returncode
        status = "OK" if rc == 0 else f"EXIT {rc}"
        done_count += 1
        print(f"  [{done_count}/{total}] [GPU {job['gpu']}] "
              f"{job['name']}: {status}")

        # ── Wait for the GPU to actually release memory ──
        # Subprocess exit triggers CUDA context teardown, but nvidia-smi
        # can lag by ~1-3s. Poll until the GPU is back to baseline so
        # the next worker on this slot starts from a clean state.
        gpu_id = job["gpu"]
        child_pid = job["proc"].pid
        time.sleep(2)  # initial driver-catchup grace period
        ok, reason = _wait_for_gpu_free(
            gpu_id,
            min_free_mb=min_free_mb,
            baseline_free_mb=gpu_baseline_free_mb.get(gpu_id, -1),
            tolerance_mb=500,
            max_wait_s=120,
            poll_interval_s=3,
            exclude_pids={parent_pid, child_pid},
        )
        if ok:
            print(f"    [GPU {gpu_id}] released: {reason}")
        else:
            print(f"    [GPU {gpu_id}] WARNING: not idle after "
                  f"completion — {reason}")

        # Return the GPU to the free pool for the next queued experiment
        free_gpus.append(gpu_id)

        if job["result_path"].exists():
            with open(job["result_path"]) as f:
                results.append(json.load(f))
        else:
            results.append({
                "name": job["name"],
                "status": "failed",
                "error": f"No result file produced (exit {rc})",
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


def _aggregate_multiseed(
    merged: list,
    results_path: Path,
    summary_name: str,
    seed_names: list[str],
    description: str,
) -> None:
    """Compute mean/std across a set of seed runs and append a summary.

    Generalised aggregator used for both S15 (S8 recipe × 3 seeds) and
    S18 (S12 recipe × 3 seeds). Reads whichever seed entries are
    present in the merged results and, if ALL of ``seed_names`` exist,
    computes mean and std of the headline metrics and writes a
    synthetic ``<summary_name>`` entry that downstream tooling can
    treat like any other experiment.
    """
    by_name = {r["name"]: r for r in merged if isinstance(r, dict)}
    seed_runs = [by_name[n] for n in seed_names if n in by_name]
    if len(seed_runs) < len(seed_names):
        return  # wait until all seeds are in

    def _final(r):
        return (r.get("metrics", {}) or {}).get("final_test", {}) or {}

    def _f1(p, r):
        return (2 * p * r / (p + r)) if (p + r) > 0 else 0.0

    def _stenosis_f1(r):
        sten = _final(r).get("per_class", {}).get("stenosis", {}) or {}
        p = sten.get("precision", 0.0)
        rv = sten.get("recall", 0.0)
        return _f1(p, rv)

    def _overall_f1(r):
        f = _final(r)
        return _f1(f.get("precision", 0.0), f.get("recall", 0.0))

    def _mean_std(xs):
        if not xs:
            return 0.0, 0.0
        n = len(xs)
        m = sum(xs) / n
        if n < 2:
            return m, 0.0
        var = sum((x - m) ** 2 for x in xs) / (n - 1)  # sample std
        return m, var ** 0.5

    syntax_mAPs = [_final(r).get("syntax_mAP50", 0.0) for r in seed_runs]
    stenosis_APs = [_final(r).get("stenosis_AP50", 0.0) for r in seed_runs]
    stenosis_F1s = [_stenosis_f1(r) for r in seed_runs]
    overall_F1s = [_overall_f1(r) for r in seed_runs]
    mAP50s = [_final(r).get("mAP50", 0.0) for r in seed_runs]

    syntax_mean, syntax_std = _mean_std(syntax_mAPs)
    sten_ap_mean, sten_ap_std = _mean_std(stenosis_APs)
    sten_f1_mean, sten_f1_std = _mean_std(stenosis_F1s)
    overall_f1_mean, overall_f1_std = _mean_std(overall_F1s)
    map_mean, map_std = _mean_std(mAP50s)

    summary = {
        "name": summary_name,
        "gpu": -1,
        "description": description,
        "elapsed_hours": round(sum(r.get("elapsed_hours", 0) for r in seed_runs), 2),
        "status": "success",
        "metrics": {
            "final_test": {
                "split": "test",
                "mAP50": round(map_mean, 4),
                "mAP50_std": round(map_std, 4),
                "syntax_mAP50": round(syntax_mean, 4),
                "syntax_mAP50_std": round(syntax_std, 4),
                "stenosis_AP50": round(sten_ap_mean, 4),
                "stenosis_AP50_std": round(sten_ap_std, 4),
                "stenosis_f1_mean": round(sten_f1_mean, 4),
                "stenosis_f1_std": round(sten_f1_std, 4),
                "overall_f1_mean": round(overall_f1_mean, 4),
                "overall_f1_std": round(overall_f1_std, 4),
                "n_seeds": len(seed_runs),
                "per_class": {
                    "stenosis": {
                        "f1": round(sten_f1_mean, 4),
                        "precision": 0.0,
                        "recall": 0.0,
                        "ap50": round(sten_ap_mean, 4),
                    },
                },
                "seed_runs": seed_names,
                "per_seed": {
                    seed_names[i]: {
                        "syntax_mAP50": round(syntax_mAPs[i], 4),
                        "stenosis_AP50": round(stenosis_APs[i], 4),
                        "stenosis_f1": round(stenosis_F1s[i], 4),
                        "overall_f1": round(overall_F1s[i], 4),
                    }
                    for i in range(len(seed_runs))
                },
            },
        },
    }

    # Replace existing summary (if any) and rewrite the merged file.
    # NOTE: we mutate the caller's ``merged`` list in place so that
    # subsequent aggregator calls (e.g. S18 after S15) see the freshly
    # written S15 entry.
    merged_by_name = {r["name"]: r for r in merged if isinstance(r, dict)}
    merged_by_name[summary_name] = summary
    new_merged = list(merged_by_name.values())
    merged[:] = new_merged
    with open(results_path, "w") as f:
        json.dump(new_merged, f, indent=2, default=str)

    print(f"\n  {summary_name} written "
          f"(n={len(seed_runs)}):")
    print(f"    syntax mAP50  : {syntax_mean:.4f} ± {syntax_std:.4f}")
    print(f"    stenosis AP50 : {sten_ap_mean:.4f} ± {sten_ap_std:.4f}")
    print(f"    stenosis F1   : {sten_f1_mean:.4f} ± {sten_f1_std:.4f}")
    print(f"    overall F1    : {overall_f1_mean:.4f} ± {overall_f1_std:.4f}")


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
    parser.add_argument(
        "--max-concurrent", type=int, default=0,
        help="Maximum concurrent subprocess runs. Default 0 = "
             "len(--gpus), i.e. one per GPU. Set to a smaller number "
             "if you want to artificially cap parallelism."
    )
    parser.add_argument(
        "--stagger-seconds", type=int, default=15,
        help="Seconds to wait between consecutive subprocess spawns "
             "(default: 15). Prevents CUDA init races when many heavy "
             "workers launch simultaneously. Set to 0 to disable."
    )
    parser.add_argument(
        "--min-free-mb", type=int, default=2000,
        help="Minimum free VRAM (MB) required on a GPU before spawning "
             "a worker there (default: 2000). Set to 0 to disable the "
             "pre-flight memory check."
    )
    parser.add_argument(
        "--reap-orphans", action="store_true",
        help="On startup, SIGTERM/SIGKILL any lingering compute "
             "processes on pool GPUs whose cmdline identifies them as "
             "stale workers from this script (ignored if cmdline "
             "doesn't match). Default: print orphans but don't kill."
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
        print(f"  advisory GPU {exp['gpu']}: {exp['name']} — {exp['description']}")

    # ── Pre-flight GPU sanity check ──
    print(f"\n  Initial GPU memory (nvidia-smi):")
    unhealthy = []
    for g in available_gpus:
        free = _gpu_free_mb(g)
        if free < 0:
            print(f"    GPU {g}: nvidia-smi unavailable (proceeding blind)")
        else:
            tag = " OK" if free >= args.min_free_mb else " LOW"
            print(f"    GPU {g}:{tag} free={free} MB")
            if free < args.min_free_mb and args.min_free_mb > 0:
                unhealthy.append(g)
    if unhealthy:
        print(f"  WARNING: GPUs {unhealthy} are below --min-free-mb "
              f"({args.min_free_mb}). The scheduler will wait for them "
              f"to free up (per-GPU timeout 10 min).")

    # ── Step 3: Launch all experiments as separate subprocesses ──
    total_start = time.time()

    results = launch_experiments_parallel(
        experiments, arcade_root, splits_dir, output_dir, args.iterations,
        max_concurrent=args.max_concurrent,
        stagger_seconds=args.stagger_seconds,
        min_free_mb=args.min_free_mb,
        gpu_pool=available_gpus,
        reap_orphans=args.reap_orphans,
    )

    total_elapsed = time.time() - total_start

    # ── Step 4: Report results ──
    print_results_table(results)

    # Save full results — append to existing file if present so that
    # results from earlier runs are preserved across sessions.
    results_path = output_dir / "strategy_results.json"
    existing = []
    if results_path.exists():
        try:
            with open(results_path) as f:
                existing = json.load(f)
            if not isinstance(existing, list):
                existing = [existing]
        except (json.JSONDecodeError, ValueError):
            existing = []

    # Merge: update entries with same name, append new ones
    existing_by_name = {r["name"]: r for r in existing}
    for r in results:
        existing_by_name[r["name"]] = r  # overwrite stale entry for same experiment
    merged = list(existing_by_name.values())

    with open(results_path, "w") as f:
        json.dump(merged, f, indent=2, default=str)

    # ── Multi-seed aggregators ──
    # If all seed runs for a given recipe are present in the merged
    # results, compute mean/std of the key metrics and append a
    # summary entry. These run after every session so they update as
    # seed runs complete across multiple invocations.
    _aggregate_multiseed(
        merged, results_path,
        summary_name="S15_multiseed_summary",
        seed_names=["S15_s8_seed42", "S15_s8_seed7", "S15_s8_seed2024"],
        description="Mean ± std of S8 recipe over 3 seeds "
                    "(42, 7, 2024) — variance baseline",
    )
    _aggregate_multiseed(
        merged, results_path,
        summary_name="S18_multiseed_summary",
        seed_names=["S18_s12_seed42", "S18_s12_seed7", "S18_s12_seed2024"],
        description="Mean ± std of S12 (yolo11l-seg) recipe over "
                    "3 seeds (42, 7, 2024) — confirms yolo11l gain",
    )

    print(f"\n{'=' * 100}")
    print(f"ALL EXPERIMENTS COMPLETE ({total_elapsed / 3600:.1f}h wall time)")
    print(f"Results saved: {results_path}")
    if existing:
        print(f"  (merged with {len(existing)} existing entries, "
              f"total {len(merged)} entries)")

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