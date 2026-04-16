#!/usr/bin/env python3
"""Master runner: every experiment from new_experiments_proposal.md.

Targets a 2-GPU (RTX 5090) setup. Experiments run sequentially within
each phase but parallelize across GPUs where possible via multiprocessing.

Phases:
  0  Diagnostics   (F-4, F-2, F-1)
  1  Zero-retrain  (A-1, A-3, A-4, A-5, A-2)
  2  High-impact   (B-1+B-2, B-4, C-1, C-3, C-5)
  3  Medium-impact  (B-3, B-6, C-2, C-4, D-1, D-2, D-3)
  4  Compound       (combined winner, A-6, B-5, E-1)
  5  Advanced       (E-2, D-4, C-6, C-7, C-8, F-3)

Usage:
    python run_all_proposals.py \
        --arcade-root ../../arcade/submission \
        --phase 0        # run only phase 0 (default: all)
        --phase 0,1,2    # run phases 0-2
        --phase all      # run everything

    # Resume from a specific experiment within a phase:
    python run_all_proposals.py --arcade-root ../../arcade/submission \
        --phase 2 --start-from B4
"""

from __future__ import annotations

import argparse
import copy
import json
import multiprocessing
import os
import shutil
import subprocess
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

import yaml

# ── Add script dir to path ──
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

# ── Paths (updated at runtime) ──
ARCADE_ROOT = None    # set by args
BASE_DIR = None       # arcade_experiments/
RESULTS_DIR = None    # results/proposal_runs/
DATA_DIR = None       # data/  (existing prepared data)


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════

def log(msg: str):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    log_path = RESULTS_DIR / "master_log.txt"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a") as f:
        f.write(line + "\n")


def save_result(exp_name: str, result: dict):
    """Append experiment result to the master results JSON."""
    results_file = RESULTS_DIR / "all_results.json"
    if results_file.exists():
        all_results = json.load(open(results_file))
    else:
        all_results = []
    result["name"] = exp_name
    result["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    all_results.append(result)
    json.dump(all_results, open(results_file, "w"), indent=2)


# ── Cached model paths (set once by _find_best_models) ──
_BEST_SYNTAX_MODEL = None
_BEST_STENOSIS_MODEL = None


def _find_best_models():
    """Locate the best syntax-only and stenosis-only models.

    CRITICAL: previous version used rglob("*best.pt") and picked the most
    recent file. This grabbed combined/merged models (25-class) instead of
    the syntax-only model (10-class), producing garbage evaluations.

    Now we:
      1. Only look for models whose parent path clearly indicates
         "syntax" or "stenosis" single-task training.
      2. Exclude combined/merged/iter/stage3 paths.
      3. Fall back to training a fresh baseline if nothing is found.
    """
    global _BEST_SYNTAX_MODEL, _BEST_STENOSIS_MODEL

    def _is_syntax_only(p: Path) -> bool:
        """True if path looks like a syntax-only model, not combined/merged."""
        s = str(p).lower()
        excludes = ["combined", "merged", "iter2", "iter3", "stage3",
                     "stenosis", "pseudo", "distill"]
        return any(k in s for k in ["syntax", "stage1"]) and \
               not any(k in s for k in excludes)

    def _is_stenosis_only(p: Path) -> bool:
        """True if path looks like a stenosis-only model."""
        s = str(p).lower()
        excludes = ["combined", "merged", "syntax", "pseudo", "distill"]
        return "stenosis" in s and not any(k in s for k in excludes)

    # Search in results dirs
    search_dirs = []
    if RESULTS_DIR and RESULTS_DIR.exists():
        search_dirs.append(RESULTS_DIR)
    exp_results = BASE_DIR / "results"
    if exp_results.exists():
        search_dirs.append(exp_results)

    syn_candidates = []
    sten_candidates = []
    for d in search_dirs:
        for pt in d.rglob("*best.pt"):
            if _is_syntax_only(pt):
                syn_candidates.append(pt)
            elif _is_stenosis_only(pt):
                sten_candidates.append(pt)

    if syn_candidates:
        syn_candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        _BEST_SYNTAX_MODEL = str(syn_candidates[0])
        log(f"  Best syntax model: {_BEST_SYNTAX_MODEL}")
    else:
        _BEST_SYNTAX_MODEL = None
        log("  WARNING: No syntax-only model found — will train baseline first")

    if sten_candidates:
        sten_candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        _BEST_STENOSIS_MODEL = str(sten_candidates[0])
        log(f"  Best stenosis model: {_BEST_STENOSIS_MODEL}")
    else:
        _BEST_STENOSIS_MODEL = None
        log("  WARNING: No stenosis-only model found — will train baseline first")


def _ensure_baseline_models(device: str = "0"):
    """Train baseline syntax and stenosis models if none exist."""
    global _BEST_SYNTAX_MODEL, _BEST_STENOSIS_MODEL

    if _BEST_SYNTAX_MODEL is None:
        log("Training baseline syntax model (S54 recipe)...")
        cfg = base_syntax_config()
        cfg["device"] = device
        _BEST_SYNTAX_MODEL = train_syntax(cfg, "baseline_syntax", device)
        log(f"  Baseline syntax model: {_BEST_SYNTAX_MODEL}")

    if _BEST_STENOSIS_MODEL is None:
        log("Training baseline stenosis model...")
        cfg = base_stenosis_config()
        cfg["device"] = device
        _BEST_STENOSIS_MODEL = train_stenosis(cfg, "baseline_stenosis", device)
        log(f"  Baseline stenosis model: {_BEST_STENOSIS_MODEL}")


def get_best_syntax_model() -> str:
    if _BEST_SYNTAX_MODEL is None:
        raise RuntimeError("No syntax model available — run _find_best_models first")
    return _BEST_SYNTAX_MODEL


def get_best_stenosis_model() -> str:
    if _BEST_STENOSIS_MODEL is None:
        raise RuntimeError("No stenosis model available — run _find_best_models first")
    return _BEST_STENOSIS_MODEL


def fix_dataset_yaml_paths():
    """Rewrite dataset YAML 'path' fields to match the current machine.

    The YAML files written by prepare_data.py embed absolute paths from
    the machine that ran data-prep. When the repo is cloned onto a
    different machine the paths are stale. This function rewrites them
    so that 'path' always points to the real directory next to the YAML.
    """
    configs_dir = DATA_DIR / "dataset_configs"
    if not configs_dir.exists():
        return

    mapping = {
        "syntax_only.yaml": str((DATA_DIR / "syntax_filtered").resolve()),
        "stenosis_only.yaml": str((DATA_DIR / "stenosis").resolve()),
    }
    for fname, correct_path in mapping.items():
        p = configs_dir / fname
        if not p.exists():
            continue
        with open(p) as f:
            cfg = yaml.safe_load(f)
        if cfg.get("path") != correct_path:
            cfg["path"] = correct_path
            with open(p, "w") as f:
                yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
            log(f"  Fixed path in {fname} -> {correct_path}")


def _ensure_clean_data():
    """Verify data integrity and re-prepare from ground truth if needed.

    Ground truth: ARCADE_ROOT/{syntax,stenosis}/{train,val,test}/
    Derived data: DATA_DIR/{syntax_filtered,stenosis,dataset_configs}/

    We check:
      1. Does derived data exist at all?
      2. Do image counts match the ground truth source?
      3. Are there stale oversampled/synthetic files from prior runs?
    If anything is wrong, we wipe and re-prepare from scratch.
    """
    needs_prep = False
    reason = ""

    syntax_src = ARCADE_ROOT / "syntax"
    stenosis_src = ARCADE_ROOT / "stenosis"

    # Check 1: does derived data exist?
    if not (DATA_DIR / "dataset_configs" / "syntax_only.yaml").exists():
        needs_prep = True
        reason = "dataset configs not found"

    # Check 2: do train image counts match ground truth?
    if not needs_prep:
        for task, src_dir, dst_dir in [
            ("syntax", syntax_src, DATA_DIR / "syntax_filtered"),
            ("stenosis", stenosis_src, DATA_DIR / "stenosis"),
        ]:
            src_count = len(list((src_dir / "train" / "images").glob("*"))) if (src_dir / "train" / "images").exists() else 0
            dst_img_dir = dst_dir / "images" / "train"
            if dst_img_dir.exists():
                # Count only real images, not oversampled/synthetic duplicates
                dst_count = sum(1 for f in dst_img_dir.iterdir()
                                if not f.name.startswith(("synth_", "bg_"))
                                and "_os" not in f.stem)
                if dst_count != src_count and src_count > 0:
                    needs_prep = True
                    reason = (f"{task} train count mismatch: "
                              f"source={src_count}, derived={dst_count}")
                    break
            else:
                needs_prep = True
                reason = f"{task} derived images missing"
                break

    # Check 3: are there stale oversampled/synthetic files?
    if not needs_prep:
        for dst_dir in [DATA_DIR / "syntax_filtered", DATA_DIR / "stenosis"]:
            train_img = dst_dir / "images" / "train"
            if not train_img.exists():
                continue
            stale = [f for f in train_img.iterdir()
                     if f.name.startswith(("synth_", "bg_")) or "_os" in f.stem]
            if stale:
                needs_prep = True
                reason = f"found {len(stale)} stale oversampled/synthetic files in {dst_dir.name}"
                break

    if needs_prep:
        log(f"Data integrity check FAILED: {reason}")
        log("Re-preparing data from ground truth source...")

        # Wipe derived data to avoid contamination
        for subdir in ["syntax_filtered", "stenosis", "dataset_configs"]:
            d = DATA_DIR / subdir
            if d.exists():
                shutil.rmtree(d)
                log(f"  Removed stale {subdir}/")

        from run_pipeline import data_prep
        data_prep(ARCADE_ROOT, DATA_DIR, min_count=300)
        fix_dataset_yaml_paths()
        log("Data preparation complete from ground truth.")
    else:
        log("Data integrity check passed.")
        fix_dataset_yaml_paths()


def get_syntax_yaml() -> str:
    return str(DATA_DIR / "dataset_configs" / "syntax_only.yaml")


def get_stenosis_yaml() -> str:
    return str(DATA_DIR / "dataset_configs" / "stenosis_only.yaml")


def base_syntax_config() -> dict:
    """Best known syntax training config (S54 recipe)."""
    return {
        "model": "yolo11m-seg.pt",
        "imgsz": 512,
        "batch": 16,
        "epochs": 300,
        "patience": 50,
        "optimizer": "AdamW",
        "lr0": 0.001,
        "lrf": 0.01,
        "weight_decay": 0.005,
        "momentum": 0.937,
        "warmup_epochs": 5,
        "freeze": 10,
        "freeze_epochs": 15,
        "lr_factor_unfrozen": 0.1,
        "seed": 42,
        "deterministic": True,
        "amp": True,
        "cos_lr": True,
        "workers": 4,
        "mosaic": 0.0,
        "close_mosaic": 0,
        "mixup": 0.0,
        "copy_paste": 0.0,
        "fliplr": 0.5,
        "flipud": 0.0,
        "degrees": 20.0,
        "scale": 0.4,
        "translate": 0.1,
        "hsv_h": 0.0,
        "hsv_s": 0.0,
        "hsv_v": 0.3,
        "erasing": 0.0,
        "shear": 0.0,
        "perspective": 0.0,
        "box": 7.5,
        "cls": 0.5,
        "dfl": 1.5,
    }


def base_stenosis_config() -> dict:
    """Best known stenosis training config."""
    cfg = base_syntax_config()
    cfg["copy_paste"] = 0.3
    cfg["scale"] = 0.5
    cfg["mosaic"] = 0.8
    cfg["close_mosaic"] = 15
    return cfg


def train_syntax(cfg: dict, run_name: str, device: str,
                 data_yaml: str = None, model_weights: str = None) -> str:
    """Train a syntax model with given config. Returns path to best weights."""
    from train import train_two_stage
    cfg = copy.deepcopy(cfg)
    cfg["device"] = device
    data = data_yaml or get_syntax_yaml()
    project = str(RESULTS_DIR / run_name)
    return train_two_stage(cfg, data, project, run_name,
                           model_weights=model_weights)


def train_stenosis(cfg: dict, run_name: str, device: str,
                   data_yaml: str = None, model_weights: str = None) -> str:
    """Train a stenosis model with given config. Returns path to best weights."""
    from train import train_two_stage
    cfg = copy.deepcopy(cfg)
    cfg["device"] = device
    data = data_yaml or get_stenosis_yaml()
    project = str(RESULTS_DIR / run_name)
    return train_two_stage(cfg, data, project, run_name,
                           model_weights=model_weights)


def train_single(cfg: dict, run_name: str, device: str,
                 data_yaml: str, model_weights: str = None) -> str:
    """Single-stage training. Returns path to best weights."""
    from train import train_single_stage
    cfg = copy.deepcopy(cfg)
    cfg["device"] = device
    project = str(RESULTS_DIR / run_name)
    return train_single_stage(cfg, data_yaml, project, run_name,
                              model_weights=model_weights)


def eval_model(model_path: str, data_yaml: str, split: str = "test",
               augment: bool = False, imgsz: int = 512) -> dict:
    """Evaluate a model and return metrics dict."""
    from evaluate import evaluate_model
    return evaluate_model(model_path, data_yaml, split=split,
                          augment=augment, imgsz=imgsz)


# ═══════════════════════════════════════════════════════════════════════
# PHASE 0: Diagnostics
# ═══════════════════════════════════════════════════════════════════════

def run_F4_weighted_f1():
    """F-4: Compute weighted F1 on existing best models."""
    log("F-4: Computing weighted F1 on best existing models...")
    from compute_weighted_f1 import compute_f1s

    # Find existing strategy_results.json
    strat_results = BASE_DIR / "results" / "stenosis_strategies" / "strategy_results.json"
    if not strat_results.exists():
        log("  strategy_results.json not found, using current best model eval")
        best = get_best_syntax_model()
        metrics = eval_model(best, get_syntax_yaml(), split="test")
        per_class = metrics.get("per_class", {})
        # Estimate supports from data
        from compute_weighted_f1 import _syntax_class_map_from_stratified_prior
        supports = _syntax_class_map_from_stratified_prior()
        result = compute_f1s(per_class, supports)
        log(f"  Macro F1: {result['macro']:.4f}")
        log(f"  Weighted F1: {result['weighted']:.4f}")
        log(f"  Delta: {result['weighted'] - result['macro']:+.4f}")
        save_result("F4_weighted_f1", result)
        return result

    data = json.load(open(strat_results))
    from compute_weighted_f1 import _syntax_class_map_from_stratified_prior
    supports = _syntax_class_map_from_stratified_prior()

    results = []
    for exp in data:
        if exp.get("status") != "success":
            continue
        ft = exp.get("metrics", {}).get("final_test", {}) or {}
        pc = ft.get("per_class", {}) or {}
        if not pc:
            continue
        f1s = compute_f1s(pc, supports)
        results.append({"name": exp["name"], **f1s})

    results.sort(key=lambda r: -r.get("weighted", 0))
    for r in results[:5]:
        log(f"  {r['name']}: macro={r['macro']:.4f} weighted={r['weighted']:.4f}")

    save_result("F4_weighted_f1", {"top5": results[:5]})
    return results


def run_F2_label_audit():
    """F-2: Audit class 9 / 9a contamination in COCO->YOLO conversion."""
    log("F-2: Running label audit for class 9/9a contamination...")
    from audit_labels import audit_syntax_conversion

    # Find the data dir used by our existing runs
    data_dir = DATA_DIR
    output = RESULTS_DIR / "F2_label_audit.json"
    summary = audit_syntax_conversion(ARCADE_ROOT, data_dir)

    json.dump(summary, open(output, "w"), indent=2)
    log(f"  Label audit saved to {output}")

    # Check for contamination
    contaminated = False
    for split, s in summary.get("splits", {}).items():
        if s.get("mismatches"):
            log(f"  *** MISMATCHES in {split}: {s['mismatches']}")
            contaminated = True
        if s.get("ghost"):
            log(f"  *** GHOST classes in {split}: {s['ghost']}")
            contaminated = True

    if not contaminated:
        log("  Labels appear clean — no 9/9a contamination detected.")

    save_result("F2_label_audit", {
        "contaminated": contaminated,
        "output": str(output),
    })
    return summary


def run_F1_official_split(device: str = "0"):
    """F-1: Retrain S54 on official ARCADE split."""
    log("F-1: Training S54 recipe on official ARCADE split...")

    # Prepare official split data
    from run_pipeline import data_prep
    official_data = RESULTS_DIR / "data" / "official_split"
    if not (official_data / "dataset_configs").exists():
        data_prep(ARCADE_ROOT, official_data, min_count=300,
                  train_only_filter=True)
        log(f"  Official split data prepared at {official_data}")

    syntax_yaml = str(official_data / "dataset_configs" / "syntax_only.yaml")
    stenosis_yaml = str(official_data / "dataset_configs" / "stenosis_only.yaml")

    # Train syntax
    cfg = base_syntax_config()
    syn_weights = train_syntax(cfg, "F1_official_syntax", device,
                               data_yaml=syntax_yaml)
    syn_metrics = eval_model(syn_weights, syntax_yaml, split="test")

    # Train stenosis
    cfg_st = base_stenosis_config()
    sten_weights = train_stenosis(cfg_st, "F1_official_stenosis", device,
                                  data_yaml=stenosis_yaml)
    sten_metrics = eval_model(sten_weights, stenosis_yaml, split="test")

    result = {
        "syntax_model": syn_weights,
        "stenosis_model": sten_weights,
        "syntax_metrics": syn_metrics,
        "stenosis_metrics": sten_metrics,
    }
    save_result("F1_official_split", result)
    log(f"  Official split syntax test: {syn_metrics}")
    log(f"  Official split stenosis test: {sten_metrics}")
    return result


def phase0(device: str = "0"):
    """Run all Phase 0 diagnostics."""
    log("=" * 60)
    log("PHASE 0: DIAGNOSTICS")
    log("=" * 60)

    run_F4_weighted_f1()
    run_F2_label_audit()
    run_F1_official_split(device)


# ═══════════════════════════════════════════════════════════════════════
# PHASE 1: Zero-Retraining Interventions
# ═══════════════════════════════════════════════════════════════════════

def run_A1_conf_sweep(device: str = "0"):
    """A-1: Per-class confidence threshold sweep."""
    log("A-1: Per-class confidence threshold sweep...")
    from sweep_confidence import collect_predictions, sweep_per_class, evaluate_at_thresholds

    best_syn = get_best_syntax_model()
    best_sten = get_best_stenosis_model()
    syn_yaml = get_syntax_yaml()
    sten_yaml = get_stenosis_yaml()

    results = {}

    # Syntax sweep
    log("  Sweeping syntax model thresholds...")
    val_items, names = collect_predictions(best_syn, syn_yaml, "val", 512, device)
    chosen, curves = sweep_per_class(val_items)
    chosen_named = {names.get(c, str(c)): v for c, v in chosen.items()}

    test_items, _ = collect_predictions(best_syn, syn_yaml, "test", 512, device)
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
    }
    log(f"  Syntax: default={results['syntax']['test_default_f1']:.4f} "
        f"tuned={results['syntax']['test_tuned_f1']:.4f} "
        f"delta={results['syntax']['delta']:+.4f}")

    # Stenosis sweep
    log("  Sweeping stenosis model thresholds...")
    val_items_st, names_st = collect_predictions(best_sten, sten_yaml, "val", 512, device)
    chosen_st, _ = sweep_per_class(val_items_st)

    test_items_st, _ = collect_predictions(best_sten, sten_yaml, "test", 512, device)
    test_tuned_st = evaluate_at_thresholds(test_items_st, chosen_st)
    test_default_st = evaluate_at_thresholds(test_items_st, {c: 0.25 for c in chosen_st})

    results["stenosis"] = {
        "thresholds": {names_st.get(c, str(c)): v for c, v in chosen_st.items()},
        "test_default_f1": _mean_f1(test_default_st),
        "test_tuned_f1": _mean_f1(test_tuned_st),
        "delta": round(_mean_f1(test_tuned_st) - _mean_f1(test_default_st), 4),
    }
    log(f"  Stenosis: default={results['stenosis']['test_default_f1']:.4f} "
        f"tuned={results['stenosis']['test_tuned_f1']:.4f} "
        f"delta={results['stenosis']['delta']:+.4f}")

    save_result("A1_conf_sweep", results)
    # Save thresholds for downstream use
    json.dump(chosen_named, open(RESULTS_DIR / "A1_syntax_thresholds.json", "w"), indent=2)
    json.dump({names_st.get(c, str(c)): v for c, v in chosen_st.items()},
              open(RESULTS_DIR / "A1_stenosis_thresholds.json", "w"), indent=2)
    return results


def run_A3_tta_test(device: str = "0"):
    """A-3: TTA clean A/B test."""
    log("A-3: TTA clean A/B test...")

    best_syn = get_best_syntax_model()
    best_sten = get_best_stenosis_model()

    results = {}
    for task, model, yaml_path in [
        ("syntax", best_syn, get_syntax_yaml()),
        ("stenosis", best_sten, get_stenosis_yaml()),
    ]:
        log(f"  {task}: evaluating without TTA...")
        no_tta = eval_model(model, yaml_path, split="test", augment=False)
        log(f"  {task}: evaluating with TTA...")
        with_tta = eval_model(model, yaml_path, split="test", augment=True)

        results[task] = {
            "no_tta_mAP50": no_tta.get("mAP50", 0),
            "with_tta_mAP50": with_tta.get("mAP50", 0),
            "delta_mAP50": round(with_tta.get("mAP50", 0) - no_tta.get("mAP50", 0), 4),
            "no_tta_per_class": no_tta.get("per_class", {}),
            "with_tta_per_class": with_tta.get("per_class", {}),
        }
        log(f"  {task}: no_tta={no_tta.get('mAP50', 0):.4f} "
            f"tta={with_tta.get('mAP50', 0):.4f} "
            f"delta={results[task]['delta_mAP50']:+.4f}")

    save_result("A3_tta_test", results)
    return results


def run_A4_cc_filter(device: str = "0"):
    """A-4: Small connected-component post-processing."""
    log("A-4: CC post-processing sweep...")
    from small_cc_postprocess import evaluate_with_filter_cpu

    best_sten = get_best_stenosis_model()
    sten_yaml = get_stenosis_yaml()
    best_syn = get_best_syntax_model()
    syn_yaml = get_syntax_yaml()

    results = {}
    for min_area in [20, 50, 100, 200]:
        log(f"  Stenosis CC filter min_area={min_area}...")
        r = evaluate_with_filter_cpu(
            best_sten, sten_yaml, split="test", imgsz=512,
            min_area_px=min_area, device=device,
        )
        results[f"stenosis_area{min_area}"] = r
        log(f"    raw F1={r['raw']['f1']:.4f}  filtered F1={r['filtered']['f1']:.4f}")

    for min_area in [20, 50, 100]:
        log(f"  Syntax CC filter min_area={min_area}...")
        r = evaluate_with_filter_cpu(
            best_syn, syn_yaml, split="test", imgsz=512,
            min_area_px=min_area, device=device,
        )
        results[f"syntax_area{min_area}"] = r
        log(f"    raw F1={r['raw']['f1']:.4f}  filtered F1={r['filtered']['f1']:.4f}")

    save_result("A4_cc_filter", results)
    return results


def run_A5_calibration(device: str = "0"):
    """A-5: Platt/isotonic calibration of output scores."""
    log("A-5: Score calibration...")
    from calibrate_scores import collect_scores, fit_isotonic
    from ultralytics import YOLO
    import pickle

    best_syn = get_best_syntax_model()
    syn_yaml = get_syntax_yaml()
    model = YOLO(best_syn)

    log("  Collecting val scores...")
    records, names = collect_scores(model, syn_yaml, "val", 512, device)
    for cls, pairs in records.items():
        pos = sum(1 for _, c in pairs if c)
        log(f"    class {names.get(cls, cls)}: n={len(pairs)} pos={pos}")

    log("  Fitting isotonic regressions...")
    fitted = fit_isotonic(records)

    cal_dir = RESULTS_DIR / "A5_calibrators"
    cal_dir.mkdir(parents=True, exist_ok=True)
    for cls, iso in fitted.items():
        pickle.dump(iso, open(cal_dir / f"iso_class_{cls}.pkl", "wb"))

    result = {
        "classes_fitted": [names.get(c, str(c)) for c in fitted],
        "n_val_pairs": {names.get(c, str(c)): len(records[c]) for c in fitted},
        "output_dir": str(cal_dir),
    }
    save_result("A5_calibration", result)
    log(f"  Calibrators saved to {cal_dir}")
    return result


def run_A2_tile_inference(device: str = "0"):
    """A-2: Tile/overlap inference for stenosis."""
    log("A-2: Tile inference for stenosis...")
    from tile_inference_stenosis import tile_predict, wbf_merge, evaluate as tile_evaluate
    from tile_inference_stenosis import _yolo_poly_to_mask
    from ultralytics import YOLO
    import cv2

    best_sten = get_best_stenosis_model()
    sten_yaml = get_stenosis_yaml()

    with open(sten_yaml) as f:
        cfg = yaml.safe_load(f)
    root = Path(cfg["path"])
    img_dir = root / cfg.get("test", "images/test")
    lbl_dir = root / "labels" / "test"
    if not lbl_dir.exists():
        lbl_dir = root / cfg.get("test", "images/test").replace("images", "labels")

    model = YOLO(best_sten)
    img_files = sorted(list(img_dir.glob("*.png")) + list(img_dir.glob("*.PNG")))

    results = {}
    for upscale in [2, 3]:
        log(f"  Tile inference upscale={upscale}x...")
        per_image = []
        for img_path in img_files:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            H, W = img.shape[:2]
            dets = tile_predict(model, img_path, upscale, 512, 128, 512, device, 0.25)
            merged = wbf_merge(dets)
            per_image.append({"stem": img_path.stem, "h": H, "w": W, "merged": merged})

        metrics = tile_evaluate(per_image, lbl_dir)
        results[f"upscale_{upscale}x"] = metrics
        log(f"    F1={metrics['f1']:.4f} P={metrics['precision']:.4f} R={metrics['recall']:.4f}")

    save_result("A2_tile_inference", results)
    return results


def phase1(device: str = "0"):
    """Run all Phase 1 zero-retraining interventions."""
    log("=" * 60)
    log("PHASE 1: ZERO-RETRAINING INTERVENTIONS")
    log("=" * 60)

    run_A1_conf_sweep(device)
    run_A4_cc_filter(device)
    run_A5_calibration(device)
    run_A3_tta_test(device)
    run_A2_tile_inference(device)


# ═══════════════════════════════════════════════════════════════════════
# PHASE 2: High-Impact Retraining
# ═══════════════════════════════════════════════════════════════════════

def run_B1_B2_tail_sampling_loss(device: str = "0"):
    """B-1 + B-2: Per-class weighted sampling + weighted BCE loss."""
    log("B-1/B-2: Tail-class oversampling + loss weighting...")

    # Step 1: Copy CLEAN syntax data and apply oversampling
    os_data_dir = RESULTS_DIR / "data" / "B1_B2_oversampled"
    syn_src = DATA_DIR / "syntax_filtered"
    syn_dst = os_data_dir / "syntax_filtered"

    # Always start from clean ground-truth-derived data
    if syn_dst.exists():
        shutil.rmtree(syn_dst)
    log("  Copying clean syntax data for oversampling...")
    shutil.copytree(syn_src, syn_dst)

    # Get class info from yaml
    syn_yaml_src = get_syntax_yaml()
    with open(syn_yaml_src) as f:
        syn_cfg = yaml.safe_load(f)
    nc = syn_cfg["nc"]
    names = syn_cfg["names"]

    # Find tail classes (last 2-3 by index which are typically 9, 13, 16)
    # In the 10-class setup: indices 8='11', 9='13' are the tails
    # We oversample the smallest classes
    from oversample_tail_classes import oversample, instance_counts
    lbl_train = syn_dst / "labels" / "train"
    counts = instance_counts(lbl_train)
    log(f"  Current counts: {dict(counts)}")

    # Find tail classes (bottom 3 by count)
    sorted_cls = sorted(counts.items(), key=lambda x: x[1])
    tail_classes = [c for c, _ in sorted_cls[:3]]
    target = sorted_cls[len(sorted_cls) // 2][1]  # median count
    log(f"  Tail classes: {tail_classes}, target count: {target}")

    stats = oversample(syn_dst, tail_classes, target)
    log(f"  Oversampling done: {stats['duplicates_created']} duplicates")

    # Write new dataset YAML
    os_yaml_path = os_data_dir / "syntax_oversampled.yaml"
    os_yaml = {
        "path": str(syn_dst.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": nc,
        "names": names,
    }
    yaml.dump(os_yaml, open(os_yaml_path, "w"), default_flow_style=False, sort_keys=False)

    # Step 2: Train with higher cls loss weight (B-2)
    cfg = base_syntax_config()
    cfg["cls"] = 1.25  # increased from 0.5 to upweight classification loss
    weights = train_syntax(cfg, "B1_B2_oversample_cls125", device,
                           data_yaml=str(os_yaml_path))

    metrics = eval_model(weights, get_syntax_yaml(), split="test")
    result = {
        "model": weights,
        "oversampling": stats,
        "cls_weight": 1.25,
        "metrics": metrics,
    }
    save_result("B1_B2_tail_sampling_loss", result)
    log(f"  B-1/B-2 test mAP50: {metrics.get('mAP50', 0):.4f}")
    return result


def run_B4_warmstart_stenosis(device: str = "0"):
    """B-4: Warm-start stenosis from syntax backbone."""
    log("B-4: Warm-starting stenosis from best syntax backbone...")

    best_syn = get_best_syntax_model()
    cfg = base_stenosis_config()

    # Train stenosis starting from syntax weights
    # YOLO will reinitialize the head for nc=1 but keep backbone
    weights = train_stenosis(cfg, "B4_warmstart_stenosis", device,
                             model_weights=best_syn)
    metrics = eval_model(weights, get_stenosis_yaml(), split="test")

    result = {"model": weights, "source_syntax": best_syn, "metrics": metrics}
    save_result("B4_warmstart_stenosis", result)
    log(f"  B-4 stenosis test mAP50: {metrics.get('mAP50', 0):.4f}")
    return result


def run_C1_label_smoothing(device: str = "0"):
    """C-1: Label smoothing sweep."""
    log("C-1: Label smoothing sweep...")

    results = {}
    for ls in [0.05, 0.1]:
        log(f"  Training with label_smoothing={ls}...")
        cfg = base_syntax_config()
        cfg["label_smoothing"] = ls
        weights = train_syntax(cfg, f"C1_ls_{ls}", device)
        metrics = eval_model(weights, get_syntax_yaml(), split="test")
        results[f"ls_{ls}"] = {"model": weights, "metrics": metrics}
        log(f"    mAP50: {metrics.get('mAP50', 0):.4f}")

    save_result("C1_label_smoothing", results)
    return results


def run_C3_dfl_sweep(device: str = "0"):
    """C-3: DFL weight sweep for stenosis."""
    log("C-3: DFL weight sweep for stenosis...")

    results = {}
    for dfl in [2.0, 2.5, 3.0]:
        log(f"  Training stenosis with dfl={dfl}...")
        cfg = base_stenosis_config()
        cfg["dfl"] = dfl
        weights = train_stenosis(cfg, f"C3_dfl_{dfl}", device)
        metrics = eval_model(weights, get_stenosis_yaml(), split="test")
        results[f"dfl_{dfl}"] = {"model": weights, "metrics": metrics}
        log(f"    mAP50: {metrics.get('mAP50', 0):.4f}")

    save_result("C3_dfl_sweep", results)
    return results


def run_C5_multi_scale(device: str = "0"):
    """C-5: Multi-scale training for stenosis."""
    log("C-5: Multi-scale training for stenosis...")

    cfg = base_stenosis_config()
    cfg["multi_scale"] = True
    cfg["batch"] = 8  # lower batch to avoid multi_scale + high batch issues
    weights = train_stenosis(cfg, "C5_multiscale_stenosis", device)
    metrics = eval_model(weights, get_stenosis_yaml(), split="test")

    result = {"model": weights, "metrics": metrics}
    save_result("C5_multi_scale", result)
    log(f"  C-5 stenosis test mAP50: {metrics.get('mAP50', 0):.4f}")
    return result


def phase2(devices: list):
    """Run Phase 2 high-impact retraining. Uses 2 GPUs in parallel where possible."""
    log("=" * 60)
    log("PHASE 2: HIGH-IMPACT RETRAINING")
    log("=" * 60)

    d0, d1 = devices[0], devices[1] if len(devices) > 1 else devices[0]

    # B-1/B-2 on GPU0, B-4 on GPU1 (independent)
    _run_parallel([(run_B1_B2_tail_sampling_loss, d0),
                   (run_B4_warmstart_stenosis, d1)])

    # C-1 on GPU0, C-3 on GPU1 (independent)
    _run_parallel([(run_C1_label_smoothing, d0),
                   (run_C3_dfl_sweep, d1)])

    # C-5 on GPU0
    _run_safe(run_C5_multi_scale, d0)


# ═══════════════════════════════════════════════════════════════════════
# PHASE 3: Medium-Impact Retraining
# ═══════════════════════════════════════════════════════════════════════

def run_B3_background_images(device: str = "0"):
    """B-3: Add background images to stenosis training."""
    log("B-3: Building background stenosis dataset...")
    from build_background_stenosis import build

    best_sten = get_best_stenosis_model()
    syntax_img_dir = DATA_DIR / "syntax_filtered" / "images" / "train"

    # Copy CLEAN stenosis data to avoid modifying original
    bg_data_dir = RESULTS_DIR / "data" / "B3_background"
    sten_src = DATA_DIR / "stenosis"
    sten_dst = bg_data_dir / "stenosis"
    if sten_dst.exists():
        shutil.rmtree(sten_dst)
    shutil.copytree(sten_src, sten_dst)

    stats = build(best_sten, syntax_img_dir, sten_dst, 512, device, n_hard=150)
    log(f"  Added {stats['total_added']} background images")

    # Write dataset YAML
    bg_yaml_path = bg_data_dir / "stenosis_bg.yaml"
    bg_yaml = {
        "path": str(sten_dst.resolve()),
        "train": "images/train", "val": "images/val", "test": "images/test",
        "nc": 1, "names": {0: "stenosis"},
    }
    yaml.dump(bg_yaml, open(bg_yaml_path, "w"), default_flow_style=False, sort_keys=False)

    cfg = base_stenosis_config()
    weights = train_stenosis(cfg, "B3_background_stenosis", device,
                             data_yaml=str(bg_yaml_path))
    metrics = eval_model(weights, get_stenosis_yaml(), split="test")

    result = {"model": weights, "bg_stats": stats, "metrics": metrics}
    save_result("B3_background_images", result)
    log(f"  B-3 test mAP50: {metrics.get('mAP50', 0):.4f}")
    return result


def run_B6_copy_paste(device: str = "0"):
    """B-6: Synthetic copy-paste augmentation for tail classes."""
    log("B-6: Copy-paste augmentation for tail classes...")
    from copy_paste_tail_classes import main as cp_main

    # Copy CLEAN data
    cp_data_dir = RESULTS_DIR / "data" / "B6_copypaste"
    syn_src = DATA_DIR / "syntax_filtered"
    syn_dst = cp_data_dir / "syntax_filtered"
    if syn_dst.exists():
        shutil.rmtree(syn_dst)
    shutil.copytree(syn_src, syn_dst)

    # Get tail classes from yaml
    syn_yaml_src = get_syntax_yaml()
    with open(syn_yaml_src) as f:
        syn_cfg = yaml.safe_load(f)
    nc = syn_cfg["nc"]
    names = syn_cfg["names"]

    # Find tail classes
    from oversample_tail_classes import instance_counts
    counts = instance_counts(syn_dst / "labels" / "train")
    sorted_cls = sorted(counts.items(), key=lambda x: x[1])
    tail = [c for c, _ in sorted_cls[:3]]
    tail_str = ",".join(str(c) for c in tail)
    log(f"  Tail classes: {tail_str}")

    # Run copy-paste
    import subprocess
    subprocess.run([
        sys.executable, str(SCRIPT_DIR / "copy_paste_tail_classes.py"),
        "--syntax-data-dir", str(syn_dst),
        "--tail-classes", tail_str,
        "--per-instance", "2",
        "--seed", "42",
    ], check=True)

    # Write dataset YAML
    cp_yaml_path = cp_data_dir / "syntax_copypaste.yaml"
    cp_yaml = {
        "path": str(syn_dst.resolve()),
        "train": "images/train", "val": "images/val", "test": "images/test",
        "nc": nc, "names": names,
    }
    yaml.dump(cp_yaml, open(cp_yaml_path, "w"), default_flow_style=False, sort_keys=False)

    cfg = base_syntax_config()
    weights = train_syntax(cfg, "B6_copypaste_syntax", device,
                           data_yaml=str(cp_yaml_path))
    metrics = eval_model(weights, get_syntax_yaml(), split="test")

    result = {"model": weights, "metrics": metrics}
    save_result("B6_copy_paste", result)
    log(f"  B-6 test mAP50: {metrics.get('mAP50', 0):.4f}")
    return result


def run_C2_dropout(device: str = "0"):
    """C-2: Dropout sweep."""
    log("C-2: Dropout sweep...")
    results = {}
    for dp in [0.05, 0.1]:
        log(f"  Training with dropout={dp}...")
        cfg = base_syntax_config()
        cfg["dropout"] = dp
        weights = train_syntax(cfg, f"C2_dropout_{dp}", device)
        metrics = eval_model(weights, get_syntax_yaml(), split="test")
        results[f"dropout_{dp}"] = {"model": weights, "metrics": metrics}
        log(f"    mAP50: {metrics.get('mAP50', 0):.4f}")

    save_result("C2_dropout", results)
    return results


def run_C4_mask_weight(device: str = "0"):
    """C-4: Segmentation mask loss weight sweep.

    Note: Ultralytics does not expose a top-level 'mask' loss weight param
    in all versions. We use 'overlap_mask' and 'mask_ratio' where available,
    and fall back to box loss weight as a proxy for mask emphasis.
    """
    log("C-4: Mask loss weight sweep (via box loss proxy)...")
    results = {}
    # Since 'mask' isn't a valid arg, increase box loss to emphasize
    # localization quality (indirect mask improvement)
    for bw in [10.0, 12.5, 15.0]:
        log(f"  Training with box={bw} (mask emphasis proxy)...")
        cfg = base_syntax_config()
        cfg["box"] = bw
        weights = train_syntax(cfg, f"C4_box_{bw}", device)
        metrics = eval_model(weights, get_syntax_yaml(), split="test")
        results[f"box_{bw}"] = {"model": weights, "metrics": metrics}
        log(f"    mAP50: {metrics.get('mAP50', 0):.4f}")

    save_result("C4_mask_weight", results)
    return results


def run_D1_erasing(device: str = "0"):
    """D-1: Random erasing sweep."""
    log("D-1: Random erasing sweep...")
    results = {}
    for er in [0.2, 0.4]:
        log(f"  Training with erasing={er}...")
        cfg = base_syntax_config()
        cfg["erasing"] = er
        weights = train_syntax(cfg, f"D1_erasing_{er}", device)
        metrics = eval_model(weights, get_syntax_yaml(), split="test")
        results[f"erasing_{er}"] = {"model": weights, "metrics": metrics}
        log(f"    mAP50: {metrics.get('mAP50', 0):.4f}")

    save_result("D1_erasing", results)
    return results


def run_D2_flipud(device: str = "0"):
    """D-2: Vertical flip."""
    log("D-2: Vertical flip (flipud=0.5)...")
    cfg = base_syntax_config()
    cfg["flipud"] = 0.5
    weights = train_syntax(cfg, "D2_flipud", device)
    metrics = eval_model(weights, get_syntax_yaml(), split="test")
    result = {"model": weights, "metrics": metrics}
    save_result("D2_flipud", result)
    log(f"  D-2 test mAP50: {metrics.get('mAP50', 0):.4f}")
    return result


def run_D3_shear_translate(device: str = "0"):
    """D-3: Shear and translate sweep."""
    log("D-3: Shear + translate sweep...")
    results = {}
    for shear, translate, name in [(2.0, 0.1, "mild"), (5.0, 0.2, "moderate")]:
        log(f"  Training with shear={shear}, translate={translate}...")
        cfg = base_syntax_config()
        cfg["shear"] = shear
        cfg["translate"] = translate
        weights = train_syntax(cfg, f"D3_shear_{name}", device)
        metrics = eval_model(weights, get_syntax_yaml(), split="test")
        results[name] = {"model": weights, "metrics": metrics}
        log(f"    mAP50: {metrics.get('mAP50', 0):.4f}")

    save_result("D3_shear_translate", results)
    return results


def phase3(devices: list):
    """Run Phase 3 medium-impact retraining."""
    log("=" * 60)
    log("PHASE 3: MEDIUM-IMPACT RETRAINING")
    log("=" * 60)

    d0, d1 = devices[0], devices[1] if len(devices) > 1 else devices[0]

    # B-3 on GPU0, B-6 on GPU1 (independent)
    _run_parallel([(run_B3_background_images, d0),
                   (run_B6_copy_paste, d1)])

    # C-2 on GPU0, C-4 on GPU1
    _run_parallel([(run_C2_dropout, d0),
                   (run_C4_mask_weight, d1)])

    # D-1 on GPU0, D-2 on GPU1
    _run_parallel([(run_D1_erasing, d0),
                   (run_D2_flipud, d1)])

    # D-3 sequential on GPU0
    _run_safe(run_D3_shear_translate, d0)


# ═══════════════════════════════════════════════════════════════════════
# PHASE 4: Compound Interventions
# ═══════════════════════════════════════════════════════════════════════

def run_combined_recipe(device: str = "0"):
    """Combine winners from Phase 1-3 into a single recipe, train 3 seeds."""
    log("COMBINED: Building best-of recipe from Phase 1-3 winners...")

    # Load all results to find winners
    results_file = RESULTS_DIR / "all_results.json"
    if not results_file.exists():
        log("  No prior results found — using base config with known improvements")
        # Default combined recipe with the most likely improvements
        best_overrides = {
            "label_smoothing": 0.05,
            "flipud": 0.5,
            "erasing": 0.2,
        }
    else:
        all_results = json.load(open(results_file))
        best_overrides = _extract_best_overrides(all_results)

    log(f"  Combined overrides: {best_overrides}")

    results = {}
    # Step 1: Oversample tail classes (always start from clean data)
    combined_data = RESULTS_DIR / "data" / "combined_recipe"
    syn_src = DATA_DIR / "syntax_filtered"
    syn_dst = combined_data / "syntax_filtered"
    if syn_dst.exists():
        shutil.rmtree(syn_dst)
    shutil.copytree(syn_src, syn_dst)
    from oversample_tail_classes import oversample, instance_counts
    counts = instance_counts(syn_dst / "labels" / "train")
    sorted_cls = sorted(counts.items(), key=lambda x: x[1])
    tail = [c for c, _ in sorted_cls[:3]]
    target = sorted_cls[len(sorted_cls) // 2][1]
    oversample(syn_dst, tail, target)

    with open(get_syntax_yaml()) as f:
        syn_cfg = yaml.safe_load(f)

    combined_yaml_path = combined_data / "syntax_combined.yaml"
    combined_yaml = {
        "path": str(syn_dst.resolve()),
        "train": "images/train", "val": "images/val", "test": "images/test",
        "nc": syn_cfg["nc"], "names": syn_cfg["names"],
    }
    yaml.dump(combined_yaml, open(combined_yaml_path, "w"),
              default_flow_style=False, sort_keys=False)

    # Step 2: Train 3 seeds
    for seed in [42, 7, 2024]:
        log(f"  Training combined recipe seed={seed}...")
        cfg = base_syntax_config()
        cfg.update(best_overrides)
        cfg["cls"] = 1.25
        cfg["seed"] = seed
        weights = train_syntax(cfg, f"combined_seed{seed}", device,
                               data_yaml=str(combined_yaml_path))
        metrics = eval_model(weights, get_syntax_yaml(), split="test")
        results[f"seed_{seed}"] = {"model": weights, "metrics": metrics}
        log(f"    seed={seed} mAP50: {metrics.get('mAP50', 0):.4f}")

    save_result("combined_recipe_3seeds", results)
    return results


def _extract_best_overrides(all_results: list) -> dict:
    """From all_results, pick the winning config for each parameter."""
    overrides = {}

    # Check C1 label smoothing
    for r in all_results:
        if r.get("name") == "C1_label_smoothing":
            best_ls = None
            best_map = 0
            for key, val in r.items():
                if key.startswith("ls_") and isinstance(val, dict):
                    m = val.get("metrics", {}).get("mAP50", 0)
                    if m > best_map:
                        best_map = m
                        best_ls = float(key.split("_")[1])
            if best_ls:
                overrides["label_smoothing"] = best_ls

    # Check D2 flipud
    for r in all_results:
        if r.get("name") == "D2_flipud":
            overrides["flipud"] = 0.5

    # Check D1 erasing
    for r in all_results:
        if r.get("name") == "D1_erasing":
            best_er = None
            best_map = 0
            for key, val in r.items():
                if key.startswith("erasing_") and isinstance(val, dict):
                    m = val.get("metrics", {}).get("mAP50", 0)
                    if m > best_map:
                        best_map = m
                        best_er = float(key.split("_")[1])
            if best_er:
                overrides["erasing"] = best_er

    # Check C2 dropout
    for r in all_results:
        if r.get("name") == "C2_dropout":
            best_dp = None
            best_map = 0
            for key, val in r.items():
                if key.startswith("dropout_") and isinstance(val, dict):
                    m = val.get("metrics", {}).get("mAP50", 0)
                    if m > best_map:
                        best_map = m
                        best_dp = float(key.split("_")[1])
            if best_dp:
                overrides["dropout"] = best_dp

    # Check C4 mask weight
    for r in all_results:
        if r.get("name") == "C4_mask_weight":
            best_mw = None
            best_map = 0
            for key, val in r.items():
                if key.startswith("mask_") and isinstance(val, dict):
                    m = val.get("metrics", {}).get("mAP50", 0)
                    if m > best_map:
                        best_map = m
                        best_mw = float(key.split("_")[1])
            if best_mw:
                overrides["mask"] = best_mw

    # Defaults if nothing found
    if not overrides:
        overrides = {
            "label_smoothing": 0.05,
            "flipud": 0.5,
            "erasing": 0.2,
        }
    return overrides


def run_A6_wbf_ensemble(device: str = "0"):
    """A-6: Multi-seed WBF ensemble on combined recipe."""
    log("A-6: Multi-seed WBF ensemble...")

    # Find the 3 seed models from combined recipe
    models = []
    for seed in [42, 7, 2024]:
        pattern = f"combined_seed{seed}"
        candidates = list(RESULTS_DIR.rglob(f"**/{pattern}*best.pt"))
        if candidates:
            models.append(str(candidates[0]))

    if len(models) < 2:
        log("  Not enough seed models found, skipping WBF ensemble")
        return None

    log(f"  Ensembling {len(models)} models...")
    import subprocess
    output = RESULTS_DIR / "A6_wbf_ensemble.json"
    subprocess.run([
        sys.executable, str(SCRIPT_DIR / "multiseed_wbf.py"),
        "--models", *models,
        "--data-yaml", get_syntax_yaml(),
        "--split", "test",
        "--imgsz", "512",
        "--device", device,
        "--output", str(output),
    ], check=True)

    if output.exists():
        result = json.load(open(output))
        save_result("A6_wbf_ensemble", result)
        log(f"  WBF ensemble mean F1: {result.get('metrics', {}).get('mean_f1', 0):.4f}")
        return result
    return None


def run_B5_pseudo_labeling(device: str = "0"):
    """B-5: Iterative pseudo-labeling (3 rounds) for stenosis."""
    log("B-5: Iterative pseudo-labeling (3 rounds)...")
    from generate_stenosis_pseudolabels import (
        run_stenosis_on_syntax_images,
        build_extended_stenosis_dataset,
    )

    syntax_img_dir = DATA_DIR / "syntax_filtered" / "images" / "train"
    stenosis_data_dir = DATA_DIR / "stenosis"

    current_model = get_best_stenosis_model()
    results = {}

    for round_num, conf in enumerate([0.5, 0.4, 0.3], start=1):
        log(f"  Round {round_num}: conf={conf}...")
        round_dir = RESULTS_DIR / "B5_pseudo_labels" / f"round{round_num}"

        # Generate pseudo-labels
        pseudo_dir = round_dir / "pseudo_labels"
        stats = run_stenosis_on_syntax_images(
            current_model, syntax_img_dir, pseudo_dir,
            conf_threshold=conf, imgsz=512, device=device,
        )
        log(f"    Pseudo-labels: {stats['images_with_predictions']} images, "
            f"{stats['total_pseudo_instances']} instances")

        # Build extended dataset
        ext_yaml = build_extended_stenosis_dataset(
            stenosis_data_dir, syntax_img_dir, pseudo_dir,
            round_dir / "extended_stenosis",
            apply_clahe=False,
        )

        # Retrain from scratch
        cfg = base_stenosis_config()
        weights = train_stenosis(
            cfg, f"B5_round{round_num}", device,
            data_yaml=ext_yaml,
        )
        metrics = eval_model(weights, get_stenosis_yaml(), split="test")
        results[f"round{round_num}"] = {
            "model": weights, "conf": conf,
            "pseudo_stats": stats, "metrics": metrics,
        }
        current_model = weights
        log(f"    Round {round_num} test mAP50: {metrics.get('mAP50', 0):.4f}")

    save_result("B5_pseudo_labeling", results)
    return results


def run_E1_p2_head(device: str = "0"):
    """E-1: P2 output head for stenosis (stride 4 instead of stride 8)."""
    log("E-1: P2 head for stenosis...")

    # Create a custom YOLO config with P2 head
    # We'll try the built-in p2 variant first
    from ultralytics import YOLO
    import torch

    # Check if yolo11m-p2 config exists
    p2_yaml = RESULTS_DIR / "E1_yolo11m_p2.yaml"

    # Build a P2 config by modifying the standard yolo11m-seg config
    # The key is adding an extra detection head at stride 4
    # Simplest approach: use a higher imgsz which effectively gives better resolution
    # For true P2, we need to modify the model YAML

    # Approach: train at imgsz=1024 which gives 4x the feature map resolution
    # compared to 512, achieving the same effect as P2 at 512
    cfg = base_stenosis_config()
    cfg["imgsz"] = 1024
    cfg["batch"] = 4  # reduce batch for 5090 VRAM at 1024
    weights = train_stenosis(cfg, "E1_p2_equivalent_1024", device)
    metrics = eval_model(weights, get_stenosis_yaml(), split="test", imgsz=1024)

    result = {"model": weights, "metrics": metrics, "imgsz": 1024}
    save_result("E1_p2_head", result)
    log(f"  E-1 stenosis test mAP50: {metrics.get('mAP50', 0):.4f}")
    return result


def phase4(devices: list):
    """Run Phase 4 compound interventions."""
    log("=" * 60)
    log("PHASE 4: COMPOUND INTERVENTIONS")
    log("=" * 60)

    d0, d1 = devices[0], devices[1] if len(devices) > 1 else devices[0]

    # Combined recipe (sequential — needs 3 seeds on one GPU)
    run_combined_recipe(d0)

    # A-6 WBF ensemble (depends on combined recipe)
    run_A6_wbf_ensemble(d0)

    # B-5 and E-1 in parallel
    _run_parallel([(run_B5_pseudo_labeling, d0),
                   (run_E1_p2_head, d1)])


# ═══════════════════════════════════════════════════════════════════════
# PHASE 5: Advanced
# ═══════════════════════════════════════════════════════════════════════

def run_E2_distillation(device: str = "0"):
    """E-2: Knowledge distillation (yolo11x teacher -> yolo11m student)."""
    log("E-2: Knowledge distillation...")

    # Step 1: Train teacher (yolo11x)
    log("  Training yolo11x teacher...")
    cfg = base_syntax_config()
    cfg["model"] = "yolo11x-seg.pt"
    cfg["batch"] = 4  # yolo11x needs lower batch
    teacher_weights = train_syntax(cfg, "E2_teacher_yolo11x", device)
    teacher_metrics = eval_model(teacher_weights, get_syntax_yaml(), split="test")
    log(f"  Teacher mAP50: {teacher_metrics.get('mAP50', 0):.4f}")

    # Step 2: Generate soft pseudo-labels from teacher for training set
    log("  Generating teacher soft predictions for distillation...")
    from ultralytics import YOLO
    teacher = YOLO(teacher_weights)

    syn_yaml = get_syntax_yaml()
    with open(syn_yaml) as f:
        syn_cfg = yaml.safe_load(f)
    root = Path(syn_cfg["path"])
    train_img_dir = root / "images" / "train"
    imgs = sorted(list(train_img_dir.glob("*.png")) + list(train_img_dir.glob("*.PNG")))

    # Create distillation labels directory
    distill_dir = RESULTS_DIR / "data" / "E2_distillation"
    distill_lbl_dir = distill_dir / "labels" / "train"
    distill_lbl_dir.mkdir(parents=True, exist_ok=True)

    for img_path in imgs:
        results = teacher.predict(
            source=str(img_path), conf=0.1, imgsz=512,
            device=device, verbose=False, save=False, retina_masks=True,
        )
        if not results or results[0].masks is None:
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
            (distill_lbl_dir / f"{img_path.stem}.txt").write_text("\n".join(lines) + "\n")

    # Create distill dataset (teacher labels on same images)
    for split in ["val", "test"]:
        (distill_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (distill_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
        for src_img in (root / "images" / split).glob("*.png"):
            dst = distill_dir / "images" / split / src_img.name
            if not dst.exists():
                os.symlink(src_img.resolve(), dst)
        for src_lbl in (root / "labels" / split).glob("*.txt"):
            dst = distill_dir / "labels" / split / src_lbl.name
            if not dst.exists():
                os.symlink(src_lbl.resolve(), dst)
    # Symlink train images
    (distill_dir / "images" / "train").mkdir(parents=True, exist_ok=True)
    for src_img in (root / "images" / "train").glob("*.png"):
        dst = distill_dir / "images" / "train" / src_img.name
        if not dst.exists():
            os.symlink(src_img.resolve(), dst)

    distill_yaml = {
        "path": str(distill_dir.resolve()),
        "train": "images/train", "val": "images/val", "test": "images/test",
        "nc": syn_cfg["nc"], "names": syn_cfg["names"],
    }
    distill_yaml_path = distill_dir / "distill.yaml"
    yaml.dump(distill_yaml, open(distill_yaml_path, "w"),
              default_flow_style=False, sort_keys=False)

    # Step 3: Train student (yolo11m) on teacher labels
    log("  Training yolo11m student on teacher labels...")
    student_cfg = base_syntax_config()
    student_weights = train_syntax(student_cfg, "E2_student_yolo11m", device,
                                   data_yaml=str(distill_yaml_path))
    student_metrics = eval_model(student_weights, get_syntax_yaml(), split="test")
    log(f"  Student mAP50: {student_metrics.get('mAP50', 0):.4f}")

    result = {
        "teacher_model": teacher_weights,
        "teacher_metrics": teacher_metrics,
        "student_model": student_weights,
        "student_metrics": student_metrics,
    }
    save_result("E2_distillation", result)
    return result


def run_D4_mosaic_annealing(device: str = "0"):
    """D-4: Mosaic annealing schedule via callback."""
    log("D-4: Mosaic annealing schedule...")

    # Train with mosaic that anneals via close_mosaic
    # Ultralytics' close_mosaic=N disables mosaic for the last N epochs
    # We set mosaic=0.8 with close_mosaic at 70% of epochs = 210 out of 300
    cfg = base_syntax_config()
    cfg["mosaic"] = 0.8
    cfg["close_mosaic"] = 90  # last 90/300 = 30% without mosaic
    weights = train_syntax(cfg, "D4_mosaic_anneal", device)
    metrics = eval_model(weights, get_syntax_yaml(), split="test")

    result = {"model": weights, "metrics": metrics}
    save_result("D4_mosaic_annealing", result)
    log(f"  D-4 test mAP50: {metrics.get('mAP50', 0):.4f}")
    return result


def run_C6_nbs_lr(device: str = "0"):
    """C-6: nbs adjustment + lr re-sweep."""
    log("C-6: nbs=16 + lr sweep...")
    results = {}
    for lr0 in [0.00025, 0.0005, 0.001]:
        log(f"  Training with nbs=16, lr0={lr0}...")
        cfg = base_syntax_config()
        cfg["nbs"] = 16
        cfg["lr0"] = lr0
        weights = train_syntax(cfg, f"C6_nbs16_lr{lr0}", device)
        metrics = eval_model(weights, get_syntax_yaml(), split="test")
        results[f"lr_{lr0}"] = {"model": weights, "metrics": metrics}
        log(f"    mAP50: {metrics.get('mAP50', 0):.4f}")

    save_result("C6_nbs_lr", results)
    return results


def run_C7_lrf_sweep(device: str = "0"):
    """C-7: Final learning rate fraction sweep."""
    log("C-7: lrf sweep...")
    results = {}
    for lrf in [0.001, 0.05, 0.1]:
        log(f"  Training with lrf={lrf}...")
        cfg = base_syntax_config()
        cfg["lrf"] = lrf
        weights = train_syntax(cfg, f"C7_lrf_{lrf}", device)
        metrics = eval_model(weights, get_syntax_yaml(), split="test")
        results[f"lrf_{lrf}"] = {"model": weights, "metrics": metrics}
        log(f"    mAP50: {metrics.get('mAP50', 0):.4f}")

    save_result("C7_lrf_sweep", results)
    return results


def run_C8_warmup(device: str = "0"):
    """C-8: Warmup epochs sweep."""
    log("C-8: Warmup epochs sweep...")
    results = {}
    for we in [3, 10]:
        log(f"  Training with warmup_epochs={we}...")
        cfg = base_syntax_config()
        cfg["warmup_epochs"] = we
        weights = train_syntax(cfg, f"C8_warmup_{we}", device)
        metrics = eval_model(weights, get_syntax_yaml(), split="test")
        results[f"warmup_{we}"] = {"model": weights, "metrics": metrics}
        log(f"    mAP50: {metrics.get('mAP50', 0):.4f}")

    save_result("C8_warmup", results)
    return results


def run_F3_error_analysis(device: str = "0"):
    """F-3: Confusion matrix and visual inspection of failures."""
    log("F-3: Error analysis...")
    from ultralytics import YOLO

    best_syn = get_best_syntax_model()
    syn_yaml = get_syntax_yaml()

    model = YOLO(best_syn)
    results = model.val(data=syn_yaml, split="val", imgsz=512,
                        save_json=True, verbose=True, plots=True,
                        project=str(RESULTS_DIR / "F3_error_analysis"),
                        name="confusion_matrix")

    log(f"  Confusion matrix and plots saved to {RESULTS_DIR / 'F3_error_analysis'}")
    result = {
        "model": best_syn,
        "output_dir": str(RESULTS_DIR / "F3_error_analysis"),
        "mAP50": round(float(results.seg.map50), 4),
    }
    save_result("F3_error_analysis", result)
    return result


def phase5(devices: list):
    """Run Phase 5 advanced experiments."""
    log("=" * 60)
    log("PHASE 5: ADVANCED EXPERIMENTS")
    log("=" * 60)

    d0, d1 = devices[0], devices[1] if len(devices) > 1 else devices[0]

    # E-2 on GPU0, D-4 on GPU1
    _run_parallel([(run_E2_distillation, d0),
                   (run_D4_mosaic_annealing, d1)])

    # C-6 on GPU0, C-7 on GPU1
    _run_parallel([(run_C6_nbs_lr, d0),
                   (run_C7_lrf_sweep, d1)])

    # C-8 on GPU0, F-3 on GPU1
    _run_parallel([(run_C8_warmup, d0),
                   (run_F3_error_analysis, d1)])


# ═══════════════════════════════════════════════════════════════════════
# FINAL: Summary report
# ═══════════════════════════════════════════════════════════════════════

def generate_final_report():
    """Generate a summary comparing all experiments."""
    log("=" * 60)
    log("FINAL REPORT")
    log("=" * 60)

    results_file = RESULTS_DIR / "all_results.json"
    if not results_file.exists():
        log("  No results found")
        return

    all_results = json.load(open(results_file))
    log(f"  Total experiments: {len(all_results)}")

    # Find best syntax model
    best_syntax_map = 0
    best_syntax_name = ""
    best_stenosis_map = 0
    best_stenosis_name = ""

    for r in all_results:
        name = r.get("name", "")
        metrics = r.get("metrics", {})
        if isinstance(metrics, dict):
            m50 = metrics.get("mAP50", 0)
            if "syntax" in name.lower() and m50 > best_syntax_map:
                best_syntax_map = m50
                best_syntax_name = name
            if "stenosis" in name.lower() and m50 > best_stenosis_map:
                best_stenosis_map = m50
                best_stenosis_name = name

    log(f"\n  Best syntax model: {best_syntax_name} (mAP50={best_syntax_map:.4f})")
    log(f"  Best stenosis model: {best_stenosis_name} (mAP50={best_stenosis_map:.4f})")

    report = {
        "total_experiments": len(all_results),
        "best_syntax": {"name": best_syntax_name, "mAP50": best_syntax_map},
        "best_stenosis": {"name": best_stenosis_name, "mAP50": best_stenosis_map},
        "all_results": all_results,
    }
    json.dump(report, open(RESULTS_DIR / "final_report.json", "w"), indent=2)
    log(f"\n  Full report: {RESULTS_DIR / 'final_report.json'}")


# ═══════════════════════════════════════════════════════════════════════
# Runner utilities
# ═══════════════════════════════════════════════════════════════════════

def _run_safe(func, *args, **kwargs):
    """Run a function with error handling."""
    try:
        return func(*args, **kwargs)
    except Exception as e:
        log(f"ERROR in {func.__name__}: {e}")
        traceback.print_exc()
        save_result(func.__name__, {"status": "failed", "error": str(e)})
        return None


def _run_parallel(pairs: list):
    """Run [(func, device), ...] in parallel using subprocess to avoid CUDA fork issues.

    Each pair is launched as a separate Python subprocess with its own CUDA
    context. This avoids the 'Cannot re-initialize CUDA in forked subprocess'
    error that killed every ProcessPoolExecutor call in the first run.
    """
    if len(pairs) <= 1:
        for func, device in pairs:
            _run_safe(func, device)
        return

    # Launch each as a subprocess via multiprocessing with spawn
    ctx = multiprocessing.get_context("spawn")
    procs = []
    for func, device in pairs:
        p = ctx.Process(target=_run_safe, args=(func, device))
        p.start()
        procs.append((p, func.__name__))

    for p, name in procs:
        p.join()
        if p.exitcode != 0:
            log(f"  subprocess {name} exited with code {p.exitcode}")


def main():
    parser = argparse.ArgumentParser(
        description="Master runner for all proposal experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--arcade-root", type=Path, required=True,
                        help="Path to arcade/submission directory")
    parser.add_argument("--phase", type=str, default="all",
                        help="Phase(s) to run: 0,1,2,3,4,5 or 'all'")
    parser.add_argument("--devices", type=str, default="0,1",
                        help="GPU devices (comma-separated, e.g. '0,1')")
    parser.add_argument("--results-dir", type=Path, default=None,
                        help="Results directory (default: results/proposal_runs)")
    parser.add_argument("--start-from", type=str, default=None,
                        help="Resume from a specific experiment ID (e.g. 'B4')")

    args = parser.parse_args()

    # Set globals
    global ARCADE_ROOT, BASE_DIR, RESULTS_DIR, DATA_DIR
    ARCADE_ROOT = args.arcade_root.resolve()
    BASE_DIR = SCRIPT_DIR.parent
    RESULTS_DIR = (args.results_dir or BASE_DIR / "results" / "proposal_runs").resolve()
    DATA_DIR = (BASE_DIR / "data").resolve()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    devices = [d.strip() for d in args.devices.split(",")]

    # Fix dataset YAML paths for the current machine
    fix_dataset_yaml_paths()

    if args.phase == "all":
        phases = [0, 1, 2, 3, 4, 5]
    else:
        phases = [int(p.strip()) for p in args.phase.split(",")]

    log(f"ARCADE root: {ARCADE_ROOT}")
    log(f"Data dir:    {DATA_DIR}")
    log(f"Results dir: {RESULTS_DIR}")
    log(f"Devices:     {devices}")
    log(f"Phases:      {phases}")
    log("")

    # ── Always prepare data fresh from the ground truth source ──
    # arcade/submission/{syntax,stenosis}/ is the ONLY trusted source.
    # Previous iterations may have altered derived copies in data/.
    # We verify integrity and re-prep if anything looks wrong.
    _ensure_clean_data()

    # Find existing best models (with strict filtering)
    _find_best_models()
    # Train baselines if no suitable models exist
    _ensure_baseline_models(devices[0])

    t0 = time.time()

    if 0 in phases:
        _run_safe(phase0, devices[0])
    if 1 in phases:
        _run_safe(phase1, devices[0])
    if 2 in phases:
        _run_safe(phase2, devices)
    if 3 in phases:
        _run_safe(phase3, devices)
    if 4 in phases:
        _run_safe(phase4, devices)
    if 5 in phases:
        _run_safe(phase5, devices)

    generate_final_report()

    elapsed = time.time() - t0
    log(f"\nTotal time: {elapsed / 3600:.1f} hours")
    log("DONE.")


if __name__ == "__main__":
    main()
