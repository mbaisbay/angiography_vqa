#!/usr/bin/env python3
"""Run the full cross-task pseudo-labeling pipeline end-to-end.

Orchestrates 9 steps to build a unified dataset where every image has BOTH
syntax segment labels AND stenosis labels:

  1. filter          — Filter classes to >=300 annotations (13 classes)
  2. train-syntax    — Train YOLOv8x-seg on filtered syntax data
  3. train-stenosis  — Train YOLOv8x-seg on filtered stenosis data
  4. cross-infer-r1  — Cross-inference: syntax→stenosis, stenosis→syntax
  5. build-combined  — Merge stenosis GT + syntax predictions
  6. train-combined  — Train on combined dataset
  7. cross-infer-r2  — Run combined model on syntax images
  8. build-final     — Quality-filtered merge of both directions
  9. train-final     — Train final unified model on ~3000 images

Usage:
    python run_pipeline.py                          # full pipeline
    python run_pipeline.py --dry-run                # print plan only
    python run_pipeline.py --start-from train-combined  # resume
    python run_pipeline.py --only build-final       # single step
    python run_pipeline.py --evaluate               # add eval after training
"""

import argparse
import copy
import subprocess
import sys
import time
from collections import OrderedDict
from pathlib import Path

import yaml


# ── Step Definitions ──────────────────────────────────────────────

STEPS = OrderedDict([
    ("filter", {
        "description": "Filter classes (>=300 annotations, 13 classes)",
        "cmd": ["filter_classes.py", "--min-count", "300"],
        "prerequisites": [],
        "outputs": lambda cfg: [
            Path(cfg["dataset_root"]) / "syntax_filtered" / "train" / "images",
            Path(cfg["dataset_root"]) / "stenosis_filtered" / "train" / "images",
        ],
        "config_overrides": None,
    }),
    ("train-syntax", {
        "description": "Train syntax segmentation model",
        "cmd": ["train.py", "--task", "syntax"],
        "prerequisites": lambda cfg: [Path(cfg["syntax_data_yaml"])],
        "outputs": lambda cfg: [Path(cfg["output_dir"]) / "syntax" / "weights" / "best.pt"],
        "config_overrides": None,
    }),
    ("train-stenosis", {
        "description": "Train stenosis detection model",
        "cmd": ["train.py", "--task", "stenosis"],
        "prerequisites": lambda cfg: [Path(cfg["stenosis_data_yaml"])],
        "outputs": lambda cfg: [Path(cfg["output_dir"]) / "stenosis" / "weights" / "best.pt"],
        "config_overrides": None,
    }),
    ("cross-infer-r1", {
        "description": "Round 1 cross-inference (syntax↔stenosis)",
        "cmd": ["cross_inference.py", "--direction", "both"],
        "prerequisites": lambda cfg: [
            Path(cfg["output_dir"]) / "syntax" / "weights" / "best.pt",
            Path(cfg["output_dir"]) / "stenosis" / "weights" / "best.pt",
        ],
        "outputs": lambda cfg: [
            Path(cfg["cross_inference"]["output_dir"]) / "syntax_on_stenosis.json",
            Path(cfg["cross_inference"]["output_dir"]) / "stenosis_on_syntax.json",
        ],
        # Override stenosis_weights to use standalone model (not combined)
        "config_overrides": lambda cfg: {
            "cross_inference": {
                "syntax_weights": str(Path(cfg["output_dir"]) / "syntax" / "weights" / "best.pt"),
                "stenosis_weights": str(Path(cfg["output_dir"]) / "stenosis" / "weights" / "best.pt"),
            }
        },
    }),
    ("build-combined", {
        "description": "Build combined dataset (stenosis GT + syntax predictions)",
        "cmd": ["build_combined_dataset.py"],
        "prerequisites": lambda cfg: [
            Path(cfg["cross_inference"]["output_dir"]) / "syntax_on_stenosis.json",
        ],
        "outputs": lambda cfg: [
            Path(cfg["combined_dataset"]["output_dir"]) / "train" / "images",
        ],
        "config_overrides": None,
    }),
    ("train-combined", {
        "description": "Train on combined dataset (26-class)",
        "cmd": ["train.py", "--task", "combined"],
        "prerequisites": lambda cfg: [Path(cfg["combined_data_yaml"])],
        "outputs": lambda cfg: [Path(cfg["output_dir"]) / "combined" / "weights" / "best.pt"],
        "config_overrides": None,
    }),
    ("cross-infer-r2", {
        "description": "Round 2 cross-inference (combined model → syntax images)",
        "cmd": ["cross_inference.py", "--direction", "combined_on_syntax"],
        "prerequisites": lambda cfg: [
            Path(cfg["output_dir"]) / "combined" / "weights" / "best.pt",
        ],
        "outputs": lambda cfg: [
            Path(cfg["cross_inference"]["output_dir"]) / "combined_on_syntax.json",
        ],
        "config_overrides": lambda cfg: {
            "cross_inference": {
                "combined_weights": str(Path(cfg["output_dir"]) / "combined" / "weights" / "best.pt"),
            }
        },
    }),
    ("build-final", {
        "description": "Build final quality-filtered dataset (both directions)",
        "cmd": ["build_final_dataset.py"],
        "prerequisites": lambda cfg: [
            Path(cfg["cross_inference"]["output_dir"]) / "syntax_on_stenosis.json",
            Path(cfg["cross_inference"]["output_dir"]) / "combined_on_syntax.json",
        ],
        "outputs": lambda cfg: [
            Path(cfg["dataset_root"]) / "final" / "train" / "images",
        ],
        "config_overrides": None,
    }),
    ("train-final", {
        "description": "Train final unified model on ~3000 images",
        "cmd": ["train.py", "--task", "final"],
        "prerequisites": lambda cfg: [Path(cfg["final_data_yaml"])],
        "outputs": lambda cfg: [Path(cfg["output_dir"]) / "final" / "weights" / "best.pt"],
        "config_overrides": None,
    }),
])

EVAL_STEPS = OrderedDict([
    ("eval-syntax", {
        "description": "Evaluate syntax model",
        "cmd": ["evaluate.py", "--task", "syntax", "--method", "both"],
    }),
    ("eval-final", {
        "description": "Evaluate final model",
        "cmd": ["evaluate.py", "--task", "syntax", "--method", "arcade"],
    }),
])


# ── Helpers ───────────────────────────────────────────────────────

def deep_merge(base: dict, overrides: dict) -> dict:
    """Deep-merge overrides into base dict (mutates base)."""
    for key, value in overrides.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def load_raw_config(config_path: str) -> dict:
    """Load raw config without path resolution (for re-serialization)."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def write_pipeline_config(base_config_path: str, overrides: dict,
                          output_path: str) -> str:
    """Create a working config with overrides applied."""
    raw = load_raw_config(base_config_path)
    if overrides:
        deep_merge(raw, overrides)
    with open(output_path, "w") as f:
        yaml.dump(raw, f, default_flow_style=False, sort_keys=False)
    return output_path


def resolve_config(config_path: str) -> dict:
    """Load and resolve config using the standard config_loader."""
    from utils.config_loader import load_config
    return load_config(config_path)


def check_prerequisites(prereqs, cfg: dict) -> list:
    """Check that prerequisite files exist. Returns list of missing paths."""
    if prereqs is None or (callable(prereqs) and prereqs is None):
        return []
    paths = prereqs(cfg) if callable(prereqs) else prereqs
    return [str(p) for p in paths if not Path(p).exists()]


def check_outputs_exist(outputs_fn, cfg: dict) -> bool:
    """Check if all output files/dirs already exist."""
    if outputs_fn is None:
        return False
    paths = outputs_fn(cfg) if callable(outputs_fn) else outputs_fn
    return all(Path(p).exists() for p in paths)


def run_step(step_name: str, cmd: list, description: str) -> tuple:
    """Run a pipeline step as a subprocess.

    Returns (success: bool, elapsed_seconds: float).
    """
    print(f"\n{'='*70}")
    print(f"  [{step_name}] {description}")
    print(f"  CMD: {' '.join(cmd)}")
    print(f"{'='*70}\n")

    start = time.time()
    result = subprocess.run(cmd)
    elapsed = time.time() - start

    status = "OK" if result.returncode == 0 else f"FAILED (exit {result.returncode})"
    print(f"\n  [{status}] {step_name} ({elapsed / 60:.1f} min)")
    return result.returncode == 0, elapsed


# ── Main ──────────────────────────────────────────────────────────

def main():
    step_names = list(STEPS.keys())

    parser = argparse.ArgumentParser(
        description="Run the full cross-task pseudo-labeling pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Steps: " + ", ".join(step_names),
    )
    parser.add_argument("--config", default="config.yaml",
                        help="Base config file (default: config.yaml)")
    parser.add_argument("--start-from", metavar="STEP", choices=step_names,
                        help="Resume from a specific step (skip earlier steps)")
    parser.add_argument("--only", metavar="STEP", choices=step_names,
                        help="Run only a single step")
    parser.add_argument("--skip", metavar="STEP", nargs="+", choices=step_names,
                        default=[], help="Skip specific steps")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print plan without executing")
    parser.add_argument("--keep-config", action="store_true",
                        help="Keep working config file after completion")
    parser.add_argument("--evaluate", action="store_true",
                        help="Run evaluation after training steps")
    parser.add_argument("--force", action="store_true",
                        help="Re-run steps even if outputs already exist")
    args = parser.parse_args()

    base_config_path = str(Path(args.config).resolve())
    base_dir = Path(base_config_path).parent
    pipeline_config_path = str(base_dir / "config_pipeline.yaml")

    # Determine which steps to run
    if args.only:
        steps_to_run = [args.only]
    else:
        steps_to_run = list(step_names)
        if args.start_from:
            start_idx = step_names.index(args.start_from)
            steps_to_run = step_names[start_idx:]
        steps_to_run = [s for s in steps_to_run if s not in args.skip]

    # ── Header ────────────────────────────────────────────────────
    print("=" * 70)
    print("  ARCADE Cross-Task Pseudo-Labeling Pipeline")
    print("=" * 70)
    print(f"  Base config:  {base_config_path}")
    print(f"  Steps to run: {len(steps_to_run)}")
    if args.dry_run:
        print(f"  Mode:         DRY RUN (no execution)")
    print()

    for i, name in enumerate(steps_to_run, 1):
        step = STEPS[name]
        marker = ">>>" if not args.dry_run else "   "
        print(f"  {marker} {i}. {name}: {step['description']}")

    if args.evaluate:
        print(f"\n  + Evaluation steps after training")

    # ── Dry run: show prerequisites and outputs ───────────────────
    if args.dry_run:
        # Resolve config once for prerequisite checking
        try:
            cfg = resolve_config(base_config_path)
        except Exception as e:
            print(f"\n  [WARN] Could not resolve config: {e}")
            print(f"  Prerequisites/outputs cannot be checked.")
            return

        print(f"\n{'='*70}")
        print("  Step Details (Prerequisites → Outputs)")
        print(f"{'='*70}")

        for name in steps_to_run:
            step = STEPS[name]
            print(f"\n  [{name}] {step['description']}")

            # Prerequisites
            prereqs = step.get("prerequisites")
            if prereqs:
                paths = prereqs(cfg) if callable(prereqs) else prereqs
                for p in paths:
                    exists = Path(p).exists()
                    status = "OK" if exists else "MISSING"
                    print(f"    prereq: {p} [{status}]")

            # Outputs
            outputs_fn = step.get("outputs")
            if outputs_fn:
                paths = outputs_fn(cfg) if callable(outputs_fn) else outputs_fn
                for p in paths:
                    exists = Path(p).exists()
                    status = "EXISTS" if exists else "pending"
                    print(f"    output: {p} [{status}]")

            # Config overrides
            overrides_fn = step.get("config_overrides")
            if overrides_fn:
                overrides = overrides_fn(cfg)
                print(f"    config overrides: {overrides}")

        print(f"\n{'='*70}")
        print("  Dry run complete. No changes made.")
        print(f"{'='*70}")
        return

    # ── Execute pipeline ──────────────────────────────────────────
    total_start = time.time()
    results = OrderedDict()
    failed_step = None

    for i, name in enumerate(steps_to_run, 1):
        step = STEPS[name]

        print(f"\n{'#'*70}")
        print(f"  STEP {i}/{len(steps_to_run)}: {name}")
        print(f"  {step['description']}")
        print(f"{'#'*70}")

        # Resolve config for this step
        cfg = resolve_config(base_config_path)

        # Check if outputs already exist (skip unless --force)
        if not args.force and check_outputs_exist(step.get("outputs"), cfg):
            print(f"\n  [SKIP] Outputs already exist. Use --force to re-run.")
            results[name] = {"status": "skipped", "elapsed": 0}
            continue

        # Check prerequisites
        prereqs = step.get("prerequisites")
        if prereqs:
            missing = check_prerequisites(prereqs, cfg)
            if missing:
                print(f"\n  [ERROR] Missing prerequisites for '{name}':")
                for m in missing:
                    print(f"    - {m}")
                print(f"\n  Fix the issue and re-run with: --start-from {name}")
                failed_step = name
                break

        # Apply config overrides if needed
        overrides_fn = step.get("config_overrides")
        if overrides_fn:
            overrides = overrides_fn(cfg)
            write_pipeline_config(base_config_path, overrides, pipeline_config_path)
            config_to_use = pipeline_config_path
            print(f"  Using pipeline config with overrides: {overrides}")
        else:
            config_to_use = base_config_path

        # Build command
        cmd = [sys.executable] + step["cmd"] + ["--config", config_to_use]

        # Execute
        ok, elapsed = run_step(name, cmd, step["description"])
        results[name] = {"status": "ok" if ok else "failed", "elapsed": elapsed}

        if not ok:
            print(f"\n  [ABORT] Step '{name}' failed.")
            print(f"  Fix the issue and re-run with: --start-from {name}")
            failed_step = name
            break

        # Progress
        remaining = len(steps_to_run) - i
        if remaining > 0:
            completed_times = [r["elapsed"] for r in results.values()
                               if r["status"] == "ok" and r["elapsed"] > 0]
            if completed_times:
                avg = sum(completed_times) / len(completed_times)
                eta = avg * remaining
                print(f"\n  Progress: {i}/{len(steps_to_run)} done, "
                      f"~{eta / 60:.0f} min remaining (estimate)")

    # ── Evaluation (optional) ─────────────────────────────────────
    if args.evaluate and failed_step is None:
        print(f"\n{'#'*70}")
        print("  EVALUATION")
        print(f"{'#'*70}")

        for eval_name, eval_step in EVAL_STEPS.items():
            cmd = [sys.executable] + eval_step["cmd"] + ["--config", base_config_path]
            ok, elapsed = run_step(eval_name, cmd, eval_step["description"])
            results[eval_name] = {"status": "ok" if ok else "failed", "elapsed": elapsed}

    # ── Cleanup ───────────────────────────────────────────────────
    pipeline_cfg = Path(pipeline_config_path)
    if pipeline_cfg.exists() and not args.keep_config:
        pipeline_cfg.unlink()

    # ── Summary ───────────────────────────────────────────────────
    total_elapsed = time.time() - total_start

    print(f"\n{'='*70}")
    print("  PIPELINE SUMMARY")
    print(f"{'='*70}")

    for name, data in results.items():
        status = data["status"].upper()
        elapsed_min = data["elapsed"] / 60 if data["elapsed"] else 0
        symbol = {"ok": "+", "failed": "X", "skipped": "-"}.get(data["status"], "?")
        print(f"  [{symbol}] {name:<20s} {status:<8s} {elapsed_min:>7.1f} min")

    print(f"\n  Total time: {total_elapsed / 60:.1f} min")

    if failed_step:
        print(f"\n  Pipeline FAILED at step: {failed_step}")
        print(f"  Resume with: python run_pipeline.py --start-from {failed_step}")
    else:
        print(f"\n  Pipeline COMPLETED successfully!")
        cfg = resolve_config(base_config_path)
        final_dir = Path(cfg["dataset_root"]) / "final"
        print(f"  Final dataset: {final_dir}")
        print(f"  Final weights: {Path(cfg['output_dir']) / 'final' / 'weights' / 'best.pt'}")

    print(f"{'='*70}")
    sys.exit(1 if failed_step else 0)


if __name__ == "__main__":
    main()
