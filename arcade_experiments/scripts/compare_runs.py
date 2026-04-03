"""Compare results across multiple experiment runs.

Loads metrics from all 4 runs and generates:
  - Overall comparison table (mAP, precision, recall)
  - Per-class AP50 comparison
  - Iteration progression tracking
  - Pseudo-label quality across iterations
"""

import argparse
import json
from pathlib import Path

import numpy as np


def load_metrics(results_dir: Path) -> dict:
    """Load all_metrics.json from a run's results directory."""
    metrics_path = results_dir / "all_metrics.json"
    if metrics_path.exists():
        with open(metrics_path, "r") as f:
            return json.load(f)

    # Fallback: load individual metric files
    metrics_dir = results_dir / "metrics"
    if not metrics_dir.exists():
        return {}

    all_metrics = {}
    for json_file in sorted(metrics_dir.glob("*.json")):
        stage_name = json_file.stem
        with open(json_file, "r") as f:
            all_metrics[stage_name] = json.load(f)
    return all_metrics


def load_pseudo_label_stats(results_dir: Path) -> dict:
    """Load pseudo-label statistics from all iterations."""
    pseudo_dir = results_dir / "pseudo_labels"
    if not pseudo_dir.exists():
        return {}

    stats = {}
    for stats_file in sorted(pseudo_dir.rglob("stats.json")):
        rel_path = stats_file.relative_to(pseudo_dir)
        stage_name = str(rel_path.parent)
        with open(stats_file, "r") as f:
            stats[stage_name] = json.load(f)
    return stats


def print_overall_comparison(runs: dict) -> None:
    """Print overall metrics comparison table across runs."""
    print("\n" + "=" * 80)
    print("OVERALL COMPARISON (Final Test Metrics)")
    print("=" * 80)

    header = f"{'Run':>20s} {'mAP50':>8s} {'mAP50:95':>10s} {'Prec':>8s} {'Recall':>8s}"
    if any("syntax_mAP50" in m.get("final_test", {}) for m in runs.values()):
        header += f" {'Syn mAP50':>10s} {'Sten AP50':>10s}"
    print(header)
    print("-" * len(header))

    for run_name, metrics in sorted(runs.items()):
        test = metrics.get("final_test", {})
        if not test:
            print(f"  {run_name:>20s}  (no test metrics)")
            continue

        line = (f"  {run_name:>20s} "
                f"{test.get('mAP50', 0):>8.4f} "
                f"{test.get('mAP50_95', 0):>10.4f} "
                f"{test.get('precision', 0):>8.4f} "
                f"{test.get('recall', 0):>8.4f}")
        if "syntax_mAP50" in test:
            line += f" {test['syntax_mAP50']:>10.4f}"
        if "stenosis_AP50" in test:
            line += f" {test['stenosis_AP50']:>10.4f}"
        print(line)


def print_per_class_comparison(runs: dict) -> None:
    """Print per-class AP50 comparison across runs."""
    print("\n" + "=" * 80)
    print("PER-CLASS AP50 COMPARISON (Final Test)")
    print("=" * 80)

    # Collect all class names
    all_classes = set()
    for metrics in runs.values():
        test = metrics.get("final_test", {})
        per_class = test.get("per_class", {})
        all_classes.update(per_class.keys())

    if not all_classes:
        print("  No per-class metrics available.")
        return

    run_names = sorted(runs.keys())
    header = f"{'Class':>10s}" + "".join(f" {n:>15s}" for n in run_names)
    print(header)
    print("-" * len(header))

    for cls_name in sorted(all_classes):
        line = f"  {cls_name:>10s}"
        for run_name in run_names:
            test = runs[run_name].get("final_test", {})
            per_class = test.get("per_class", {})
            cls_metrics = per_class.get(cls_name, {})
            ap50 = cls_metrics.get("ap50", 0)
            line += f" {ap50:>15.4f}"
        print(line)


def print_iteration_progression(runs: dict) -> None:
    """Print mAP progression across iterations for each run."""
    print("\n" + "=" * 80)
    print("ITERATION PROGRESSION (mAP@50)")
    print("=" * 80)

    for run_name, metrics in sorted(runs.items()):
        print(f"\n  {run_name}:")
        stages = sorted(
            [(k, v) for k, v in metrics.items()
             if isinstance(v, dict) and "mAP50" in v],
            key=lambda x: x[0]
        )
        for stage_name, stage_metrics in stages:
            map50 = stage_metrics.get("mAP50", 0)
            print(f"    {stage_name:>30s}: mAP50={map50:.4f}")


def print_pseudo_label_summary(all_pl_stats: dict) -> None:
    """Print pseudo-label statistics across runs."""
    print("\n" + "=" * 80)
    print("PSEUDO-LABEL STATISTICS")
    print("=" * 80)

    for run_name, pl_stats in sorted(all_pl_stats.items()):
        print(f"\n  {run_name}:")
        for stage, stats in sorted(pl_stats.items()):
            total = stats.get("total_images", 0)
            labeled = stats.get("images_with_predictions", 0)
            preds = stats.get("total_predictions", 0)
            conf = stats.get("confidence", {})
            conf_mean = conf.get("mean", 0) if conf else 0

            print(f"    {stage:>40s}: "
                  f"{labeled}/{total} images, "
                  f"{preds} predictions, "
                  f"conf_mean={conf_mean:.3f}")


def generate_comparison_json(runs: dict, all_pl_stats: dict,
                              output_path: Path) -> None:
    """Save full comparison data as JSON."""
    comparison = {
        "overall": {},
        "per_class": {},
        "iterations": {},
        "pseudo_labels": all_pl_stats,
    }

    for run_name, metrics in runs.items():
        test = metrics.get("final_test", {})
        comparison["overall"][run_name] = {
            "mAP50": test.get("mAP50", 0),
            "mAP50_95": test.get("mAP50_95", 0),
            "precision": test.get("precision", 0),
            "recall": test.get("recall", 0),
        }
        comparison["per_class"][run_name] = test.get("per_class", {})

        # Iteration progression
        stages = {
            k: {"mAP50": v.get("mAP50", 0)}
            for k, v in metrics.items()
            if isinstance(v, dict) and "mAP50" in v
        }
        comparison["iterations"][run_name] = stages

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"\n  Full comparison saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare results across experiment runs"
    )
    parser.add_argument(
        "--results-root", type=str, default="results",
        help="Root directory containing run result directories"
    )
    parser.add_argument(
        "--runs", type=str, nargs="+", default=None,
        help="Specific run directories to compare (default: all in results-root)"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Path to save comparison JSON"
    )
    args = parser.parse_args()

    results_root = Path(args.results_root)

    # Find run directories
    if args.runs:
        run_dirs = {name: results_root / name for name in args.runs}
    else:
        run_dirs = {
            d.name: d for d in sorted(results_root.iterdir())
            if d.is_dir() and not d.name.startswith(".")
        }

    if not run_dirs:
        print(f"No run directories found in {results_root}")
        return

    print(f"Comparing {len(run_dirs)} runs: {list(run_dirs.keys())}")

    # Load metrics
    all_runs = {}
    all_pl_stats = {}
    for run_name, run_dir in run_dirs.items():
        all_runs[run_name] = load_metrics(run_dir)
        all_pl_stats[run_name] = load_pseudo_label_stats(run_dir)

    # Print comparisons
    print_overall_comparison(all_runs)
    print_per_class_comparison(all_runs)
    print_iteration_progression(all_runs)
    print_pseudo_label_summary(all_pl_stats)

    # Save comparison JSON
    output_path = Path(args.output) if args.output else results_root / "comparison.json"
    generate_comparison_json(all_runs, all_pl_stats, output_path)


if __name__ == "__main__":
    main()
