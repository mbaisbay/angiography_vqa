"""Run model comparison across two dataset variants.

Dataset A: Stratified splits + ALL 25 SYNTAX classes (min_count=0)
Dataset B: Stratified splits + 10 SYNTAX classes (min_count=300)

For each dataset, trains each model through the full iterative pipeline,
then outputs a comparison table with F1 scores.

Usage:
    python run_comparison.py \
        --arcade-root ../../arcade/submission \
        --models yolo11m-seg yolov8m-seg \
        --iterations 3
"""

import argparse
import json
import sys
import time
from pathlib import Path

from create_stratified_splits import create_stratified_splits


def build_run_config(model: str, results_dir: str, batch: int = 16) -> dict:
    """Build a run config dict for a given model."""
    # Smaller batch for large models
    if "11x" in model or "v8x" in model or "11l" in model or "v8l" in model:
        batch = 8

    cfg = {
        "run_name": Path(model).stem.replace("-", "_").replace(".", ""),
        "model": model,
        "imgsz": 512,
        "batch": batch,
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
        "device": "0",
        "workers": 4,
        # Augmentation
        "mosaic": 0.0,
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
        # Pseudo-labeling
        "pseudo_label": {
            "initial_conf": 0.85,
            "conf_decay": 0.05,
            "min_conf": 0.65,
            "use_one_to_many": False,
        },
        "data_dir": "../data",
        "results_dir": results_dir,
    }
    return cfg


def run_single_experiment(
    model: str,
    dataset_name: str,
    min_count: int,
    arcade_root: Path,
    splits_dir: Path,
    base_results_dir: Path,
    iterations: int,
) -> dict:
    """Run a single model on a single dataset variant."""
    from run_pipeline import run_pipeline
    import yaml

    run_name = f"{Path(model).stem.replace('-', '_')}_{dataset_name}"
    results_dir = base_results_dir / run_name
    results_dir.mkdir(parents=True, exist_ok=True)

    # Create a temporary config file
    cfg = build_run_config(model, str(results_dir))
    config_path = results_dir / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    print(f"\n{'#' * 60}")
    print(f"# EXPERIMENT: {run_name}")
    print(f"#   Model: {model}")
    print(f"#   Dataset: {dataset_name} (min_count={min_count})")
    print(f"#   Results: {results_dir}")
    print(f"{'#' * 60}")

    start = time.time()

    run_pipeline(
        config_path=str(config_path),
        arcade_root=str(arcade_root),
        iterations=iterations,
        splits_dir=str(splits_dir),
        min_count=min_count,
    )

    elapsed = time.time() - start

    # Load final metrics
    metrics_path = results_dir / "all_metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            all_metrics = json.load(f)
    else:
        all_metrics = {}

    return {
        "model": model,
        "dataset": dataset_name,
        "min_count": min_count,
        "run_name": run_name,
        "elapsed_hours": round(elapsed / 3600, 2),
        "metrics": all_metrics,
    }


def compute_f1(precision: float, recall: float) -> float:
    """Compute F1 score from precision and recall."""
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def print_comparison_table(results: list) -> None:
    """Print a formatted comparison table with F1 scores."""
    print("\n" + "=" * 90)
    print("COMPARISON RESULTS")
    print("=" * 90)

    # Header
    print(f"\n{'Model':<20} {'Dataset':<15} {'mAP50':>8} {'mAP50:95':>10} "
          f"{'Prec':>8} {'Recall':>8} {'F1':>8} "
          f"{'Syn mAP50':>10} {'Sten AP50':>10}")
    print("-" * 90)

    for r in results:
        test = r["metrics"].get("final_test", {})
        p = test.get("precision", 0)
        rec = test.get("recall", 0)
        f1 = compute_f1(p, rec)

        print(f"{r['model']:<20} {r['dataset']:<15} "
              f"{test.get('mAP50', 0):>8.4f} "
              f"{test.get('mAP50_95', 0):>10.4f} "
              f"{p:>8.4f} {rec:>8.4f} {f1:>8.4f} "
              f"{test.get('syntax_mAP50', 0):>10.4f} "
              f"{test.get('stenosis_AP50', 0):>10.4f}")

    # Per-class F1 table
    print(f"\n{'=' * 90}")
    print("PER-CLASS F1 SCORES (Final Test)")
    print("=" * 90)

    # Collect all class names
    all_classes = set()
    for r in results:
        test = r["metrics"].get("final_test", {})
        per_class = test.get("per_class", {})
        all_classes.update(per_class.keys())

    if all_classes:
        sorted_classes = sorted(all_classes)
        header = f"{'Model':<20} {'Dataset':<15}"
        for cls in sorted_classes:
            header += f" {cls:>8}"
        print(header)
        print("-" * len(header))

        for r in results:
            test = r["metrics"].get("final_test", {})
            per_class = test.get("per_class", {})
            line = f"{r['model']:<20} {r['dataset']:<15}"
            for cls in sorted_classes:
                cls_m = per_class.get(cls, {})
                f1 = cls_m.get("f1", 0)
                line += f" {f1:>8.4f}"
            print(line)

    # Iteration progression
    print(f"\n{'=' * 90}")
    print("ITERATION PROGRESSION (Val mAP50)")
    print("=" * 90)

    for r in results:
        metrics = r["metrics"]
        print(f"\n  {r['model']} / {r['dataset']}:")
        for stage in sorted(metrics.keys()):
            if stage != "final_test":
                m = metrics[stage]
                print(f"    {stage:<30s}: mAP50={m.get('mAP50', 0):.4f}")
        test = metrics.get("final_test", {})
        print(f"    {'final_test':<30s}: mAP50={test.get('mAP50', 0):.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare models across dataset variants"
    )
    parser.add_argument(
        "--arcade-root", type=str, default="../../arcade/submission",
        help="Path to arcade/submission directory"
    )
    parser.add_argument(
        "--models", nargs="+",
        default=["yolo11m-seg.pt", "yolov8m-seg.pt"],
        help="YOLO model names to compare (default: yolo11m-seg.pt yolov8m-seg.pt)"
    )
    parser.add_argument(
        "--iterations", type=int, default=3,
        help="Number of pseudo-label iterations (default: 3)"
    )
    parser.add_argument(
        "--output-dir", type=str, default="../results/comparison",
        help="Base output directory for all results"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for stratified splits (default: 42)"
    )
    parser.add_argument(
        "--skip-splits", action="store_true",
        help="Skip stratified split creation (if already done)"
    )
    args = parser.parse_args()

    arcade_root = Path(args.arcade_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    splits_dir = output_dir / "stratified_splits"

    # ── Step 1: Create stratified splits (shared by both datasets) ──
    if not args.skip_splits:
        print("\n" + "#" * 60)
        print("# STEP 1: Create stratified splits")
        print("#" * 60)
        create_stratified_splits(arcade_root, splits_dir, args.seed)
    else:
        print("\n[SKIP] Stratified split creation (--skip-splits)")
        if not splits_dir.exists():
            print(f"ERROR: {splits_dir} does not exist. Remove --skip-splits.")
            sys.exit(1)

    # ── Step 2: Run experiments ──
    # Dataset A: all classes (min_count=0)
    # Dataset B: filtered classes (min_count=300)
    datasets = [
        ("all_classes", 0),
        ("filtered_classes", 300),
    ]

    all_results = []
    total_start = time.time()

    for model in args.models:
        for dataset_name, min_count in datasets:
            result = run_single_experiment(
                model=model,
                dataset_name=dataset_name,
                min_count=min_count,
                arcade_root=arcade_root,
                splits_dir=splits_dir,
                base_results_dir=output_dir,
                iterations=args.iterations,
            )
            all_results.append(result)

            # Save intermediate results
            with open(output_dir / "comparison_results.json", "w") as f:
                json.dump(all_results, f, indent=2, default=str)

    total_elapsed = time.time() - total_start

    # ── Step 3: Print comparison ──
    print_comparison_table(all_results)

    # Save final results
    with open(output_dir / "comparison_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n{'=' * 90}")
    print(f"ALL EXPERIMENTS COMPLETE ({total_elapsed / 3600:.1f} hours)")
    print(f"Results saved: {output_dir / 'comparison_results.json'}")
    print(f"{'=' * 90}")


if __name__ == "__main__":
    main()
