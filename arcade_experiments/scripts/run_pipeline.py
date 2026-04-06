"""Orchestrate the full iterative cross-training pipeline.

Pipeline flow:
  0. Data prep: filter SYNTAX -> convert COCO->YOLO -> grayscale->3ch -> dataset YAMLs
  1. Train on SYNTAX only (10 classes, IDs 0-9) -> model_syntax_v1
  2. model_syntax_v1 inference on STENOSIS train images -> pseudo SYNTAX labels
  3. Merge: stenosis GT (class 10) + pseudo SYNTAX (0-9) -> train model_combined_v1
  4. model_combined_v1 on SYNTAX train images -> pseudo stenosis (class 10)
  5+ Iterate: merge ALL + retrain from COCO pretrained + lower confidence

Key rules:
  - Always retrain from COCO pretrained weights (not previous iteration)
  - Confidence schedule: initial_conf, initial_conf - decay, ...
  - val/test NEVER get pseudo labels
  - If mAP degrades vs previous iteration, log warning
"""

import argparse
import json
import shutil
import sys
import time
import yaml
from pathlib import Path

from train import load_run_config, train_two_stage
from predict_pseudo_labels import generate_pseudo_labels, print_stats, save_stats
from merge_datasets import merge_datasets
from evaluate import evaluate_model


def data_prep(arcade_root: Path, data_dir: Path, min_count: int = 300,
              skip_images: bool = False) -> dict:
    """Run data preparation (Step 0)."""
    print("\n" + "#" * 60)
    print("# STEP 0: Data Preparation")
    print("#" * 60)

    from prepare_data import (
        prepare_syntax, prepare_stenosis,
        convert_images, generate_dataset_yamls,
    )

    syntax_mapping = prepare_syntax(arcade_root, data_dir, min_count)
    prepare_stenosis(arcade_root, data_dir)

    if not skip_images:
        convert_images(arcade_root, data_dir)

    generate_dataset_yamls(data_dir, syntax_mapping)

    return syntax_mapping


def stage1_train_syntax(cfg: dict, data_dir: Path, results_dir: Path) -> str:
    """Stage 1: Train on SYNTAX only (10 classes)."""
    print("\n" + "#" * 60)
    print("# STAGE 1: Train on SYNTAX only")
    print("#" * 60)

    syntax_yaml = str(data_dir / "dataset_configs" / "syntax_only.yaml")
    weights_path = train_two_stage(
        cfg, syntax_yaml,
        project=str(results_dir),
        run_name="stage1_syntax",
    )

    return weights_path


def stage2_pseudo_label_stenosis(
    model_path: str, data_dir: Path, results_dir: Path,
    conf: float, use_o2m: bool = False,
) -> Path:
    """Stage 2: Generate pseudo SYNTAX labels for stenosis images."""
    print("\n" + "#" * 60)
    print(f"# STAGE 2: Pseudo-label stenosis images (conf>={conf})")
    print("#" * 60)

    image_dir = str(data_dir / "stenosis" / "images" / "train")
    output_dir = results_dir / "pseudo_labels" / "syntax_on_stenosis"
    output_dir.mkdir(parents=True, exist_ok=True)

    stats = generate_pseudo_labels(
        model_path=model_path,
        image_dir=image_dir,
        output_label_dir=str(output_dir),
        conf_threshold=conf,
        class_offset=0,  # syntax model outputs 0-9, no offset needed
        use_one_to_many=use_o2m,
    )
    print_stats(stats)
    save_stats(stats, str(output_dir / "stats.json"))

    return output_dir


def stage3_merge_and_train_combined(
    cfg: dict, data_dir: Path, results_dir: Path,
    pseudo_syntax_dir: Path, iteration: int,
) -> str:
    """Stage 3: Merge stenosis GT + pseudo SYNTAX, train combined model."""
    print("\n" + "#" * 60)
    print(f"# STAGE 3: Merge + Train combined (iteration {iteration})")
    print("#" * 60)

    merged_dir = data_dir / f"merged_iter{iteration}_stage3"
    merged_yaml = data_dir / "dataset_configs" / f"merged_iter{iteration}_stage3.yaml"
    class_mapping = str(data_dir / "syntax_filtered" / "class_mapping.json")

    # Merge: stenosis images get GT stenosis + pseudo SYNTAX
    # syntax images are NOT included yet (only stenosis images for Stage 3)
    merge_datasets(
        syntax_data_dir=data_dir / "syntax_filtered",
        stenosis_data_dir=data_dir / "stenosis",
        pseudo_syntax_dir=pseudo_syntax_dir,  # pseudo SYNTAX for stenosis images
        pseudo_stenosis_dir=None,              # no pseudo stenosis for syntax yet
        output_dir=merged_dir,
        class_names_json=class_mapping,
        yaml_output=merged_yaml,
    )

    # Train from COCO pretrained weights (NOT from previous model)
    weights_path = train_two_stage(
        cfg, str(merged_yaml),
        project=str(results_dir),
        run_name=f"stage3_combined_iter{iteration}",
    )

    return weights_path


def stage4_pseudo_label_syntax(
    model_path: str, data_dir: Path, results_dir: Path,
    conf: float, use_o2m: bool = False,
) -> Path:
    """Stage 4: Generate pseudo stenosis labels for syntax images."""
    print("\n" + "#" * 60)
    print(f"# STAGE 4: Pseudo-label syntax images (conf>={conf})")
    print("#" * 60)

    image_dir = str(data_dir / "syntax_filtered" / "images" / "train")
    output_dir = results_dir / "pseudo_labels" / "stenosis_on_syntax"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Combined model has 11 classes (0-9 syntax, 10 stenosis)
    # We only want the stenosis predictions (class 10)
    # The model outputs class 10 for stenosis, no offset needed
    stats = generate_pseudo_labels(
        model_path=model_path,
        image_dir=image_dir,
        output_label_dir=str(output_dir),
        conf_threshold=conf,
        class_offset=0,
        use_one_to_many=use_o2m,
    )
    print_stats(stats)
    save_stats(stats, str(output_dir / "stats.json"))

    # Filter to only keep stenosis predictions (class 10)
    _filter_pseudo_to_class(output_dir, target_class=10)

    return output_dir


def _filter_pseudo_to_class(labels_dir: Path, target_class: int) -> None:
    """Keep only lines with the target class in pseudo-label files."""
    for label_file in labels_dir.glob("*.txt"):
        lines = []
        with open(label_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if parts and int(parts[0]) == target_class:
                    lines.append(line.strip())
        with open(label_file, "w") as f:
            f.write("\n".join(lines) + "\n" if lines else "")


def full_merge_and_train(
    cfg: dict, data_dir: Path, results_dir: Path,
    pseudo_syntax_dir: Path, pseudo_stenosis_dir: Path,
    iteration: int,
) -> str:
    """Full merge: ALL images with GT + pseudo, then retrain."""
    print("\n" + "#" * 60)
    print(f"# ITERATION {iteration}: Full merge + retrain")
    print("#" * 60)

    merged_dir = data_dir / f"merged_iter{iteration}"
    merged_yaml = data_dir / "dataset_configs" / f"merged_iter{iteration}.yaml"
    class_mapping = str(data_dir / "syntax_filtered" / "class_mapping.json")

    # Clean previous merge if exists
    if merged_dir.exists():
        shutil.rmtree(merged_dir)

    merge_datasets(
        syntax_data_dir=data_dir / "syntax_filtered",
        stenosis_data_dir=data_dir / "stenosis",
        pseudo_syntax_dir=pseudo_syntax_dir,
        pseudo_stenosis_dir=pseudo_stenosis_dir,
        output_dir=merged_dir,
        class_names_json=class_mapping,
        yaml_output=merged_yaml,
    )

    # Always retrain from COCO pretrained weights
    weights_path = train_two_stage(
        cfg, str(merged_yaml),
        project=str(results_dir),
        run_name=f"iter{iteration}_merged",
    )

    return weights_path


def run_pipeline(
    config_path: str,
    arcade_root: str,
    iterations: int = 3,
    initial_conf: float = 0.85,
    conf_decay: float = 0.05,
    skip_data_prep: bool = False,
    skip_stage1: bool = False,
    stage1_weights: str = None,
) -> None:
    """Run the full iterative cross-training pipeline."""
    cfg = load_run_config(config_path)
    config_dir = Path(config_path).resolve().parent

    arcade_root = Path(arcade_root).resolve()
    # Resolve config paths relative to the config file's directory
    data_dir = (config_dir / cfg.get("data_dir", "../data")).resolve()
    results_dir = (config_dir / cfg.get("results_dir", "../results")).resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    use_o2m = cfg.get("pseudo_label", {}).get("use_one_to_many", False)

    # Override pseudo-label config from CLI if provided
    pl_cfg = cfg.get("pseudo_label", {})
    if initial_conf != 0.85:
        pl_cfg["initial_conf"] = initial_conf
    if conf_decay != 0.05:
        pl_cfg["conf_decay"] = conf_decay

    start_time = time.time()

    # Save pipeline config for reproducibility
    pipeline_state = {
        "config_path": config_path,
        "arcade_root": str(arcade_root),
        "iterations": iterations,
        "initial_conf": initial_conf,
        "conf_decay": conf_decay,
        "run_config": cfg,
    }
    with open(results_dir / "pipeline_config.json", "w") as f:
        json.dump(pipeline_state, f, indent=2)

    # ── Step 0: Data Preparation ──
    if not skip_data_prep:
        data_prep(arcade_root, data_dir)
    else:
        print("\n[SKIP] Data preparation (--skip-data-prep)")

    # ── Stage 1: Train on SYNTAX only ──
    if skip_stage1 and stage1_weights:
        syntax_weights = stage1_weights
        print(f"\n[SKIP] Stage 1 (using provided weights: {syntax_weights})")
    else:
        syntax_weights = stage1_train_syntax(cfg, data_dir, results_dir)

    # Evaluate Stage 1
    syntax_yaml = str(data_dir / "dataset_configs" / "syntax_only.yaml")
    print("\n  Evaluating Stage 1 model on syntax val...")
    stage1_metrics = evaluate_model(syntax_weights, syntax_yaml, split="val")
    _save_metrics(results_dir, "stage1_syntax", stage1_metrics)

    # ── Stage 2: Pseudo-label stenosis images ──
    conf = initial_conf
    pseudo_syntax_dir = stage2_pseudo_label_stenosis(
        syntax_weights, data_dir, results_dir, conf, use_o2m
    )

    # ── Stage 3: Merge stenosis + train combined ──
    combined_weights = stage3_merge_and_train_combined(
        cfg, data_dir, results_dir, pseudo_syntax_dir, iteration=1
    )

    # Evaluate Stage 3
    merged_yaml_3 = str(data_dir / "dataset_configs" / "merged_iter1_stage3.yaml")
    print("\n  Evaluating Stage 3 model...")
    stage3_metrics = evaluate_model(combined_weights, merged_yaml_3, split="val")
    _save_metrics(results_dir, "stage3_combined_iter1", stage3_metrics)

    # ── Stage 4: Pseudo-label syntax images ──
    pseudo_stenosis_dir = stage4_pseudo_label_syntax(
        combined_weights, data_dir, results_dir, conf, use_o2m
    )

    # ── Iterations 2+ ──
    prev_metrics = stage3_metrics
    all_metrics = {
        "stage1_syntax": stage1_metrics,
        "stage3_combined_iter1": stage3_metrics,
    }

    for iteration in range(2, iterations + 1):
        conf = max(initial_conf - conf_decay * (iteration - 1),
                   pl_cfg.get("min_conf", 0.65))

        print(f"\n{'#' * 60}")
        print(f"# ITERATION {iteration} (conf >= {conf})")
        print(f"{'#' * 60}")

        # Full merge and retrain
        iter_weights = full_merge_and_train(
            cfg, data_dir, results_dir,
            pseudo_syntax_dir, pseudo_stenosis_dir,
            iteration=iteration,
        )

        # Evaluate
        merged_yaml = str(data_dir / "dataset_configs" / f"merged_iter{iteration}.yaml")
        iter_metrics = evaluate_model(iter_weights, merged_yaml, split="val")
        _save_metrics(results_dir, f"iter{iteration}_merged", iter_metrics)
        all_metrics[f"iter{iteration}_merged"] = iter_metrics

        # Check for degradation
        if prev_metrics and iter_metrics:
            prev_map = prev_metrics.get("mAP50", 0)
            curr_map = iter_metrics.get("mAP50", 0)
            if curr_map < prev_map - 0.01:
                print(f"\n  WARNING: mAP50 degraded: {prev_map:.4f} -> {curr_map:.4f}")
                print(f"  Consider stopping iterations.")
            else:
                print(f"\n  mAP50: {prev_map:.4f} -> {curr_map:.4f} "
                      f"({'improved' if curr_map > prev_map else 'stable'})")

        prev_metrics = iter_metrics

        # Update pseudo labels for next iteration
        pseudo_syntax_dir_new = results_dir / "pseudo_labels" / f"syntax_on_stenosis_iter{iteration}"
        pseudo_syntax_dir_new.mkdir(parents=True, exist_ok=True)
        stats = generate_pseudo_labels(
            model_path=iter_weights,
            image_dir=str(data_dir / "stenosis" / "images" / "train"),
            output_label_dir=str(pseudo_syntax_dir_new),
            conf_threshold=conf,
            class_offset=0,
            use_one_to_many=use_o2m,
        )
        save_stats(stats, str(pseudo_syntax_dir_new / "stats.json"))
        pseudo_syntax_dir = pseudo_syntax_dir_new

        pseudo_stenosis_dir_new = results_dir / "pseudo_labels" / f"stenosis_on_syntax_iter{iteration}"
        pseudo_stenosis_dir_new.mkdir(parents=True, exist_ok=True)
        stats = generate_pseudo_labels(
            model_path=iter_weights,
            image_dir=str(data_dir / "syntax_filtered" / "images" / "train"),
            output_label_dir=str(pseudo_stenosis_dir_new),
            conf_threshold=conf,
            class_offset=0,
            use_one_to_many=use_o2m,
        )
        save_stats(stats, str(pseudo_stenosis_dir_new / "stats.json"))
        _filter_pseudo_to_class(pseudo_stenosis_dir_new, target_class=10)
        pseudo_stenosis_dir = pseudo_stenosis_dir_new

    # ── Final evaluation on test set ──
    print("\n" + "#" * 60)
    print("# FINAL: Test set evaluation")
    print("#" * 60)

    # Use the last iteration's weights and merged YAML
    last_iter = iterations
    final_weights = iter_weights if iterations >= 2 else combined_weights
    final_yaml = str(data_dir / "dataset_configs" / (
        f"merged_iter{last_iter}.yaml" if iterations >= 2
        else "merged_iter1_stage3.yaml"
    ))

    test_metrics = evaluate_model(final_weights, final_yaml, split="test")
    _save_metrics(results_dir, "final_test", test_metrics)
    all_metrics["final_test"] = test_metrics

    # Save all metrics
    with open(results_dir / "all_metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)

    elapsed = time.time() - start_time
    print(f"\n{'#' * 60}")
    print(f"# PIPELINE COMPLETE ({elapsed / 3600:.1f} hours)")
    print(f"# Results: {results_dir}")
    print(f"# Final weights: {final_weights}")
    print(f"{'#' * 60}")


def _save_metrics(results_dir: Path, stage_name: str, metrics: dict) -> None:
    """Save metrics for a pipeline stage."""
    metrics_dir = results_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    with open(metrics_dir / f"{stage_name}.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Metrics saved: {metrics_dir / f'{stage_name}.json'}")


def main():
    parser = argparse.ArgumentParser(
        description="Run the full iterative cross-training pipeline"
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to run config YAML (e.g., configs/run1_yolo11m.yaml)"
    )
    parser.add_argument(
        "--arcade-root", type=str, default="../arcade/submission",
        help="Path to arcade/submission directory"
    )
    parser.add_argument(
        "--iterations", type=int, default=3,
        help="Number of pseudo-label iterations (default: 3)"
    )
    parser.add_argument(
        "--initial-conf", type=float, default=0.85,
        help="Starting confidence threshold (default: 0.85)"
    )
    parser.add_argument(
        "--conf-decay", type=float, default=0.05,
        help="Confidence reduction per iteration (default: 0.05)"
    )
    parser.add_argument(
        "--skip-data-prep", action="store_true",
        help="Skip data preparation (already done)"
    )
    parser.add_argument(
        "--skip-stage1", action="store_true",
        help="Skip Stage 1 training (provide --stage1-weights)"
    )
    parser.add_argument(
        "--stage1-weights", type=str, default=None,
        help="Pre-trained Stage 1 weights (used with --skip-stage1)"
    )
    args = parser.parse_args()

    run_pipeline(
        config_path=args.config,
        arcade_root=args.arcade_root,
        iterations=args.iterations,
        initial_conf=args.initial_conf,
        conf_decay=args.conf_decay,
        skip_data_prep=args.skip_data_prep,
        skip_stage1=args.skip_stage1,
        stage1_weights=args.stage1_weights,
    )


if __name__ == "__main__":
    main()
