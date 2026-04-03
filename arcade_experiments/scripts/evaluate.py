"""Comprehensive evaluation for ARCADE YOLO segmentation models.

Reports:
  - Overall mAP@50, mAP@50:95
  - Per-class Precision, Recall, AP50
  - Separate metrics for SYNTAX classes vs stenosis
  - For YOLO26: evaluates both end2end heads
  - Saves results to JSON + prints human-readable table
"""

import argparse
import json
from pathlib import Path

import yaml
from ultralytics import YOLO


def evaluate_model(model_path: str, data_yaml: str,
                   split: str = "val") -> dict:
    """Run standard ultralytics validation and extract metrics.

    Args:
        model_path: Path to trained model weights.
        data_yaml: Path to YOLO dataset YAML.
        split: Which split to evaluate on (val/test).

    Returns:
        Dict with mAP, per-class metrics, etc.
    """
    model = YOLO(model_path)
    results = model.val(
        data=data_yaml,
        split=split,
        imgsz=512,
        save_json=True,
        verbose=False,
    )

    # Overall segmentation metrics
    metrics = {
        "split": split,
        "model": model_path,
        "mAP50": round(float(results.seg.map50), 4),
        "mAP50_95": round(float(results.seg.map), 4),
        "precision": round(float(results.seg.mp), 4),
        "recall": round(float(results.seg.mr), 4),
    }

    # Per-class metrics
    with open(data_yaml, "r") as f:
        data_cfg = yaml.safe_load(f)
    names = data_cfg.get("names", {})
    if isinstance(names, list):
        names = {i: n for i, n in enumerate(names)}
    else:
        names = {int(k): v for k, v in names.items()}

    per_class = {}
    if hasattr(results.seg, "ap50") and results.seg.ap50 is not None:
        ap50 = results.seg.ap50
        for i, ap_val in enumerate(ap50):
            cls_name = names.get(i, str(i))
            per_class[cls_name] = {
                "ap50": round(float(ap_val), 4),
            }

    if hasattr(results.seg, "p") and results.seg.p is not None:
        for i, p_val in enumerate(results.seg.p):
            cls_name = names.get(i, str(i))
            if cls_name not in per_class:
                per_class[cls_name] = {}
            per_class[cls_name]["precision"] = round(float(p_val), 4)

    if hasattr(results.seg, "r") and results.seg.r is not None:
        for i, r_val in enumerate(results.seg.r):
            cls_name = names.get(i, str(i))
            if cls_name not in per_class:
                per_class[cls_name] = {}
            per_class[cls_name]["recall"] = round(float(r_val), 4)

    # Compute F1 for each class
    for cls_name, cls_metrics in per_class.items():
        p = cls_metrics.get("precision", 0)
        r = cls_metrics.get("recall", 0)
        if p + r > 0:
            cls_metrics["f1"] = round(2 * p * r / (p + r), 4)
        else:
            cls_metrics["f1"] = 0.0

    metrics["per_class"] = per_class

    # Separate SYNTAX vs stenosis metrics
    syntax_aps, stenosis_aps = [], []
    for cls_name, cls_metrics in per_class.items():
        ap = cls_metrics.get("ap50", 0)
        if cls_name == "stenosis":
            stenosis_aps.append(ap)
        else:
            syntax_aps.append(ap)

    if syntax_aps:
        metrics["syntax_mAP50"] = round(sum(syntax_aps) / len(syntax_aps), 4)
    if stenosis_aps:
        metrics["stenosis_AP50"] = round(sum(stenosis_aps) / len(stenosis_aps), 4)

    return metrics


def evaluate_yolo26_both_heads(model_path: str, data_yaml: str,
                                split: str = "val") -> dict:
    """Evaluate YOLO26 with both end2end (one-to-one) and one-to-many heads."""
    # One-to-one (default, deployment)
    print("  Evaluating YOLO26 one-to-one head (deployment)...")
    metrics_o2o = evaluate_model(model_path, data_yaml, split)
    metrics_o2o["head"] = "one-to-one"

    # One-to-many
    print("  Evaluating YOLO26 one-to-many head...")
    model = YOLO(model_path)
    try:
        model.model.model[-1].end2end = False
    except (AttributeError, IndexError):
        print("  WARNING: Could not set end2end=False, skipping o2m eval")
        return {"one_to_one": metrics_o2o}

    results = model.val(data=data_yaml, split=split, imgsz=512, verbose=False)
    metrics_o2m = {
        "head": "one-to-many",
        "split": split,
        "mAP50": round(float(results.seg.map50), 4),
        "mAP50_95": round(float(results.seg.map), 4),
        "precision": round(float(results.seg.mp), 4),
        "recall": round(float(results.seg.mr), 4),
    }

    return {
        "one_to_one": metrics_o2o,
        "one_to_many": metrics_o2m,
    }


def print_metrics(metrics: dict) -> None:
    """Print metrics as a human-readable table."""
    print(f"\n  Overall Metrics ({metrics.get('split', 'val')}):")
    print(f"    mAP@50:    {metrics.get('mAP50', 0):.4f}")
    print(f"    mAP@50:95: {metrics.get('mAP50_95', 0):.4f}")
    print(f"    Precision: {metrics.get('precision', 0):.4f}")
    print(f"    Recall:    {metrics.get('recall', 0):.4f}")

    if "syntax_mAP50" in metrics:
        print(f"    SYNTAX mAP@50:  {metrics['syntax_mAP50']:.4f}")
    if "stenosis_AP50" in metrics:
        print(f"    Stenosis AP@50: {metrics['stenosis_AP50']:.4f}")

    per_class = metrics.get("per_class", {})
    if per_class:
        print(f"\n  Per-class metrics:")
        print(f"    {'Class':>10s} {'AP50':>8s} {'Prec':>8s} {'Recall':>8s} {'F1':>8s}")
        print(f"    {'-'*42}")
        for cls_name in sorted(per_class.keys()):
            cm = per_class[cls_name]
            print(f"    {cls_name:>10s} "
                  f"{cm.get('ap50', 0):>8.4f} "
                  f"{cm.get('precision', 0):>8.4f} "
                  f"{cm.get('recall', 0):>8.4f} "
                  f"{cm.get('f1', 0):>8.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate YOLO segmentation model on ARCADE dataset"
    )
    parser.add_argument(
        "--model", type=str, required=True,
        help="Path to model weights (.pt)"
    )
    parser.add_argument(
        "--data-yaml", type=str, required=True,
        help="Path to YOLO dataset YAML"
    )
    parser.add_argument(
        "--split", type=str, default="val", choices=["val", "test"],
        help="Which split to evaluate (default: val)"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Path to save metrics JSON"
    )
    parser.add_argument(
        "--yolo26", action="store_true",
        help="Evaluate both YOLO26 heads (one-to-one and one-to-many)"
    )
    args = parser.parse_args()

    print("=" * 60)
    print(f"Evaluating: {args.model}")
    print(f"Data:       {args.data_yaml}")
    print(f"Split:      {args.split}")
    print("=" * 60)

    if args.yolo26:
        metrics = evaluate_yolo26_both_heads(
            args.model, args.data_yaml, args.split
        )
        for head_name, head_metrics in metrics.items():
            print(f"\n  [{head_name.upper()}]")
            print_metrics(head_metrics)
    else:
        metrics = evaluate_model(args.model, args.data_yaml, args.split)
        print_metrics(metrics)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"\n  Saved: {output_path}")


if __name__ == "__main__":
    main()
