"""Extract final results from the 4-step YOLO pipeline and validate prediction quality.

This script:
1. Extracts and summarizes predictions from both cross-inference directions
2. Validates quality through multiple lenses:
   - Confidence distribution analysis
   - Spatial consistency checks (stenosis should be ON vessels)
   - Per-class prediction frequency analysis
   - Visual samples for manual inspection
3. Produces a quality report + visualizations

Usage:
    python extract_and_validate.py --config config.yaml
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from utils.config_loader import load_config
from utils.mask_utils import polygon_to_binary_mask, compute_intersection_over_smaller
from utils.visualization import draw_masks_overlay, VESSEL_PALETTE, STENOSIS_COLOR


SPLITS = ["train", "val", "test"]

# Combined class mapping
SYNTAX_CLASSES = list(range(25))   # 0-24
STENOSIS_CLASS = 25


def load_cross_inference(json_path: Path) -> dict:
    """Load cross-inference JSON results."""
    with open(json_path, "r") as f:
        return json.load(f)


# =========================================================================
# 1. EXTRACTION: Summarize what we got
# =========================================================================

def extract_summary(ci_data: dict, direction: str) -> dict:
    """Extract summary statistics from cross-inference results."""
    summary = {"direction": direction, "splits": {}}
    total_images = 0
    total_preds = 0
    total_with_stenosis = 0
    total_with_syntax = 0
    total_with_both = 0
    all_confidences = []
    stenosis_confidences = []
    syntax_confidences = []
    class_counts = Counter()

    for split in SPLITS:
        results = ci_data.get(split, [])
        n_images = len(results)
        n_with_stenosis = 0
        n_with_syntax = 0
        n_with_both = 0
        n_preds = 0

        for entry in results:
            preds = entry["predictions"]
            n_preds += len(preds)

            has_stenosis = False
            has_syntax = False
            for p in preds:
                cls = p["class_id"]
                conf = p["confidence"]
                all_confidences.append(conf)
                class_counts[cls] += 1

                if cls == STENOSIS_CLASS:
                    has_stenosis = True
                    stenosis_confidences.append(conf)
                else:
                    has_syntax = True
                    syntax_confidences.append(conf)

            if has_stenosis:
                n_with_stenosis += 1
            if has_syntax:
                n_with_syntax += 1
            if has_stenosis and has_syntax:
                n_with_both += 1

        summary["splits"][split] = {
            "images": n_images,
            "predictions": n_preds,
            "with_stenosis": n_with_stenosis,
            "with_syntax": n_with_syntax,
            "with_both": n_with_both,
        }
        total_images += n_images
        total_preds += n_preds
        total_with_stenosis += n_with_stenosis
        total_with_syntax += n_with_syntax
        total_with_both += n_with_both

    summary["total"] = {
        "images": total_images,
        "predictions": total_preds,
        "with_stenosis": total_with_stenosis,
        "with_syntax": total_with_syntax,
        "with_both": total_with_both,
    }
    summary["confidences"] = {
        "all": all_confidences,
        "stenosis": stenosis_confidences,
        "syntax": syntax_confidences,
    }
    summary["class_counts"] = class_counts
    return summary


def print_extraction_report(summary: dict, class_names: dict) -> None:
    """Print extraction summary."""
    d = summary["direction"]
    print(f"\n{'='*60}")
    print(f"  {d}")
    print(f"{'='*60}")
    print(f"  {'Split':<8} {'Images':>7} {'Preds':>7} {'w/ Sten':>8} {'w/ Syn':>8} {'w/ Both':>8}")
    print(f"  {'-'*50}")

    for split in SPLITS:
        s = summary["splits"].get(split, {})
        print(f"  {split:<8} {s.get('images',0):>7} {s.get('predictions',0):>7} "
              f"{s.get('with_stenosis',0):>8} {s.get('with_syntax',0):>8} "
              f"{s.get('with_both',0):>8}")

    t = summary["total"]
    print(f"  {'-'*50}")
    print(f"  {'TOTAL':<8} {t['images']:>7} {t['predictions']:>7} "
          f"{t['with_stenosis']:>8} {t['with_syntax']:>8} {t['with_both']:>8}")

    # Top predicted classes
    cc = summary["class_counts"]
    print(f"\n  Top predicted classes:")
    for cls_id, count in cc.most_common(10):
        name = class_names.get(cls_id, str(cls_id))
        print(f"    class {cls_id:>2d} ({name:>8s}): {count}")


# =========================================================================
# 2. VALIDATION: Confidence analysis
# =========================================================================

def validate_confidence(summary: dict, output_dir: Path) -> dict:
    """Analyze confidence distribution of predictions."""
    confs = summary["confidences"]
    direction = summary["direction"]
    report = {}

    for key, values in confs.items():
        if not values:
            continue
        arr = np.array(values)
        report[key] = {
            "count": len(arr),
            "mean": float(arr.mean()),
            "median": float(np.median(arr)),
            "std": float(arr.std()),
            "min": float(arr.min()),
            "max": float(arr.max()),
            "below_0.3": int((arr < 0.3).sum()),
            "below_0.5": int((arr < 0.5).sum()),
            "above_0.7": int((arr > 0.7).sum()),
            "above_0.9": int((arr > 0.9).sum()),
        }

    # Plot confidence distributions
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (key, values) in zip(axes, confs.items()):
        if not values:
            ax.set_title(f"{key}: no data")
            continue
        arr = np.array(values)
        ax.hist(arr, bins=50, color="steelblue", edgecolor="black", linewidth=0.3, alpha=0.8)
        ax.axvline(arr.mean(), color="red", linestyle="--", label=f"mean={arr.mean():.3f}")
        ax.axvline(np.median(arr), color="green", linestyle="--", label=f"median={np.median(arr):.3f}")
        ax.set_xlabel("Confidence")
        ax.set_ylabel("Count")
        ax.set_title(f"{direction}\n{key} (n={len(arr)})")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    safe_name = direction.replace("/", "_")
    fig.savefig(output_dir / f"{safe_name}_confidence.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_dir / f'{safe_name}_confidence.png'}")

    return report


# =========================================================================
# 3. VALIDATION: Spatial consistency (stenosis should overlap vessels)
# =========================================================================

def validate_spatial_consistency(ci_data: dict, direction: str,
                                mask_size: int = 512,
                                num_samples: int = 100) -> dict:
    """Check that stenosis predictions spatially overlap with vessel predictions.

    For each image that has BOTH stenosis and syntax predictions,
    compute overlap between each stenosis mask and the union of all vessel masks.
    High overlap = stenosis is ON a vessel = good.
    Low overlap = stenosis floating in empty space = suspicious.
    """
    overlaps = []
    unmatched_count = 0
    total_stenoses = 0
    sampled = 0

    for split in SPLITS:
        results = ci_data.get(split, [])
        for entry in results:
            preds = entry["predictions"]
            stenosis_preds = [p for p in preds if p["class_id"] == STENOSIS_CLASS]
            vessel_preds = [p for p in preds if p["class_id"] != STENOSIS_CLASS]

            if not stenosis_preds or not vessel_preds:
                continue
            if sampled >= num_samples:
                continue
            sampled += 1

            # Build union vessel mask
            vessel_union = np.zeros((mask_size, mask_size), dtype=np.uint8)
            for vp in vessel_preds:
                poly = vp.get("polygon_normalized", [])
                if len(poly) >= 3:
                    mask = polygon_to_binary_mask(poly, mask_size, mask_size)
                    vessel_union = np.maximum(vessel_union, mask)

            # Check each stenosis against vessel union
            for sp in stenosis_preds:
                total_stenoses += 1
                poly = sp.get("polygon_normalized", [])
                if len(poly) < 3:
                    continue
                sten_mask = polygon_to_binary_mask(poly, mask_size, mask_size)
                sten_area = sten_mask.sum()
                if sten_area == 0:
                    continue
                overlap_area = (sten_mask & vessel_union).sum()
                overlap_ratio = overlap_area / sten_area
                overlaps.append(overlap_ratio)
                if overlap_ratio < 0.1:
                    unmatched_count += 1

    if not overlaps:
        return {"status": "no_data"}

    arr = np.array(overlaps)
    report = {
        "total_stenoses_checked": total_stenoses,
        "images_sampled": sampled,
        "mean_overlap": float(arr.mean()),
        "median_overlap": float(np.median(arr)),
        "min_overlap": float(arr.min()),
        "pct_above_50": float((arr > 0.5).sum() / len(arr) * 100),
        "pct_above_80": float((arr > 0.8).sum() / len(arr) * 100),
        "pct_below_10": float((arr < 0.1).sum() / len(arr) * 100),
        "unmatched_count": unmatched_count,
    }
    return report


# =========================================================================
# 4. VALIDATION: Visual samples for manual QC
# =========================================================================

def generate_visual_samples(ci_data: dict, direction: str,
                            images_dir: Path, output_dir: Path,
                            class_names: dict, num_samples: int = 10) -> None:
    """Generate visual overlays for manual quality inspection.

    Picks images with both stenosis and vessel predictions, draws overlays.
    """
    vis_dir = output_dir / f"visual_qc_{direction}"
    vis_dir.mkdir(parents=True, exist_ok=True)

    # Collect images with both types
    candidates = []
    for split in SPLITS:
        results = ci_data.get(split, [])
        for entry in results:
            preds = entry["predictions"]
            has_sten = any(p["class_id"] == STENOSIS_CLASS for p in preds)
            has_syn = any(p["class_id"] != STENOSIS_CLASS for p in preds)
            if has_sten and has_syn:
                candidates.append((split, entry))

    if not candidates:
        print(f"  No images with both stenosis + syntax predictions")
        return

    rng = np.random.RandomState(42)
    indices = rng.choice(len(candidates), size=min(num_samples, len(candidates)), replace=False)

    for idx in indices:
        split, entry = candidates[idx]
        img_name = entry["image_name"]
        img_path = images_dir / split / "images" / img_name
        if not img_path.exists():
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        preds = entry["predictions"]
        vessel_preds = [p for p in preds if p["class_id"] != STENOSIS_CLASS]
        stenosis_preds = [p for p in preds if p["class_id"] == STENOSIS_CLASS]

        # Add class_name if missing
        for p in vessel_preds:
            if "class_name" not in p:
                p["class_name"] = class_names.get(p["class_id"], str(p["class_id"]))
        for p in stenosis_preds:
            if "class_name" not in p:
                p["class_name"] = "stenosis"

        # Draw vessel masks
        vis = draw_masks_overlay(img, vessel_preds, alpha=0.35, is_stenosis=False)

        # Draw stenosis on top (red, thicker outline)
        h, w = vis.shape[:2]
        for sp in stenosis_preds:
            poly = sp.get("polygon_normalized", [])
            if len(poly) < 3:
                continue
            points = np.array(poly, dtype=np.float32)
            points[:, 0] *= w
            points[:, 1] *= h
            points = points.astype(np.int32)

            overlay = vis.copy()
            cv2.fillPoly(overlay, [points], STENOSIS_COLOR)
            cv2.addWeighted(overlay, 0.4, vis, 0.6, 0, vis)
            cv2.polylines(vis, [points], True, (255, 255, 255), 2)

            cx = int(points[:, 0].mean())
            cy = int(points[:, 1].mean())
            conf = sp.get("confidence", 0)
            text = f"STEN {conf:.2f}"
            cv2.putText(vis, text, (cx - 20, cy - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        save_path = vis_dir / f"{split}_{img_name}"
        cv2.imwrite(str(save_path), vis)

    print(f"  Saved {min(num_samples, len(candidates))} visual QC samples to: {vis_dir}")


# =========================================================================
# 5. VALIDATION: Prediction frequency analysis per confidence threshold
# =========================================================================

def threshold_sensitivity(ci_data: dict, direction: str, output_dir: Path) -> dict:
    """Show how many images retain stenosis predictions at various thresholds."""
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    results = {}

    total_images = 0
    for split in SPLITS:
        total_images += len(ci_data.get(split, []))

    for thresh in thresholds:
        n_with_sten = 0
        n_sten_preds = 0
        for split in SPLITS:
            for entry in ci_data.get(split, []):
                preds = entry["predictions"]
                sten = [p for p in preds if p["class_id"] == STENOSIS_CLASS and p["confidence"] >= thresh]
                if sten:
                    n_with_sten += 1
                n_sten_preds += len(sten)
        results[thresh] = {
            "images_with_stenosis": n_with_sten,
            "total_stenosis_preds": n_sten_preds,
            "pct_images": n_with_sten / max(total_images, 1) * 100,
        }

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    xs = thresholds
    y_imgs = [results[t]["images_with_stenosis"] for t in xs]
    y_preds = [results[t]["total_stenosis_preds"] for t in xs]

    ax1.plot(xs, y_imgs, "o-", color="steelblue", linewidth=2)
    ax1.set_xlabel("Confidence Threshold")
    ax1.set_ylabel("Images with Stenosis")
    ax1.set_title(f"{direction}\nImages with Stenosis vs Threshold")
    ax1.grid(alpha=0.3)
    for x, y in zip(xs, y_imgs):
        ax1.annotate(str(y), (x, y), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)

    ax2.plot(xs, y_preds, "o-", color="coral", linewidth=2)
    ax2.set_xlabel("Confidence Threshold")
    ax2.set_ylabel("Total Stenosis Predictions")
    ax2.set_title(f"{direction}\nStenosis Count vs Threshold")
    ax2.grid(alpha=0.3)
    for x, y in zip(xs, y_preds):
        ax2.annotate(str(y), (x, y), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)

    plt.tight_layout()
    safe_name = direction.replace("/", "_")
    fig.savefig(output_dir / f"{safe_name}_threshold_sensitivity.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_dir / f'{safe_name}_threshold_sensitivity.png'}")

    return results


# =========================================================================
# MAIN
# =========================================================================

def build_class_names(config: dict) -> dict:
    """Build combined 26-class name mapping."""
    names = {}
    syntax_cats = config["syntax_categories"]
    for i, coco_id in enumerate(sorted(syntax_cats.keys())):
        names[i] = syntax_cats[coco_id]
    names[STENOSIS_CLASS] = "stenosis"
    return names


def main():
    parser = argparse.ArgumentParser(
        description="Extract final results and validate prediction quality"
    )
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--visual-samples", type=int, default=20,
                        help="Number of visual QC samples per direction")
    parser.add_argument("--spatial-samples", type=int, default=200,
                        help="Number of images to check spatial consistency")
    args = parser.parse_args()

    config = load_config(args.config)
    dataset_root = Path(config["dataset_root"])
    ci_dir = Path(config["cross_inference"]["output_dir"])
    output_dir = Path(config["output_dir"]) / "validation"
    output_dir.mkdir(parents=True, exist_ok=True)

    class_names = build_class_names(config)

    # Load cross-inference results
    ci_files = {
        "syntax_on_stenosis": ci_dir / "syntax_on_stenosis.json",
        "combined_on_syntax": ci_dir / "combined_on_syntax.json",
    }

    full_report = {}

    for direction, ci_path in ci_files.items():
        if not ci_path.exists():
            print(f"  [SKIP] {ci_path} not found")
            continue

        ci_data = load_cross_inference(ci_path)

        # Determine which images directory to use
        if direction == "syntax_on_stenosis":
            images_dir = dataset_root / "stenosis"
        else:
            images_dir = dataset_root / "syntax"

        # 1. Extract summary
        summary = extract_summary(ci_data, direction)
        print_extraction_report(summary, class_names)

        # 2. Confidence analysis
        print(f"\n  Confidence Analysis:")
        conf_report = validate_confidence(summary, output_dir)
        for key, stats in conf_report.items():
            print(f"    {key}: mean={stats['mean']:.3f}, median={stats['median']:.3f}, "
                  f"<0.5: {stats['below_0.5']}/{stats['count']} "
                  f"({stats['below_0.5']/stats['count']*100:.1f}%), "
                  f">0.7: {stats['above_0.7']}/{stats['count']} "
                  f"({stats['above_0.7']/stats['count']*100:.1f}%)")

        # 3. Spatial consistency (only for combined_on_syntax where we have both types)
        if direction == "combined_on_syntax":
            print(f"\n  Spatial Consistency (stenosis should overlap vessels):")
            spatial_report = validate_spatial_consistency(
                ci_data, direction, num_samples=args.spatial_samples
            )
            if spatial_report.get("status") != "no_data":
                print(f"    Stenoses checked: {spatial_report['total_stenoses_checked']}")
                print(f"    Mean overlap with vessels: {spatial_report['mean_overlap']:.3f}")
                print(f"    Median overlap: {spatial_report['median_overlap']:.3f}")
                print(f"    >50% overlap: {spatial_report['pct_above_50']:.1f}%")
                print(f"    >80% overlap: {spatial_report['pct_above_80']:.1f}%")
                print(f"    <10% overlap (suspicious): {spatial_report['pct_below_10']:.1f}% "
                      f"({spatial_report['unmatched_count']} stenoses)")
            full_report[f"{direction}_spatial"] = spatial_report

        # 4. Threshold sensitivity
        print(f"\n  Threshold Sensitivity:")
        thresh_report = threshold_sensitivity(ci_data, direction, output_dir)
        print(f"    {'Threshold':>10} {'Images w/ Sten':>15} {'Total Preds':>12} {'% Images':>10}")
        for thresh, stats in sorted(thresh_report.items()):
            print(f"    {thresh:>10.1f} {stats['images_with_stenosis']:>15} "
                  f"{stats['total_stenosis_preds']:>12} {stats['pct_images']:>9.1f}%")

        # 5. Visual QC samples
        print(f"\n  Visual QC:")
        generate_visual_samples(
            ci_data, direction, images_dir, output_dir,
            class_names, num_samples=args.visual_samples
        )

        full_report[direction] = {
            "summary": summary["total"],
            "splits": summary["splits"],
            "confidence": conf_report,
            "threshold_sensitivity": {
                str(k): v for k, v in thresh_report.items()
            },
        }

    # Save full report as JSON
    # Strip non-serializable data
    report_path = output_dir / "validation_report.json"
    with open(report_path, "w") as f:
        json.dump(full_report, f, indent=2, default=str)
    print(f"\n  Full report saved: {report_path}")

    # Final count
    print(f"\n{'='*60}")
    print("FINAL DATASET COUNTS")
    print(f"{'='*60}")
    for direction, ci_path in ci_files.items():
        if not ci_path.exists():
            continue
        r = full_report.get(direction, {}).get("summary", {})
        print(f"  {direction}: {r.get('with_both', 0)} images with BOTH syntax + stenosis "
              f"(out of {r.get('images', 0)})")

    both_total = sum(
        full_report.get(d, {}).get("summary", {}).get("with_both", 0)
        for d in ci_files
    )
    print(f"\n  TOTAL images with both syntax + stenosis: ~{both_total}")
    print(f"  All outputs saved to: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
