"""Generate pseudo-labels by running a trained model on target images.

Saves predictions as YOLO segmentation format .txt files.
Supports confidence thresholding, class ID offsetting, and YOLO26
one-to-many head mode for more detection candidates.
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from ultralytics import YOLO


def generate_pseudo_labels(
    model_path: str,
    image_dir: str,
    output_label_dir: str,
    conf_threshold: float = 0.85,
    class_offset: int = 0,
    use_one_to_many: bool = False,
    imgsz: int = 512,
) -> dict:
    """Run inference and save predictions as YOLO label files.

    Args:
        model_path: Path to trained YOLO model weights.
        image_dir: Directory containing images to predict on.
        output_label_dir: Directory to save YOLO .txt label files.
        conf_threshold: Minimum confidence to keep a prediction.
        class_offset: Offset to add to predicted class IDs.
        use_one_to_many: For YOLO26, use one-to-many head (more candidates).
        imgsz: Image size for inference.

    Returns:
        Stats dict with prediction counts and confidence distributions.
    """
    model = YOLO(model_path)

    # For YOLO26 one-to-many head
    if use_one_to_many:
        try:
            model.model.model[-1].end2end = False
            print("  YOLO26: Using one-to-many head for pseudo-labels")
        except (AttributeError, IndexError):
            print("  WARNING: Could not set end2end=False (not YOLO26?)")

    output_dir = Path(output_label_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stats = {
        "total_images": 0,
        "images_with_predictions": 0,
        "total_predictions": 0,
        "predictions_filtered": 0,
        "confidences": [],
        "class_distribution": Counter(),
        "predictions_per_image": [],
    }

    print(f"  Running inference on {image_dir}...")
    results = model.predict(
        source=image_dir,
        conf=conf_threshold,
        imgsz=imgsz,
        save=False,
        save_txt=False,
        verbose=False,
        stream=True,
        retina_masks=True,
    )

    for result in results:
        stats["total_images"] += 1
        image_name = Path(result.path).stem
        label_path = output_dir / f"{image_name}.txt"

        lines = []

        if result.masks is not None and len(result.masks) > 0:
            for i in range(len(result.masks)):
                conf = float(result.boxes.conf[i].item())
                cls_id = int(result.boxes.cls[i].item())

                # Apply class offset
                out_cls_id = cls_id + class_offset

                # Get normalized polygon coordinates
                polygon = result.masks.xyn[i]
                if len(polygon) < 3:
                    stats["predictions_filtered"] += 1
                    continue

                # Format polygon as flat coordinates
                coords = []
                for pt in polygon:
                    coords.append(f"{pt[0]:.6f}")
                    coords.append(f"{pt[1]:.6f}")

                line = f"{out_cls_id} " + " ".join(coords)
                lines.append(line)

                stats["total_predictions"] += 1
                stats["confidences"].append(conf)
                stats["class_distribution"][out_cls_id] += 1

        # Write label file (even if empty)
        with open(label_path, "w") as f:
            if lines:
                f.write("\n".join(lines) + "\n")
                stats["images_with_predictions"] += 1

        stats["predictions_per_image"].append(len(lines))

    return stats


def print_stats(stats: dict) -> None:
    """Print pseudo-label generation statistics."""
    print(f"\n  Pseudo-label statistics:")
    print(f"    Total images:         {stats['total_images']}")
    print(f"    Images with labels:   {stats['images_with_predictions']}")
    print(f"    Total predictions:    {stats['total_predictions']}")
    print(f"    Predictions filtered: {stats['predictions_filtered']}")

    if stats["confidences"]:
        confs = np.array(stats["confidences"])
        print(f"    Confidence: mean={confs.mean():.3f}, "
              f"median={np.median(confs):.3f}, "
              f"min={confs.min():.3f}, max={confs.max():.3f}")

    ppi = np.array(stats["predictions_per_image"])
    if len(ppi) > 0:
        print(f"    Predictions/image: mean={ppi.mean():.1f}, "
              f"max={ppi.max()}")

    if stats["class_distribution"]:
        print(f"    Class distribution:")
        for cls_id in sorted(stats["class_distribution"]):
            print(f"      Class {cls_id}: {stats['class_distribution'][cls_id]}")


def save_stats(stats: dict, output_path: str) -> None:
    """Save stats to JSON (converting non-serializable types)."""
    serializable = {
        "total_images": stats["total_images"],
        "images_with_predictions": stats["images_with_predictions"],
        "total_predictions": stats["total_predictions"],
        "predictions_filtered": stats["predictions_filtered"],
        "class_distribution": dict(stats["class_distribution"]),
        "predictions_per_image_mean": float(np.mean(stats["predictions_per_image"])) if stats["predictions_per_image"] else 0,
        "predictions_per_image_max": int(np.max(stats["predictions_per_image"])) if stats["predictions_per_image"] else 0,
    }
    if stats["confidences"]:
        confs = np.array(stats["confidences"])
        serializable["confidence"] = {
            "mean": float(confs.mean()),
            "median": float(np.median(confs)),
            "min": float(confs.min()),
            "max": float(confs.max()),
            "std": float(confs.std()),
        }

    with open(output_path, "w") as f:
        json.dump(serializable, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Generate pseudo-labels from a trained YOLO model"
    )
    parser.add_argument(
        "--model", type=str, required=True,
        help="Path to trained model weights (.pt)"
    )
    parser.add_argument(
        "--image-dir", type=str, required=True,
        help="Directory containing images to generate labels for"
    )
    parser.add_argument(
        "--output-dir", type=str, required=True,
        help="Output directory for YOLO .txt label files"
    )
    parser.add_argument(
        "--conf", type=float, default=0.85,
        help="Minimum confidence threshold (default: 0.85)"
    )
    parser.add_argument(
        "--class-offset", type=int, default=0,
        help="Offset to add to predicted class IDs (default: 0)"
    )
    parser.add_argument(
        "--one-to-many", action="store_true",
        help="Use YOLO26 one-to-many head (more candidates)"
    )
    parser.add_argument(
        "--imgsz", type=int, default=512,
        help="Image size for inference (default: 512)"
    )
    parser.add_argument(
        "--stats-output", type=str, default=None,
        help="Path to save stats JSON (default: output_dir/pseudo_label_stats.json)"
    )
    args = parser.parse_args()

    print(f"Generating pseudo-labels:")
    print(f"  Model:        {args.model}")
    print(f"  Images:       {args.image_dir}")
    print(f"  Output:       {args.output_dir}")
    print(f"  Confidence:   {args.conf}")
    print(f"  Class offset: {args.class_offset}")

    stats = generate_pseudo_labels(
        model_path=args.model,
        image_dir=args.image_dir,
        output_label_dir=args.output_dir,
        conf_threshold=args.conf,
        class_offset=args.class_offset,
        use_one_to_many=args.one_to_many,
        imgsz=args.imgsz,
    )

    print_stats(stats)

    stats_path = args.stats_output or str(
        Path(args.output_dir) / "pseudo_label_stats.json"
    )
    save_stats(stats, stats_path)
    print(f"\n  Stats saved: {stats_path}")


if __name__ == "__main__":
    main()
