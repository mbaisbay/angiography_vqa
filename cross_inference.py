"""Run each trained model on the other task's images (cross-inference)."""

import argparse
import json
from pathlib import Path

from ultralytics import YOLO

from utils.config_loader import load_config, get_inference_args
from utils.coco_to_yolo import load_category_mapping


def run_cross_inference(model_path: str, images_dir: str, inference_args: dict,
                        category_mapping: dict) -> list:
    """Run a model on a set of images and collect predictions.

    Args:
        model_path: Path to trained model weights.
        images_dir: Directory containing images to run inference on.
        inference_args: Dict of inference parameters (conf, iou, max_det).
        category_mapping: Category mapping dict with yolo_to_name.

    Returns:
        List of prediction dicts, one per image.
    """
    model = YOLO(model_path)
    yolo_to_name = category_mapping["yolo_to_name"]

    results_list = []
    images_dir = Path(images_dir)
    image_files = sorted(images_dir.glob("*.png")) + sorted(images_dir.glob("*.PNG"))

    print(f"    Running inference on {len(image_files)} images...")

    for result in model.predict(
        source=str(images_dir),
        stream=True,
        save=False,
        verbose=False,
        **inference_args,
    ):
        image_name = Path(result.path).name
        predictions = []

        if result.masks is not None and len(result.masks) > 0:
            for i in range(len(result.masks)):
                cls_id = int(result.boxes.cls[i].item())
                conf = float(result.boxes.conf[i].item())
                bbox = result.boxes.xywh[i].tolist()

                # Get normalized polygon coordinates
                polygon = result.masks.xyn[i].tolist()

                class_name = yolo_to_name.get(cls_id, str(cls_id))

                predictions.append({
                    "class_id": cls_id,
                    "class_name": class_name,
                    "confidence": round(conf, 4),
                    "bbox_xywh": [round(v, 2) for v in bbox],
                    "polygon_normalized": polygon,
                })

        results_list.append({
            "image_name": image_name,
            "num_predictions": len(predictions),
            "predictions": predictions,
        })

    print(f"    Processed {len(results_list)} images, "
          f"{sum(r['num_predictions'] for r in results_list)} total predictions")

    return results_list


def main():
    parser = argparse.ArgumentParser(
        description="Run cross-inference: each model on the other task's images"
    )
    parser.add_argument(
        "--config", type=str, default="config.yaml",
        help="Path to pipeline config YAML (default: config.yaml)"
    )
    parser.add_argument(
        "--direction", type=str,
        choices=["syntax_on_stenosis", "stenosis_on_syntax", "both"],
        default="both",
        help="Which cross-inference to run (default: both)"
    )
    args = parser.parse_args()

    config = load_config(args.config)
    ci = config["cross_inference"]
    inference_args = get_inference_args(config)
    dataset_root = Path(config["dataset_root"])
    output_dir = Path(ci["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    splits = ci.get("splits", ["test"])
    mappings_dir = Path(config["mappings_dir"])

    directions = []
    if args.direction in ("syntax_on_stenosis", "both"):
        directions.append("syntax_on_stenosis")
    if args.direction in ("stenosis_on_syntax", "both"):
        directions.append("stenosis_on_syntax")

    for direction in directions:
        print(f"\n{'='*60}")
        print(f"Cross-inference: {direction}")
        print(f"{'='*60}")

        if direction == "syntax_on_stenosis":
            model_path = ci["syntax_weights"]
            target_task = "stenosis"
            mapping_path = mappings_dir / "syntax_categories.json"
        else:
            model_path = ci["stenosis_weights"]
            target_task = "syntax"
            mapping_path = mappings_dir / "stenosis_categories.json"

        if not Path(model_path).exists():
            print(f"  [ERROR] Model weights not found: {model_path}")
            print(f"  Run train.py first.")
            continue

        category_mapping = load_category_mapping(str(mapping_path))
        print(f"  Model: {model_path}")
        print(f"  Target images: {target_task}")
        print(f"  Splits: {splits}")

        all_results = {}

        for split in splits:
            images_dir = dataset_root / target_task / split / "images"
            if not images_dir.exists():
                print(f"  [SKIP] {split}: {images_dir} not found")
                continue

            print(f"\n  [{split.upper()}]")
            results = run_cross_inference(
                str(model_path), str(images_dir), inference_args, category_mapping
            )
            all_results[split] = results

        # Save results
        output_file = output_dir / f"{direction}.json"
        with open(output_file, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\n  Saved: {output_file}")

    print(f"\n{'='*60}")
    print("Cross-inference complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
