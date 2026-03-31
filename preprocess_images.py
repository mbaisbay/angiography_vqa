"""Optional image preprocessing: CLAHE and white top-hat morphological transform."""

import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np

from utils.config_loader import load_config


SPLITS = ["train", "val", "test"]


def apply_tophat(image: np.ndarray, kernel_size: int) -> np.ndarray:
    """Apply white top-hat transform following the ARCADE paper recipe.

    Steps: compute negative, apply morphological top-hat on negative,
    subtract result from negative, clip to [0, 255].
    """
    negative = 255 - image
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    tophat = cv2.morphologyEx(negative, cv2.MORPH_TOPHAT, kernel)
    result = cv2.subtract(negative, tophat)
    return result


def apply_clahe(image: np.ndarray, clip_limit: float, grid_size: int) -> np.ndarray:
    """Apply Contrast Limited Adaptive Histogram Equalization."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid_size, grid_size))
    return clahe.apply(image)


def preprocess_image(image: np.ndarray, config: dict) -> np.ndarray:
    """Apply configured preprocessing steps to a single grayscale image."""
    prep = config["preprocessing"]

    if prep["tophat"]["enabled"]:
        image = apply_tophat(image, prep["tophat"]["kernel_size"])

    if prep["clahe"]["enabled"]:
        image = apply_clahe(
            image,
            prep["clahe"]["clip_limit"],
            prep["clahe"]["grid_size"],
        )

    return image


def preprocess_task(config: dict, task: str) -> None:
    """Preprocess all images for a given task.

    Handles both original (syntax/stenosis) and filtered (syntax_filtered/stenosis_filtered)
    directories. Prefers the filtered directory if it exists.
    """
    dataset_root = Path(config["dataset_root"])

    # Prefer filtered directory if it exists (pipeline uses filtered data)
    filtered_dir = dataset_root / f"{task}_filtered"
    if filtered_dir.exists():
        task_dir = filtered_dir
    else:
        task_dir = dataset_root / task

    prep = config["preprocessing"]
    mode = prep["mode"]

    if mode == "separate":
        output_base = Path(prep["output_dir"]) / task
    else:
        output_base = task_dir  # inplace

    print(f"\n  Processing {task.upper()} task (mode: {mode})")

    for split in SPLITS:
        images_dir = task_dir / split / "images"
        if not images_dir.exists():
            print(f"    [SKIP] {split}: {images_dir} not found")
            continue

        if mode == "separate":
            output_dir = output_base / split / "images"
            output_dir.mkdir(parents=True, exist_ok=True)
            # Copy labels too if they exist
            labels_src = task_dir / split / "labels"
            labels_dst = output_base / split / "labels"
            if labels_src.exists() and not labels_dst.exists():
                shutil.copytree(str(labels_src), str(labels_dst))
            # Copy annotations too
            ann_src = task_dir / split / "annotations"
            ann_dst = output_base / split / "annotations"
            if ann_src.exists() and not ann_dst.exists():
                shutil.copytree(str(ann_src), str(ann_dst))
        else:
            output_dir = images_dir

        image_files = sorted(images_dir.glob("*.png")) + sorted(images_dir.glob("*.PNG"))
        print(f"    [{split.upper()}] Processing {len(image_files)} images...")

        for img_path in image_files:
            image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                print(f"      [WARN] Failed to read: {img_path}")
                continue

            processed = preprocess_image(image, config)
            out_path = output_dir / img_path.name
            cv2.imwrite(str(out_path), processed)

        print(f"    [{split.upper()}] Done -> {output_dir}")


def update_data_yamls(config: dict) -> None:
    """If mode is 'separate', regenerate data YAMLs pointing to preprocessed images."""
    import yaml

    prep = config["preprocessing"]
    if prep["mode"] != "separate":
        return

    preprocessed_root = Path(prep["output_dir"])

    for task in ["syntax", "stenosis"]:
        yaml_key = f"{task}_data_yaml"
        yaml_path = Path(config[yaml_key])

        if not yaml_path.exists():
            continue

        with open(yaml_path, "r") as f:
            data_yaml = yaml.safe_load(f)

        # Update path to point to preprocessed directory
        data_yaml["path"] = str((preprocessed_root / task).resolve())

        with open(yaml_path, "w") as f:
            yaml.dump(data_yaml, f, default_flow_style=False, sort_keys=False)

        print(f"\n  Updated {yaml_path} -> path: {data_yaml['path']}")


def main():
    parser = argparse.ArgumentParser(
        description="Apply CLAHE and top-hat preprocessing to ARCADE images"
    )
    parser.add_argument(
        "--config", type=str, default="config.yaml",
        help="Path to pipeline config YAML (default: config.yaml)"
    )
    parser.add_argument(
        "--task", type=str, choices=["syntax", "stenosis", "both"], default="both",
        help="Which task to preprocess (default: both)"
    )
    args = parser.parse_args()

    config = load_config(args.config)

    if not config["preprocessing"]["enabled"]:
        print("Preprocessing is disabled in config. Set preprocessing.enabled: true to run.")
        return

    print("="*60)
    print("Image Preprocessing")
    print("="*60)

    tasks = ["syntax", "stenosis"] if args.task == "both" else [args.task]

    for task in tasks:
        preprocess_task(config, task)

    update_data_yamls(config)

    print(f"\n{'='*60}")
    print("Preprocessing complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
