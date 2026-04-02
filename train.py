"""Train YOLOv8x-seg instance segmentation model for syntax or stenosis task."""

import argparse
from pathlib import Path

import yaml
from ultralytics import YOLO

from utils.config_loader import load_config, get_training_args


def main():
    parser = argparse.ArgumentParser(
        description="Train YOLOv8x-seg on ARCADE dataset"
    )
    parser.add_argument(
        "--config", type=str, default="config.yaml",
        help="Path to pipeline config YAML (default: config.yaml)"
    )
    parser.add_argument(
        "--task", type=str, required=True, choices=["syntax", "stenosis", "combined", "final"],
        help="Which task to train: 'syntax' (vessel segments), 'stenosis' (plaque detection), or 'combined' (both)"
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to checkpoint to resume training from (optional)"
    )
    args = parser.parse_args()

    config = load_config(args.config)
    train_args = get_training_args(config, args.task)

    # Verify data YAML exists
    data_yaml = Path(train_args["data"])
    if not data_yaml.exists():
        raise FileNotFoundError(
            f"Data YAML not found: {data_yaml}\n"
            f"Run prepare_data.py first to generate it."
        )

    # Verify dataset path inside data YAML points to an existing directory
    with open(data_yaml) as f:
        _data = yaml.safe_load(f)
    dataset_path = Path(_data.get("path", ""))
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset path in {data_yaml} does not exist: {dataset_path}\n"
            f"The path may be from a different machine.\n"
            f"Run filter_classes.py (or fix_data_yaml.py) to regenerate data YAMLs with correct paths."
        )

    print("=" * 60)
    print(f"Training {args.task.upper()} model")
    print("=" * 60)
    print(f"  Model: {config['model_variant']}")
    print(f"  Data:  {train_args['data']}")
    print(f"  Epochs: {train_args['epochs']}")
    print(f"  Batch:  {train_args['batch']}")
    print(f"  ImgSz:  {train_args['imgsz']}")
    print(f"  Device: {train_args['device']}")
    print(f"  Output: {train_args['project']}/{train_args['name']}")
    print("=" * 60)

    # Load model
    if args.resume:
        print(f"\n  Resuming from: {args.resume}")
        model = YOLO(args.resume)
        train_args["resume"] = True
    else:
        weights = config["pretrained_weights"]
        print(f"\n  Loading pretrained weights: {weights}")
        model = YOLO(weights)

    # Train
    results = model.train(**train_args)

    # Report results
    output_dir = Path(train_args["project"]) / train_args["name"]
    best_weights = output_dir / "weights" / "best.pt"
    last_weights = output_dir / "weights" / "last.pt"

    print(f"\n{'='*60}")
    print(f"Training complete for {args.task.upper()}")
    print(f"  Best weights: {best_weights}")
    print(f"  Last weights: {last_weights}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
