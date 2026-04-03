"""Two-stage YOLO segmentation training with backbone freeze/unfreeze.

Stage A: Train with frozen backbone (layers 0-9) for freeze_epochs.
Stage B: Unfreeze all layers, reduce LR by 10x, train for remaining epochs.

Always starts from COCO pretrained weights (or specified checkpoint).
"""

import argparse
import yaml
from pathlib import Path

from ultralytics import YOLO


def load_run_config(config_path: str) -> dict:
    """Load a run configuration YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def build_train_args(cfg: dict, data_yaml: str, stage: str,
                     project: str, name: str) -> dict:
    """Build ultralytics model.train() arguments from config.

    Args:
        cfg: Run config dict.
        data_yaml: Path to YOLO dataset YAML.
        stage: "frozen" or "unfrozen".
        project: Results project directory.
        name: Run name within project.
    """
    args = {
        "data": data_yaml,
        "imgsz": cfg["imgsz"],
        "batch": cfg["batch"],
        "optimizer": cfg["optimizer"],
        "momentum": cfg.get("momentum", 0.937),
        "weight_decay": cfg["weight_decay"],
        "warmup_epochs": cfg.get("warmup_epochs", 5),
        "seed": cfg.get("seed", 42),
        "deterministic": cfg.get("deterministic", True),
        "amp": cfg.get("amp", True),
        "cos_lr": cfg.get("cos_lr", True),
        "device": cfg.get("device", "0"),
        "workers": cfg.get("workers", 4),
        "project": project,
        "name": name,
        "exist_ok": True,
        # Augmentation
        "mosaic": cfg.get("mosaic", 0.0),
        "mixup": cfg.get("mixup", 0.0),
        "copy_paste": cfg.get("copy_paste", 0.0),
        "fliplr": cfg.get("fliplr", 0.5),
        "flipud": cfg.get("flipud", 0.0),
        "degrees": cfg.get("degrees", 20.0),
        "scale": cfg.get("scale", 0.4),
        "translate": cfg.get("translate", 0.1),
        "hsv_h": cfg.get("hsv_h", 0.0),
        "hsv_s": cfg.get("hsv_s", 0.0),
        "hsv_v": cfg.get("hsv_v", 0.3),
        "erasing": cfg.get("erasing", 0.0),
        "shear": cfg.get("shear", 0.0),
        "perspective": cfg.get("perspective", 0.0),
    }

    if stage == "frozen":
        args["epochs"] = cfg.get("freeze_epochs", 15)
        args["freeze"] = cfg.get("freeze", 10)
        args["lr0"] = cfg["lr0"]
        args["lrf"] = cfg.get("lrf", 0.01)
        args["patience"] = 0  # no early stopping during frozen stage
    elif stage == "unfrozen":
        remaining = cfg["epochs"] - cfg.get("freeze_epochs", 15)
        args["epochs"] = max(remaining, 10)
        args["freeze"] = 0
        args["lr0"] = cfg["lr0"] * 0.1  # 10x lower for fine-tuning
        args["lrf"] = cfg.get("lrf", 0.01)
        args["patience"] = cfg.get("patience", 25)
        args["resume"] = False

    return args


def train_two_stage(cfg: dict, data_yaml: str, project: str,
                    run_name: str, model_weights: str = None) -> str:
    """Run two-stage training: frozen backbone -> unfrozen fine-tuning.

    Args:
        cfg: Run config dict.
        data_yaml: Path to YOLO dataset YAML.
        project: Results project directory.
        run_name: Name for this training run.
        model_weights: Path to starting weights (None = use cfg['model']).

    Returns:
        Path to best weights from Stage B.
    """
    project = str(Path(project).resolve())
    weights = model_weights or cfg["model"]

    # ── Stage A: Frozen backbone ──
    print("=" * 60)
    print(f"Stage A: Frozen backbone training ({cfg.get('freeze_epochs', 15)} epochs)")
    print(f"  Model: {weights}")
    print(f"  Data:  {data_yaml}")
    print("=" * 60)

    model = YOLO(weights)
    stage_a_name = f"{run_name}_stage_a_frozen"
    stage_a_args = build_train_args(cfg, data_yaml, "frozen",
                                     project, stage_a_name)
    model.train(**stage_a_args)

    # Get best weights from Stage A
    stage_a_best = Path(project) / stage_a_name / "weights" / "best.pt"
    if not stage_a_best.exists():
        stage_a_best = Path(project) / stage_a_name / "weights" / "last.pt"

    print(f"\nStage A complete. Best weights: {stage_a_best}")

    # ── Stage B: Unfrozen fine-tuning ──
    remaining_epochs = cfg["epochs"] - cfg.get("freeze_epochs", 15)
    print("\n" + "=" * 60)
    print(f"Stage B: Unfrozen fine-tuning ({remaining_epochs} epochs)")
    print(f"  Starting from: {stage_a_best}")
    print(f"  LR: {cfg['lr0'] * 0.1}")
    print("=" * 60)

    model = YOLO(str(stage_a_best))
    stage_b_name = f"{run_name}_stage_b_unfrozen"
    stage_b_args = build_train_args(cfg, data_yaml, "unfrozen",
                                     project, stage_b_name)
    model.train(**stage_b_args)

    # Get best weights from Stage B
    stage_b_best = Path(project) / stage_b_name / "weights" / "best.pt"
    if not stage_b_best.exists():
        stage_b_best = Path(project) / stage_b_name / "weights" / "last.pt"

    print(f"\nStage B complete. Best weights: {stage_b_best}")

    # Copy best weights to a canonical location
    final_weights = Path(project) / f"{run_name}_best.pt"
    import shutil
    shutil.copy2(stage_b_best, final_weights)
    print(f"Final weights saved: {final_weights}")

    return str(final_weights)


def main():
    parser = argparse.ArgumentParser(
        description="Two-stage YOLO segmentation training"
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to run config YAML (e.g., configs/run1_yolo11m.yaml)"
    )
    parser.add_argument(
        "--data-yaml", type=str, required=True,
        help="Path to YOLO dataset YAML"
    )
    parser.add_argument(
        "--project", type=str, default=None,
        help="Results directory (overrides config results_dir)"
    )
    parser.add_argument(
        "--name", type=str, default=None,
        help="Run name (default: from config run_name)"
    )
    parser.add_argument(
        "--weights", type=str, default=None,
        help="Starting model weights (overrides config model)"
    )
    parser.add_argument(
        "--single-stage", action="store_true",
        help="Skip two-stage training, use freeze for all epochs"
    )
    args = parser.parse_args()

    cfg = load_run_config(args.config)
    project = args.project or cfg.get("results_dir", "results")
    run_name = args.name or cfg.get("run_name", "train")

    # Verify data YAML exists
    data_yaml = Path(args.data_yaml)
    if not data_yaml.exists():
        raise FileNotFoundError(f"Data YAML not found: {data_yaml}")

    if args.single_stage:
        # Simple single-stage training with freeze
        print("=" * 60)
        print(f"Single-stage training ({cfg['epochs']} epochs, freeze={cfg.get('freeze', 10)})")
        print("=" * 60)

        weights = args.weights or cfg["model"]
        model = YOLO(weights)
        train_args = build_train_args(cfg, str(data_yaml), "frozen",
                                       str(project), run_name)
        train_args["epochs"] = cfg["epochs"]
        train_args["patience"] = cfg.get("patience", 25)
        model.train(**train_args)
    else:
        train_two_stage(
            cfg, str(data_yaml), str(project), run_name,
            model_weights=args.weights
        )


if __name__ == "__main__":
    main()
