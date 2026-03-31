"""Configuration loader and parameter extraction for the ARCADE pipeline."""

import yaml
from pathlib import Path


def load_config(config_path: str) -> dict:
    """Load and validate the pipeline configuration from a YAML file.

    All relative paths are resolved to absolute paths using the config file's
    parent directory as the base.
    """
    config_path = Path(config_path).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    base_dir = config_path.parent

    # Resolve path fields to absolute
    path_keys = [
        "dataset_root", "output_dir", "mappings_dir",
        "syntax_data_yaml", "stenosis_data_yaml",
    ]
    for key in path_keys:
        if key in config:
            config[key] = str((base_dir / config[key]).resolve())

    # Resolve nested path fields
    nested_paths = [
        ("preprocessing", "output_dir"),
        ("cross_inference", "syntax_weights"),
        ("cross_inference", "stenosis_weights"),
        ("cross_inference", "output_dir"),
        ("intersection", "overlay_output_dir"),
        ("intersection", "results_output_dir"),
    ]
    for section, key in nested_paths:
        if section in config and key in config[section]:
            config[section][key] = str(
                (base_dir / config[section][key]).resolve()
            )

    # Validate required top-level keys
    required = [
        "dataset_root", "output_dir", "training", "augmentation",
        "inference", "syntax_categories", "stenosis_categories",
    ]
    missing = [k for k in required if k not in config]
    if missing:
        raise ValueError(f"Missing required config keys: {missing}")

    config["_base_dir"] = str(base_dir)
    return config


def get_training_args(config: dict, task: str) -> dict:
    """Build a flat dict of arguments for ultralytics model.train()."""
    t = config["training"]
    a = config["augmentation"]

    data_yaml = config["syntax_data_yaml"] if task == "syntax" else config["stenosis_data_yaml"]

    args = {
        "data": data_yaml,
        "epochs": t["epochs"],
        "batch": t["batch_size"],
        "imgsz": t["image_size"],
        "lr0": t["lr0"],
        "lrf": t["lrf"],
        "optimizer": t["optimizer"],
        "momentum": t["momentum"],
        "weight_decay": t["weight_decay"],
        "patience": t["patience"],
        "device": t["device"],
        "workers": t["workers"],
        "seed": t["seed"],
        "amp": t["amp"],
        "cos_lr": t["cos_lr"],
        "project": config["output_dir"],
        "name": task,
        "exist_ok": True,
        # Augmentation
        "hsv_h": a["hsv_h"],
        "hsv_s": a["hsv_s"],
        "hsv_v": a["hsv_v"],
        "degrees": a["degrees"],
        "translate": a["translate"],
        "scale": a["scale"],
        "shear": a["shear"],
        "perspective": a["perspective"],
        "flipud": a["flipud"],
        "fliplr": a["fliplr"],
        "mosaic": a["mosaic"],
        "mixup": a["mixup"],
        "copy_paste": a["copy_paste"],
    }
    return args


def get_inference_args(config: dict) -> dict:
    """Build a dict of arguments for ultralytics model.predict()."""
    inf = config["inference"]
    return {
        "conf": inf["confidence_threshold"],
        "iou": inf["iou_threshold"],
        "max_det": inf["max_detections"],
    }


def get_data_yaml_path(config: dict, task: str) -> str:
    """Return the path to the YOLO data YAML for the given task."""
    if task == "syntax":
        return config["syntax_data_yaml"]
    elif task == "stenosis":
        return config["stenosis_data_yaml"]
    else:
        raise ValueError(f"Unknown task: {task}. Must be 'syntax' or 'stenosis'.")
