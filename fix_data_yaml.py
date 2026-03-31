"""Generate corrected YOLO data.yaml files for the ARCADE dataset.

Use this when the existing data.yaml files have issues (e.g., val pointing to
test, wrong nc). This does NOT convert annotations — labels must already exist.

The ARCADE dataset uses a shared 26-class scheme:
  Classes 0-24: Syntax vessel segments
  Class 25: Stenosis
"""

import argparse
import yaml
from pathlib import Path

from utils.config_loader import load_config


# Shared 26-class names used across all ARCADE tasks
SHARED_CLASS_NAMES = {
    0: "1", 1: "2", 2: "3", 3: "4", 4: "5", 5: "6", 6: "7", 7: "8",
    8: "9", 9: "9a", 10: "10", 11: "10a", 12: "11", 13: "12", 14: "12a",
    15: "13", 16: "14", 17: "14a", 18: "15", 19: "16", 20: "16a",
    21: "16b", 22: "16c", 23: "12b", 24: "14b", 25: "stenosis",
}


def generate_yaml(task_dir: Path, yaml_path: Path, nc: int = 26,
                   names: dict = None) -> None:
    """Write a data.yaml for a given task directory."""
    if names is None:
        names = SHARED_CLASS_NAMES

    data = {
        "path": str(task_dir.resolve()),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "nc": nc,
        "names": names,
    }

    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    with open(yaml_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    print(f"  Written: {yaml_path}")
    print(f"    path: {data['path']}")
    print(f"    val: {data['val']}  (NOT test/images)")
    print(f"    nc: {data['nc']}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate corrected YOLO data.yaml files"
    )
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--task", type=str,
                        choices=["syntax", "stenosis", "combined", "all"],
                        default="all")
    args = parser.parse_args()

    config = load_config(args.config)
    dataset_root = Path(config["dataset_root"])

    tasks = ["syntax", "stenosis", "combined"] if args.task == "all" else [args.task]

    for task in tasks:
        task_dir = dataset_root / task

        if task == "syntax":
            yaml_path = Path(config["syntax_data_yaml"])
        elif task == "stenosis":
            yaml_path = Path(config["stenosis_data_yaml"])
        elif task == "combined":
            yaml_path = Path(config["combined_data_yaml"])

        if not task_dir.exists():
            print(f"  [SKIP] {task}: {task_dir} not found")
            continue

        print(f"\n  Generating data.yaml for {task.upper()}")
        generate_yaml(task_dir, yaml_path, nc=26, names=SHARED_CLASS_NAMES)

        # Also write data.yaml inside the task directory for convenience
        local_yaml = task_dir / "data.yaml"
        generate_yaml(task_dir, local_yaml, nc=26, names=SHARED_CLASS_NAMES)


if __name__ == "__main__":
    main()
