"""Build combined syntax-PL dataset for the Cross-Task experiment (E4).

Train set:
    1000 syntax GT images + GT labels
  + 1000 stenosis images + syntax pseudo-labels (prefixed `sten_pl_`)

Val/test:
    Original syntax val (200) and test (300), unchanged.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import yaml


def build(
    syntax_data_dir: Path,
    stenosis_img_dir: Path,
    pseudo_label_dir: Path,
    output_dir: Path,
    syntax_yaml: Path,
) -> str:
    output_dir = Path(output_dir)
    train_img = output_dir / "images" / "train"
    train_lbl = output_dir / "labels" / "train"
    train_img.mkdir(parents=True, exist_ok=True)
    train_lbl.mkdir(parents=True, exist_ok=True)

    src_syn_train_img = syntax_data_dir / "images" / "train"
    src_syn_train_lbl = syntax_data_dir / "labels" / "train"

    n_syn = 0
    for img in list(src_syn_train_img.glob("*.png")) + list(src_syn_train_img.glob("*.PNG")):
        dst = train_img / img.name
        if not dst.exists() and not dst.is_symlink():
            os.symlink(img.resolve(), dst)
        n_syn += 1
    for lbl in src_syn_train_lbl.glob("*.txt"):
        dst = train_lbl / lbl.name
        if not dst.exists() and not dst.is_symlink():
            os.symlink(lbl.resolve(), dst)

    n_pseudo = 0
    for lbl_path in pseudo_label_dir.glob("*.txt"):
        stem = lbl_path.stem
        src_img = None
        for ext in (".png", ".PNG", ".jpg"):
            cand = stenosis_img_dir / (stem + ext)
            if cand.exists():
                src_img = cand
                break
        if src_img is None:
            continue
        dst_img = train_img / f"sten_pl_{src_img.name}"
        if not dst_img.exists() and not dst_img.is_symlink():
            os.symlink(src_img.resolve(), dst_img)
        dst_lbl = train_lbl / f"sten_pl_{stem}.txt"
        if not dst_lbl.exists() and not dst_lbl.is_symlink():
            os.symlink(lbl_path.resolve(), dst_lbl)
        n_pseudo += 1

    for split in ("val", "test"):
        src_img_dir = syntax_data_dir / "images" / split
        src_lbl_dir = syntax_data_dir / "labels" / split
        dst_img_dir = output_dir / "images" / split
        dst_lbl_dir = output_dir / "labels" / split
        dst_img_dir.mkdir(parents=True, exist_ok=True)
        dst_lbl_dir.mkdir(parents=True, exist_ok=True)
        for img in list(src_img_dir.glob("*.png")) + list(src_img_dir.glob("*.PNG")):
            dst = dst_img_dir / img.name
            if not dst.exists() and not dst.is_symlink():
                os.symlink(img.resolve(), dst)
        if src_lbl_dir.exists():
            for lbl in src_lbl_dir.glob("*.txt"):
                dst = dst_lbl_dir / lbl.name
                if not dst.exists() and not dst.is_symlink():
                    os.symlink(lbl.resolve(), dst)

    # Inherit class names from the source syntax YAML
    with open(syntax_yaml) as f:
        src_cfg = yaml.safe_load(f)
    names = src_cfg.get("names", {})

    cfg = {
        "path": str(output_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(names),
        "names": names,
    }
    cfgs_dir = output_dir / "dataset_configs"
    cfgs_dir.mkdir(parents=True, exist_ok=True)
    yaml_path = cfgs_dir / "syntax_combined_pl.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

    print(f"  Combined syntax-PL dataset:")
    print(f"    syntax GT:     {n_syn}")
    print(f"    stenosis PL:   {n_pseudo}")
    print(f"    total train:   {n_syn + n_pseudo}")
    return str(yaml_path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--syntax-data-dir", type=Path, required=True)
    p.add_argument("--stenosis-img-dir", type=Path, required=True)
    p.add_argument("--pseudo-label-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--syntax-yaml", type=Path, required=True)
    args = p.parse_args()
    print(build(args.syntax_data_dir, args.stenosis_img_dir,
                args.pseudo_label_dir, args.output_dir, args.syntax_yaml))


if __name__ == "__main__":
    main()
