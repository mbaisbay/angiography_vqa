"""Re-evaluate the top-5 S-series models on the *official* ARCADE 300-test.

Why
---
The S-series experiments were all trained and evaluated on the stratified
splits (999 train / 200 val / 301 test resampled from the 1500-image
train∪val∪test pool). Their stratified-test scores (e.g. S54 mean F1 =
0.7181, syntax 0.7398) are not directly comparable to the H/E-series
scores on the official 300-image held-out test. This script closes that
gap: load each S-run's trained weights and evaluate them against the
official ARCADE test set, producing numbers comparable to
H1/E1/E2/E3/E4.

Models evaluated
----------------
  S54_s43b_clahe         best overall on stratified (0.7181)
  S31_sgd_lr005          best stenosis on stratified (0.4613)
  S51_s31_clahe          2nd overall on stratified
  S33_sgd_wd001          stenosis 0.4549
  S12_stenosis_yolo11l   stenosis 0.4548, yolo11l capacity

Notes
-----
- S51 and S54 trained stenosis on CLAHE-preprocessed images, so we
  build a shadow test set with CLAHE applied before running their
  stenosis models. Syntax models were trained on raw images for all 5.
- All runs trained at imgsz=768 with augment=True (TTA) at eval time,
  matching their original strategy_results entries.
- Uses data_prep on the raw arcade_root (splits_dir=None) so the test
  yaml points at the real official 300-image test.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import yaml


# ── Experiments to re-evaluate ─────────────────────────────────────
EXPERIMENTS = [
    # (name, stenosis_clahe?)
    ("S54_s43b_clahe",       True),
    ("S31_sgd_lr005",        False),
    ("S51_s31_clahe",        True),
    ("S33_sgd_wd001",        False),
    ("S12_stenosis_yolo11l", False),
]


def _find_weight(results_dir: Path, exp_name: str, kind: str) -> Path:
    """Locate syntax_768_best.pt / stenosis_768_best.pt for an S-run.

    Handles both the stored absolute paths from the runner PC and the
    local repo layout.
    """
    assert kind in ("syntax", "stenosis")
    fname = f"{kind}_768_best.pt"
    candidates = [
        results_dir / exp_name / f"{kind}_model" / fname,
        results_dir / exp_name / f"{kind}_model" / f"{kind}_768" / "weights" / "best.pt",
    ]
    for c in candidates:
        if c.exists():
            return c
    # Fall back to a glob
    hits = list((results_dir / exp_name).rglob(f"{kind}*best.pt"))
    if hits:
        return hits[0]
    raise FileNotFoundError(
        f"Could not find {kind} weights for {exp_name} under {results_dir / exp_name}")


def _prepare_eval_data(arcade_root: Path, output_root: Path) -> Path:
    """Build a fresh data dir from the raw arcade root (original splits)."""
    from run_pipeline import data_prep
    data_dir = output_root / "_eval_official_data"
    if (data_dir / "dataset_configs" / "syntax_only.yaml").exists():
        return data_dir
    data_prep(arcade_root, data_dir, min_count=300, splits_dir=None)
    return data_dir


def _make_stenosis_clahe_shadow(data_dir: Path, shadow_dir: Path) -> Path:
    """Build a CLAHE-preprocessed shadow of the stenosis dataset.

    We copy (not symlink) the test images because we apply CLAHE
    in-place to the shadow. Labels and the train/val splits are
    symlinked since the eval script only touches test.
    """
    from run_stenosis_strategies_v2 import _apply_image_preprocessing

    if (shadow_dir / "dataset_configs" / "stenosis_only.yaml").exists():
        return shadow_dir / "dataset_configs" / "stenosis_only.yaml"

    src = data_dir / "stenosis"
    for sub in ("images", "labels"):
        for split in ("train", "val", "test"):
            src_sub = src / sub / split
            dst_sub = shadow_dir / sub / split
            dst_sub.mkdir(parents=True, exist_ok=True)
            for p in src_sub.iterdir() if src_sub.exists() else []:
                dst = dst_sub / p.name
                if dst.exists() or dst.is_symlink():
                    continue
                if sub == "images" and split == "test":
                    # Real copy so we can mutate it with CLAHE
                    shutil.copy2(p, dst)
                else:
                    os.symlink(p.resolve(), dst)

    # Apply CLAHE in-place to the *test* split only
    _apply_image_preprocessing(
        shadow_dir / "images" / "test",
        {"clahe_preprocess": True, "clahe_clip_limit": 2.0, "clahe_tile_size": 8},
        tag="eval_clahe_stenosis_test",
    )

    # Write the stenosis_only yaml for this shadow dir
    configs_dir = shadow_dir / "dataset_configs"
    configs_dir.mkdir(parents=True, exist_ok=True)
    yaml_cfg = {
        "path": str(shadow_dir.resolve()),
        "train": "images/train",
        "val":   "images/val",
        "test":  "images/test",
        "nc":    1,
        "names": {0: "stenosis"},
    }
    out_yaml = configs_dir / "stenosis_only.yaml"
    with open(out_yaml, "w") as f:
        yaml.dump(yaml_cfg, f, default_flow_style=False, sort_keys=False)
    return out_yaml


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arcade-root", required=True, type=Path,
                   help="Path to arcade/submission (raw, original splits)")
    p.add_argument("--results-dir", type=Path,
                   default=Path(__file__).resolve().parent.parent
                            / "results" / "stenosis_strategies",
                   help="Where the S* experiment outputs live")
    p.add_argument("--output", type=Path, default=None,
                   help="Where to write the comparison JSON "
                        "(default: <results-dir>/top5_on_official.json)")
    p.add_argument("--imgsz", type=int, default=768)
    p.add_argument("--device", type=str, default="0")
    p.add_argument("--no-tta", action="store_true",
                   help="Disable test-time augmentation")
    args = p.parse_args()

    script_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(script_dir))

    from evaluate import evaluate_model

    arcade_root = args.arcade_root.resolve()
    results_dir = args.results_dir.resolve()
    out_path = args.output or (results_dir / "top5_on_official.json")

    # ── Build the shared eval data dir once ──
    eval_root = results_dir / "_eval_official"
    eval_root.mkdir(parents=True, exist_ok=True)
    data_dir = _prepare_eval_data(arcade_root, eval_root)

    syntax_yaml = str(data_dir / "dataset_configs" / "syntax_only.yaml")
    raw_stenosis_yaml = str(data_dir / "dataset_configs" / "stenosis_only.yaml")

    # ── CLAHE shadow for S51 / S54 ──
    clahe_shadow_dir = eval_root / "_clahe_stenosis_shadow"
    clahe_stenosis_yaml = str(
        _make_stenosis_clahe_shadow(data_dir, clahe_shadow_dir))

    augment = not args.no_tta
    all_results = {}

    for exp_name, needs_clahe in EXPERIMENTS:
        print(f"\n{'#'*60}\n# {exp_name}\n{'#'*60}")
        try:
            syn_w  = _find_weight(results_dir, exp_name, "syntax")
            sten_w = _find_weight(results_dir, exp_name, "stenosis")
        except FileNotFoundError as exc:
            print(f"  SKIP: {exc}")
            all_results[exp_name] = {"error": str(exc)}
            continue

        print(f"  syntax weights  : {syn_w}")
        print(f"  stenosis weights: {sten_w}")
        print(f"  stenosis CLAHE  : {needs_clahe}")

        try:
            syn_m = evaluate_model(
                str(syn_w), syntax_yaml, split="test",
                augment=augment, imgsz=args.imgsz,
            )
        except Exception as exc:
            print(f"  SYNTAX eval failed: {exc}")
            syn_m = {"error": str(exc)}

        sten_yaml = clahe_stenosis_yaml if needs_clahe else raw_stenosis_yaml
        try:
            sten_m = evaluate_model(
                str(sten_w), sten_yaml, split="test",
                augment=augment, imgsz=args.imgsz,
            )
        except Exception as exc:
            print(f"  STENOSIS eval failed: {exc}")
            sten_m = {"error": str(exc)}

        # ── Combine per-class metrics ──
        combined_pc = {}
        for cls, m in (syn_m.get("per_class") or {}).items():
            if cls == "stenosis":
                continue
            combined_pc[cls] = m
        for cls, m in (sten_m.get("per_class") or {}).items():
            combined_pc["stenosis"] = m

        syn_f1s  = [m.get("f1", 0) for k, m in combined_pc.items() if k != "stenosis"]
        sten_f1  = combined_pc.get("stenosis", {}).get("f1", 0)
        all_f1s  = list(syn_f1s) + ([sten_f1] if "stenosis" in combined_pc else [])
        syn_aps  = [m.get("ap50", 0) for k, m in combined_pc.items() if k != "stenosis"]
        sten_ap  = combined_pc.get("stenosis", {}).get("ap50", 0)

        summary = {
            "syntax_mean_f1":  round(sum(syn_f1s) / len(syn_f1s), 4) if syn_f1s else 0,
            "stenosis_f1":     round(sten_f1, 4),
            "mean_f1":         round(sum(all_f1s) / len(all_f1s), 4) if all_f1s else 0,
            "syntax_mAP50":    round(sum(syn_aps) / len(syn_aps), 4) if syn_aps else 0,
            "stenosis_AP50":   round(sten_ap, 4),
            "n_classes":       len(combined_pc),
            "per_class":       combined_pc,
            "syntax_weights":  str(syn_w),
            "stenosis_weights": str(sten_w),
            "stenosis_clahe":  needs_clahe,
        }

        print(f"  syntax mean F1: {summary['syntax_mean_f1']}")
        print(f"  stenosis F1   : {summary['stenosis_f1']}")
        print(f"  overall mean  : {summary['mean_f1']}")

        all_results[exp_name] = summary

        # Write incremental output so a crash doesn't lose earlier runs
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2)

    print(f"\n==> Wrote {out_path}")

    # ── Summary table ──
    print(f"\n{'name':<25s} {'syn_mean':>9s} {'sten_f1':>8s} {'mean_f1':>8s}")
    print("-" * 55)
    for name, r in all_results.items():
        if "error" in r and "mean_f1" not in r:
            print(f"{name:<25s}  ERROR: {r['error']}")
            continue
        print(f"{name:<25s} {r.get('syntax_mean_f1',0):>9.4f} "
              f"{r.get('stenosis_f1',0):>8.4f} {r.get('mean_f1',0):>8.4f}")


if __name__ == "__main__":
    main()
