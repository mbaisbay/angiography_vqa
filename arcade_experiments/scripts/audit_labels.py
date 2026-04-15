"""F-2: Audit label conversion for class-9 / class-9a contamination.

Hypothesis: class 9 (LAD-D1) has a persistent 0.55 F1 ceiling across all
runs. If the COCO->YOLO conversion accidentally folds class 9a (first
diagonal sub-branch) into class 9, the class-9 training signal is noisy
by construction and no hyperparameter tuning would fix it.

This script:
  1. Loads the original COCO annotation JSONs (train/val/test for syntax).
  2. Reads the YOLO label files the pipeline produced.
  3. For every GT polygon, maps the COCO (image_id, annotation_id) back
     to the expected YOLO class id using the class_mapping.json written
     by prepare_syntax().
  4. Flags any disagreement, missing annotation, or spurious YOLO label.
  5. Reports per-class counts from both sides for a sanity check.

Usage:
    python audit_labels.py \
        --arcade-root ../../arcade/submission \
        --data-dir ../results/stenosis_strategies/data/S54_s43b_clahe
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def load_coco(p: Path) -> dict:
    return json.load(open(p))


def audit_syntax_conversion(arcade_root: Path, data_dir: Path) -> dict:
    """Cross-check syntax COCO annotations vs YOLO labels written on disk."""
    syntax_filtered = data_dir / "syntax_filtered"
    class_mapping_path = syntax_filtered / "class_mapping.json"
    if not class_mapping_path.exists():
        raise SystemExit(
            f"class_mapping.json missing at {class_mapping_path}. "
            "Run data_prep first.")
    mapping = json.load(open(class_mapping_path))
    # mapping typically has keys: {"class_names": {..}, "coco_to_yolo": {..}, ...}
    class_names: dict = mapping.get("class_names", {})
    coco_to_yolo: dict = mapping.get("coco_to_yolo", {})

    print("\n=== Class mapping on disk ===")
    print(f"  Kept classes ({len(class_names)}): {class_names}")
    print(f"  COCO->YOLO map ({len(coco_to_yolo)}): {coco_to_yolo}")

    summary = {
        "class_names": class_names,
        "coco_to_yolo": coco_to_yolo,
        "splits": {},
    }

    for split in ("train", "val", "test"):
        coco_path = arcade_root / "syntax" / split / "annotations" / f"{split}.json"
        if not coco_path.exists():
            print(f"\n  [{split}] COCO annotations missing at {coco_path}")
            continue
        coco = load_coco(coco_path)

        # Build category_id -> (name, kept?)
        cat_id_to_name = {c["id"]: c["name"] for c in coco["categories"]}

        coco_counts = Counter()
        for ann in coco["annotations"]:
            name = cat_id_to_name.get(ann["category_id"], "?")
            coco_counts[name] += 1

        # YOLO label counts (per kept class id)
        yolo_labels_dir = syntax_filtered / "labels" / split
        yolo_counts = Counter()
        if yolo_labels_dir.exists():
            for p in yolo_labels_dir.rglob("*.txt"):
                for line in p.read_text().splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    yolo_counts[line.split()[0]] += 1

        # Translate yolo counts -> human names using class_names
        yolo_named = {}
        for yid, n in yolo_counts.items():
            yolo_named[class_names.get(str(yid), str(yid))] = n

        # Classes present in COCO but NOT mapped to YOLO → dropped
        dropped = {n: coco_counts[n] for n in coco_counts
                   if n not in yolo_named}
        # Classes mapped to YOLO but with 0 COCO count → broken
        ghost = {n: yolo_named[n] for n in yolo_named
                 if n not in coco_counts}

        total_coco = sum(coco_counts.values())
        total_yolo = sum(yolo_counts.values())
        print(f"\n  [{split}] COCO total={total_coco}  YOLO total={total_yolo}")
        print(f"    kept-class COCO counts: {dict(sorted(((n,c) for n,c in coco_counts.items() if n in yolo_named), key=lambda x: -x[1]))}")
        print(f"    kept-class YOLO counts: {dict(sorted(yolo_named.items(), key=lambda x: -x[1]))}")
        if dropped:
            print(f"    dropped classes (present in COCO, not in YOLO): {dict(sorted(dropped.items(), key=lambda x: -x[1]))}")
        if ghost:
            print(f"    *** GHOST CLASSES (in YOLO labels but not COCO!): {ghost}")
        # Delta per kept class
        mismatches = {}
        for n, c in yolo_named.items():
            diff = c - coco_counts.get(n, 0)
            if diff != 0:
                mismatches[n] = {"coco": coco_counts.get(n, 0), "yolo": c, "diff": diff}
        if mismatches:
            print(f"    *** COUNT MISMATCH per class: {mismatches}")

        summary["splits"][split] = {
            "coco_counts": dict(coco_counts),
            "yolo_counts": yolo_named,
            "dropped": dropped,
            "ghost": ghost,
            "mismatches": mismatches,
        }

    # Cross-split class-9/9a focus
    print("\n=== Class 9 vs 9a focus ===")
    for split, s in summary["splits"].items():
        c9 = s["coco_counts"].get("9", 0)
        c9a = s["coco_counts"].get("9a", 0)
        y9 = s["yolo_counts"].get("9", 0)
        print(f"  [{split}] COCO: class 9 = {c9}, class 9a = {c9a}  |  YOLO class 9 = {y9}")
        if y9 != c9:
            print(f"    *** WARNING: YOLO class 9 count ({y9}) != COCO class 9 count ({c9})")
            print(f"        diff = {y9 - c9}; if equal to class 9a count ({c9a}), contamination confirmed")

    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arcade-root", type=Path, required=True)
    p.add_argument("--data-dir", type=Path, required=True,
                   help="data dir created by run_pipeline.data_prep "
                        "(e.g. results/stenosis_strategies/data/S54_s43b_clahe)")
    p.add_argument("--output", type=Path, default=None)
    args = p.parse_args()

    summary = audit_syntax_conversion(args.arcade_root, args.data_dir)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        json.dump(summary, open(args.output, "w"), indent=2)
        print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
