"""B-1: Oversample images containing tail syntax classes.

Problem: classes 9, 13, 16 cap at F1 ~0.55-0.70 across every experiment
because the model sees ~3x fewer images of them. Simplest fix: duplicate
the images that contain them so the training loader sees them more often.

Strategy (Option 1 from the proposal — image-level duplication):
  For each tail class c with count N_c, compute multiplier such that the
  post-duplication effective count approaches the max non-tail count.
  Create symlinks named `<stem>_os{i}.png` / `.txt` in the train folder.

Usage:
    python oversample_tail_classes.py \
        --syntax-data-dir .../data/<exp>/syntax_filtered \
        --tail-classes 9,13,16 \
        --target-count 900
"""

from __future__ import annotations

import argparse
import os
from collections import Counter, defaultdict
from pathlib import Path


def load_image_class_map(labels_dir: Path) -> dict:
    """Return {image_stem: set(class_ids)} for a YOLO labels dir."""
    m = defaultdict(set)
    for p in labels_dir.glob("*.txt"):
        for line in p.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            m[p.stem].add(int(line.split()[0]))
    return dict(m)


def instance_counts(labels_dir: Path) -> Counter:
    counts: Counter = Counter()
    for p in labels_dir.glob("*.txt"):
        for line in p.read_text().splitlines():
            line = line.strip()
            if line:
                counts[int(line.split()[0])] += 1
    return counts


def oversample(
    syntax_data_dir: Path,
    tail_classes: list,
    target_count: int,
) -> dict:
    img_train = syntax_data_dir / "images" / "train"
    lbl_train = syntax_data_dir / "labels" / "train"

    counts = instance_counts(lbl_train)
    img_cls = load_image_class_map(lbl_train)

    print(f"Current instance counts: {dict(counts)}")

    # Build per-image multipliers: max over tail classes of ceil(target/count_c)
    dup_factor: dict = {}
    for stem, classes in img_cls.items():
        factor = 1
        for c in classes:
            if c in tail_classes and counts.get(c, 0) > 0:
                ratio = target_count / counts[c]
                if ratio > 1:
                    factor = max(factor, int(round(ratio)))
        dup_factor[stem] = factor

    # Create duplicates
    created = 0
    for stem, factor in dup_factor.items():
        if factor <= 1:
            continue
        # Find the image file (png/PNG/jpg)
        src_img = None
        for ext in (".png", ".PNG", ".jpg"):
            cand = img_train / (stem + ext)
            if cand.exists():
                src_img = cand
                break
        if src_img is None:
            continue
        src_lbl = lbl_train / (stem + ".txt")
        if not src_lbl.exists():
            continue

        for i in range(1, factor):  # 1..factor-1 additional copies
            dup_stem = f"{stem}_os{i}"
            dst_img = img_train / (dup_stem + src_img.suffix)
            dst_lbl = lbl_train / (dup_stem + ".txt")
            if dst_img.exists() or dst_img.is_symlink():
                continue
            os.symlink(src_img.resolve(), dst_img)
            os.symlink(src_lbl.resolve(), dst_lbl)
            created += 1

    new_counts = instance_counts(lbl_train)
    print(f"New instance counts: {dict(new_counts)}")
    print(f"  Created {created} duplicate symlinks")
    return {"original": dict(counts), "after": dict(new_counts),
            "duplicates_created": created}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--syntax-data-dir", type=Path, required=True)
    p.add_argument("--tail-classes", type=str, default="9,13,16",
                   help="Comma-separated YOLO class ids (NOT ARCADE names)")
    p.add_argument("--target-count", type=int, default=900)
    args = p.parse_args()

    tail = [int(x) for x in args.tail_classes.split(",")]
    stats = oversample(args.syntax_data_dir, tail, args.target_count)
    import json
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
