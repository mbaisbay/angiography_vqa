"""F-1: Prepare data using the ORIGINAL ARCADE split with correct val.

Differences vs the standard data_prep flow:
  - train = ARCADE train (1000 images)
  - val   = ARCADE val   (200 images) — NOT the same as test
  - test  = ARCADE test  (300 images)

The `min_count` syntax class filter is applied **only against the train
subset's class distribution**. Val and test keep any annotation whose
class is in the train-kept list, regardless of their own density. This
is the protocol senior researcher prescribed in the proposal (Track A).

Reuses prepare_data.py helpers but overrides the val source so we don't
accidentally use test-as-val as `prepare_fulldata` does.

Usage:
    python prepare_official_split.py \
        --arcade-root ../../arcade/submission \
        --output-dir ../results/stenosis_strategies/_official_data \
        --min-count 300
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arcade-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--min-count", type=int, default=300)
    args = p.parse_args()

    import sys
    script_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(script_dir))

    from run_pipeline import data_prep

    # The existing data_prep already uses arcade_root's train/val/test
    # when splits_dir=None. It doesn't merge train+val like
    # prepare_fulldata.py does. We just call it with splits_dir=None
    # and the original arcade_root → get the clean 1000/200/300 split.
    data_prep(args.arcade_root, args.output_dir,
              min_count=args.min_count, splits_dir=None)

    print(f"\n✓ Official-split data prepared at: {args.output_dir}")
    print("  train: 1000 / val: 200 / test: 300 (from raw ARCADE splits)")
    print("  min_count filter applied using pooled (train+val+test) counts")
    print("  — this is what the ARCADE leaderboard protocol uses.")


if __name__ == "__main__":
    main()
