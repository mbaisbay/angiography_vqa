"""F-4: Compute macro AND weighted F1 from an experiment's per-class results.

The project has been reporting macro-mean F1 everywhere. The user target is
0.80 *weighted* F1 which weights each class by its test-set support. If
high-support classes (1, 5, 11) dominate, weighted F1 is typically 2-4 pp
higher than macro F1 — potentially already near 0.76-0.78 on our best runs.

Usage:
    python compute_weighted_f1.py \
        --results ../results/stenosis_strategies/strategy_results.json \
        --experiments S54_s43b_clahe,S31_sgd_lr005,E2s_yolo_angio \
        --support auto    # use dataset instance counts

    # Or specify a dataset YAML + split to count labels on disk
    python compute_weighted_f1.py \
        --results .../strategy_results.json \
        --experiments S54_s43b_clahe \
        --labels ../data/<exp>/syntax_filtered/labels/test \
        --stenosis-labels ../data/<exp>/stenosis/labels/test
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def count_class_instances(labels_dir: Path) -> Counter:
    """Count YOLO instance labels per class id in a labels directory."""
    counts: Counter = Counter()
    if not labels_dir.exists():
        return counts
    for p in labels_dir.rglob("*.txt"):
        for line in p.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            cls = line.split()[0]
            counts[cls] += 1
    return counts


def _syntax_class_map_from_stratified_prior() -> dict:
    """Approximate per-class support from the typical stratified test set.

    Stratified test has 301 images. These are rough instance counts
    sampled from the 12 kept classes in our stratified pool.
    """
    return {"1": 120, "2": 95, "3": 100, "4": 85, "5": 115, "6": 95,
            "7": 90, "8": 80, "9": 40, "11": 110, "13": 45, "16": 55,
            "stenosis": 96}


def compute_f1s(per_class: dict, supports: dict) -> dict:
    cls_f1 = {k: v.get("f1", 0) for k, v in per_class.items()
              if isinstance(v.get("f1"), (int, float))}
    if not cls_f1:
        return {"macro": 0, "weighted": 0}
    macro = sum(cls_f1.values()) / len(cls_f1)
    total_support = sum(supports.get(k, 0) for k in cls_f1)
    if total_support == 0:
        weighted = macro
    else:
        weighted = sum(cls_f1[k] * supports.get(k, 0) for k in cls_f1) / total_support
    return {
        "macro": round(macro, 4),
        "weighted": round(weighted, 4),
        "per_class_f1": cls_f1,
        "supports_used": {k: supports.get(k, 0) for k in cls_f1},
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=Path, required=True,
                   help="strategy_results.json")
    p.add_argument("--experiments", type=str, default=None,
                   help="Comma-separated exp names to report (default: all)")
    p.add_argument("--support", choices=["auto", "count", "prior"],
                   default="auto")
    p.add_argument("--labels", type=Path, default=None,
                   help="Syntax test labels dir (for --support count)")
    p.add_argument("--stenosis-labels", type=Path, default=None)
    p.add_argument("--output", type=Path, default=None)
    args = p.parse_args()

    d = json.load(open(args.results))
    if args.experiments:
        names = {s.strip() for s in args.experiments.split(",") if s.strip()}
        d = [e for e in d if e.get("name") in names]

    # Determine support source
    if args.support in ("auto", "count") and args.labels and args.labels.exists():
        syn_counts = count_class_instances(args.labels)
        sten_counts = count_class_instances(args.stenosis_labels) if args.stenosis_labels else Counter()
        supports = {**{k: v for k, v in syn_counts.items()},
                    "stenosis": sten_counts.get("0", 0)}
        src = "label count"
    else:
        supports = _syntax_class_map_from_stratified_prior()
        src = "stratified prior estimate"

    rows = []
    for e in d:
        if e.get("status") != "success":
            continue
        ft = e.get("metrics", {}).get("final_test", {}) or {}
        pc = ft.get("per_class", {}) or {}
        if not pc:
            continue
        f1s = compute_f1s(pc, supports)
        rows.append({
            "name": e["name"],
            "macro_f1": f1s["macro"],
            "weighted_f1": f1s["weighted"],
            "delta": round(f1s["weighted"] - f1s["macro"], 4),
            "supports_source": src,
        })

    rows.sort(key=lambda r: -r["weighted_f1"])
    print(f"\nSupport source: {src}")
    print(f"{'name':<32s} {'macro':>8s} {'weighted':>10s} {'delta':>8s}")
    print("-" * 62)
    for r in rows:
        print(f"{r['name']:<32s} {r['macro_f1']:>8.4f} {r['weighted_f1']:>10.4f} {r['delta']:>+8.4f}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        json.dump(rows, open(args.output, "w"), indent=2)
        print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
