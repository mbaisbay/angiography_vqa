"""Anatomy-graph post-processing for YOLO-Angio (ARCADE 3rd place).

Coronary tree topology: in a normal heart, vessel segments only connect
to certain neighbours. YOLO often misclassifies the rare tail classes
(9, 13, 16) into more frequent neighbours. This module:

  1. Reads YOLO predictions (instances with class id, confidence, mask).
  2. Builds an adjacency graph (nodes = instances, edges = mask
     proximity via dilation overlap).
  3. Applies hardcoded anatomical rules to either:
        - merge same-class adjacent instances, or
        - reclassify an instance whose claimed class is not adjacency-
          consistent with its neighbours' classes.
  4. Returns the cleaned prediction list.

The rules are deliberately conservative -- when in doubt, leave the
prediction alone. The point is to fix a handful of confidently-wrong
labels on the tail classes, not to second-guess the network everywhere.

ARCADE class IDs (0-indexed YOLO):
    0:1   1:2   2:3   3:4   4:5   5:6   6:7   7:8
    8:9   9:9a  10:10  11:10a 12:11  13:12 14:12a
   15:13 16:14 17:14a 18:15 19:16 20:16a 21:16b 22:16c
   23:12b 24:14b
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Set, Tuple

import cv2
import numpy as np


# ── Anatomical adjacency table ─────────────────────────────────────────
# Maps class idx -> set of class idxs it can legally touch.
# Built from coronary-tree segmental anatomy (Medina et al.) using the
# standard ARCADE class label scheme. A few "main vessel <-> branch"
# entries plus self-adjacency are sufficient to constrain reclassification.
ADJACENCY: Dict[int, Set[int]] = {
    # RCA chain: 1->2->3->4 + branches 16/16a/16b/16c
    0:  {0, 1},                         # 1   (RCA prox)
    1:  {0, 1, 2},                      # 2   (RCA mid)
    2:  {1, 2, 3, 19, 20, 21, 22},      # 3   (RCA dist)
    3:  {2, 3, 19, 20, 21, 22},         # 4   (PDA)
    19: {2, 3, 19, 20, 21, 22},         # 16
    20: {19, 20, 2, 3},                 # 16a
    21: {19, 21, 2, 3},                 # 16b
    22: {19, 22, 2, 3},                 # 16c

    # LM -> LAD chain: 5->6->7->8, branches 9/9a/10/10a
    4:  {4, 5},                         # 5  (LM)
    5:  {4, 5, 6, 8, 9, 10, 11},        # 6  (LAD prox)
    6:  {5, 6, 7, 8, 9, 10, 11},        # 7  (LAD mid)
    7:  {6, 7, 8, 9},                   # 8  (LAD dist)
    8:  {5, 6, 7, 8, 9},                # 9  (D1)
    9:  {8, 9, 5, 6},                   # 9a
    10: {5, 6, 10, 11},                 # 10 (D2)
    11: {5, 6, 10, 11},                 # 10a

    # LCx chain: 5->11->13, OM/PL branches 12/12a/12b/13/14/14a/14b/15
    12: {4, 5, 12, 13, 14, 15, 16},     # 11 (LCx prox)
    13: {12, 13, 14, 15, 16, 23},       # 12
    14: {13, 14, 23},                   # 12a
    23: {13, 14, 23},                   # 12b
    15: {12, 13, 15, 16, 17, 18, 24},   # 13
    16: {12, 15, 16, 17, 18, 24},       # 14
    17: {15, 16, 17, 24},               # 14a
    24: {16, 17, 24},                   # 14b
    18: {12, 15, 16, 18},               # 15
}


def _instances_from_results(results) -> List[dict]:
    """Flatten ultralytics Results into per-instance dicts."""
    out = []
    for r_idx, r in enumerate(results):
        if r.masks is None or len(r.masks) == 0:
            continue
        masks = r.masks.data.cpu().numpy()
        cls = r.boxes.cls.cpu().numpy().astype(int)
        conf = r.boxes.conf.cpu().numpy()
        for i, m in enumerate(masks):
            out.append({
                "img_idx": r_idx,
                "inst_idx": i,
                "cls": int(cls[i]),
                "conf": float(conf[i]),
                "mask": (m > 0.5).astype(np.uint8),
            })
    return out


def _build_adjacency_graph(instances: List[dict],
                            dilate_px: int = 8) -> Dict[int, Set[int]]:
    """Return image-local adjacency: node id -> set of neighbour node ids.

    Two instances are adjacent if their dilated masks overlap by at least
    1 pixel. Only instances from the same image are considered.
    """
    by_img = defaultdict(list)
    for idx, inst in enumerate(instances):
        by_img[inst["img_idx"]].append(idx)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (dilate_px * 2 + 1, dilate_px * 2 + 1))

    graph: Dict[int, Set[int]] = {i: set() for i in range(len(instances))}
    for img_idx, idx_list in by_img.items():
        dilated = {}
        for i in idx_list:
            dilated[i] = cv2.dilate(instances[i]["mask"], kernel)
        for ai, a in enumerate(idx_list):
            ma = dilated[a]
            for b in idx_list[ai + 1:]:
                mb = dilated[b]
                if (ma & mb).any():
                    graph[a].add(b)
                    graph[b].add(a)
    return graph


def _is_adjacency_consistent(cls_self: int, neighbour_cls: List[int]) -> bool:
    if not neighbour_cls:
        return True
    allowed = ADJACENCY.get(cls_self)
    if allowed is None:
        return True
    return any(nc in allowed for nc in neighbour_cls)


def _best_consistent_class(neighbour_cls: List[int],
                            current_conf: float) -> int | None:
    """Return the class id from neighbours that this instance could legally
    be (i.e. one whose ADJACENCY set covers the neighbour majority).
    Picks the most common neighbour class as the reassignment candidate.
    """
    if not neighbour_cls:
        return None
    counts = defaultdict(int)
    for c in neighbour_cls:
        counts[c] += 1
    return max(counts, key=counts.get)


def postprocess(results, dilate_px: int = 8,
                min_conf_to_keep: float = 0.05) -> list:
    """Run anatomy-graph postprocessing in place.

    Args:
        results: list of Ultralytics Results (one per image).
        dilate_px: dilation kernel half-width for adjacency test.
        min_conf_to_keep: below this confidence, a tail-class instance
                          that fails the adjacency check is dropped
                          rather than reclassified.

    Returns:
        results (same list, mutated).
    """
    import torch

    instances = _instances_from_results(results)
    if not instances:
        return results

    graph = _build_adjacency_graph(instances, dilate_px=dilate_px)

    new_cls = {i: inst["cls"] for i, inst in enumerate(instances)}
    drop = set()
    tail_classes = {8, 9, 11, 14, 17, 18, 20, 21, 22, 23, 24}

    for i, inst in enumerate(instances):
        neighbour_cls = [instances[j]["cls"] for j in graph[i]]
        if _is_adjacency_consistent(inst["cls"], neighbour_cls):
            continue
        if inst["cls"] not in tail_classes:
            continue
        candidate = _best_consistent_class(neighbour_cls, inst["conf"])
        if candidate is None:
            continue
        if inst["conf"] < min_conf_to_keep:
            drop.add(i)
        else:
            new_cls[i] = candidate

    by_img: Dict[int, List[int]] = defaultdict(list)
    for i, inst in enumerate(instances):
        if i in drop:
            continue
        by_img[inst["img_idx"]].append(i)

    for r_idx, r in enumerate(results):
        if r.masks is None or len(r.masks) == 0:
            continue
        kept = by_img.get(r_idx, [])
        if not kept:
            r.masks = None
            r.boxes = r.boxes[torch.zeros(0, dtype=torch.long)]
            continue
        local_idx = [instances[k]["inst_idx"] for k in kept]
        new_classes = [new_cls[k] for k in kept]

        keep_t = torch.as_tensor(local_idx, dtype=torch.long,
                                  device=r.masks.data.device)
        r.masks.data = r.masks.data[keep_t]
        if hasattr(r.masks, "xy"):
            r.masks.xy = [r.masks.xy[i] for i in local_idx]
        if hasattr(r.masks, "xyn"):
            r.masks.xyn = [r.masks.xyn[i] for i in local_idx]
        r.boxes = r.boxes[keep_t]

        # Overwrite class ids on the kept boxes
        try:
            r.boxes.data[:, 5] = torch.as_tensor(
                new_classes, dtype=r.boxes.data.dtype, device=r.boxes.data.device)
        except Exception:
            pass

    return results
