"""Mask operations: polygon rasterization, IoU computation, vessel-stenosis assignment."""

import numpy as np
import cv2


def polygon_to_binary_mask(polygon_normalized: list, mask_size: int = 512) -> np.ndarray:
    """Convert normalized polygon coordinates to a binary mask.

    Args:
        polygon_normalized: List of [x, y] pairs, each in [0, 1].
        mask_size: Output mask dimension (square).

    Returns:
        Binary mask of shape (mask_size, mask_size), dtype uint8.
    """
    mask = np.zeros((mask_size, mask_size), dtype=np.uint8)

    if not polygon_normalized or len(polygon_normalized) < 3:
        return mask

    # Denormalize and convert to int32 array for cv2.fillPoly
    points = np.array(polygon_normalized, dtype=np.float32)
    points[:, 0] *= mask_size  # x
    points[:, 1] *= mask_size  # y
    points = points.astype(np.int32)

    cv2.fillPoly(mask, [points], 1)
    return mask


def compute_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Compute Intersection over Union between two binary masks."""
    intersection = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    if union == 0:
        return 0.0
    return float(intersection / union)


def compute_intersection_over_smaller(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Compute intersection divided by the smaller mask's area.

    Preferred for stenosis-vessel assignment since stenoses are much smaller
    than vessel segments. Standard IoU would give misleadingly low values.
    """
    intersection = np.logical_and(mask_a, mask_b).sum()
    smaller_area = min(mask_a.sum(), mask_b.sum())
    if smaller_area == 0:
        return 0.0
    return float(intersection / smaller_area)


def compute_dice(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Compute Dice coefficient (F1) between two binary masks."""
    intersection = np.logical_and(mask_a, mask_b).sum()
    total = mask_a.sum() + mask_b.sum()
    if total == 0:
        return 0.0
    return float(2 * intersection / total)


METRIC_FUNCTIONS = {
    "iou": compute_iou,
    "intersection_over_smaller": compute_intersection_over_smaller,
    "dice": compute_dice,
}


def assign_stenosis_to_vessels(stenosis_preds: list, vessel_preds: list,
                               mask_size: int = 512, threshold: float = 0.01,
                               metric: str = "intersection_over_smaller") -> dict:
    """Match each stenosis prediction to the best-overlapping vessel segment.

    Args:
        stenosis_preds: List of stenosis prediction dicts with 'polygon_normalized'.
        vessel_preds: List of vessel prediction dicts with 'polygon_normalized',
                      'class_id', 'class_name', 'confidence'.
        mask_size: Size of the binary mask for rasterization.
        threshold: Minimum overlap score to count as a match.
        metric: Overlap metric to use.

    Returns:
        Dict with 'matched', 'unmatched', and 'summary' fields.
    """
    metric_fn = METRIC_FUNCTIONS.get(metric, compute_intersection_over_smaller)

    # Pre-compute vessel masks
    vessel_masks = []
    for vp in vessel_preds:
        vmask = polygon_to_binary_mask(vp["polygon_normalized"], mask_size)
        vessel_masks.append(vmask)

    matched = []
    unmatched = []

    for s_idx, sp in enumerate(stenosis_preds):
        s_mask = polygon_to_binary_mask(sp["polygon_normalized"], mask_size)

        if s_mask.sum() == 0:
            unmatched.append({
                "stenosis_id": s_idx,
                "confidence": sp.get("confidence", 0),
                "bbox": sp.get("bbox_xywh", []),
                "reason": "empty_mask",
            })
            continue

        best_score = 0.0
        best_vessel_idx = -1

        for v_idx, v_mask in enumerate(vessel_masks):
            score = metric_fn(s_mask, v_mask)
            if score > best_score:
                best_score = score
                best_vessel_idx = v_idx

        if best_score >= threshold and best_vessel_idx >= 0:
            vp = vessel_preds[best_vessel_idx]
            matched.append({
                "stenosis_id": s_idx,
                "stenosis_confidence": sp.get("confidence", 0),
                "stenosis_bbox": sp.get("bbox_xywh", []),
                "matched_vessel_segment": vp.get("class_name", str(vp.get("class_id", "?"))),
                "vessel_class_id": vp.get("class_id", -1),
                "vessel_confidence": vp.get("confidence", 0),
                "overlap_score": round(best_score, 4),
                "overlap_metric": metric,
            })
        else:
            unmatched.append({
                "stenosis_id": s_idx,
                "confidence": sp.get("confidence", 0),
                "bbox": sp.get("bbox_xywh", []),
                "best_overlap": round(best_score, 4),
                "reason": "below_threshold",
            })

    # Build summary
    affected_segments = sorted(set(m["matched_vessel_segment"] for m in matched))

    return {
        "matched": matched,
        "unmatched": unmatched,
        "summary": {
            "total_stenoses": len(stenosis_preds),
            "matched_count": len(matched),
            "unmatched_count": len(unmatched),
            "affected_segments": affected_segments,
        },
    }
