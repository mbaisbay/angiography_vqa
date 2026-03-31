"""Visualization utilities for vessel segment masks and stenosis overlays."""

import numpy as np
import cv2

# 25-color palette for vessel segment classes (BGR format for OpenCV)
VESSEL_PALETTE = [
    (255, 0, 0),     # 1  - blue
    (0, 255, 0),     # 2  - green
    (0, 0, 255),     # 3  - red
    (255, 255, 0),   # 4  - cyan
    (255, 0, 255),   # 5  - magenta
    (0, 255, 255),   # 6  - yellow
    (128, 0, 0),     # 7  - dark blue
    (0, 128, 0),     # 8  - dark green
    (0, 0, 128),     # 9  - dark red
    (128, 128, 0),   # 9a - teal
    (128, 0, 128),   # 10 - purple
    (0, 128, 128),   # 10a - olive
    (255, 128, 0),   # 11 - light blue
    (128, 255, 0),   # 12 - lime
    (0, 128, 255),   # 12a - orange
    (255, 0, 128),   # 13 - rose
    (128, 255, 128), # 14 - light green
    (128, 128, 255), # 14a - light red
    (255, 128, 128), # 15 - light blue 2
    (64, 0, 128),    # 16 - indigo
    (0, 64, 128),    # 16a - brown
    (128, 64, 0),    # 16b - navy green
    (64, 128, 0),    # 16c - dark lime
    (0, 255, 128),   # 12b - spring green
    (128, 0, 255),   # 14b - violet
]

STENOSIS_COLOR = (0, 0, 255)  # Red in BGR


def draw_masks_overlay(image: np.ndarray, predictions: list,
                       alpha: float = 0.4, is_stenosis: bool = False) -> np.ndarray:
    """Draw semi-transparent colored masks on a grayscale or BGR image.

    Args:
        image: Input image (grayscale or BGR).
        predictions: List of prediction dicts with 'polygon_normalized' and 'class_id'.
        alpha: Transparency of the overlay (0=transparent, 1=opaque).
        is_stenosis: If True, use red for all masks.

    Returns:
        BGR image with overlaid masks.
    """
    if len(image.shape) == 2:
        vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        vis = image.copy()

    h, w = vis.shape[:2]
    overlay = vis.copy()

    for pred in predictions:
        polygon = pred.get("polygon_normalized", [])
        if not polygon or len(polygon) < 3:
            continue

        points = np.array(polygon, dtype=np.float32)
        points[:, 0] *= w
        points[:, 1] *= h
        points = points.astype(np.int32)

        if is_stenosis:
            color = STENOSIS_COLOR
        else:
            cls_id = pred.get("class_id", 0)
            color = VESSEL_PALETTE[cls_id % len(VESSEL_PALETTE)]

        cv2.fillPoly(overlay, [points], color)
        cv2.polylines(vis, [points], isClosed=True, color=color, thickness=1)

    cv2.addWeighted(overlay, alpha, vis, 1 - alpha, 0, vis)

    # Add class labels
    for pred in predictions:
        polygon = pred.get("polygon_normalized", [])
        if not polygon or len(polygon) < 3:
            continue

        points = np.array(polygon, dtype=np.float32)
        points[:, 0] *= w
        points[:, 1] *= h

        # Label at centroid
        cx = int(points[:, 0].mean())
        cy = int(points[:, 1].mean())
        label = pred.get("class_name", str(pred.get("class_id", "?")))
        conf = pred.get("confidence", 0)
        text = f"{label} {conf:.2f}"

        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
        cv2.rectangle(vis, (cx - 1, cy - th - 2), (cx + tw + 1, cy + 2), (0, 0, 0), -1)
        cv2.putText(vis, text, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                    (255, 255, 255), 1, cv2.LINE_AA)

    return vis


def draw_assignment_visualization(image: np.ndarray, vessel_preds: list,
                                  stenosis_preds: list, assignments: dict,
                                  alpha: float = 0.3) -> np.ndarray:
    """Draw both vessel and stenosis masks with assignment connections.

    Args:
        image: Input grayscale or BGR image.
        vessel_preds: Vessel segment predictions.
        stenosis_preds: Stenosis predictions.
        assignments: Output from assign_stenosis_to_vessels().
        alpha: Mask transparency.

    Returns:
        BGR visualization image.
    """
    # Draw vessel masks first
    vis = draw_masks_overlay(image, vessel_preds, alpha=alpha, is_stenosis=False)

    # Draw stenosis masks on top
    h, w = vis.shape[:2]
    for sp in stenosis_preds:
        polygon = sp.get("polygon_normalized", [])
        if not polygon or len(polygon) < 3:
            continue

        points = np.array(polygon, dtype=np.float32)
        points[:, 0] *= w
        points[:, 1] *= h
        points = points.astype(np.int32)

        # Hatched fill for stenosis
        cv2.fillPoly(vis, [points], STENOSIS_COLOR)
        cv2.polylines(vis, [points], isClosed=True, color=(255, 255, 255), thickness=2)

    # Draw assignment connections
    for match in assignments.get("matched", []):
        s_idx = match["stenosis_id"]
        if s_idx >= len(stenosis_preds):
            continue

        sp = stenosis_preds[s_idx]
        s_poly = np.array(sp.get("polygon_normalized", []), dtype=np.float32)
        if len(s_poly) < 3:
            continue

        s_cx = int(s_poly[:, 0].mean() * w)
        s_cy = int(s_poly[:, 1].mean() * h)

        # Find matching vessel
        v_cls_id = match.get("vessel_class_id", -1)
        for vp in vessel_preds:
            if vp.get("class_id") == v_cls_id:
                v_poly = np.array(vp.get("polygon_normalized", []), dtype=np.float32)
                if len(v_poly) >= 3:
                    v_cx = int(v_poly[:, 0].mean() * w)
                    v_cy = int(v_poly[:, 1].mean() * h)
                    cv2.arrowedLine(vis, (s_cx, s_cy), (v_cx, v_cy),
                                    (255, 255, 255), 1, tipLength=0.05)
                break

        # Annotation
        segment = match.get("matched_vessel_segment", "?")
        score = match.get("overlap_score", 0)
        text = f"-> seg {segment} ({score:.2f})"
        cv2.putText(vis, text, (s_cx + 5, s_cy - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1, cv2.LINE_AA)

    return vis
