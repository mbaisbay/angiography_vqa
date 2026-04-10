"""Shared utilities for crop-per-vessel stenosis detection.

Provides coordinate transformation helpers used by both
build_crop_dataset.py (training data) and crop_inference.py (inference).
"""

import cv2
import numpy as np
from shapely.geometry import Polygon, box as shapely_box
from shapely.validation import make_valid


# ---------------------------------------------------------------------------
# Bounding box helpers
# ---------------------------------------------------------------------------

def bbox_xywh_to_xyxy(bbox_xywh):
    """Convert YOLO [cx, cy, w, h] to [x1, y1, x2, y2]."""
    cx, cy, w, h = bbox_xywh
    return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]


def compute_padded_bbox(bbox_xyxy, padding_frac, img_size, min_dim=16):
    """Add fractional padding to bbox and clip to image bounds.

    Args:
        bbox_xyxy: [x1, y1, x2, y2] in pixel coords.
        padding_frac: Fraction of bbox dimension to add as padding.
        img_size: Image dimension (assumes square).
        min_dim: Minimum crop dimension in pixels.

    Returns:
        (x1, y1, x2, y2) as integers clipped to [0, img_size],
        or None if the crop is smaller than min_dim.
    """
    x1, y1, x2, y2 = bbox_xyxy
    w = x2 - x1
    h = y2 - y1
    pad_x = w * padding_frac
    pad_y = h * padding_frac
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(img_size, x2 + pad_x)
    y2 = min(img_size, y2 + pad_y)
    x1, y1, x2, y2 = int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))

    if (x2 - x1) < min_dim or (y2 - y1) < min_dim:
        return None

    return (x1, y1, x2, y2)


def crop_and_resize(image, crop_bbox, target_size):
    """Crop image region and resize to target_size x target_size.

    Args:
        image: numpy array (H, W) or (H, W, C).
        crop_bbox: (x1, y1, x2, y2) in integer pixel coords.
        target_size: Output dimension.

    Returns:
        Resized crop as numpy array.
    """
    x1, y1, x2, y2 = crop_bbox
    crop = image[y1:y2, x1:x2]
    # Use INTER_AREA for downscale, INTER_LINEAR for upscale
    h, w = crop.shape[:2]
    if h > target_size or w > target_size:
        interp = cv2.INTER_AREA
    else:
        interp = cv2.INTER_LINEAR
    return cv2.resize(crop, (target_size, target_size), interpolation=interp)


# ---------------------------------------------------------------------------
# Polygon coordinate remapping
# ---------------------------------------------------------------------------

def remap_polygon_to_crop(polygon_pixels, crop_bbox, normalize=True):
    """Remap polygon from original image pixel coords to crop space.

    Args:
        polygon_pixels: List of (x, y) tuples in original image pixel coords,
                        OR flat list [x1, y1, x2, y2, ...].
        crop_bbox: (x1, y1, x2, y2) of the crop in original image.
        normalize: If True, return coords normalized to [0, 1].

    Returns:
        List of (x, y) tuples in crop space (normalized or pixel).
        Points are clipped to crop bounds.
    """
    cx1, cy1, cx2, cy2 = crop_bbox
    crop_w = cx2 - cx1
    crop_h = cy2 - cy1
    if crop_w <= 0 or crop_h <= 0:
        return []

    # Handle flat list format
    if polygon_pixels and not isinstance(polygon_pixels[0], (list, tuple)):
        polygon_pixels = [
            (polygon_pixels[i], polygon_pixels[i + 1])
            for i in range(0, len(polygon_pixels), 2)
        ]

    remapped = []
    for px, py in polygon_pixels:
        rx = (px - cx1) / crop_w
        ry = (py - cy1) / crop_h
        rx = max(0.0, min(1.0, rx))
        ry = max(0.0, min(1.0, ry))
        if normalize:
            remapped.append((rx, ry))
        else:
            remapped.append((rx * crop_w, ry * crop_h))

    return remapped


def remap_crop_to_original(polygon_crop_normalized, crop_bbox, img_size=512):
    """Remap normalized crop-space polygon back to original image pixel coords.

    Args:
        polygon_crop_normalized: List of [x, y] pairs in [0, 1] crop space.
        crop_bbox: (x1, y1, x2, y2) of the crop in original image.
        img_size: Original image dimension.

    Returns:
        List of [x, y] pairs in original image pixel coords.
    """
    cx1, cy1, cx2, cy2 = crop_bbox
    crop_w = cx2 - cx1
    crop_h = cy2 - cy1

    remapped = []
    for nx, ny in polygon_crop_normalized:
        px = cx1 + nx * crop_w
        py = cy1 + ny * crop_h
        px = max(0, min(img_size, px))
        py = max(0, min(img_size, py))
        remapped.append([px, py])

    return remapped


# ---------------------------------------------------------------------------
# Stenosis-vessel overlap detection
# ---------------------------------------------------------------------------

def find_overlapping_stenoses(stenosis_anns, crop_bbox):
    """Find stenosis annotations whose polygons overlap with the crop bbox.

    Args:
        stenosis_anns: List of COCO annotation dicts with 'segmentation'
                       (pixel coords, flat list [x1,y1,x2,y2,...]).
        crop_bbox: (x1, y1, x2, y2) of the crop region.

    Returns:
        List of (annotation, polygon_points) tuples for overlapping stenoses.
        polygon_points is a list of (x, y) tuples in original pixel coords.
    """
    x1, y1, x2, y2 = crop_bbox
    crop_box = shapely_box(x1, y1, x2, y2)
    results = []

    for ann in stenosis_anns:
        for seg in ann.get("segmentation", []):
            if len(seg) < 6:
                continue
            points = [(seg[i], seg[i + 1]) for i in range(0, len(seg), 2)]
            if len(points) < 3:
                continue
            try:
                poly = Polygon(points)
                if not poly.is_valid:
                    poly = poly.buffer(0)
                if poly.is_empty:
                    continue
                intersection = poly.intersection(crop_box)
                if not intersection.is_empty and intersection.area > 0:
                    results.append((ann, points))
            except Exception:
                continue

    return results


# ---------------------------------------------------------------------------
# YOLO label I/O
# ---------------------------------------------------------------------------

def save_yolo_label(label_path, annotations):
    """Write YOLO segmentation label file.

    Args:
        label_path: Output .txt path.
        annotations: List of (class_id, [(x_norm, y_norm), ...]) tuples.
    """
    with open(label_path, "w") as f:
        for cls_id, polygon_norm in annotations:
            if len(polygon_norm) < 3:
                continue
            coords = " ".join(f"{x:.6f} {y:.6f}" for x, y in polygon_norm)
            f.write(f"{cls_id} {coords}\n")


# ---------------------------------------------------------------------------
# Polygon NMS
# ---------------------------------------------------------------------------

def polygon_nms(predictions, iou_threshold=0.5):
    """Non-maximum suppression for polygon predictions.

    Args:
        predictions: List of (polygon_pixel_coords, confidence) tuples.
                     polygon_pixel_coords is a list of [x, y] pairs.
        iou_threshold: IoU threshold for suppression.

    Returns:
        Filtered list of (polygon_pixel_coords, confidence) tuples.
    """
    if len(predictions) <= 1:
        return predictions

    # Sort by confidence descending
    predictions = sorted(predictions, key=lambda x: x[1], reverse=True)

    # Build Shapely polygons
    polys = []
    for coords, _conf in predictions:
        try:
            p = Polygon(coords)
            if not p.is_valid:
                p = make_valid(p)
            polys.append(p)
        except Exception:
            polys.append(Polygon())

    keep = []
    suppressed = set()

    for i in range(len(predictions)):
        if i in suppressed:
            continue
        keep.append(predictions[i])

        for j in range(i + 1, len(predictions)):
            if j in suppressed:
                continue
            if polys[i].is_empty or polys[j].is_empty:
                continue
            try:
                inter = polys[i].intersection(polys[j]).area
                union = polys[i].union(polys[j]).area
                if union > 0 and inter / union >= iou_threshold:
                    suppressed.add(j)
            except Exception:
                continue

    return keep
