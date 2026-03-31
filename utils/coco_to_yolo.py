"""Convert COCO JSON annotations to YOLO segmentation format."""

import json
from pathlib import Path


def build_category_mapping(categories: list, task: str) -> tuple:
    """Build COCO category ID → YOLO 0-indexed class mapping.

    Args:
        categories: List of category dicts from COCO JSON.
        task: "syntax" or "stenosis".

    Returns:
        (coco_to_yolo, yolo_to_coco, class_names)
        - coco_to_yolo: {coco_id: yolo_index}
        - yolo_to_coco: {yolo_index: coco_id}
        - class_names: ordered list of class names by YOLO index
    """
    # Sort categories by their COCO ID for deterministic ordering
    sorted_cats = sorted(categories, key=lambda c: c["id"])

    coco_to_yolo = {}
    yolo_to_coco = {}
    class_names = []

    for yolo_idx, cat in enumerate(sorted_cats):
        coco_id = cat["id"]
        name = str(cat["name"])
        coco_to_yolo[coco_id] = yolo_idx
        yolo_to_coco[yolo_idx] = coco_id
        class_names.append(name)

    return coco_to_yolo, yolo_to_coco, class_names


def convert_annotation(annotation: dict, img_width: int, img_height: int,
                       coco_to_yolo: dict) -> list:
    """Convert a single COCO annotation to YOLO segmentation format lines.

    Handles multi-polygon annotations by producing one line per polygon part.

    Args:
        annotation: Single COCO annotation dict.
        img_width: Image width for normalization.
        img_height: Image height for normalization.
        coco_to_yolo: COCO category ID to YOLO index mapping.

    Returns:
        List of YOLO format strings, one per polygon part.
        Empty list if category not in mapping or polygon is degenerate.
    """
    coco_id = annotation["category_id"]
    if coco_id not in coco_to_yolo:
        return []

    yolo_cls = coco_to_yolo[coco_id]
    segmentation = annotation.get("segmentation", [])
    lines = []

    for polygon in segmentation:
        # Each polygon is a flat list: [x1, y1, x2, y2, ..., xn, yn]
        if len(polygon) < 6:  # Need at least 3 points
            continue

        # Normalize coordinates
        normalized = []
        for i in range(0, len(polygon), 2):
            x = polygon[i] / img_width
            y = polygon[i + 1] / img_height
            # Clamp to [0, 1]
            x = max(0.0, min(1.0, x))
            y = max(0.0, min(1.0, y))
            normalized.extend([f"{x:.6f}", f"{y:.6f}"])

        line = f"{yolo_cls} " + " ".join(normalized)
        lines.append(line)

    return lines


def convert_coco_to_yolo(coco_json_path: str, output_labels_dir: str,
                         coco_to_yolo: dict) -> dict:
    """Convert a full COCO JSON annotation file to YOLO label .txt files.

    Args:
        coco_json_path: Path to the COCO JSON file.
        output_labels_dir: Directory to write YOLO .txt label files.
        coco_to_yolo: COCO category ID to YOLO index mapping.

    Returns:
        Stats dict with conversion counts.
    """
    coco_json_path = Path(coco_json_path)
    output_labels_dir = Path(output_labels_dir)
    output_labels_dir.mkdir(parents=True, exist_ok=True)

    with open(coco_json_path, "r") as f:
        coco_data = json.load(f)

    # Build explicit image_id → image_info lookup (not positional!)
    images_by_id = {img["id"]: img for img in coco_data["images"]}

    # Group annotations by image_id
    anns_by_image = {}
    for ann in coco_data["annotations"]:
        img_id = ann["image_id"]
        if img_id not in anns_by_image:
            anns_by_image[img_id] = []
        anns_by_image[img_id].append(ann)

    stats = {"total_images": 0, "images_with_labels": 0, "total_labels": 0, "skipped": 0}

    # Process each image
    for img_id, img_info in images_by_id.items():
        stats["total_images"] += 1
        file_name = img_info["file_name"]
        label_name = Path(file_name).stem + ".txt"
        label_path = output_labels_dir / label_name

        img_w = img_info["width"]
        img_h = img_info["height"]

        annotations = anns_by_image.get(img_id, [])
        all_lines = []

        for ann in annotations:
            lines = convert_annotation(ann, img_w, img_h, coco_to_yolo)
            all_lines.extend(lines)
            if not lines:
                stats["skipped"] += 1

        # Write label file (even if empty — YOLO expects one per image)
        with open(label_path, "w") as f:
            if all_lines:
                f.write("\n".join(all_lines) + "\n")
                stats["images_with_labels"] += 1
                stats["total_labels"] += len(all_lines)

    return stats


def save_category_mapping(coco_to_yolo: dict, class_names: list,
                          output_path: str) -> None:
    """Save category mapping to JSON for cross-script consistency."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    mapping = {
        "coco_to_yolo": {str(k): v for k, v in coco_to_yolo.items()},
        "yolo_to_name": {str(i): name for i, name in enumerate(class_names)},
        "class_names": class_names,
        "num_classes": len(class_names),
    }

    with open(output_path, "w") as f:
        json.dump(mapping, f, indent=2)


def load_category_mapping(mapping_path: str) -> dict:
    """Load a previously saved category mapping from JSON."""
    with open(mapping_path, "r") as f:
        mapping = json.load(f)

    # Convert string keys back to int
    mapping["coco_to_yolo"] = {int(k): v for k, v in mapping["coco_to_yolo"].items()}
    mapping["yolo_to_name"] = {int(k): v for k, v in mapping["yolo_to_name"].items()}
    return mapping
