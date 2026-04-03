"""Convert COCO JSON segmentation annotations to YOLO segmentation format.

YOLO seg format: one .txt file per image, each line:
  class_id x1 y1 x2 y2 ... xN yN  (normalized 0-1 polygon coordinates)

Supports custom category ID remapping via --class-offset for merged datasets.
"""

import argparse
import json
from pathlib import Path


def convert_annotation(annotation: dict, img_width: int, img_height: int,
                       coco_to_yolo: dict) -> list:
    """Convert a single COCO annotation to YOLO segmentation format lines.

    Returns list of YOLO format strings, one per polygon part.
    """
    coco_id = annotation["category_id"]
    if coco_id not in coco_to_yolo:
        return []

    yolo_cls = coco_to_yolo[coco_id]
    segmentation = annotation.get("segmentation", [])
    lines = []

    for polygon in segmentation:
        if len(polygon) < 6:
            continue

        normalized = []
        for i in range(0, len(polygon), 2):
            x = max(0.0, min(1.0, polygon[i] / img_width))
            y = max(0.0, min(1.0, polygon[i + 1] / img_height))
            normalized.extend([f"{x:.6f}", f"{y:.6f}"])

        line = f"{yolo_cls} " + " ".join(normalized)
        lines.append(line)

    return lines


def convert_coco_to_yolo(coco_json_path: str, output_labels_dir: str,
                         coco_to_yolo: dict) -> dict:
    """Convert a full COCO JSON annotation file to YOLO label .txt files.

    Returns stats dict with conversion counts.
    """
    coco_json_path = Path(coco_json_path)
    output_labels_dir = Path(output_labels_dir)
    output_labels_dir.mkdir(parents=True, exist_ok=True)

    with open(coco_json_path, "r") as f:
        coco_data = json.load(f)

    images_by_id = {img["id"]: img for img in coco_data["images"]}

    # Group annotations by image_id
    anns_by_image = {}
    for ann in coco_data["annotations"]:
        img_id = ann["image_id"]
        if img_id not in anns_by_image:
            anns_by_image[img_id] = []
        anns_by_image[img_id].append(ann)

    stats = {
        "total_images": 0,
        "images_with_labels": 0,
        "total_labels": 0,
        "skipped": 0,
    }

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

        with open(label_path, "w") as f:
            if all_lines:
                f.write("\n".join(all_lines) + "\n")
                stats["images_with_labels"] += 1
                stats["total_labels"] += len(all_lines)

    return stats


def build_identity_mapping(coco_json_path: str, class_offset: int = 0) -> dict:
    """Build identity COCO→YOLO mapping with optional class offset.

    For a COCO JSON where categories are already 0-indexed (filtered),
    maps each category ID to itself + offset.
    """
    with open(coco_json_path, "r") as f:
        coco_data = json.load(f)

    mapping = {}
    for cat in coco_data["categories"]:
        mapping[cat["id"]] = cat["id"] + class_offset
    return mapping


def main():
    parser = argparse.ArgumentParser(
        description="Convert COCO JSON annotations to YOLO segmentation format"
    )
    parser.add_argument(
        "--coco-json", type=str, required=True,
        help="Path to COCO JSON annotation file"
    )
    parser.add_argument(
        "--output-dir", type=str, required=True,
        help="Output directory for YOLO .txt label files"
    )
    parser.add_argument(
        "--class-offset", type=int, default=0,
        help="Offset to add to all class IDs (for merging datasets)"
    )
    parser.add_argument(
        "--mapping-json", type=str, default=None,
        help="Optional JSON file with custom coco_to_yolo mapping"
    )
    args = parser.parse_args()

    if args.mapping_json:
        with open(args.mapping_json, "r") as f:
            mapping_data = json.load(f)
        coco_to_yolo = {int(k): v for k, v in mapping_data["old_to_new"].items()}
    else:
        coco_to_yolo = build_identity_mapping(args.coco_json, args.class_offset)

    print(f"Converting: {args.coco_json}")
    print(f"Output:     {args.output_dir}")
    print(f"Mapping:    {coco_to_yolo}")

    stats = convert_coco_to_yolo(args.coco_json, args.output_dir, coco_to_yolo)

    print(f"\nConversion complete:")
    print(f"  Total images:      {stats['total_images']}")
    print(f"  Images with labels: {stats['images_with_labels']}")
    print(f"  Total labels:      {stats['total_labels']}")
    print(f"  Skipped:           {stats['skipped']}")


if __name__ == "__main__":
    main()
