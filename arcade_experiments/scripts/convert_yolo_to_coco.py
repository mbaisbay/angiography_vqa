"""Convert YOLO segmentation label files back to COCO JSON format.

Reads YOLO .txt label files (class_id x1 y1 x2 y2 ... xN yN with normalized
coordinates) and reconstructs a COCO-format JSON annotation file.

Useful for evaluation with COCO-format metrics on pseudo-labeled data.
"""

import argparse
import json
from pathlib import Path


def yolo_line_to_coco_annotation(line: str, ann_id: int, image_id: int,
                                 img_width: int, img_height: int) -> dict:
    """Convert a single YOLO label line to a COCO annotation dict.

    Args:
        line: YOLO format line: "class_id x1 y1 x2 y2 ... xN yN"
        ann_id: Annotation ID for COCO format.
        image_id: Image ID this annotation belongs to.
        img_width: Image width for denormalization.
        img_height: Image height for denormalization.

    Returns:
        COCO annotation dict with segmentation, bbox, area, etc.
    """
    parts = line.strip().split()
    if len(parts) < 7:  # class_id + at least 3 points (6 coords)
        return None

    category_id = int(parts[0])
    coords = [float(x) for x in parts[1:]]

    # Denormalize polygon coordinates
    polygon = []
    xs, ys = [], []
    for i in range(0, len(coords), 2):
        x = coords[i] * img_width
        y = coords[i + 1] * img_height
        polygon.extend([x, y])
        xs.append(x)
        ys.append(y)

    if len(xs) < 3:
        return None

    # Compute bbox [x, y, width, height]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    bbox = [x_min, y_min, x_max - x_min, y_max - y_min]

    # Compute area using shoelace formula
    area = 0.0
    n = len(xs)
    for i in range(n):
        j = (i + 1) % n
        area += xs[i] * ys[j]
        area -= xs[j] * ys[i]
    area = abs(area) / 2.0

    return {
        "id": ann_id,
        "image_id": image_id,
        "category_id": category_id,
        "segmentation": [polygon],
        "bbox": [round(v, 2) for v in bbox],
        "area": round(area, 2),
        "iscrowd": 0,
    }


def convert_yolo_to_coco(labels_dir: str, images_dir: str,
                         class_names: dict, img_width: int = 512,
                         img_height: int = 512) -> dict:
    """Convert a directory of YOLO label files to COCO JSON format.

    Args:
        labels_dir: Directory containing YOLO .txt label files.
        images_dir: Directory containing corresponding images.
        class_names: {class_id: name} mapping.
        img_width: Image width (default 512 for ARCADE).
        img_height: Image height (default 512 for ARCADE).

    Returns:
        COCO-format dict with images, categories, and annotations.
    """
    labels_dir = Path(labels_dir)
    images_dir = Path(images_dir)

    # Build categories list
    categories = []
    for cls_id in sorted(class_names.keys()):
        categories.append({
            "id": cls_id,
            "name": class_names[cls_id],
            "supercategory": "",
        })

    # Collect images and annotations
    images = []
    annotations = []
    ann_id = 1

    image_files = sorted(images_dir.glob("*.png")) + sorted(images_dir.glob("*.PNG"))

    for img_idx, img_path in enumerate(image_files, start=1):
        image_id = img_idx
        images.append({
            "id": image_id,
            "file_name": img_path.name,
            "width": img_width,
            "height": img_height,
        })

        label_path = labels_dir / (img_path.stem + ".txt")
        if not label_path.exists():
            continue

        with open(label_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                ann = yolo_line_to_coco_annotation(
                    line, ann_id, image_id, img_width, img_height
                )
                if ann is not None:
                    annotations.append(ann)
                    ann_id += 1

    return {
        "images": images,
        "categories": categories,
        "annotations": annotations,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Convert YOLO segmentation labels to COCO JSON format"
    )
    parser.add_argument(
        "--labels-dir", type=str, required=True,
        help="Directory containing YOLO .txt label files"
    )
    parser.add_argument(
        "--images-dir", type=str, required=True,
        help="Directory containing corresponding images"
    )
    parser.add_argument(
        "--class-names-json", type=str, required=True,
        help="JSON file with class_names: {id: name} mapping"
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Output COCO JSON file path"
    )
    parser.add_argument(
        "--img-width", type=int, default=512,
        help="Image width (default: 512)"
    )
    parser.add_argument(
        "--img-height", type=int, default=512,
        help="Image height (default: 512)"
    )
    args = parser.parse_args()

    with open(args.class_names_json, "r") as f:
        mapping = json.load(f)

    # Support both {id: name} and {"class_names": {id: name}} formats
    if "class_names" in mapping:
        class_names = {int(k): v for k, v in mapping["class_names"].items()}
    else:
        class_names = {int(k): v for k, v in mapping.items()}

    print(f"Labels:  {args.labels_dir}")
    print(f"Images:  {args.images_dir}")
    print(f"Classes: {class_names}")

    coco_data = convert_yolo_to_coco(
        args.labels_dir, args.images_dir, class_names,
        args.img_width, args.img_height
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(coco_data, f)

    print(f"\nConversion complete:")
    print(f"  Images:      {len(coco_data['images'])}")
    print(f"  Annotations: {len(coco_data['annotations'])}")
    print(f"  Saved:       {output_path}")


if __name__ == "__main__":
    main()
