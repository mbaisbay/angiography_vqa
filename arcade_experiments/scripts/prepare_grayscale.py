"""Convert single-channel grayscale PNGs to 3-channel for COCO pretrained weights.

ARCADE images are 512x512 8-bit grayscale (color_type=0 in PNG header).
YOLO models with COCO pretrained weights expect 3-channel input.
This script replicates the single channel to create 3-channel images.
"""

import argparse
import struct
from pathlib import Path
import shutil


def is_grayscale_png(png_path: Path) -> bool:
    """Check if a PNG file is grayscale by reading the IHDR chunk."""
    with open(png_path, "rb") as f:
        f.read(8)   # PNG signature
        f.read(4)   # IHDR length
        f.read(4)   # IHDR chunk type
        f.read(4)   # width
        f.read(4)   # height
        f.read(1)   # bit depth
        color_type = struct.unpack("B", f.read(1))[0]
    # color_type 0 = Grayscale, 4 = Grayscale+Alpha
    return color_type in (0, 4)


def convert_grayscale_to_rgb(input_path: Path, output_path: Path) -> bool:
    """Convert a grayscale PNG to 3-channel RGB by replicating the channel.

    Uses cv2 if available, falls back to numpy+PNG manual approach.
    Returns True if conversion was performed, False if already RGB.
    """
    try:
        import cv2
        img = cv2.imread(str(input_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise RuntimeError(f"Failed to read: {input_path}")
        if len(img.shape) == 2:
            # Single channel -> replicate to 3 channels
            img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            cv2.imwrite(str(output_path), img_rgb)
            return True
        elif img.shape[2] == 1:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            cv2.imwrite(str(output_path), img_rgb)
            return True
        else:
            # Already multi-channel, just copy
            shutil.copy2(input_path, output_path)
            return False
    except ImportError:
        # Fallback: use numpy + imageio or PIL
        try:
            from PIL import Image
            import numpy as np
            img = Image.open(input_path)
            if img.mode in ("L", "I", "F"):
                arr = np.array(img)
                arr_rgb = np.stack([arr, arr, arr], axis=-1)
                Image.fromarray(arr_rgb.astype(np.uint8)).save(output_path)
                return True
            else:
                shutil.copy2(input_path, output_path)
                return False
        except ImportError:
            raise RuntimeError(
                "Neither cv2 nor PIL available. Install opencv-python or Pillow."
            )


def process_directory(input_dir: Path, output_dir: Path,
                      force: bool = False) -> dict:
    """Convert all grayscale PNGs in a directory to 3-channel.

    Args:
        input_dir: Source image directory.
        output_dir: Destination directory for converted images.
        force: If True, reconvert even if output exists.

    Returns:
        Stats dict.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    image_files = sorted(input_dir.glob("*.png")) + sorted(input_dir.glob("*.PNG"))
    stats = {"total": 0, "converted": 0, "skipped": 0, "already_rgb": 0}

    for img_path in image_files:
        stats["total"] += 1
        out_path = output_dir / img_path.name

        if out_path.exists() and not force:
            stats["skipped"] += 1
            continue

        if is_grayscale_png(img_path):
            converted = convert_grayscale_to_rgb(img_path, out_path)
            if converted:
                stats["converted"] += 1
            else:
                stats["already_rgb"] += 1
        else:
            shutil.copy2(img_path, out_path)
            stats["already_rgb"] += 1

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Convert grayscale PNGs to 3-channel for YOLO compatibility"
    )
    parser.add_argument(
        "--input-dir", type=str, required=True,
        help="Input directory containing grayscale PNGs"
    )
    parser.add_argument(
        "--output-dir", type=str, required=True,
        help="Output directory for 3-channel PNGs"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Reconvert even if output files exist"
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    print(f"Converting grayscale -> 3-channel RGB")
    print(f"  Input:  {input_dir}")
    print(f"  Output: {output_dir}")

    stats = process_directory(input_dir, output_dir, force=args.force)

    print(f"\nResults:")
    print(f"  Total images:  {stats['total']}")
    print(f"  Converted:     {stats['converted']}")
    print(f"  Already RGB:   {stats['already_rgb']}")
    print(f"  Skipped:       {stats['skipped']}")


if __name__ == "__main__":
    main()
