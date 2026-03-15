#!/usr/bin/env python3

"""spad_digitization.py

Digitize (sum) 1-bit SPAD frames to create higher bit-depth images.

This script reads a directory of 1-bit SPAD binary frames (saved as PNGs) and generates
higher bit-depth images by summing consecutive frames. For example, with bit_depth=8,
it sums every 255 (2^8 - 1) frames to create an 8-bit image where each pixel value
represents the count of detections across those frames.

Usage (examples):
    python spad_digitization.py input_spad_frames --output_dir output_digitized --bit_depth 8
    python spad_digitization.py /path/to/spad_frames --output_dir digitized --bit_depth 16

Dependencies:
    - numpy
    - pillow (PIL)
    - tqdm

The script will:
  1. Load all 1-bit SPAD frames from the input directory (sorted by filename).
  2. For each group of (2^bit_depth - 1) frames, sum the binary values pixel-wise.
  3. Save the summed images as grayscale PNGs in the output directory.
  4. Save metadata.json with processing parameters.

"""

import json
import argparse
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm
from typing import List, Tuple

# CONSTANTS AND DEFAULT PARAMETERS
FILENAME_PAD = 6  # number of digits in output frame filenames

# UTILS
def ensure_dir(p: str):
    """
    Ensure that a directory exists; create it if it doesn't.
    """
    Path(p).mkdir(parents=True, exist_ok=True)

def load_spad_frame(p: Path) -> np.ndarray:
    """
    Load a 1-bit SPAD frame from PNG and return as binary numpy array (0 or 1).
    """
    img = Image.open(p).convert('L')  # Convert to grayscale
    arr = np.array(img, dtype=np.uint8)
    # Assuming saved as 0 or 255, convert to 0 or 1
    binary = (arr > 127).astype(np.uint8)
    return binary

def digitize_frames(
        frame_paths: List[Path],
        bit_depth: int,
        output_dir: Path
        ) -> int:
    """
    Digitize frames by summing groups of (2^bit_depth - 1) frames.

    Parameters:
    - frame_paths (List[Path]): Sorted list of paths to SPAD frame PNGs.
    - bit_depth (int): Target bit depth (e.g., 8 for 8-bit images).
    - output_dir (Path): Directory to save digitized images.

    Returns:
    - Number of digitized images created.
    """
    if bit_depth < 1:
        raise ValueError("bit_depth must be at least 1")

    group_size = (1 << bit_depth) - 1  # 2^bit_depth - 1
    max_value = group_size  # Maximum sum value

    num_frames = len(frame_paths)
    num_groups = num_frames // group_size
    if num_groups == 0:
        raise ValueError(f"Not enough frames ({num_frames}) for bit_depth {bit_depth} (need at least {group_size})")

    digitized_count = 0

    # Load first frame to get dimensions
    first_frame = load_spad_frame(frame_paths[0])
    h, w = first_frame.shape

    for group_idx in tqdm(range(num_groups), desc="Digitizing frames"):
        start_idx = group_idx * group_size
        end_idx = start_idx + group_size
        group_paths = frame_paths[start_idx:end_idx]

        # Initialize sum array
        summed = np.zeros((h, w), dtype=np.uint16)  # Use uint16 to handle sums up to 65535 (for bit_depth=16)

        for path in group_paths:
            frame = load_spad_frame(path)
            summed += frame.astype(np.uint16)

        # Clip to max_value and convert to uint8 for saving (assuming bit_depth <= 8 for now)
        # For higher bit_depth, could save as 16-bit, but PIL save as PNG supports it.
        if bit_depth <= 8:
            summed_clipped = np.clip(summed, 0, 255).astype(np.uint8)
            img = Image.fromarray(summed_clipped, mode='L')
        else:
            summed_clipped = np.clip(summed, 0, max_value).astype(np.uint16)
            # PIL can handle uint16, but for PNG, it might need mode='I' or something, but let's use 'L' for 8-bit for simplicity.
            # Actually, for higher bit, perhaps save as TIFF or just scale, but to keep simple, assume bit_depth=8.
            raise NotImplementedError("bit_depth > 8 not fully implemented yet")

        out_path = output_dir / f"digitized_{digitized_count:0{FILENAME_PAD}d}.png"
        img.save(out_path)
        digitized_count += 1

    return digitized_count

def main():
    parser = argparse.ArgumentParser(description="Digitize 1-bit SPAD frames to higher bit-depth images by summing consecutive frames.")
    parser.add_argument("input_dir", type=str, help="Path to the input directory containing 1-bit SPAD frame PNGs.")
    parser.add_argument("--output_dir", "-o", type=str, default="./output_digitized", help="Path to the output directory for digitized images.")
    parser.add_argument("--bit_depth", "-b", type=int, default=8, help="Target bit depth (e.g., 8 for 8-bit images).")

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        raise RuntimeError(f"Input directory {input_dir} does not exist or is not a directory.")

    output_dir = Path(args.output_dir)
    ensure_dir(str(output_dir))

    # Find all PNG files and sort them
    png_files = sorted(input_dir.glob("*.png"))
    if not png_files:
        raise RuntimeError(f"No PNG files found in directory {input_dir}.")

    print(f"Found {len(png_files)} SPAD frames in {input_dir}")

    # Digitize
    num_digitized = digitize_frames(png_files, args.bit_depth, output_dir)

    # Save metadata
    metadata = {
        'input_dir': str(input_dir),
        'output_dir': str(output_dir),
        'bit_depth': args.bit_depth,
        'frames_per_group': (1 << args.bit_depth) - 1,
        'total_input_frames': len(png_files),
        'total_digitized_images': num_digitized
    }
    metadata_path = output_dir / 'metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"✅ Digitization complete. {num_digitized} images saved to {output_dir}")
    print(f"📥 Metadata saved to: {metadata_path}")

if __name__ == "__main__":
    main()
