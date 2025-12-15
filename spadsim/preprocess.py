# preprocess.py

import numpy as np
import cv2
from PIL import Image
from pathlib import Path
from typing import Optional

def quantize_image_bits(img_rgb: np.ndarray, bits: int) -> np.ndarray:
    """
    Uniform quantization of N bits per channel.
    """
    if bits >= 8:
        return img_rgb

    levels = 2 ** bits
    img = img_rgb.astype(np.float32) / 255.0
    img_q = np.floor(img * (levels - 1)) / (levels - 1)
    return (img_q * 255).astype(np.uint8)


def resize_image(img_rgb: np.ndarray, size_str: str) -> np.ndarray:
    """
    Resizes the image to WxH, e.g. '256x256'
    """
    try:
        w, h = [int(x) for x in size_str.lower().split('x')]
    except:
        raise ValueError("Resize format must be WxH (e.g., 320x240)")

    return cv2.resize(img_rgb, (w, h), interpolation=cv2.INTER_AREA)


def save_image(img: np.ndarray, out_path: Path):
    Image.fromarray(img).save(out_path)

def apply_degradation(
    image: np.ndarray,
    model: object = None
) -> np.ndarray:
    """
    Apply a degradation model if provided.
    """
    if model is None:
        return image

    return model.apply(image)