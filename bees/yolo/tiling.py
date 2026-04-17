"""
Shared tiling utilities for large-image training export and tiled inference.

Grid uses a fixed tile size, fractional overlap, and a final anchor at the far
edge so the image is fully covered without variable-size crops.
"""

from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np


def tile_axis_origins(axis_length: int, tile_size: int, overlap: float) -> List[int]:
    """
    Return start indices along one axis so tiles cover [0, axis_length].

    The last origin is axis_length - tile_size when axis_length > tile_size,
    matching the edge-aligned grid used for spore tiling.
    """
    if axis_length <= tile_size:
        return [0]
    stride = max(1, int(tile_size * (1.0 - overlap)))
    origins = list(range(0, axis_length - tile_size + 1, stride))
    last = axis_length - tile_size
    if origins[-1] != last:
        origins.append(last)
    return origins


def crop_tile_to_square(
    image_bgr: np.ndarray,
    y1: int,
    x1: int,
    tile_size: int,
    border_mode: int = cv2.BORDER_REFLECT_101,
) -> Tuple[np.ndarray, int, int]:
    """
    Crop a tile and pad to tile_size x tile_size.

    Returns:
        Padded BGR image, crop height, crop width (unpadded region in tile coords).
    """
    h, w = image_bgr.shape[:2]
    y2 = min(y1 + tile_size, h)
    x2 = min(x1 + tile_size, w)
    crop = image_bgr[y1:y2, x1:x2]
    ch, cw = crop.shape[:2]
    pad_bottom = tile_size - ch
    pad_right = tile_size - cw
    if pad_bottom == 0 and pad_right == 0:
        return crop, ch, cw
    padded = cv2.copyMakeBorder(crop, 0, pad_bottom, 0, pad_right, border_mode)
    return padded, ch, cw


def apply_clahe_bgr(image_bgr: np.ndarray, clip_limit: float = 3.0, grid_size: int = 8) -> np.ndarray:
    """Apply CLAHE to the L channel in LAB (helps faint spores on bright backgrounds)."""
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid_size, grid_size))
    l_eq = clahe.apply(l_channel)
    merged = cv2.merge((l_eq, a_channel, b_channel))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
