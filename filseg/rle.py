"""COCO RLE helpers matching the competition submission format.

The submission stores only the compressed COCO RLE *counts* string of each
mask; the size is fixed to 2048 x 2048 for every image.
"""
from __future__ import annotations

import numpy as np
from pycocotools import mask as mask_util

from . import IMAGE_SIZE


def encode(mask: np.ndarray) -> str:
    """Binary (H, W) mask -> compressed COCO RLE counts string."""
    rle = mask_util.encode(np.asfortranarray(mask.astype(np.uint8)))
    counts = rle["counts"]
    return counts.decode("ascii") if isinstance(counts, bytes) else counts


def decode(counts: str, height: int = IMAGE_SIZE, width: int = IMAGE_SIZE) -> np.ndarray:
    """Compressed COCO RLE counts string -> boolean (H, W) mask."""
    rle = {"size": [height, width], "counts": counts.encode("ascii")}
    return mask_util.decode(rle).astype(bool)


def polygon_to_mask(polygon: list[float], height: int = IMAGE_SIZE, width: int = IMAGE_SIZE) -> np.ndarray:
    """Rasterise one COCO polygon ``[x0, y0, x1, y1, ...]`` exactly as pycocotools does."""
    rles = mask_util.frPyObjects([polygon], height, width)
    return mask_util.decode(mask_util.merge(rles)).astype(bool)


def area(counts: str, height: int = IMAGE_SIZE, width: int = IMAGE_SIZE) -> int:
    return int(mask_util.area({"size": [height, width], "counts": counts.encode("ascii")}))
