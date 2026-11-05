"""Traditional image-processing baseline (no learning).

Filaments are elongated features that are darker than the surrounding
chromosphere. We estimate the local background with a large median filter
and score each pixel by its relative darkness inside the solar disk.
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage as ndi


def disk_mask(img: np.ndarray, level: int = 20, erode: int = 8) -> np.ndarray:
    """Boolean mask of the solar disk (largest bright component, holes filled)."""
    fg = img > level
    lab, n = ndi.label(fg)
    if n == 0:
        return fg
    sizes = ndi.sum(fg, lab, index=np.arange(1, n + 1))
    disk = ndi.binary_fill_holes(lab == (1 + int(np.argmax(sizes))))
    if erode > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * erode + 1, 2 * erode + 1))
        disk = cv2.erode(disk.astype(np.uint8), k).astype(bool)
    return disk


def relative_darkness(img: np.ndarray, bg_kernel: int = 31, scale: int = 4, smooth: float = 1.5) -> np.ndarray:
    """``(background - I) / background`` inside the disk, 0 outside.

    The background is a median filter computed on a ``scale``-times
    downsampled image (kernel ``bg_kernel`` there, i.e. ~``bg_kernel*scale`` px
    at full resolution), with off-disk pixels filled by the disk median.
    """
    disk = disk_mask(img)
    f = img.astype(np.float32)
    med = float(np.median(f[disk])) if disk.any() else float(f.mean())
    filled = np.where(disk, f, med).astype(np.float32)
    h, w = img.shape
    small = cv2.resize(filled, (w // scale, h // scale), interpolation=cv2.INTER_AREA)
    bg_small = cv2.medianBlur(np.clip(small, 0, 255).astype(np.uint8), bg_kernel).astype(np.float32)
    bg = cv2.resize(bg_small, (w, h), interpolation=cv2.INTER_LINEAR)
    if smooth > 0:
        filled = cv2.GaussianBlur(filled, (0, 0), smooth)
    dark = (bg - filled) / np.maximum(bg, 1.0)
    dark[~disk] = 0
    return dark


def pseudo_probability(img: np.ndarray, lo: float = 0.06, hi: float = 0.16, **kw) -> np.ndarray:
    """Map relative darkness linearly to [0, 1]; 0.5 corresponds to ``(lo+hi)/2``."""
    d = relative_darkness(img, **kw)
    return np.clip((d - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)
