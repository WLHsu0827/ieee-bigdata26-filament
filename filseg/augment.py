"""Data augmentation for (image, soft target[, weight map]) samples, implemented with OpenCV.

Solar H-alpha full-disk images have no preferred orientation for the purpose
of segmentation, so we use flips, arbitrary rotations, mild scaling and
photometric jitter that mimics seeing / exposure changes between GONG sites.

All geometric functions take a list of arrays of equal height and width
(``[image, target, ...]``, any dtype) and apply the *same* transform to each.
They work on uint8 arrays so that large (2048 px) images are never converted
to float before cropping.
"""
from __future__ import annotations

import cv2
import numpy as np


def geometric(arrays: list[np.ndarray], rng: np.random.Generator, rot: float = 180.0,
              scale: tuple[float, float] = (0.9, 1.1), shift: float = 0.03) -> list[np.ndarray]:
    """Random horizontal / vertical flips, rotation, scaling and shift (zero padding)."""
    h, w = arrays[0].shape[:2]
    if rng.random() < 0.5:
        arrays = [a[:, ::-1] for a in arrays]
    if rng.random() < 0.5:
        arrays = [a[::-1] for a in arrays]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), rng.uniform(-rot, rot), rng.uniform(*scale))
    m[:, 2] += rng.uniform(-shift, shift, size=2) * (w, h)
    return [cv2.warpAffine(np.ascontiguousarray(a), m, (w, h), flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=0) for a in arrays]


def focus_center(tgt: np.ndarray, rng: np.random.Generator, p_focus: float = 0.7) -> tuple[int, int]:
    """With probability ``p_focus`` a random filament pixel (target >= half its max), else a random pixel."""
    h, w = tgt.shape[:2]
    if rng.random() < p_focus:
        mx = float(tgt.max())
        if mx > 0:
            ys, xs = np.nonzero(tgt >= 0.5 * mx)
            k = rng.integers(len(ys))
            return int(ys[k]), int(xs[k])
    return int(rng.integers(h)), int(rng.integers(w))


def window(a: np.ndarray, cy: int, cx: int, size: int) -> np.ndarray:
    """``size`` x ``size`` window centred at (cy, cx); zero-padded where it leaves the array."""
    h, w = a.shape[:2]
    y0, x0 = cy - size // 2, cx - size // 2
    out = np.zeros((size, size) + a.shape[2:], a.dtype)
    ys, ye, xs, xe = max(y0, 0), min(y0 + size, h), max(x0, 0), min(x0 + size, w)
    if ys < ye and xs < xe:
        out[ys - y0:ye - y0, xs - x0:xe - x0] = a[ys:ye, xs:xe]
    return out


def crop_geometric(arrays: list[np.ndarray], size: int, rng: np.random.Generator, focus: int = 1,
                   jitter: float = 0.25, src_factor: float = 1.6, p_focus: float = 0.7,
                   **geo) -> list[np.ndarray]:
    """Random ``size`` x ``size`` crop with the full :func:`geometric` augmentation.

    A window of ``src_factor * size`` px (large enough to cover any rotation at
    scale <= 1.1) is cut around a (usually filament-centred) point of
    ``arrays[focus]``, transformed, and centre-cropped to ``size``.
    """
    cy, cx = focus_center(arrays[focus], rng, p_focus)
    j = int(size * jitter)
    cy, cx = cy + int(rng.integers(-j, j + 1)), cx + int(rng.integers(-j, j + 1))
    src = int(round(size * src_factor))
    out = geometric([window(a, cy, cx, src) for a in arrays], rng, **geo)
    o = (src - size) // 2
    return [a[o:o + size, o:o + size] for a in out]


def photometric(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Gamma, contrast / brightness, blur and noise jitter; ``img`` float32 in [0, 1]."""
    if rng.random() < 0.8:
        img = np.power(np.clip(img, 0, 1), rng.uniform(0.75, 1.35))
    if rng.random() < 0.8:
        c = rng.uniform(0.8, 1.25)
        img = (img - 0.5) * c + 0.5 + rng.uniform(-0.06, 0.06)
    if rng.random() < 0.2:
        img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.4, 1.2))
    if rng.random() < 0.2:
        img = img + rng.normal(0, rng.uniform(0.005, 0.025), img.shape).astype(np.float32)
    return np.clip(img, 0, 1).astype(np.float32)
