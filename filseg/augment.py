"""Data augmentation for (image, soft target) pairs, implemented with OpenCV.

Solar H-alpha full-disk images have no preferred orientation for the purpose
of segmentation, so we use flips, arbitrary rotations about the image centre,
mild scaling, and photometric jitter that mimics seeing / exposure changes
between GONG sites.
"""
from __future__ import annotations

import cv2
import numpy as np


def geometric(img: np.ndarray, tgt: np.ndarray, rng: np.random.Generator,
              rot: float = 180.0, scale: tuple[float, float] = (0.9, 1.1),
              shift: float = 0.03) -> tuple[np.ndarray, np.ndarray]:
    """Random flips, rotation, scaling and shift. ``tgt`` may be (H, W) or (H, W, C<=4)."""
    h, w = img.shape
    if rng.random() < 0.5:
        img, tgt = img[:, ::-1], tgt[:, ::-1]
    if rng.random() < 0.5:
        img, tgt = img[::-1], tgt[::-1]
    angle = rng.uniform(-rot, rot)
    s = rng.uniform(*scale)
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, s)
    m[:, 2] += rng.uniform(-shift, shift, size=2) * (w, h)
    img = cv2.warpAffine(np.ascontiguousarray(img), m, (w, h), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    tgt = cv2.warpAffine(np.ascontiguousarray(tgt), m, (w, h), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return img, tgt


def photometric(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """img float32 in [0, 1]."""
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


def random_crop(img: np.ndarray, tgt: np.ndarray, size: int, rng: np.random.Generator,
                p_focus: float = 0.7) -> tuple[np.ndarray, np.ndarray]:
    """Crop ``size`` x ``size``; with prob ``p_focus`` centre it on a filament pixel.

    ``tgt`` may be (H, W) or (H, W, C); channel 0 is the filament target."""
    h, w = img.shape
    if size >= h and size >= w:
        return img, tgt
    cy, cx = None, None
    if rng.random() < p_focus:
        t0 = tgt[..., 0] if tgt.ndim == 3 else tgt
        ys, xs = np.nonzero(t0 > 0.5 * t0.max()) if t0.max() > 0 else ([], [])
        if len(ys):
            k = rng.integers(len(ys))
            cy, cx = int(ys[k]), int(xs[k])
    if cy is None:
        cy, cx = int(rng.integers(h)), int(rng.integers(w))
    y0 = int(np.clip(cy - size // 2 + rng.integers(-size // 4, size // 4 + 1), 0, h - size))
    x0 = int(np.clip(cx - size // 2 + rng.integers(-size // 4, size // 4 + 1), 0, w - size))
    return img[y0:y0 + size, x0:x0 + size], tgt[y0:y0 + size, x0:x0 + size]
