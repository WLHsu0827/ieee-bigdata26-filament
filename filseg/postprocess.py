"""Turn a per-pixel filament probability map into individual filament instances.

Pipeline (all steps optional / parameterised so they can be tuned on
validation PQ):

1. threshold the probability map,
2. morphological closing to bridge small gaps inside a filament,
3. fill holes (annotations are filled polygons),
4. connected components (8-connectivity) = filament instances,
5. drop components that are too small or not confident enough,
6. optional merging of nearby components into one instance.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy import ndimage as ndi

from . import IMAGE_SIZE
from .metrics import Crop


def upsample(prob: np.ndarray, size: int = IMAGE_SIZE) -> np.ndarray:
    """Probability map -> float32 in [0, 1] at full resolution (bilinear).

    uint8 maps are interpreted as probabilities scaled to 0..255 (the storage format
    of ``val_probs.npz`` / ``test_probs.npz``).
    """
    p = prob.astype(np.float32)
    if prob.dtype == np.uint8:
        p /= 255.0
    if p.shape[0] == size:
        return p
    return cv2.resize(p, (size, size), interpolation=cv2.INTER_LINEAR)


@dataclass
class PostConfig:
    threshold: float = 0.5
    close_radius: int = 0          # px at full (2048) resolution; 0 disables
    fill_holes: bool = True
    min_area: int = 150            # px at full resolution
    min_mean_prob: float = 0.0     # mean probability inside the component
    min_max_prob: float = 0.0      # peak probability inside the component
    merge_distance: int = 0        # merge components closer than this (px); 0 disables
    dilate_radius: int = 0         # grow (+) each final instance; applied after CC
    mask_threshold: float = 0.0    # redraw kept instances at this probability level (0 = keep the detection outline)


def _disk(r: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def instances_from_prob(prob: np.ndarray, cfg: PostConfig = PostConfig()) -> tuple[np.ndarray, list[dict]]:
    """Probability map (H, W) in [0, 1] -> (label map int32, per-instance stats).

    Instances in the label map are disjoint, numbered 1..n.
    """
    fg = prob > cfg.threshold
    if cfg.close_radius > 0:
        fg = cv2.morphologyEx(fg.astype(np.uint8), cv2.MORPH_CLOSE, _disk(cfg.close_radius)).astype(bool)
    if cfg.fill_holes:
        fg = ndi.binary_fill_holes(fg)

    if cfg.merge_distance > 0:
        # group components whose gap is below merge_distance, but keep the
        # pixel support of the original (un-dilated) foreground
        grouped = cv2.dilate(fg.astype(np.uint8), _disk(cfg.merge_distance // 2 + 1)).astype(bool)
        glab, _ = ndi.label(grouped, structure=np.ones((3, 3)))
        lab = np.where(fg, glab, 0).astype(np.int32)
    else:
        lab, _ = ndi.label(fg, structure=np.ones((3, 3)))
        lab = lab.astype(np.int32)

    out = np.zeros_like(lab)
    stats = []
    objs = ndi.find_objects(lab)
    k = 0
    for i, sl in enumerate(objs, start=1):
        if sl is None:
            continue
        m = lab[sl] == i
        area = int(m.sum())
        if area < cfg.min_area:
            continue
        p = prob[sl][m]
        mean_p, max_p = float(p.mean()), float(p.max())
        if mean_p < cfg.min_mean_prob or max_p < cfg.min_max_prob:
            continue
        k += 1
        out[sl][m] = k
        stats.append(dict(label=k, area=area, mean_prob=mean_p, max_prob=max_p,
                          y0=sl[0].start, x0=sl[1].start, y1=sl[0].stop, x1=sl[1].stop))

    if cfg.mask_threshold > 0 and cfg.mask_threshold != cfg.threshold and k > 0:
        out = refine_extent(out, prob, cfg.mask_threshold, cfg.threshold, cfg.fill_holes)
    if cfg.dilate_radius > 0 and k > 0:
        out = grow_labels(out, cfg.dilate_radius)
    return out, stats


def refine_extent(lab: np.ndarray, prob: np.ndarray, t_mask: float, t_detect: float,
                  fill_holes: bool = True) -> np.ndarray:
    """Redraw the outline of every instance at probability level ``t_mask``.

    Detection (which components exist) and extent (where each outline lies) are
    decoupled: with ``t_mask < t_detect`` an instance grows into the connected region
    ``p > t_mask`` around it (pixels reachable from several instances go to the one
    reached along the highest probabilities: watershed on ``-p``, so instances stay
    disjoint); with ``t_mask > t_detect`` it shrinks to its pixels with ``p > t_mask``.
    Instances that vanish are dropped and labels renumbered 1..n. Per-instance
    statistics returned by :func:`instances_from_prob` refer to the detection outline.
    """
    if t_mask < t_detect:
        from skimage.segmentation import watershed

        out = watershed(-prob, markers=lab, mask=(prob > t_mask) | (lab > 0)).astype(np.int32)
    else:
        out = np.where(prob > t_mask, lab, 0).astype(np.int32)
    if fill_holes:
        for i, sl in enumerate(ndi.find_objects(out), start=1):
            if sl is None:
                continue
            filled = ndi.binary_fill_holes(out[sl] == i)
            out[sl][filled & (out[sl] == 0)] = i
    ids = np.unique(out)
    ids = ids[ids > 0]
    if len(ids) and ids[-1] != len(ids):
        remap = np.zeros(int(ids[-1]) + 1, np.int32)
        remap[ids] = np.arange(1, len(ids) + 1, dtype=np.int32)
        out = remap[out]
    return out


def grow_labels(lab: np.ndarray, radius: int) -> np.ndarray:
    """Dilate every instance by ``radius`` px without letting instances overlap."""
    dist, (iy, ix) = ndi.distance_transform_edt(lab == 0, return_indices=True)
    grown = lab[iy, ix]
    grown[dist > radius] = 0
    return grown.astype(np.int32)


def label_to_crops(lab: np.ndarray) -> list[Crop]:
    crops = []
    for i, sl in enumerate(ndi.find_objects(lab), start=1):
        if sl is None:
            continue
        crops.append(Crop(sl[0].start, sl[1].start, lab[sl] == i))
    return crops


def label_to_masks(lab: np.ndarray):
    """Yield full-size boolean masks, one per instance (for RLE encoding)."""
    for i, sl in enumerate(ndi.find_objects(lab), start=1):
        if sl is None:
            continue
        m = np.zeros(lab.shape, bool)
        m[sl] = lab[sl] == i
        yield m
