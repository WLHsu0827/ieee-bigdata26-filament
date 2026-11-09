"""Loading MAGFiLO (COCO-style) annotations and building ground-truth masks.

Terminology used throughout the package:

* ``image_id``  -- observation name without extension, e.g. ``20140609195854Bh``.
* ``entry_id``  -- one annotator's annotation set of an image,
  ``<annotator>-<image_id>`` (e.g. ``040301-20140609195854Bh``). An image may be
  annotated independently by 1-3 annotators. The official metric scores every
  entry separately against the same predictions of that image.
"""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from . import IMAGE_SIZE
from . import rle as rle_util

TRAIN_JSON = "MAGFiLO_1.0_Annotations_kaggle2026_train.json"


@dataclass
class Entry:
    """One annotator's annotation set of one image (``entry_id = <annotator>-<image_id>``): the COCO
    polygon and category id of every filament the annotator drew."""

    entry_id: str
    annotator: str
    image_id: str
    file_name: str
    polygons: list[list[float]] = field(default_factory=list)
    categories: list[int] = field(default_factory=list)

    def masks(self, height: int = IMAGE_SIZE, width: int = IMAGE_SIZE) -> list[np.ndarray]:
        """One boolean mask per annotated filament (pycocotools rasterisation)."""
        return [rle_util.polygon_to_mask(p, height, width) for p in self.polygons]


def _parse_segmentation(seg) -> list[float]:
    if isinstance(seg, str):
        seg = json.loads(seg)
    if len(seg) != 1:
        raise ValueError(f"expected a single polygon, got {len(seg)}")
    return [float(v) for v in seg[0]]


def load_entries(json_path: str | Path) -> list[Entry]:
    """Parse the training JSON into a list of annotator-image entries."""
    with open(json_path, encoding="utf-8") as f:
        coco = json.load(f)
    entries: dict[str, Entry] = {}
    for img in coco["images"]:
        annotator, image_id = img["id"].split("-", maxsplit=1)
        entries[img["id"]] = Entry(img["id"], annotator, image_id, img["file_name"])
    for ann in coco["annotations"]:
        e = entries[ann["image_id"]]
        e.polygons.append(_parse_segmentation(ann["segmentation"]))
        e.categories.append(int(ann["category_id"]))
    return sorted(entries.values(), key=lambda e: e.entry_id)


def group_by_image(entries: list[Entry]) -> dict[str, list[Entry]]:
    """``image_id -> entries`` (one to three annotators per image), sorted by image id."""
    out: dict[str, list[Entry]] = defaultdict(list)
    for e in entries:
        out[e.image_id].append(e)
    return dict(sorted(out.items()))


def gt_frame(entries: list[Entry]) -> pd.DataFrame:
    """Ground truth in the official evaluation format.

    ``filament_id`` is ``<entry_id>_<k>`` and ``segmentation_rle`` holds the
    compressed RLE counts, exactly as consumed by the self-evaluation notebook.
    """
    rows = []
    for e in entries:
        for k, m in enumerate(e.masks(), start=1):
            rows.append((f"{e.entry_id}_{k}", rle_util.encode(m)))
    return pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])


def read_image(path: str | Path) -> np.ndarray:
    """Read a GONG H-alpha JPEG as a single-channel uint8 array."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return img


def soft_target(entries: list[Entry], size: int = IMAGE_SIZE) -> np.ndarray:
    """Fraction of annotators that mark each pixel as filament, at ``size`` x ``size``.

    Polygons are rasterised at full resolution and area-averaged when
    ``size`` is smaller, giving a soft target in [0, 1].
    """
    acc = np.zeros((IMAGE_SIZE, IMAGE_SIZE), np.float32)
    for e in entries:
        union = np.zeros((IMAGE_SIZE, IMAGE_SIZE), bool)
        for m in e.masks():
            union |= m
        acc += union
    acc /= max(len(entries), 1)
    if size != IMAGE_SIZE:
        acc = cv2.resize(acc, (size, size), interpolation=cv2.INTER_AREA)
    return acc


def image_folds(image_ids: list[str], n_folds: int = 5, seed: int = 2026) -> dict[str, int]:
    """Deterministic random image-level fold assignment (all annotators of an image share a fold)."""
    ids = sorted(set(image_ids))
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(ids))
    return {ids[i]: int(k % n_folds) for k, i in enumerate(order)}
