"""Submission file creation and validation.

The competition expects one row per predicted filament with columns
``filament_id`` (``<image_id>_<k>``) and ``segmentation_rle`` (compressed COCO
RLE counts of a 2048 x 2048 mask, no quotes).
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from . import IMAGE_SIZE, rle

COLUMNS = ["filament_id", "segmentation_rle"]


def rows_from_label(image_id: str, lab: np.ndarray) -> list[tuple[str, str]]:
    """Label map (0 = background, 1..n = instances) -> submission rows."""
    from .postprocess import label_to_masks

    return [(f"{image_id}_{k}", rle.encode(m)) for k, m in enumerate(label_to_masks(lab), start=1)]


def check(df: pd.DataFrame, test_ids: set[str] | None = None) -> dict:
    """Validate a submission frame; ``report["problems"]`` is empty when it is well-formed.

    Checks the header, unique ids, quote-free RLE strings, image ids against the
    test set, non-empty masks, and that predictions of one image do not overlap
    (our instances are disjoint by construction; duplicated or overlapping masks
    would distort Panoptic Quality and are rejected).
    """
    problems: list[str] = []
    if list(df.columns) != COLUMNS:
        problems.append(f"unexpected columns {list(df.columns)}")
        return {"problems": problems}
    if df["filament_id"].duplicated().any():
        problems.append(f"{int(df['filament_id'].duplicated().sum())} duplicated filament_id")
    if df["segmentation_rle"].astype(str).str.contains("[\"']", regex=True).any():
        problems.append("RLE strings contain quotes")
    image_ids = df["filament_id"].str.rsplit("_", n=1).str[0]
    missing = None
    if test_ids is not None:
        unknown = set(image_ids) - test_ids
        if unknown:
            problems.append(f"{len(unknown)} image ids not in the test set, e.g. {sorted(unknown)[:3]}")
        missing = len(test_ids - set(image_ids))

    by_image = defaultdict(list)
    for iid, counts in zip(image_ids, df["segmentation_rle"]):
        by_image[iid].append(counts)
    n_empty = n_overlap = 0
    areas = []
    for rles in by_image.values():
        cover = np.zeros((IMAGE_SIZE, IMAGE_SIZE), np.uint8)
        for counts in rles:
            m = rle.decode(counts)
            areas.append(int(m.sum()))
            n_empty += areas[-1] == 0
            cover += m
        n_overlap += int((cover > 1).any())
    if n_empty:
        problems.append(f"{n_empty} empty masks")
    if n_overlap:
        problems.append(f"{n_overlap} images with overlapping predicted masks")
    a = np.array(areas) if areas else np.zeros(1)
    return {"rows": len(df), "images": len(by_image), "test_images_without_prediction": missing,
            "filaments_per_image": round(len(df) / max(len(by_image), 1), 2),
            "area_p10_p50_p90": [int(np.percentile(a, q)) for q in (10, 50, 90)],
            "problems": problems}


def write(rows: list[tuple[str, str]], path: Path, test_ids: set[str] | None = None) -> tuple[pd.DataFrame, dict]:
    """Write rows to ``path`` after validating them; raises if the file would be malformed."""
    df = pd.DataFrame(rows, columns=COLUMNS)
    report = check(df, test_ids)
    if report["problems"]:
        raise ValueError(f"invalid submission: {report['problems']}")
    df.to_csv(path, index=False)
    return df, report
