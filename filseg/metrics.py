"""Panoptic Quality exactly as defined by the competition's self-evaluation notebook.

Official procedure (``get_overlap_df`` + ``get_pq_score``):

* Every ground-truth *annotator-image entry* (``<annotator>-<image_id>``) is
  scored separately against all predictions of ``image_id``.
* A (GT, prediction) pair is a true positive when ``IoU > 0.5`` (float32).
* A prediction is a false positive if it hits no GT of the entry, a GT is a
  false negative if no prediction hits it.
* ``PQ = sum(IoU of TP pairs) / (|TP| + 0.5 |FP| + 0.5 |FN|)`` pooled over all
  entries of the data set (not averaged per image).

Masks are handled as bounding-box crops so that scoring the full training
set takes seconds instead of materialising 2048 x 2048 stacks.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import rle as rle_util

IOU_THRESHOLD = 0.5


@dataclass
class Crop:
    """A binary mask stored as its tight bounding-box crop."""

    y0: int
    x0: int
    mask: np.ndarray  # bool (h, w)

    @property
    def area(self) -> int:
        """Number of mask pixels."""
        return int(self.mask.sum())

    @staticmethod
    def from_full(mask: np.ndarray) -> "Crop":
        """Crop a full-size boolean mask to its bounding box (an empty mask gives a 0 x 0 crop)."""
        ys = np.flatnonzero(mask.any(axis=1))
        xs = np.flatnonzero(mask.any(axis=0))
        if ys.size == 0:
            return Crop(0, 0, np.zeros((0, 0), bool))
        y0, y1, x0, x1 = ys[0], ys[-1] + 1, xs[0], xs[-1] + 1
        return Crop(int(y0), int(x0), mask[y0:y1, x0:x1].copy())

    @staticmethod
    def from_rle(counts: str) -> "Crop":
        """Decode a compressed COCO RLE counts string of a 2048 x 2048 mask into a crop."""
        return Crop.from_full(rle_util.decode(counts))


def _intersection(a: Crop, b: Crop) -> int:
    ay1, ax1 = a.y0 + a.mask.shape[0], a.x0 + a.mask.shape[1]
    by1, bx1 = b.y0 + b.mask.shape[0], b.x0 + b.mask.shape[1]
    y0, y1 = max(a.y0, b.y0), min(ay1, by1)
    x0, x1 = max(a.x0, b.x0), min(ax1, bx1)
    if y0 >= y1 or x0 >= x1:
        return 0
    sa = a.mask[y0 - a.y0:y1 - a.y0, x0 - a.x0:x1 - a.x0]
    sb = b.mask[y0 - b.y0:y1 - b.y0, x0 - b.x0:x1 - b.x0]
    return int(np.count_nonzero(sa & sb))


def overlap_matrices(gts: list[Crop], preds: list[Crop]) -> tuple[np.ndarray, np.ndarray]:
    """IoU and Dice matrices of shape (n_gt, n_pred), float32 as in the official code."""
    n_gt, n_pred = len(gts), len(preds)
    inter = np.zeros((n_gt, n_pred), np.float32)
    for i, g in enumerate(gts):
        for j, p in enumerate(preds):
            inter[i, j] = _intersection(g, p)
    ga = np.array([g.area for g in gts], np.float32).reshape(-1, 1)
    pa = np.array([p.area for p in preds], np.float32).reshape(1, -1)
    union = ga + pa - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = np.where(union == 0, np.float32(0), inter / union).astype(np.float32)
        dice = np.where(union == 0, np.float32(0), 2 * inter / (ga + pa)).astype(np.float32)
    return iou, dice


@dataclass
class PQResult:
    """Pooled Panoptic Quality and its counts: ``pq = sum_iou / (tp + fp / 2 + fn / 2)``."""

    pq: float
    tp: int
    fp: int
    fn: int
    sum_iou: float

    @property
    def sq(self) -> float:
        """Segmentation quality: mean IoU of matched pairs."""
        return self.sum_iou / self.tp if self.tp else 0.0

    @property
    def rq(self) -> float:
        """Recognition quality (F1-like): TP / (TP + FP/2 + FN/2)."""
        d = self.tp + 0.5 * self.fp + 0.5 * self.fn
        return self.tp / d if d else 0.0

    def __str__(self) -> str:
        return (f"PQ={self.pq:.4f} SQ={self.sq:.4f} RQ={self.rq:.4f} "
                f"TP={self.tp} FP={self.fp} FN={self.fn}")


def entry_counts(iou: np.ndarray, n_gt: int, n_pred: int) -> tuple[list[float], int, int]:
    """TP IoUs, FP count, FN count for one entry (official ``get_pq_score`` loop body)."""
    if n_gt == 0:
        return [], n_pred, 0
    if n_pred == 0:
        return [], 0, n_gt
    hit = iou > IOU_THRESHOLD
    tp = iou[hit].tolist()
    fp = int((hit.sum(axis=0) == 0).sum())
    fn = int((hit.sum(axis=1) == 0).sum())
    return tp, fp, fn


def pq_from_overlaps(rows) -> PQResult:
    """``rows`` iterates over ``(iou_matrix, n_gt, n_pred)`` per entry."""
    tp_scores: list[float] = []
    fp = fn = 0
    for iou, n_gt, n_pred in rows:
        t, f_p, f_n = entry_counts(iou, n_gt, n_pred)
        tp_scores.extend(t)
        fp += f_p
        fn += f_n
    tp = len(tp_scores)
    denom = tp + 0.5 * fp + 0.5 * fn
    s = float(sum(tp_scores))
    return PQResult(s / denom if denom > 0 else 0.0, tp, fp, fn, s)


def evaluate(gt: dict[str, list[Crop]], preds: dict[str, list[Crop]]) -> tuple[PQResult, pd.DataFrame]:
    """Score predictions against ground truth.

    Args:
        gt: ``entry_id -> GT crops`` (entry_id = ``<annotator>-<image_id>``).
        preds: ``image_id -> predicted crops``. Images without predictions count
            all their GT as false negatives.

    Returns:
        Pooled :class:`PQResult` and a per-entry table with the IoU/Dice matrices.
    """
    records = []
    for entry_id, gts in gt.items():
        image_id = entry_id.split("-", maxsplit=1)[1]
        ps = preds.get(image_id, [])
        iou, dice = overlap_matrices(gts, ps)
        t, f_p, f_n = entry_counts(iou, len(gts), len(ps))
        records.append(dict(entry_id=entry_id, image_id=image_id, n_gt=len(gts), n_pred=len(ps),
                            tp=len(t), fp=f_p, fn=f_n, sum_iou=float(sum(t)),
                            iou_matrix=iou, dice_matrix=dice))
    df = pd.DataFrame.from_records(records)
    res = pq_from_overlaps(zip(df["iou_matrix"], df["n_gt"], df["n_pred"]))
    return res, df


def frame_to_crops(df: pd.DataFrame) -> dict[str, list[Crop]]:
    """Official-format frame (``filament_id``, ``segmentation_rle``) -> ``key -> crops``.

    The key is the part of ``filament_id`` before the first ``_``; that is the
    entry id for GT frames and the image id for prediction frames.
    """
    out: dict[str, list[Crop]] = {}
    keys = df["filament_id"].str.split("_", n=1).str[0]
    for key, counts in zip(keys, df["segmentation_rle"]):
        out.setdefault(key, []).append(Crop.from_rle(counts))
    return out


def evaluate_frames(gt_df: pd.DataFrame, pred_df: pd.DataFrame) -> tuple[PQResult, pd.DataFrame]:
    """:func:`evaluate` on two official-format frames (``filament_id``, ``segmentation_rle``): the GT
    frame keyed by entry id, the prediction frame by image id (see :func:`frame_to_crops`)."""
    return evaluate(frame_to_crops(gt_df), frame_to_crops(pred_df))


def matching_degrees(overlap_df: pd.DataFrame) -> tuple[Counter, Counter]:
    """One-to-many / many-to-one statistics (IoU > 0), as in ``plot_m2n_counts``.

    Returns ``(gt_degree_counts, pred_degree_counts)``: how many GTs overlap
    0, 1, 2, ... predictions and vice versa.
    """
    g, p = Counter(), Counter()
    for iou, n_gt, n_pred in zip(overlap_df["iou_matrix"], overlap_df["n_gt"], overlap_df["n_pred"]):
        if n_gt == 0:
            p[0] += n_pred
            continue
        if n_pred == 0:
            g[0] += n_gt
            continue
        hit = iou > 0
        g.update(hit.sum(axis=1).tolist())
        p.update(hit.sum(axis=0).tolist())
    return g, p
