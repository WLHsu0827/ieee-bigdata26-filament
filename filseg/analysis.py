"""Error analysis of instance predictions (used for validation reports and the paper).

Given the per-entry table returned by :func:`filseg.metrics.evaluate`, computes

* recall / mean IoU of ground-truth filaments by size,
* false positives by predicted size,
* IoU and Dice distributions of overlapping pairs (as in the organisers' notebook),
* one-to-many / many-to-one matching degrees.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import IOU_THRESHOLD, Crop, matching_degrees

AREA_BINS = [0, 300, 600, 1200, 2500, 5000, 10_000, np.inf]


def gt_table(gt: dict[str, list[Crop]], overlap_df: pd.DataFrame) -> pd.DataFrame:
    """One row per GT filament: area, best IoU with any prediction, matched flag."""
    rows = []
    for r in overlap_df.itertuples(index=False):
        crops = gt[r.entry_id]
        best = r.iou_matrix.max(axis=1) if r.n_pred else np.zeros(r.n_gt)
        for c, b in zip(crops, best):
            rows.append((r.entry_id, r.image_id, c.area, float(b), bool(b > IOU_THRESHOLD)))
    return pd.DataFrame(rows, columns=["entry_id", "image_id", "area", "best_iou", "matched"])


def pred_table(preds: dict[str, list[Crop]], overlap_df: pd.DataFrame) -> pd.DataFrame:
    """One row per (entry, prediction): area, best IoU with that entry's GT, matched flag."""
    rows = []
    for r in overlap_df.itertuples(index=False):
        crops = preds.get(r.image_id, [])
        best = r.iou_matrix.max(axis=0) if r.n_gt else np.zeros(r.n_pred)
        for c, b in zip(crops, best):
            rows.append((r.entry_id, r.image_id, c.area, float(b), bool(b > IOU_THRESHOLD)))
    return pd.DataFrame(rows, columns=["entry_id", "image_id", "area", "best_iou", "matched"])


def by_size(table: pd.DataFrame, bins=AREA_BINS) -> pd.DataFrame:
    t = table.assign(size=pd.cut(table["area"], bins, right=False))
    g = t.groupby("size", observed=True)
    return pd.DataFrame({"n": g.size(), "match_rate": g["matched"].mean(),
                         "mean_best_iou": g["best_iou"].mean()}).reset_index()


def overlap_distributions(overlap_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """IoU and Dice values of all overlapping (score > 0) GT/prediction pairs."""
    iou = np.concatenate([np.asarray(m).ravel() for m in overlap_df["iou_matrix"]] or [np.zeros(0)])
    dice = np.concatenate([np.asarray(m).ravel() for m in overlap_df["dice_matrix"]] or [np.zeros(0)])
    return iou[iou > 0], dice[dice > 0]


def summary(gt: dict[str, list[Crop]], preds: dict[str, list[Crop]], overlap_df: pd.DataFrame) -> dict:
    g, p = gt_table(gt, overlap_df), pred_table(preds, overlap_df)
    gdeg, pdeg = matching_degrees(overlap_df)
    iou, dice = overlap_distributions(overlap_df)
    return {
        "gt_by_size": by_size(g),
        "pred_by_size": by_size(p),
        "gt_degrees": dict(sorted(gdeg.items())),
        "pred_degrees": dict(sorted(pdeg.items())),
        "mean_overlap_iou": float(iou.mean()) if iou.size else 0.0,
        "mean_overlap_dice": float(dice.mean()) if dice.size else 0.0,
    }


def _pq(t: pd.DataFrame) -> float:
    den = t["tp"].sum() + 0.5 * t["fp"].sum() + 0.5 * t["fn"].sum()
    return float(t["sum_iou"].sum() / den) if den else 0.0


def aggregation_breakdown(overlap_df: pd.DataFrame) -> dict:
    """PQ under different aggregations and annotator subsets (per-entry table of ``metrics.evaluate``).

    ``pooled`` is the official definition; the per-entry / per-image means show how
    sensitive the score is to the aggregation, ``by_n_annotators`` to the image mix.
    """
    t = overlap_df.copy()
    den = t["tp"] + 0.5 * t["fp"] + 0.5 * t["fn"]
    t["pq_entry"] = np.where(den > 0, t["sum_iou"] / den.where(den > 0, 1), np.nan)
    n_ann = t.groupby("image_id")["entry_id"].transform("count")
    per_image = t.groupby("image_id").apply(_pq, include_groups=False)
    return {
        "pooled": _pq(t),
        "mean_per_entry": float(np.nanmean(t["pq_entry"])),
        "mean_per_image": float(per_image.mean()),
        "by_n_annotators": {int(k): round(_pq(g), 4) for k, g in t.groupby(n_ann)},
        "n_entries_by_n_annotators": {int(k): int(len(g)) for k, g in t.groupby(n_ann)},
    }
