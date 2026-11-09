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
    """Count, match rate and mean best IoU per area bin (px at 2048) of a :func:`gt_table` or
    :func:`pred_table`."""
    t = table.assign(size=pd.cut(table["area"], bins, right=False))
    g = t.groupby("size", observed=True)
    return pd.DataFrame({"n": g.size(), "match_rate": g["matched"].mean(),
                         "mean_best_iou": g["best_iou"].mean()}).reset_index()


def by_size_bootstrap(table: pd.DataFrame, bins=AREA_BINS, draws: int = 2000, seed: int = 0) -> pd.DataFrame:
    """Match rate per area bin of a :func:`gt_table` with a 95 % bootstrap interval over images.

    Whole images are resampled with replacement, so the filaments of one image (which share its
    observing conditions and annotators) stay together. Bins without filaments are omitted, as in
    :func:`by_size`.
    """
    codes = pd.cut(table["area"], bins, right=False, labels=False)
    labels = pd.cut(table["area"], bins, right=False).cat.categories
    images, img_idx = np.unique(table["image_id"].to_numpy(), return_inverse=True)
    counts = np.zeros((len(images), len(bins) - 1, 2))
    ok = codes.notna().to_numpy()
    rows, cols = img_idx[ok], codes[ok].astype(int).to_numpy()
    np.add.at(counts, (rows, cols, 0), table["matched"].to_numpy(float)[ok])
    np.add.at(counts, (rows, cols, 1), 1.0)
    rng = np.random.default_rng(seed)
    boot = np.empty((draws, len(bins) - 1))
    for d in range(draws):
        s = counts[rng.integers(0, len(images), len(images))].sum(0)
        boot[d] = s[:, 0] / np.maximum(s[:, 1], 1)
    total = counts.sum(0)
    keep = total[:, 1] > 0
    lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
    return pd.DataFrame({"size": labels[keep], "n": total[keep, 1].astype(int),
                         "match_rate": total[keep, 0] / total[keep, 1], "ci_low": lo[keep], "ci_high": hi[keep]})


def pq_bootstrap(overlap_df: pd.DataFrame, draws: int = 2000, seed: int = 0) -> tuple[float, float]:
    """95 % bootstrap interval of the pooled PQ over images (per-entry table of ``metrics.evaluate``)."""
    per_image = overlap_df.groupby("image_id")[["sum_iou", "tp", "fp", "fn"]].sum().to_numpy(float)
    rng = np.random.default_rng(seed)
    values = np.empty(draws)
    for d in range(draws):
        s = per_image[rng.integers(0, len(per_image), len(per_image))].sum(0)
        den = s[1] + 0.5 * s[2] + 0.5 * s[3]
        values[d] = s[0] / den if den else 0.0
    lo, hi = np.percentile(values, [2.5, 97.5])
    return float(lo), float(hi)


def overlap_distributions(overlap_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """IoU and Dice values of all overlapping (score > 0) GT/prediction pairs."""
    iou = np.concatenate([np.asarray(m).ravel() for m in overlap_df["iou_matrix"]] or [np.zeros(0)])
    dice = np.concatenate([np.asarray(m).ravel() for m in overlap_df["dice_matrix"]] or [np.zeros(0)])
    return iou[iou > 0], dice[dice > 0]


def summary(gt: dict[str, list[Crop]], preds: dict[str, list[Crop]], overlap_df: pd.DataFrame) -> dict:
    """Validation summary: GT and prediction tables by size, one-to-many / many-to-one matching
    degrees and the mean IoU / Dice of overlapping pairs."""
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


GT_CAUSES = ["tp", "missed", "fragmented", "merged", "too_large", "partial"]
PRED_CAUSES = ["tp", "spurious", "overlapping"]


def error_taxonomy(gt: dict[str, list[Crop]], preds: dict[str, list[Crop]], overlap_df: pd.DataFrame) -> dict:
    """Why GT filaments are missed and predictions are false positives (per annotator entry, pooled).

    Intersections follow exactly from IoU and areas (``I = IoU (|g| + |p|) / (1 + IoU)``), so
    no masks are needed. Every GT filament without a match (IoU > 0.5) gets one cause:

    * ``missed``: no prediction overlaps it;
    * ``fragmented``: >= 2 predictions lie mostly (>= 50 % of their area) inside it and their
      union would match (IoU > 0.5) -- fixable by grouping fragments;
    * ``merged``: its best-overlapping prediction covers >= 50 % of it but also overlaps another
      GT filament of the same annotator -- fixable by splitting;
    * ``too_large``: the best prediction covers >= 50 % of it but is too large (no other GT);
    * ``partial``: the best prediction covers < 50 % of it.

    Predictions without a match are ``spurious`` (no overlap with any GT of that annotator) or
    ``overlapping``. ``upper_bounds`` gives the pooled PQ if all fragmented filaments were
    grouped (fragments replaced by their union) or all spurious predictions removed.
    """
    gt_count = dict.fromkeys(GT_CAUSES, 0)
    pred_count = dict.fromkeys(PRED_CAUSES, 0)
    s_tot = tp_tot = fp_tot = fn_tot = 0.0
    frag_gain_s = frag_tp = frag_fp_removed = 0.0
    for r in overlap_df.itertuples(index=False):
        s_tot += r.sum_iou
        tp_tot += r.tp
        fp_tot += r.fp
        fn_tot += r.fn
        if r.n_gt == 0:
            pred_count["spurious"] += r.n_pred
            continue
        if r.n_pred == 0:
            gt_count["missed"] += r.n_gt
            continue
        iou = np.asarray(r.iou_matrix, np.float64)
        ga = np.array([c.area for c in gt[r.entry_id]], np.float64)[:, None]
        pa = np.array([c.area for c in preds[r.image_id]], np.float64)[None, :]
        inter = iou * (ga + pa) / (1.0 + iou)
        hit = iou > IOU_THRESHOLD
        for i in range(r.n_gt):
            if hit[i].any():
                gt_count["tp"] += 1
                continue
            if not (inter[i] > 0).any():
                gt_count["missed"] += 1
                continue
            inside = (inter[i] > 0) & (inter[i] >= 0.5 * pa[0]) & ~hit.any(axis=0)
            if inside.sum() >= 2:
                si = inter[i, inside].sum()
                u = si / (ga[i, 0] + pa[0, inside].sum() - si)
                if u > IOU_THRESHOLD:
                    gt_count["fragmented"] += 1
                    frag_gain_s += u
                    frag_tp += 1
                    frag_fp_removed += int(inside.sum())
                    continue
            j = int(np.argmax(iou[i]))
            if inter[i, j] >= 0.5 * ga[i, 0]:
                others = (inter[:, j] > 0)
                others[i] = False
                gt_count["merged" if others.any() else "too_large"] += 1
            else:
                gt_count["partial"] += 1
        matched_pred = hit.any(axis=0)
        pred_count["tp"] += int(matched_pred.sum())
        overl = (inter > 0).any(axis=0)
        pred_count["spurious"] += int((~matched_pred & ~overl).sum())
        pred_count["overlapping"] += int((~matched_pred & overl).sum())

    def pq(s, tp, fp, fn):
        d = tp + 0.5 * fp + 0.5 * fn
        return float(s / d) if d else 0.0

    base = pq(s_tot, tp_tot, fp_tot, fn_tot)
    return {
        "gt": gt_count,
        "pred": pred_count,
        "pq": base,
        "upper_bounds": {
            "group_fragments": pq(s_tot + frag_gain_s, tp_tot + frag_tp, fp_tot - frag_fp_removed, fn_tot - frag_tp),
            "remove_spurious": pq(s_tot, tp_tot, fp_tot - pred_count["spurious"], fn_tot),
        },
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
