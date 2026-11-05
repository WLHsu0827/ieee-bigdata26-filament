"""Component-level evaluation and selection.

For disjoint predicted instances the official PQ decomposes over components.
Let ``E(c)`` be the annotator entries of the image of component ``c`` and
``v_c = sum_{e in E(c)} IoU_e(c) * [IoU_e(c) > 0.5]`` (a component matches at
most one GT per entry because matches need IoU > 0.5). Then for a kept set K

    PQ(K) = sum_{c in K} v_c / (0.5 * sum_{c in K} |E(c)| + 0.5 * G),

with G the total number of GT filaments. Keeping ``c`` therefore increases PQ
iff ``v_c / |E(c)| > PQ / 2``. This module extracts all candidate components
once per *shape* configuration (threshold, closing, ...), records their
features and match statistics, and evaluates any *selection* rule (area /
confidence filters or a learned score) instantly.
"""
from __future__ import annotations

import multiprocessing as mp
import os
from dataclasses import dataclass, replace

import cv2
import numpy as np
import pandas as pd
from scipy import ndimage as ndi

from .metrics import IOU_THRESHOLD, Crop, overlap_matrices
from .postprocess import PostConfig, instances_from_prob, upsample

FEATURES = ["log_area", "mean_prob", "max_prob", "p90_prob", "std_prob", "elongation", "fill_ratio",
            "radius", "n_comp_image", "area_rank"]


def component_features(prob: np.ndarray, lab: np.ndarray) -> list[dict]:
    """Shape / confidence features of every labelled component of a full-resolution map."""
    h, w = lab.shape
    feats = []
    for i, sl in enumerate(ndi.find_objects(lab), start=1):
        if sl is None:
            continue
        m = lab[sl] == i
        p = prob[sl][m]
        ys, xs = np.nonzero(m)
        area = int(m.sum())
        if area >= 3:
            cov = np.cov(np.stack([ys, xs]).astype(np.float64))
            ev = np.sort(np.linalg.eigvalsh(cov))
            elong = float(np.sqrt(max(ev[1], 1e-6) / max(ev[0], 1e-6)))
        else:
            elong = 1.0
        cy, cx = sl[0].start + ys.mean(), sl[1].start + xs.mean()
        feats.append(dict(area=area, log_area=float(np.log(area)), mean_prob=float(p.mean()),
                          max_prob=float(p.max()), p90_prob=float(np.percentile(p, 90)),
                          std_prob=float(p.std()), elongation=elong,
                          fill_ratio=area / float(m.shape[0] * m.shape[1]),
                          radius=float(np.hypot(cy - h / 2, cx - w / 2) / (h / 2)),
                          y0=sl[0].start, x0=sl[1].start))
    n = len(feats)
    order = np.argsort([-f["area"] for f in feats])
    for rank, k in enumerate(order):
        feats[k]["n_comp_image"] = n
        feats[k]["area_rank"] = rank
    return feats


def _job(args):
    image_id, prob, shape_cfg, gt_entries = args
    p = upsample(prob)
    lab, _ = instances_from_prob(p, shape_cfg)
    feats = component_features(p, lab)
    crops = [Crop(sl[0].start, sl[1].start, lab[sl] == i)
             for i, sl in enumerate(ndi.find_objects(lab), start=1) if sl is not None]
    n_e = len(gt_entries)
    v = np.zeros(len(crops))
    best = np.zeros((len(crops), max(n_e, 1)))
    for k, (entry_id, gts) in enumerate(gt_entries):
        iou, _ = overlap_matrices(gts, crops) if crops else (np.zeros((len(gts), 0)), None)
        if iou.size:
            b = iou.max(axis=0)
            best[:, k] = b
            v += np.where(b > IOU_THRESHOLD, b, 0.0)
    for f, vc, bc in zip(feats, v, best):
        f.update(image_id=image_id, v=float(vc), n_entries=n_e, n_matched=int((bc > IOU_THRESHOLD).sum()),
                 mean_best_iou=float(bc[:n_e].mean()) if n_e else 0.0)
    return feats


@dataclass
class ComponentSet:
    table: pd.DataFrame      # one row per candidate component
    n_gt: int                # total number of GT filaments over all entries
    shape_cfg: PostConfig

    def pq(self, keep: np.ndarray | None = None) -> tuple[float, int, int, int]:
        """PQ, TP, FP, FN when keeping the components flagged in ``keep``."""
        t = self.table if keep is None else self.table[np.asarray(keep, bool)]
        num = t["v"].sum()
        den = 0.5 * t["n_entries"].sum() + 0.5 * self.n_gt
        tp = int(t["n_matched"].sum())
        fp = int((t["n_entries"] - t["n_matched"]).sum())
        return (num / den if den else 0.0), tp, fp, self.n_gt - tp

    def rule(self, min_area: int = 0, min_max_prob: float = 0.0, min_mean_prob: float = 0.0) -> np.ndarray:
        t = self.table
        return ((t["area"] >= min_area) & (t["max_prob"] >= min_max_prob)
                & (t["mean_prob"] >= min_mean_prob)).to_numpy()


def build(probs: dict[str, np.ndarray], gt: dict[str, list[Crop]], shape_cfg: PostConfig,
          workers: int | None = None) -> ComponentSet:
    """Extract candidate components (no area / confidence filtering) and their match statistics."""
    shape_cfg = replace(shape_cfg, min_area=min(shape_cfg.min_area, 20), min_mean_prob=0.0, min_max_prob=0.0)
    by_image: dict[str, list] = {}
    for entry_id, crops in gt.items():
        by_image.setdefault(entry_id.split("-", 1)[1], []).append((entry_id, crops))
    jobs = [(k, p, shape_cfg, by_image.get(k, [])) for k, p in probs.items()]
    workers = workers if workers is not None else (os.cpu_count() or 1)
    if workers > 1 and len(jobs) > 1:
        method = "forkserver" if "forkserver" in mp.get_all_start_methods() else "spawn"
        with mp.get_context(method).Pool(workers) as pool:
            rows = [f for fs in pool.imap_unordered(_job, jobs, chunksize=2) for f in fs]
    else:
        rows = [f for j in jobs for f in _job(j)]
    n_gt = sum(len(c) for e, c in gt.items() if e.split("-", 1)[1] in probs)
    return ComponentSet(pd.DataFrame(rows), n_gt, shape_cfg)


def fit_scorer(table: pd.DataFrame, n_splits: int = 5, seed: int = 0):
    """Out-of-fold predictions of the per-entry matched IoU ``v / n_entries`` (GroupKFold by image).

    Returns ``(oof_prediction, fitted_model_on_all_rows)``.
    """
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.model_selection import GroupKFold

    x = table[FEATURES].to_numpy(np.float64)
    y = (table["v"] / table["n_entries"].clip(lower=1)).to_numpy()
    groups = table["image_id"].to_numpy()
    oof = np.zeros(len(table))
    params = dict(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=20,
                  l2_regularization=1.0, random_state=seed)
    for tr, va in GroupKFold(n_splits=n_splits).split(x, y, groups):
        oof[va] = HistGradientBoostingRegressor(**params).fit(x[tr], y[tr]).predict(x[va])
    return oof, HistGradientBoostingRegressor(**params).fit(x, y)


def best_threshold(cs: ComponentSet, score: np.ndarray, grid=None) -> tuple[float, float]:
    """Score threshold maximising PQ when keeping components with ``score > t``."""
    grid = np.linspace(0.05, 0.6, 56) if grid is None else grid
    res = [(cs.pq(score > t)[0], t) for t in grid]
    pq, t = max(res)
    return float(t), float(pq)
