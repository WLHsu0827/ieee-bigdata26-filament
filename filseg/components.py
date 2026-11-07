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
from dataclasses import dataclass, field, replace

import cv2
import numpy as np
import pandas as pd
from scipy import ndimage as ndi

from .metrics import IOU_THRESHOLD, Crop, overlap_matrices
from .postprocess import PostConfig, instances_from_prob, upsample

FEATURES = ["log_area", "mean_prob", "max_prob", "p90_prob", "std_prob", "elongation", "fill_ratio",
            "radius", "n_comp_image", "area_rank", "length", "width"]


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
            length, width = 4.0 * float(np.sqrt(max(ev[1], 0.0))), 4.0 * float(np.sqrt(max(ev[0], 0.0)))
        else:
            elong, length, width = 1.0, 1.0, 1.0
        cy, cx = sl[0].start + ys.mean(), sl[1].start + xs.mean()
        feats.append(dict(area=area, log_area=float(np.log(area)), mean_prob=float(p.mean()),
                          max_prob=float(p.max()), p90_prob=float(np.percentile(p, 90)),
                          std_prob=float(p.std()), elongation=elong, length=length, width=width,
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
    # candidates and their features come from the detection outline (as in instances_from_prob,
    # where selection rules see detection statistics); an outline threshold above the detection
    # threshold then shrinks every candidate independently, so the decomposition stays exact
    t_mask = shape_cfg.mask_threshold
    if 0 < t_mask < shape_cfg.threshold:
        raise ValueError("component decomposition supports only mask_threshold above the detection threshold")
    lab, _ = instances_from_prob(p, replace(shape_cfg, mask_threshold=0.0))
    feats = component_features(p, lab)
    crops = []
    for i, sl in enumerate(ndi.find_objects(lab), start=1):
        if sl is None:
            continue
        m = lab[sl] == i
        if t_mask > shape_cfg.threshold:
            m = m & (p[sl] > t_mask)
            if shape_cfg.fill_holes:
                m = ndi.binary_fill_holes(m)
        crops.append(Crop(sl[0].start, sl[1].start, m))
    n_e = len(gt_entries)
    v = np.zeros(len(crops))
    best = np.zeros((len(crops), max(n_e, 1)))
    for k, (entry_id, gts) in enumerate(gt_entries):
        iou, _ = overlap_matrices(gts, crops) if crops else (np.zeros((len(gts), 0)), None)
        if iou.size:
            b = iou.max(axis=0)
            best[:, k] = b
            v += np.where(b > IOU_THRESHOLD, b, 0.0)
    for f, vc, bc, c in zip(feats, v, best, crops):
        gone = c.area == 0  # vanished after shrinking: dropped by the pipeline, counts nowhere
        f.update(image_id=image_id, v=float(vc), n_entries=0 if gone else n_e,
                 n_matched=0 if gone else int((bc > IOU_THRESHOLD).sum()),
                 mean_best_iou=float(bc[:n_e].mean()) if n_e else 0.0)
    return feats


@dataclass
class ComponentSet:
    table: pd.DataFrame      # one row per candidate component
    n_gt: int                # total number of GT filaments over all entries
    shape_cfg: PostConfig
    gt_per_image: dict = field(default_factory=dict)  # image_id -> GT filaments summed over its entries

    def counts(self, keep: np.ndarray | None = None, images=None) -> tuple[float, int, int, int]:
        """(sum of matched IoU, TP, FP, FN) for the kept components of ``images`` (default: all)."""
        n_gt = self.n_gt if images is None else sum(self.gt_per_image.get(i, 0) for i in set(images))
        if len(self.table) == 0:
            return 0.0, 0, 0, n_gt
        sel = np.ones(len(self.table), bool) if keep is None else np.asarray(keep, bool).copy()
        if images is not None:
            sel &= self.table["image_id"].isin(set(images)).to_numpy()
        t = self.table[sel]
        tp = int(t["n_matched"].sum())
        return float(t["v"].sum()), tp, int((t["n_entries"] - t["n_matched"]).sum()), n_gt - tp

    def pq(self, keep: np.ndarray | None = None, images=None) -> tuple[float, int, int, int]:
        """PQ, TP, FP, FN when keeping the components flagged in ``keep`` (optionally on a subset of images)."""
        s, tp, fp, fn = self.counts(keep, images)
        den = tp + 0.5 * fp + 0.5 * fn
        return (s / den if den else 0.0), tp, fp, fn

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
    gt_per_image: dict[str, int] = {}
    for e, c in gt.items():
        image_id = e.split("-", 1)[1]
        if image_id in probs:
            gt_per_image[image_id] = gt_per_image.get(image_id, 0) + len(c)
    return ComponentSet(pd.DataFrame(rows), sum(gt_per_image.values()), shape_cfg, gt_per_image)


def fit_scorer(table: pd.DataFrame, n_splits: int = 5, seed: int = 0):
    """Out-of-fold predictions of the per-entry matched IoU ``v / n_entries`` (GroupKFold by image).

    Returns ``(oof_prediction, fitted_model_on_all_rows)``.
    """
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.model_selection import GroupKFold

    feats = [f for f in FEATURES if f in table.columns]
    x = table[feats].to_numpy(np.float64)
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


# ----------------------------------------------------------------------------- rule selection
RULE_GRID = {
    "min_area": [0, 100, 200, 300, 400, 500, 650, 800, 1000],
    "min_max_prob": [0.0, 0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99],
    "min_mean_prob": [0.0, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
}


def _rule_masks(cs: ComponentSet, grid: dict) -> list[tuple[dict, np.ndarray]]:
    import itertools

    keys = list(grid)
    return [(dict(zip(keys, v)), cs.rule(**dict(zip(keys, v)))) for v in itertools.product(*(grid[k] for k in keys))]


def pq_sparse(s: float, tp: float, fp: float, fn: float, r: float = 1.0) -> float:
    """Expected PQ if each GT filament were labelled only with probability ``r`` (``r = 1``: plain PQ).

    A matched prediction then loses its label with probability 1 - r (becomes an FP) and
    unmatched GT count only with weight r. Used to study sparser test labelling; with
    r = 1 this is exactly the official pooled PQ.
    """
    den = r * tp + 0.5 * fp + 0.5 * (1 - r) * tp + 0.5 * r * fn
    return r * s / den if den else 0.0


def nested_select(sets: dict, folds: dict[str, int], grid: dict | None = None, r: float = 1.0) -> dict:
    """Nested cross-validation of the post-processing choice.

    ``sets`` maps a shape configuration (any hashable key) to its :class:`ComponentSet`
    (all built on the same out-of-fold images); ``folds`` maps image_id -> fold.
    For every outer fold the (shape, rule) with the best objective on the *other* folds is
    applied to the held-out fold; the pooled counts give an unbiased estimate. The
    objective is :func:`pq_sparse` with label rate ``r`` (``r = 1``: official PQ).
    Returns the nested estimate, the per-fold choices and the final choice (best on all images).
    """
    grid = grid or RULE_GRID
    images = set().union(*(cs.gt_per_image.keys() for cs in sets.values()))
    fold_ids = sorted({folds[i] for i in images})
    gt_per_image = next(iter(sets.values())).gt_per_image
    subsets = {k: {i for i in images if folds[i] != k} for k in fold_ids}  # inner images of each outer fold
    subsets[None] = images
    n_gt = {k: sum(gt_per_image.get(i, 0) for i in imgs) for k, imgs in subsets.items()}
    # per shape: numpy columns and, per subset, the rows belonging to it (vectorised counting)
    cols = {}
    for key, cs in sets.items():
        t = cs.table
        img = t["image_id"].to_numpy() if len(t) else np.array([], object)
        cols[key] = (t["v"].to_numpy(float) if len(t) else np.zeros(0),
                     t["n_matched"].to_numpy(int) if len(t) else np.zeros(0, int),
                     t["n_entries"].to_numpy(int) if len(t) else np.zeros(0, int),
                     {k: np.isin(img, list(imgs)) for k, imgs in subsets.items()})

    def counts(key, mask, k):
        v, nm, ne, rows = cols[key]
        sel = mask & rows[k] if len(v) else np.zeros(0, bool)
        tp = int(nm[sel].sum())
        return float(v[sel].sum()), tp, int((ne[sel] - nm[sel]).sum()), n_gt[k] - tp

    def objective(key, mask, k):
        return pq_sparse(*counts(key, mask, k), r=r)

    cands = [(key, rule, mask) for key, cs in sets.items() for rule, mask in _rule_masks(cs, grid)]
    pooled = np.zeros(4)
    choices = []
    for k in fold_ids:
        key, rule, mask = max(cands, key=lambda c: objective(c[0], c[2], k))
        outer = images - subsets[k]
        pooled += np.array(sets[key].counts(mask, outer))
        choices.append({"fold": k, "shape": key, "rule": rule, "inner": objective(key, mask, k),
                        "outer": pq_sparse(*sets[key].counts(mask, outer), r=r)})
    s, tp, fp, fn = pooled
    key, rule, mask = max(cands, key=lambda c: objective(c[0], c[2], None))
    fs, ftp, ffp, ffn = sets[key].counts(mask)
    return {"r": r, "nested_pq": pq_sparse(s, tp, fp, fn, 1.0), "nested_objective": pq_sparse(s, tp, fp, fn, r),
            "nested_counts": {"sum_iou": float(s), "tp": int(tp), "fp": int(fp), "fn": int(fn)},
            "fold_choices": choices, "final_shape": key, "final_rule": rule,
            "final_pq_in_sample": pq_sparse(fs, ftp, ffp, ffn, 1.0),
            "final_objective_in_sample": pq_sparse(fs, ftp, ffp, ffn, r)}

LEARNED_PARAMS = dict(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=20,
                      l2_regularization=1.0, random_state=0)


def nested_learned(cs: ComponentSet, folds: dict[str, int], r: float = 1.0, inner_splits: int = 5,
                   grid=None) -> dict:
    """Nested cross-validation of a *learned* component filter on one shape configuration.

    For every outer fold a gradient-boosted regressor predicts each candidate's per-entry matched
    IoU ``v / |E|`` from its features (:data:`FEATURES`); the keep threshold is chosen on inner
    out-of-fold predictions of the other folds (objective :func:`pq_sparse` with label rate ``r``),
    and the model refitted on those folds filters the held-out fold. Comparable with
    ``nested_select({shape: cs}, folds, r=r)`` on the same shape.
    """
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.model_selection import GroupKFold

    grid = np.linspace(0.05, 0.6, 56) if grid is None else grid
    t = cs.table.reset_index(drop=True)
    feats = [f for f in FEATURES if f in t.columns]
    x = t[feats].to_numpy(np.float64)
    y = (t["v"] / t["n_entries"].clip(lower=1)).to_numpy()
    img = t["image_id"].to_numpy()
    v, nm, ne = t["v"].to_numpy(float), t["n_matched"].to_numpy(int), t["n_entries"].to_numpy(int)
    images = set(cs.gt_per_image)
    fold_of = np.array([folds[i] for i in img])

    def counts(sel, imgs):
        tp = int(nm[sel].sum())
        return float(v[sel].sum()), tp, int((ne[sel] - nm[sel]).sum()), sum(cs.gt_per_image.get(i, 0) for i in imgs) - tp

    def fit_predict(tr, te):
        return HistGradientBoostingRegressor(**LEARNED_PARAMS).fit(x[tr], y[tr]).predict(x[te])

    pooled, lams = np.zeros(4), []
    for k in sorted({folds[i] for i in images}):
        inner = np.flatnonzero(fold_of != k)
        outer = np.flatnonzero(fold_of == k)
        inner_imgs = {i for i in images if folds[i] != k}
        oof = np.zeros(len(inner))
        for a, b in GroupKFold(inner_splits).split(inner, groups=img[inner]):
            oof[b] = fit_predict(inner[a], inner[b])
        score = np.zeros(len(t))
        score[inner] = oof
        best = max((pq_sparse(*counts(np.isin(np.arange(len(t)), inner) & (score > lam), inner_imgs), r=r), lam)
                   for lam in grid)
        lam = float(best[1])
        lams.append(round(lam, 3))
        score[outer] = fit_predict(inner, outer)
        sel = np.zeros(len(t), bool)
        sel[outer] = score[outer] > lam
        pooled += np.array(counts(sel, images - inner_imgs))
    s, tp, fp, fn = pooled
    return {"r": r, "nested_pq": pq_sparse(s, tp, fp, fn, 1.0), "nested_objective": pq_sparse(s, tp, fp, fn, r),
            "nested_counts": {"sum_iou": float(s), "tp": int(tp), "fp": int(fp), "fn": int(fn)},
            "thresholds": lams, "features": feats}
