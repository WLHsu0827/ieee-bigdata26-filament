"""Per-component outline level and extra evidence for the learned component filter.

``build``: on every out-of-fold image, candidate components are detected once on the ensemble map
(detection fields of ``--post``: threshold, closing, hole filling; all components, as the learned
filter sees them). For every candidate the script records the detection features of
:mod:`filseg.components`, extra evidence (mean / peak probability under each network separately and
their disagreement, distance to the nearest other candidate, number of confident candidates in the
image) and, at several outline levels, the area and the match statistics (``v``: sum over the
image's annotator sets of the matched IoU, ``nm``: number of matches) of the candidate redrawn where
the ensemble probability exceeds the level. Writes a pickle with the table and the GT counts.

``analyse``: nested cross-validation (outer folds = CV folds, thresholds chosen on inner out-of-fold
predictions) of
  * ``base``: the current learned filter (detection features, outline 0.5);
  * ``extra``: the same with the extra evidence;
  * ``levels`` / ``levels_extra``: one regressor over (candidate, level) rows predicts the per-entry
    matched IoU at every level; each candidate takes its best predicted level (only if it beats
    outline 0.5 by a margin chosen on inner predictions) and is kept if that prediction exceeds a
    threshold;
plus in-sample oracle bounds with the true values (fixed outline 0.5 and best outline per candidate).

Usage:
    python scripts/extent_levels.py build --probs "runs/k1/**/val_probs.npz" --gt-cache gt.pkl \
        --post configs/post_final_full.json --out levels.pkl
    python scripts/extent_levels.py analyse levels.pkl [--r 1 0.77]
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import pickle
import re
import sys
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from filseg.components import FEATURES, LEARNED_PARAMS, component_features, pq_sparse  # noqa: E402
from filseg.metrics import IOU_THRESHOLD, Crop, overlap_matrices  # noqa: E402
from filseg.postprocess import PostConfig, instances_from_prob, upsample  # noqa: E402

LEVELS = (0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7)
EXTRA = ["mean_net0", "max_net0", "mean_net1", "max_net1", "mean_diff", "abs_mean_diff", "nn_dist",
         "n_conf_image"]
LEVEL_FEATURES = ["level", "frac_level", "log_area_level"]


def _job(args) -> list[dict]:
    image_id, maps, cfg, levels, gt_entries = args
    acc = np.zeros(maps[0].shape, np.float32)
    for m in maps:
        acc += m
    ens = upsample(np.round(acc / len(maps)).astype(np.uint8))  # as scripts/analyze_oof.load_probs
    nets = [upsample(m) for m in maps]
    nets = nets if len(nets) > 1 else nets * 2  # a single network: both "networks" are the same map
    det = replace(cfg, mask_threshold=0.0, min_area=0, min_mean_prob=0.0, min_max_prob=0.0, scorer=None,
                  dilate_radius=0, merge_distance=0)
    lab, _ = instances_from_prob(ens, det)
    feats = component_features(ens, lab)
    slices = [sl for sl in ndi.find_objects(lab) if sl is not None]
    cents = []
    for i, (sl, f) in enumerate(zip(slices, feats, strict=True), start=1):
        m = lab[sl] == i
        ys, xs = np.nonzero(m)
        cents.append((sl[0].start + ys.mean(), sl[1].start + xs.mean()))
        for j, p in enumerate(nets[:2]):
            q = p[sl][m]
            f[f"mean_net{j}"], f[f"max_net{j}"] = float(q.mean()), float(q.max())
        f["mean_diff"] = f["mean_net0"] - f["mean_net1"]
        f["abs_mean_diff"] = abs(f["mean_diff"])
    c = np.array(cents) if cents else np.zeros((0, 2))
    n_conf = int(sum(f["max_prob"] >= 0.99 for f in feats))
    for k, f in enumerate(feats):
        d = np.hypot(*(c - c[k]).T) if len(c) > 1 else np.array([np.inf])
        d = d[np.arange(len(d)) != k] if len(c) > 1 else d
        f["nn_dist"] = float(min(d.min(), 2048.0))
        f["n_conf_image"] = n_conf
    n_e = len(gt_entries)
    for lev in levels:
        crops = []
        for i, sl in enumerate(slices, start=1):
            m = (lab[sl] == i) & (ens[sl] > lev)
            if cfg.fill_holes:
                m = ndi.binary_fill_holes(m)
            crops.append(Crop(sl[0].start, sl[1].start, m))
        v = np.zeros(len(crops))
        nm = np.zeros(len(crops), int)
        for _, gts in gt_entries:
            if not crops:
                break
            iou, _ = overlap_matrices(gts, crops)
            if iou.size:
                b = iou.max(axis=0)
                v += np.where(b > IOU_THRESHOLD, b, 0.0)
                nm += b > IOU_THRESHOLD
        for f, cr, vc, nc in zip(feats, crops, v, nm, strict=True):
            f[f"a@{lev:.2f}"], f[f"v@{lev:.2f}"], f[f"nm@{lev:.2f}"] = int(cr.area), float(vc), int(nc)
    for f in feats:
        f.update(image_id=image_id, n_entries=n_e)
    return feats


def build(args) -> None:
    from analyze_oof import load_gt
    from cli_utils import expand

    cfg = PostConfig(**json.loads(Path(args.post).read_text()))
    by_fold: dict[int, list[Path]] = {}
    for p in expand(args.probs):
        by_fold.setdefault(int(re.search(r"fold(\d+)", str(p)).group(1)), []).append(Path(p))
    rows, gt_per_image, fold_of_image, t0 = [], {}, {}, time.time()
    for fold, files in sorted(by_fold.items()):
        files = sorted(files)  # network order: alphabetical paths (r34 before v2s)
        zs = [np.load(f) for f in files]
        keys = sorted(set.intersection(*(set(z.files) for z in zs)))
        if args.limit:
            keys = keys[:args.limit]
        gt = load_gt(SimpleNamespace(gt_cache=args.gt_cache, data=args.data), set(keys))
        by_image: dict[str, list] = {}
        for entry_id, crops in gt.items():
            by_image.setdefault(entry_id.split("-", 1)[1], []).append((entry_id, crops))
        for k in keys:
            gt_per_image[k] = sum(len(c) for _, c in by_image.get(k, []))
            fold_of_image[k] = fold
        jobs = ((k, [z[k] for z in zs], cfg, LEVELS, by_image.get(k, [])) for k in keys)
        if args.workers > 1:
            method = "forkserver" if "forkserver" in mp.get_all_start_methods() else "spawn"
            with mp.get_context(method).Pool(args.workers) as pool:
                results = list(pool.imap_unordered(_job, jobs, chunksize=2))
        else:
            results = [_job(j) for j in jobs]
        for feats in results:
            for f in feats:
                f["fold"] = fold
            rows.extend(feats)
        print(f"fold {fold}: {len(keys)} images ({[f.parent.parent.name for f in files]}), "
              f"{len(rows)} candidates so far, {time.time() - t0:.0f} s", flush=True)
    out = {"table": pd.DataFrame(rows), "gt_per_image": gt_per_image, "fold_of_image": fold_of_image,
           "levels": LEVELS, "post": json.loads(Path(args.post).read_text()) | {"scorer": None}}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "wb") as fh:
        pickle.dump(out, fh, protocol=pickle.HIGHEST_PROTOCOL)


# ----------------------------------------------------------------------------- analysis
class Levels:
    """Candidates x levels arrays and exact pooled counts for (keep, level) decisions."""

    def __init__(self, data: dict):
        t = data["table"].reset_index(drop=True)
        self.t, self.levels = t, list(data["levels"])
        self.gt = data["gt_per_image"]
        self.fold_of_image = data["fold_of_image"]
        self.img = t["image_id"].to_numpy()
        self.fold = t["fold"].to_numpy()
        self.ne = t["n_entries"].to_numpy(int)
        self.area = t["area"].to_numpy(float)
        self.a = np.stack([t[f"a@{x:.2f}"].to_numpy(float) for x in self.levels], 1)
        self.v = np.stack([t[f"v@{x:.2f}"].to_numpy(float) for x in self.levels], 1)
        self.nm = np.stack([t[f"nm@{x:.2f}"].to_numpy(int) for x in self.levels], 1)
        self.y = self.v / np.maximum(self.ne, 1)[:, None]
        self.i05 = self.levels.index(0.5)

    def counts(self, keep: np.ndarray, lev: np.ndarray, rows: np.ndarray, images: set) -> np.ndarray:
        sel = rows & keep & (self.a[np.arange(len(lev)), lev] > 0)
        idx = np.flatnonzero(sel)
        tp = int(self.nm[idx, lev[idx]].sum())
        return np.array([self.v[idx, lev[idx]].sum(), tp, int((self.ne[idx] - self.nm[idx, lev[idx]]).sum()),
                         sum(self.gt.get(i, 0) for i in images) - tp], float)

    def oracle(self, per_component: bool) -> float:
        lev = self.y.argmax(1) if per_component else np.full(len(self.y), self.i05)
        val = self.y[np.arange(len(lev)), lev]
        pq, images = 0.4, set(self.gt)
        for _ in range(50):  # Dinkelbach: keep iff value > PQ / 2
            new = pq_sparse(*self.counts(val > pq / 2, lev, np.ones(len(lev), bool), images))
            if abs(new - pq) < 1e-9:
                break
            pq = new
        return pq


def _x(lv: Levels, feats: list[str]) -> np.ndarray:
    return lv.t[feats].to_numpy(np.float64)


def _stack(lv: Levels, rows: np.ndarray, feats: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """(candidate, level) design matrix and targets for the given candidate rows."""
    base = _x(lv, feats)[rows]
    n, nl = len(base), len(lv.levels)
    lev = np.repeat(np.array(lv.levels)[None, :], n, 0)
    frac = lv.a[rows] / np.maximum(lv.area[rows], 1)[:, None]
    loga = np.log1p(lv.a[rows])
    x = np.concatenate([np.repeat(base, nl, 0), lev.reshape(-1, 1), frac.reshape(-1, 1), loga.reshape(-1, 1)], 1)
    return x, lv.y[rows].reshape(-1)


def _fit_predict(x_tr, y_tr, x_te):
    from sklearn.ensemble import HistGradientBoostingRegressor

    return HistGradientBoostingRegressor(**LEARNED_PARAMS).fit(x_tr, y_tr).predict(x_te)


def _predict(lv: Levels, tr: np.ndarray, te: np.ndarray, feats: list[str], per_level: bool) -> np.ndarray:
    """Predicted per-entry matched IoU of the ``te`` rows: (n, levels) if per_level else (n, 1) at 0.5."""
    if not per_level:
        y = lv.y[:, lv.i05]
        return _fit_predict(_x(lv, feats)[tr], y[tr], _x(lv, feats)[te])[:, None]
    xs, ys = _stack(lv, tr, feats)
    xt, _ = _stack(lv, te, feats)
    return _fit_predict(xs, ys, xt).reshape(len(te), len(lv.levels))


def _decide(lv: Levels, pred: np.ndarray, rows: np.ndarray, margin: float, lam: float) -> tuple[np.ndarray, np.ndarray]:
    """Per candidate: level (best prediction if it beats outline 0.5 by ``margin``) and keep flag."""
    n = len(lv.t)
    lev, keep = np.full(n, lv.i05), np.zeros(n, bool)
    if pred.shape[1] == 1:
        score = pred[:, 0]
        lev_r = np.full(len(rows), lv.i05)
    else:
        p = np.where(lv.a[rows] > 0, pred, -np.inf)
        if np.isinf(margin):
            lev_r = np.full(len(rows), lv.i05)
        else:
            best = p.argmax(1)
            lev_r = np.where(p[np.arange(len(rows)), best] > p[:, lv.i05] + margin, best, lv.i05)
        score = p[np.arange(len(rows)), lev_r]
    lev[rows], keep[rows] = lev_r, score > lam
    return lev, keep


def nested(lv: Levels, feats: list[str], per_level: bool, r: float, inner_splits: int = 5) -> dict:
    from sklearn.model_selection import GroupKFold

    lams = np.linspace(0.05, 0.6, 56)
    margins = [0.0, 0.01, 0.02, 0.05, np.inf] if per_level else [np.inf]
    all_images = set(lv.gt)
    pooled, chosen = np.zeros(4), []
    for k in sorted(set(lv.fold_of_image.values())):
        inner = np.flatnonzero(lv.fold != k)
        outer = np.flatnonzero(lv.fold == k)
        inner_imgs = {i for i in all_images if lv.fold_of_image[i] != k}
        mask_inner = np.zeros(len(lv.t), bool)
        mask_inner[inner] = True
        oof = np.zeros((len(inner), len(lv.levels) if per_level else 1))
        for a, b in GroupKFold(inner_splits).split(inner, groups=lv.img[inner]):
            oof[b] = _predict(lv, inner[a], inner[b], feats, per_level)

        def inner_objective(m, lam, oof=oof, inner=inner, mask_inner=mask_inner, inner_imgs=inner_imgs):
            lev, keep = _decide(lv, oof, inner, m, lam)
            return pq_sparse(*lv.counts(keep, lev, mask_inner, inner_imgs), r=r)

        _, m, lam = max((inner_objective(m, lam), m, lam) for m in margins for lam in lams)
        pred = _predict(lv, inner, outer, feats, per_level)
        lev, keep = _decide(lv, pred, outer, m, lam)
        mask_outer = np.zeros(len(lv.t), bool)
        mask_outer[outer] = True
        pooled += lv.counts(keep, lev, mask_outer, all_images - inner_imgs)
        moved = int((lev[outer][keep[outer]] != lv.i05).sum())
        chosen.append({"fold": int(k), "margin": float(m), "lambda": round(float(lam), 3), "kept": int(keep[outer].sum()),
                       "kept_with_other_level": moved})
    s, tp, fp, fn = pooled
    return {"nested_pq": pq_sparse(s, tp, fp, fn), "objective": pq_sparse(s, tp, fp, fn, r), "tp": int(tp),
            "fp": int(fp), "fn": int(fn), "folds": chosen}


def analyse(args) -> None:
    with open(args.table, "rb") as fh:
        data = pickle.load(fh)
    lv = Levels(data)
    print(f"{len(lv.t)} candidates on {len(lv.gt)} images, levels {lv.levels}", flush=True)
    res = {"oracle_outline_0.5": lv.oracle(False), "oracle_best_outline": lv.oracle(True)}
    print(json.dumps(res), flush=True)
    variants = {"base": (FEATURES, False), "extra": (FEATURES + EXTRA, False),
                "levels": (FEATURES, True), "levels_extra": (FEATURES + EXTRA, True)}
    for r in args.r:
        for name, (feats, per_level) in variants.items():
            if args.only and name not in args.only:
                continue
            t0 = time.time()
            out = nested(lv, feats, per_level, r, args.inner_splits)
            res[f"{name}@r{r}"] = out
            print(f"{name:13s} r={r}: nested PQ {out['nested_pq']:.4f} (TP {out['tp']}, FP {out['fp']}, "
                  f"FN {out['fn']}) {time.time() - t0:.0f} s; folds {out['folds']}", flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--probs", nargs="+", required=True)
    b.add_argument("--post", required=True, help="post-processing JSON whose detection fields are used")
    b.add_argument("--gt-cache", type=Path, default=None)
    b.add_argument("--data", default=None)
    b.add_argument("--workers", type=int, default=4)
    b.add_argument("--limit", type=int, default=0, help="first n images per fold only (tests)")
    b.add_argument("--out", required=True)
    a = sub.add_parser("analyse")
    a.add_argument("table")
    a.add_argument("--r", type=float, nargs="+", default=[1.0, 0.77])
    a.add_argument("--only", nargs="*", default=None, help="variants to run (default: all)")
    a.add_argument("--inner-splits", type=int, default=5)
    a.add_argument("--out", default=None)
    args = ap.parse_args()
    build(args) if args.cmd == "build" else analyse(args)


if __name__ == "__main__":
    main()
