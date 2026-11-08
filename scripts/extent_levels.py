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
EXTRA_NETS = EXTRA[:6]
EXTRA_CONTEXT = EXTRA[6:]
LEVEL_FEATURES = ["level", "frac_level", "log_area_level"]
VARIANTS = {"rule": (None, False), "base": (FEATURES, False), "extra": (FEATURES + EXTRA, False),
            "extra_nets": (FEATURES + EXTRA_NETS, False), "extra_context": (FEATURES + EXTRA_CONTEXT, False),
            "levels": (FEATURES, True), "levels_extra": (FEATURES + EXTRA, True)}
DEFAULT_VARIANTS = ["rule", "base", "extra", "levels", "levels_extra"]


def candidate_features(maps: list[np.ndarray], cfg: PostConfig):
    """Ensemble map, detection label map, component slices and features (detection features of
    filseg.components plus the extra evidence) of one image from its networks' uint8 maps."""
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
    return ens, lab, slices, feats


def outline(lab: np.ndarray, ens: np.ndarray, slices: list, i: int, level: float, fill_holes: bool) -> np.ndarray:
    """Mask (within ``slices[i - 1]``) of candidate ``i`` redrawn where the ensemble exceeds ``level``."""
    sl = slices[i - 1]
    m = (lab[sl] == i) & (ens[sl] > level)
    return ndi.binary_fill_holes(m) if fill_holes else m


def _job(args) -> list[dict]:
    image_id, maps, cfg, levels, gt_entries = args
    ens, lab, slices, feats = candidate_features(maps, cfg)
    n_e = len(gt_entries)
    for lev in levels:
        crops = [Crop(sl[0].start, sl[1].start, outline(lab, ens, slices, i, lev, cfg.fill_holes))
                 for i, sl in enumerate(slices, start=1)]
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
        if args.subfolds:  # nested CV inside one fold (pilots trained on a single fold)
            from filseg.data import image_folds

            sub = image_folds(keys, args.subfolds, 2026)
            fold_of_image.update({k: fold * 100 + sub[k] for k in keys})
        jobs = ((k, [z[k] for z in zs], cfg, LEVELS, by_image.get(k, [])) for k in keys)
        if args.workers > 1:
            method = "forkserver" if "forkserver" in mp.get_all_start_methods() else "spawn"
            with mp.get_context(method).Pool(args.workers) as pool:
                results = list(pool.imap_unordered(_job, jobs, chunksize=2))
        else:
            results = [_job(j) for j in jobs]
        for feats in results:
            for f in feats:
                f["fold"] = fold_of_image[f["image_id"]]
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


def nested(lv: Levels, feats: list[str], per_level: bool, r: float, inner_splits: int = 5,
           with_decisions: bool = False) -> dict:
    from sklearn.model_selection import GroupKFold

    lams = np.linspace(0.05, 0.6, 56)
    margins = [0.0, 0.01, 0.02, 0.05, np.inf] if per_level else [np.inf]
    all_images = set(lv.gt)
    pooled, chosen = np.zeros(4), []
    keep_all, lev_all = np.zeros(len(lv.t), bool), np.full(len(lv.t), lv.i05)
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
        keep_all[outer], lev_all[outer] = keep[outer], lev[outer]
        moved = int((lev[outer][keep[outer]] != lv.i05).sum())
        chosen.append({"fold": int(k), "margin": float(m), "lambda": round(float(lam), 3), "kept": int(keep[outer].sum()),
                       "kept_with_other_level": moved})
    s, tp, fp, fn = pooled
    out = {"nested_pq": pq_sparse(s, tp, fp, fn), "objective": pq_sparse(s, tp, fp, fn, r), "tp": int(tp),
           "fp": int(fp), "fn": int(fn), "folds": chosen}
    if with_decisions:
        out.update(keep=keep_all, lev=lev_all)
    return out


def per_image_counts(lv: Levels, keep: np.ndarray, lev: np.ndarray) -> pd.DataFrame:
    """Per image: sum of matched IoU, TP, FP, FN of a (keep, level) decision."""
    ok = keep & (lv.a[np.arange(len(lev)), lev] > 0)
    rows = np.arange(len(lev))
    df = pd.DataFrame({"image_id": lv.img, "s": np.where(ok, lv.v[rows, lev], 0.0),
                       "tp": np.where(ok, lv.nm[rows, lev], 0), "fp": np.where(ok, lv.ne - lv.nm[rows, lev], 0)})
    per = df.groupby("image_id")[["s", "tp", "fp"]].sum().reindex(sorted(lv.gt), fill_value=0)
    per["fn"] = [lv.gt[i] for i in per.index] - per["tp"]
    return per


def paired_bootstrap(a: pd.DataFrame, b: pd.DataFrame, draws: int = 4000, seed: int = 0) -> dict:
    """Difference of pooled PQ (b - a) under resampling of images with replacement."""
    rng = np.random.default_rng(seed)
    xa, xb = a[["s", "tp", "fp", "fn"]].to_numpy(float), b[["s", "tp", "fp", "fn"]].to_numpy(float)

    def pq(x):
        return x[0] / max(x[1] + 0.5 * x[2] + 0.5 * x[3], 1e-9)

    d = []
    for _ in range(draws):
        i = rng.integers(0, len(xa), len(xa))
        d.append(pq(xb[i].sum(0)) - pq(xa[i].sum(0)))
    d = np.array(d)
    return {"diff": pq(xb.sum(0)) - pq(xa.sum(0)), "sd": float(d.std()), "p_le_0": float((d <= 0).mean())}


def nested_rule(lv: Levels, r: float) -> dict:
    """Nested CV of the three-parameter rule (area, peak, mean probability) at outline 0.5."""
    from filseg.components import ComponentSet, nested_select

    t = lv.t.copy()
    gone = t["a@0.50"].to_numpy() == 0
    t["v"], t["n_matched"] = t["v@0.50"], t["nm@0.50"]
    t["n_entries"] = np.where(gone, 0, t["n_entries"])
    cs = ComponentSet(t, sum(lv.gt.values()), PostConfig(**{k: v for k, v in lv.post.items() if k != "scorer"}),
                      dict(lv.gt))
    out = nested_select({"shape": cs}, lv.fold_of_image, r=r)
    c = out["nested_counts"]
    return {"nested_pq": out["nested_pq"], "objective": out["nested_objective"], "tp": c["tp"], "fp": c["fp"],
            "fn": c["fn"], "folds": [{"fold": f["fold"], "rule": f["rule"]} for f in out["fold_choices"]]}


def analyse(args) -> None:
    with open(args.table, "rb") as fh:
        data = pickle.load(fh)
    lv = Levels(data)
    lv.post = data["post"]
    print(f"{len(lv.t)} candidates on {len(lv.gt)} images, levels {lv.levels}", flush=True)
    res = {"oracle_outline_0.5": lv.oracle(False), "oracle_best_outline": lv.oracle(True)}
    print(json.dumps(res), flush=True)
    variants = {k: VARIANTS[k] for k in (args.only or DEFAULT_VARIANTS)}
    for r in args.r:
        for name, (feats, per_level) in variants.items():
            t0 = time.time()
            out = nested_rule(lv, r) if name == "rule" else nested(lv, feats, per_level, r, args.inner_splits)
            res[f"{name}@r{r}"] = out
            print(f"{name:13s} r={r}: nested PQ {out['nested_pq']:.4f} (TP {out['tp']}, FP {out['fp']}, "
                  f"FN {out['fn']}) {time.time() - t0:.0f} s; folds {out['folds']}", flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=1))


def compare(args) -> None:
    """Paired bootstrap over images of two learned variants' nested decisions."""
    with open(args.table, "rb") as fh:
        data = pickle.load(fh)
    lv = Levels(data)
    for r in args.r:
        per = {}
        for name in (args.a, args.b):
            feats, per_level = VARIANTS[name]
            out = nested(lv, feats, per_level, r, args.inner_splits, with_decisions=True)
            per[name] = per_image_counts(lv, out["keep"], out["lev"])
            print(f"{name} r={r}: nested PQ {out['nested_pq']:.4f}", flush=True)
        print(f"{args.b} - {args.a} (r={r}): {paired_bootstrap(per[args.a], per[args.b])}", flush=True)


def compare_tables(args) -> None:
    """Paired bootstrap over images: the same variant on two candidate tables (e.g. two ensembles)."""
    lvs = []
    for path in args.tables:
        with open(path, "rb") as fh:
            data = pickle.load(fh)
        lv = Levels(data)
        lv.post = data["post"]
        lvs.append(lv)
    if set(lvs[0].gt) != set(lvs[1].gt):
        raise SystemExit("the tables cover different images")
    res = {"tables": [str(p) for p in args.tables], "variant": args.variant}
    for r in args.r:
        per, pqs = [], []
        for lv in lvs:
            if args.variant == "rule":
                from filseg.components import ComponentSet, nested_select

                t = lv.t.copy()
                t["v"], t["n_matched"] = t["v@0.50"], t["nm@0.50"]
                t["n_entries"] = np.where(t["a@0.50"].to_numpy() == 0, 0, t["n_entries"])
                cs = ComponentSet(t, sum(lv.gt.values()), PostConfig(threshold=0.25), dict(lv.gt))
                out = nested_select({"shape": cs}, lv.fold_of_image, r=r)
                keep = np.zeros(len(t), bool)
                for ch in out["fold_choices"]:
                    rows = np.isin(lv.img, [i for i, f in lv.fold_of_image.items() if f == ch["fold"]])
                    keep |= rows & cs.rule(**ch["rule"])
                lev = np.full(len(t), lv.i05)
                pqs.append(out["nested_pq"])
            else:
                feats, per_level = VARIANTS[args.variant]
                out = nested(lv, feats, per_level, r, args.inner_splits, with_decisions=True)
                keep, lev = out["keep"], out["lev"]
                pqs.append(out["nested_pq"])
            per.append(per_image_counts(lv, keep, lev))
        boot = paired_bootstrap(per[0], per[1])
        res[f"r{r}"] = {"nested_pq": pqs, **{k: float(v) for k, v in boot.items()}}
        print(f"{args.variant} r={r}: {pqs[0]:.4f} -> {pqs[1]:.4f}; paired {boot}", flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=1))


_TS: dict = {}


def _ts_init(files: dict, cfg: PostConfig, models: dict, level: float) -> None:
    import cv2

    cv2.setNumThreads(1)
    _TS.update(files={ek: np.load(p) for ek, p in files.items()}, cfg=cfg, models=models, level=level)


def _kept_labels(maps: list[np.ndarray], feats_cols: list[str], model, lam: float) -> np.ndarray:
    cfg, level = _TS["cfg"], _TS["level"]
    ens, lab, slices, feats = candidate_features(maps, cfg)
    kept = np.zeros(lab.shape, np.int32)
    if not feats:
        return kept
    keep = model.predict(np.array([[f[c] for c in feats_cols] for f in feats], np.float64)) > lam
    n = 0
    for i in np.flatnonzero(keep) + 1:
        m = outline(lab, ens, slices, int(i), level, cfg.fill_holes)
        if m.any():
            n += 1
            kept[slices[i - 1]][m] = n
    return kept


def _ts_job(key: str) -> dict:
    import itertools

    from ensemble_shift import agreement

    maps = {ek: z[key] for ek, z in _TS["files"].items()}
    encs, folds = sorted({e for e, _ in maps}), sorted({k for _, k in maps})
    settings = {f"two{k}": [maps[(e, k)] for e in encs] for k in folds}
    settings["ten"] = [np.round(np.mean([maps[(e, k)].astype(np.float32) for k in folds], 0)).astype(np.uint8)
                       for e in encs]  # per-encoder averages; their mean is the ten-network ensemble
    out = {"image": key}
    for name, (cols, model, lam) in _TS["models"].items():
        labs = {s: _kept_labels(ms, cols, model, lam) for s, ms in settings.items()}
        out[name] = {"n": {s: int(lab.max()) for s, lab in labs.items()},
                     "vs_ten": [agreement(labs["ten"], labs[f"two{k}"]) for k in folds],
                     "pairs": [agreement(labs[f"two{a}"], labs[f"two{b}"]) for a, b in itertools.combinations(folds, 2)]}
    return out


def _fit_full(lv: Levels, feats: list[str], r: float):
    """Filter fitted on all out-of-fold candidates; keep threshold from grouped out-of-fold predictions."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.model_selection import GroupKFold

    x, y = _x(lv, feats), lv.y[:, lv.i05]
    oof = np.zeros(len(y))
    for a, b in GroupKFold(5).split(x, y, lv.img):
        oof[b] = _fit_predict(x[a], y[a], x[b])
    rows, lev = np.ones(len(y), bool), np.full(len(y), lv.i05)
    lam = max((pq_sparse(*lv.counts(oof > t, lev, rows, set(lv.gt)), r=r), t) for t in np.linspace(0.05, 0.6, 56))[1]
    return HistGradientBoostingRegressor(**LEARNED_PARAMS).fit(x, y), float(lam)


def test_shift(args) -> None:
    """Label-free ensemble-size check of learned filters on the test images (as scripts/ensemble_shift.py):
    decisions on the ten-network average (per-network features from the per-encoder averages of five) vs on
    each fold's two-network average (per-network features from single networks, as out of fold)."""
    import itertools

    from cli_utils import expand

    with open(args.table, "rb") as fh:
        data = pickle.load(fh)
    lv = Levels(data)
    cfg = PostConfig(**data["post"])
    models = {}
    for name in args.variants:
        cols = VARIANTS[name][0]
        model, lam = _fit_full(lv, cols, args.r)
        models[name] = (cols, model, lam)
        print(f"{name}: filter fitted on {len(lv.t)} candidates, keep threshold {lam:.2f}", flush=True)
    files = {}
    for p in expand(args.test_probs):
        enc = "r34" if "r34" in str(p) else "v2s"
        files[(enc, int(re.search(r"fold(\d+)", str(p)).group(1)))] = str(p)
    with np.load(next(iter(files.values()))) as z:
        keys = sorted(z.files)[:args.limit] if args.limit else sorted(z.files)
    method = "forkserver" if "forkserver" in mp.get_all_start_methods() else "spawn"
    with mp.get_context(method).Pool(args.workers, initializer=_ts_init,
                                     initargs=(files, cfg, models, cfg.mask_threshold)) as pool:
        rows = list(pool.imap_unordered(_ts_job, keys, chunksize=2))
    folds = sorted({k for _, k in files})
    res = {"images": len(rows), "r": args.r}
    for name in args.variants:
        n_ten = sum(r_[name]["n"]["ten"] for r_ in rows)
        n_two = [sum(r_[name]["n"][f"two{k}"] for r_ in rows) for k in folds]

        def pq(tp, s, na, nb):
            return s / max(tp + 0.5 * (na - tp) + 0.5 * (nb - tp), 1e-9)

        vs = [pq(sum(r_[name]["vs_ten"][j][0] for r_ in rows), sum(r_[name]["vs_ten"][j][1] for r_ in rows),
                 n_ten, n_two[j]) for j in range(len(folds))]
        pairs = [pq(sum(r_[name]["pairs"][j][0] for r_ in rows), sum(r_[name]["pairs"][j][1] for r_ in rows),
                    n_two[a], n_two[b]) for j, (a, b) in enumerate(itertools.combinations(range(len(folds)), 2))]
        res[name] = {"instances_per_image_ten": n_ten / len(rows),
                     "instances_per_image_two": float(np.mean(n_two)) / len(rows),
                     "two_range": [min(n_two) / len(rows), max(n_two) / len(rows)],
                     "ten_vs_two_pq": float(np.mean(vs)), "two_vs_two_pq": float(np.mean(pairs)),
                     "threshold": models[name][2]}
        print(name, json.dumps(res[name]), flush=True)
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
    b.add_argument("--subfolds", type=int, default=0,
                   help="split every fold into this many sub-folds for nested CV (single-fold pilots)")
    b.add_argument("--out", required=True)
    a = sub.add_parser("analyse")
    a.add_argument("table")
    a.add_argument("--r", type=float, nargs="+", default=[1.0, 0.77])
    a.add_argument("--only", nargs="*", default=None, help=f"variants to run (default: {' '.join(DEFAULT_VARIANTS)})")
    a.add_argument("--inner-splits", type=int, default=5)
    a.add_argument("--out", default=None)
    c = sub.add_parser("compare")
    c.add_argument("table")
    c.add_argument("--a", default="base")
    c.add_argument("--b", default="extra")
    c.add_argument("--r", type=float, nargs="+", default=[1.0, 0.77])
    c.add_argument("--inner-splits", type=int, default=5)
    ct = sub.add_parser("compare-tables")
    ct.add_argument("tables", nargs=2, help="two candidate tables on the same images (baseline first)")
    ct.add_argument("--variant", default="base", choices=sorted(VARIANTS))
    ct.add_argument("--r", type=float, nargs="+", default=[1.0, 0.77])
    ct.add_argument("--inner-splits", type=int, default=5)
    ct.add_argument("--out", default=None)
    ts = sub.add_parser("test-shift")
    ts.add_argument("table", help="out-of-fold candidate table (build) to fit the filters on")
    ts.add_argument("--test-probs", nargs="+", required=True, help="test_probs.npz of every network (glob)")
    ts.add_argument("--variants", nargs="+", default=["base", "extra_nets", "extra"])
    ts.add_argument("--r", type=float, default=0.77)
    ts.add_argument("--workers", type=int, default=4)
    ts.add_argument("--limit", type=int, default=0)
    ts.add_argument("--out", default=None)
    args = ap.parse_args()
    {"build": build, "analyse": analyse, "compare": compare, "compare-tables": compare_tables,
     "test-shift": test_shift}[args.cmd](args)


if __name__ == "__main__":
    main()
