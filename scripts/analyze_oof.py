"""Analyse out-of-fold probability maps: tune post-processing and component re-scoring.

For every *shape* configuration (threshold, closing radius, merge distance)
candidate components are extracted once; selection rules (minimum area /
peak / mean probability) and a learned component scorer are then evaluated
with the exact PQ decomposition of :mod:`filseg.components`.

Usage:
    python scripts/analyze_oof.py --data <MAGFiLO dir> --probs run_a/val_probs.npz [run_b/val_probs.npz ...] \
        --out <dir>
Several ``--probs`` files are averaged (ensemble). Candidate components of every
shape configuration are cached in ``--out``, so an interrupted run resumes where it stopped.
"""
from __future__ import annotations

import argparse
import itertools
import json
import pickle
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from filseg import analysis, components, data  # noqa: E402
from filseg.metrics import Crop, evaluate  # noqa: E402
from filseg.postprocess import PostConfig  # noqa: E402
from filseg.train import find_data_dir  # noqa: E402
from filseg.tune import Scorer  # noqa: E402


def load_probs(paths: list[Path]) -> dict[str, np.ndarray]:
    """Load ``val_probs.npz`` files as uint8 maps (probability x 255).

    Maps of the same image are averaged (ensembles; resized to the largest resolution),
    files with disjoint images (different CV folds) are concatenated. Files are read lazily
    and averaged one image at a time, so 707 maps at 2048 px need ~3 GB.
    """
    import cv2

    files = [np.load(p) for p in paths]
    keys = sorted(set().union(*(z.files for z in files)))
    out = {}
    for k in keys:
        arrs = [z[k] for z in files if k in z.files]
        size = max(a.shape[0] for a in arrs)
        acc = np.zeros((size, size), np.float32)
        for a in arrs:
            a = a.astype(np.float32)
            acc += a if a.shape[0] == size else cv2.resize(a, (size, size), interpolation=cv2.INTER_LINEAR)
        out[k] = np.round(acc / len(arrs)).astype(np.uint8)
    return out


def load_gt(args, image_ids: set[str]) -> dict[str, list[Crop]]:
    """Ground-truth crops of the entries of ``image_ids``.

    With ``args.gt_cache`` set, the crops of *all* entries are read from that pickle, or
    built from the competition data and written there first (so later runs reuse them).
    """
    cache = Path(args.gt_cache) if getattr(args, "gt_cache", None) else None
    if cache is not None and cache.exists():
        with open(cache, "rb") as f:
            crops = pickle.load(f)
        return {k: v for k, v in crops.items() if k.split("-", 1)[1] in image_ids}
    data_dir = find_data_dir(args.data)
    entries = data.load_entries(data_dir / "train" / data.TRAIN_JSON)
    if cache is None:
        entries = [e for e in entries if e.image_id in image_ids]
    crops = {e.entry_id: [Crop.from_full(m) for m in e.masks()] for e in entries}
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        with open(cache.with_suffix(".tmp"), "wb") as f:
            pickle.dump(crops, f, protocol=pickle.HIGHEST_PROTOCOL)
        cache.with_suffix(".tmp").replace(cache)
    return {k: v for k, v in crops.items() if k.split("-", 1)[1] in image_ids}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--probs", nargs="+", type=Path, required=True)
    ap.add_argument("--data", default=None)
    ap.add_argument("--gt-cache", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--thresholds", type=float, nargs="+", default=[0.3, 0.35, 0.4, 0.45, 0.5, 0.55])
    ap.add_argument("--close", type=int, nargs="+", default=[0, 3])
    ap.add_argument("--merge", type=int, nargs="+", default=[0])
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    probs = load_probs(args.probs)
    gt = load_gt(args, set(probs))
    print(f"{len(probs)} images, {len(gt)} GT entries, {sum(map(len, gt.values()))} GT filaments", flush=True)

    rules = list(itertools.product([0, 100, 200, 300, 400, 500, 650, 800, 1000, 1200, 1600, 2000],
                                   [0.0, 0.5, 0.6, 0.7, 0.8, 0.9],
                                   [0.0, 0.4, 0.5, 0.6]))
    rows, best = [], None
    for thr, close, merge in itertools.product(args.thresholds, args.close, args.merge):
        shape = PostConfig(threshold=thr, close_radius=close, merge_distance=merge, min_area=0)
        cache = args.out / f"components_t{thr}_c{close}_m{merge}.pkl"  # checkpoint: one file per shape config
        if cache.exists():
            with open(cache, "rb") as f:
                cs = pickle.load(f)
        else:
            cs = components.build(probs, gt, shape, args.workers)
            with open(cache.with_suffix(".tmp"), "wb") as f:
                pickle.dump(cs, f, protocol=pickle.HIGHEST_PROTOCOL)
            cache.with_suffix(".tmp").replace(cache)
        for a, mx, mn in rules:
            pq, tp, fp, fn = cs.pq(cs.rule(a, mx, mn))
            rows.append(dict(threshold=thr, close=close, merge=merge, min_area=a, min_max_prob=mx,
                             min_mean_prob=mn, pq=pq, tp=tp, fp=fp, fn=fn))
        sub = pd.DataFrame(rows[-len(rules):]).sort_values("pq", ascending=False).iloc[0]
        oof, _ = components.fit_scorer(cs.table)
        lam, pq_learned = components.best_threshold(cs, oof)
        print(f"thr={thr} close={close} merge={merge}: candidates={len(cs.table)} "
              f"best rule PQ={sub.pq:.4f} (area>={sub.min_area}, max>={sub.min_max_prob}, mean>={sub.min_mean_prob}) "
              f"| learned scorer PQ={pq_learned:.4f} (score>{lam:.2f}) [{time.time() - t0:.0f}s]", flush=True)
        if best is None or sub.pq > best[0]:
            best = (sub.pq, sub, cs)

    grid = pd.DataFrame(rows).sort_values("pq", ascending=False)
    grid.to_csv(args.out / "rule_grid.csv", index=False)
    _, b, cs = best
    cfg = PostConfig(threshold=float(b.threshold), close_radius=int(b["close"]), merge_distance=int(b.merge),
                     min_area=int(b.min_area), min_max_prob=float(b.min_max_prob), min_mean_prob=float(b.min_mean_prob))
    (args.out / "best_post.json").write_text(json.dumps(asdict(cfg), indent=1))

    # error analysis of the best rule-based configuration, scored with the official code path
    s = Scorer(probs, gt, args.workers)
    try:
        preds = s.predict(cfg)
    finally:
        s.close()
    res, odf = evaluate(gt, preds)
    summ = analysis.summary(gt, preds, odf)
    print(f"\nbest rule config {asdict(cfg)}\n  -> {res}")
    print("GT filaments by area (px at 2048):\n", summ["gt_by_size"].to_string(index=False))
    print("Predictions by area:\n", summ["pred_by_size"].to_string(index=False))
    print("GT overlap degrees (1:n):", summ["gt_degrees"], " prediction degrees (n:1):", summ["pred_degrees"])
    print(f"mean IoU / Dice of overlapping pairs: {summ['mean_overlap_iou']:.3f} / {summ['mean_overlap_dice']:.3f}")
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
