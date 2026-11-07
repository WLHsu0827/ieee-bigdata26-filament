"""Fit the learned component filter for the final post-processing.

Input: the candidate components of one *shape* configuration on all out-of-fold images (a
``components_*.pkl`` written by ``scripts/select_post.py``). The keep threshold is chosen on
out-of-fold scores (GroupKFold by image) for the objective ``pq_sparse`` with label rate ``--r``;
the final tree ensemble is fitted on all candidates and stored, with the shape, in a self-contained
post-processing JSON (``PostConfig.scorer`` / ``min_score``) that ``filseg.postprocess`` applies.

With ``--probs`` (the same out-of-fold maps) the written configuration is also run through the
official evaluation path, which must reproduce the component-table PQ of the final model exactly.

Usage:
    python scripts/fit_scorer.py --components sel/components_t0.25_c0_o0.5.pkl --r 0.77 \
        --out sel/post_learned_r0.77.json [--probs runs/**/val_probs.npz --gt-cache gt_crops.pkl]
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from filseg import components, data  # noqa: E402
from filseg.components import FEATURES, LEARNED_PARAMS, pq_sparse  # noqa: E402
from filseg.postprocess import PostConfig  # noqa: E402
from filseg.scorer import export_hgb, predict  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--components", type=Path, required=True)
    ap.add_argument("--r", type=float, default=1.0)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--probs", nargs="*", default=[], help="out-of-fold maps for the end-to-end check")
    ap.add_argument("--gt-cache", type=Path, default=None)
    ap.add_argument("--data", default=None)
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.model_selection import GroupKFold

    with open(args.components, "rb") as f:
        cs = pickle.load(f)
    t = cs.table.reset_index(drop=True)
    feats = [k for k in FEATURES if k in t.columns]
    x = t[feats].to_numpy(np.float64)
    y = (t["v"] / t["n_entries"].clip(lower=1)).to_numpy()
    groups = t["image_id"].to_numpy()
    oof = np.zeros(len(t))
    for a, b in GroupKFold(5).split(x, y, groups):
        oof[b] = HistGradientBoostingRegressor(**LEARNED_PARAMS).fit(x[a], y[a]).predict(x[b])
    grid = np.linspace(0.05, 0.6, 56)
    res = [(pq_sparse(*cs.counts(oof > lam), r=args.r), float(lam)) for lam in grid]
    obj, lam = max(res)
    model = HistGradientBoostingRegressor(**LEARNED_PARAMS).fit(x, y)
    spec = export_hgb(model, feats)
    assert np.allclose(predict(spec, x), model.predict(x), atol=1e-9), "exported trees differ from sklearn"
    in_sample = cs.counts(predict(spec, x) > lam)
    folds = data.image_folds(sorted(cs.gt_per_image), 5, args.seed)
    nested = components.nested_learned(cs, folds, r=args.r)
    print(f"{len(t)} candidates on {len(cs.gt_per_image)} images, {len(feats)} features")
    print(f"out-of-fold threshold {lam:.3f} (objective {obj:.4f}, PQ {pq_sparse(*cs.counts(oof > lam)):.4f}); "
          f"nested PQ {nested['nested_pq']:.4f}; final model in-sample PQ {pq_sparse(*in_sample):.4f} "
          f"(TP {in_sample[1]}, FP {in_sample[2]}, FN {in_sample[3]})")

    s = cs.shape_cfg
    cfg = PostConfig(threshold=s.threshold, close_radius=s.close_radius, fill_holes=s.fill_holes, min_area=0,
                     min_mean_prob=0.0, min_max_prob=0.0, merge_distance=s.merge_distance, dilate_radius=0,
                     mask_threshold=s.mask_threshold, scorer=spec, min_score=lam)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(asdict(cfg)))
    print(f"wrote {args.out} ({args.out.stat().st_size / 1e3:.0f} kB)")

    if args.probs:
        from analyze_oof import load_gt, load_probs
        from cli_utils import expand

        from filseg.metrics import evaluate
        from filseg.tune import Scorer

        probs = load_probs(expand(args.probs))
        gt = load_gt(args, set(probs))
        sc = Scorer(probs, gt, args.workers)
        try:
            out, _ = evaluate(gt, sc.predict(PostConfig(**json.loads(args.out.read_text()))))
        finally:
            sc.close()
        # candidates rebuilt here: features computed on another machine can differ in the last float
        # digits, which can move a candidate lying exactly at a tree split across it
        local = components.build(probs, gt, replace(cfg, scorer=None, min_score=0.0), args.workers)
        lt = local.table.reset_index(drop=True)
        expect = pq_sparse(*local.counts(predict(spec, lt[feats].to_numpy(np.float64)) > lam))
        print(f"end-to-end check on {len(probs)} images: official PQ {out.pq:.4f} (TP {out.tp}, FP {out.fp}, "
              f"FN {out.fn}) vs component table {expect:.4f} -> {'OK' if abs(out.pq - expect) < 1e-6 else 'MISMATCH'}")
        if abs(out.pq - expect) >= 1e-6:
            sys.exit(1)


if __name__ == "__main__":
    main()
