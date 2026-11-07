"""Compare models / ensembles on the same out-of-fold images (ablation tables).

Each ``--group NAME=PATTERN[,PATTERN...]`` averages the probability maps of its files (an
ensemble when several models share images). Every group is scored on the images common to all
groups with (a) a fixed post-processing JSON and (b) nested cross-validation of the
post-processing (5 image folds; shapes from ``--thresholds`` x ``--close`` x ``--mask`` and the
rule grid of :mod:`filseg.components`), so models with different calibration are compared fairly.

Usage:
    python scripts/compare_models.py --gt-cache gt_crops.pkl --post configs/post_c1_nested.json \
        --group "teacher_v2s=/kaggle/input/**/v2s_2048/fold0/val_probs.npz" \
        --group "student_v2s=/kaggle/input/**/st_v2s/fold0/val_probs.npz" --out cmp.json
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_oof import load_gt, load_probs  # noqa: E402
from cli_utils import expand  # noqa: E402

from filseg import components, data  # noqa: E402
from filseg.metrics import evaluate  # noqa: E402
from filseg.postprocess import PostConfig  # noqa: E402
from filseg.tune import Scorer  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--group", action="append", required=True, help="NAME=PATTERN[,PATTERN...]")
    ap.add_argument("--post", type=Path, required=True, help="fixed post-processing JSON")
    ap.add_argument("--data", default=None)
    ap.add_argument("--gt-cache", type=Path, default=None)
    ap.add_argument("--thresholds", type=float, nargs="+", default=[0.3, 0.35, 0.4])
    ap.add_argument("--close", type=int, nargs="+", default=[4])
    ap.add_argument("--mask", type=float, nargs="+", default=[0.0, 0.5])
    ap.add_argument("--r", type=float, default=1.0, help="label rate of the nested objective")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--out", type=Path, default=None, help="JSON with all results")
    args = ap.parse_args()
    t0 = time.time()

    groups = {}
    for g in args.group:
        name, pats = g.split("=", 1)
        groups[name] = expand(pats.split(","))
    probs = {name: load_probs(files) for name, files in groups.items()}
    common = set.intersection(*(set(p) for p in probs.values()))
    probs = {name: {k: v for k, v in p.items() if k in common} for name, p in probs.items()}
    gt = load_gt(args, common)
    folds = data.image_folds(sorted(common), 5, args.seed)
    fixed = PostConfig(**json.loads(args.post.read_text()))
    print(f"{len(common)} common images, {len(gt)} GT entries; groups: "
          + ", ".join(f"{n} ({len(f)} file(s))" for n, f in groups.items()), flush=True)

    results = {}
    for name, p in probs.items():
        s = Scorer(p, gt, args.workers)
        try:
            res, _ = evaluate(gt, s.predict(fixed))
        finally:
            s.close()
        sets = {}
        for thr, close, mask in itertools.product(args.thresholds, args.close, args.mask):
            if 0 < mask <= thr:
                continue
            sets[(thr, close, 0, mask)] = components.build(
                p, gt, PostConfig(threshold=thr, close_radius=close, mask_threshold=mask, min_area=0), args.workers)
        nest = components.nested_select(sets, folds, r=args.r)
        results[name] = {"files": [str(f) for f in groups[name]], "fixed_pq": res.pq, "fixed_sq": res.sq,
                         "fixed_rq": res.rq, "fixed_counts": [res.tp, res.fp, res.fn],
                         "nested_pq": nest["nested_pq"], "nested_counts": nest["nested_counts"],
                         "final_shape": list(nest["final_shape"]), "final_rule": nest["final_rule"],
                         "final_pq_in_sample": nest["final_pq_in_sample"]}
        print(f"{name:24s} fixed rule PQ {res.pq:.4f} (SQ {res.sq:.3f} RQ {res.rq:.3f}) | nested PQ "
              f"{nest['nested_pq']:.4f} | in-sample best {nest['final_pq_in_sample']:.4f} "
              f"[{time.time() - t0:.0f}s]", flush=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"images": len(common), "post": str(args.post), "r": args.r,
                                        "results": results}, indent=1))


if __name__ == "__main__":
    main()
