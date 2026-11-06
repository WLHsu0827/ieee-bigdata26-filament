"""Choose the post-processing by nested cross-validation on out-of-fold probability maps.

Candidate components are extracted once per *shape* configuration (threshold,
closing) and cached in ``--out`` (resumable); selection rules (minimum area,
peak and mean probability) are then evaluated exactly with the component
decomposition of PQ. The outer loop holds out one fold at a time, so the
reported ``nested_pq`` is not tuned on the images it is measured on.

Usage:
    python scripts/select_post.py --probs runs/*/val_probs.npz --gt-cache gt_crops.pkl --out sel/
Several files with the same images are averaged (ensemble); files of different folds are concatenated.
"""
from __future__ import annotations

import argparse
import itertools
import json
import pickle
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analyze_oof import load_gt, load_probs  # noqa: E402

from filseg import components, data  # noqa: E402
from filseg.postprocess import PostConfig  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--probs", nargs="+", type=Path, required=True)
    ap.add_argument("--data", default=None)
    ap.add_argument("--gt-cache", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--thresholds", type=float, nargs="+", default=[0.3, 0.35, 0.4, 0.45, 0.5])
    ap.add_argument("--close", type=int, nargs="+", default=[0, 4])
    ap.add_argument("--nfolds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    probs = load_probs(args.probs)
    gt = load_gt(args, set(probs))
    folds = data.image_folds(sorted({e.split("-", 1)[1] for e in gt} | set(probs)), args.nfolds, args.seed)
    if len({folds[i] for i in probs}) < 2:  # a single CV fold: nest over pseudo-folds of its images
        folds = data.image_folds(sorted(probs), args.nfolds, args.seed + 1)
    print(f"{len(probs)} images in folds {sorted({folds[i] for i in probs})}, {len(gt)} GT entries", flush=True)

    sets = {}
    for thr, close in itertools.product(args.thresholds, args.close):
        cache = args.out / f"components_t{thr}_c{close}.pkl"
        if cache.exists():
            with open(cache, "rb") as f:
                cs = pickle.load(f)
        else:
            cs = components.build(probs, gt, PostConfig(threshold=thr, close_radius=close, min_area=0), args.workers)
            with open(cache.with_suffix(".tmp"), "wb") as f:
                pickle.dump(cs, f, protocol=pickle.HIGHEST_PROTOCOL)
            cache.with_suffix(".tmp").replace(cache)
        sets[(thr, close)] = cs
        print(f"shape t={thr} close={close}: {len(cs.table)} candidates [{time.time() - t0:.0f}s]", flush=True)

    res = components.nested_select(sets, folds)
    thr, close = res["final_shape"]
    cfg = replace(PostConfig(threshold=thr, close_radius=close), **res["final_rule"])
    for c in res["fold_choices"]:
        print(f"outer fold {c['fold']}: shape {c['shape']} rule {c['rule']} inner {c['inner_pq']:.4f} "
              f"outer {c['outer_pq']:.4f}")
    print(f"nested PQ {res['nested_pq']:.4f} {res['nested_counts']} | final {asdict(cfg)} "
          f"(in-sample {res['final_pq_in_sample']:.4f}) [{time.time() - t0:.0f}s]")
    (args.out / "post_selected.json").write_text(json.dumps(asdict(cfg), indent=1))
    res["fold_choices"] = [{**c, "shape": list(c["shape"])} for c in res["fold_choices"]]
    res["final_shape"] = list(res["final_shape"])
    (args.out / "selection.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
