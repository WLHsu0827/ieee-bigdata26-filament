"""How noisy is a leaderboard score computed on a subset of the test images?

Scores post-processing configurations on out-of-fold probability maps per image (pooled over the
image's annotator sets: sum of matched IoU, TP, FP, FN) and draws random subsets of ``--subset``
images (the public split has about 90 of the 180 test images): the spread of the pooled PQ over the
subsets estimates the sampling noise of a public score, and the paired difference between two
configurations on the same subsets shows how often a better configuration scores lower.

Usage:
    python scripts/split_noise.py --probs "runs/k1/**/val_probs.npz" --gt-cache cache/gt_crops.pkl \
        --post configs/post_final_rule.json configs/post_final_full.json --out split_noise.json

Maps of the same fold (``fold<k>`` in the path) are averaged; folds are processed one at a time.
"""
from __future__ import annotations

import argparse
import itertools
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
sys.path.insert(0, str(ROOT))

from analyze_oof import load_gt, load_probs  # noqa: E402
from cli_utils import expand  # noqa: E402

from filseg.metrics import evaluate  # noqa: E402
from filseg.postprocess import PostConfig  # noqa: E402
from filseg.tune import Scorer  # noqa: E402


def per_image_counts(probs_files: list[Path], gt_cache: Path | None, data: str | None, cfgs: dict,
                     workers: int) -> dict[str, dict[str, list[float]]]:
    """``config -> image -> [sum of matched IoU, TP, FP, FN]`` (pooled over the image's annotator sets)."""
    by_fold: dict[str, list[Path]] = {}
    for p in probs_files:
        m = re.search(r"fold(\d+)", str(p))
        by_fold.setdefault(m.group(1) if m else "all", []).append(p)
    out: dict[str, dict[str, list[float]]] = {name: {} for name in cfgs}
    for fold, files in sorted(by_fold.items()):
        probs = load_probs(files)
        gt = load_gt(SimpleNamespace(gt_cache=gt_cache, data=data), set(probs))
        scorer = Scorer(probs, gt, workers=workers)
        try:
            for name, cfg in cfgs.items():
                res, odf = evaluate(gt, scorer.predict(cfg))
                per = odf.groupby("image_id")[["sum_iou", "tp", "fp", "fn"]].sum()
                out[name].update({i: [float(r.sum_iou), int(r.tp), int(r.fp), int(r.fn)] for i, r in per.iterrows()})
                print(f"fold {fold} ({len(files)} maps averaged, {len(probs)} images) {name}: {res}", flush=True)
        finally:
            scorer.close()
        del probs, gt
    return out


def pooled_pq(a: np.ndarray) -> float:
    s = a.sum(0)
    return float(s[0] / max(s[1] + 0.5 * s[2] + 0.5 * s[3], 1e-9))


def subset_stats(counts: dict, size: int, draws: int, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    names = list(counts)
    images = sorted(set.intersection(*(set(counts[n]) for n in names)))
    arrays = {n: np.array([counts[n][i] for i in images], np.float64) for n in names}
    idx = [rng.choice(len(images), size, replace=False) for _ in range(draws)]
    summary = {"images": len(images), "subset": size, "draws": draws, "configs": {}}
    for n, a in arrays.items():
        pqs = np.array([pooled_pq(a[i]) for i in idx])
        summary["configs"][n] = {"pq_all": pooled_pq(a), "subset_sd": float(pqs.std()),
                                 "subset_p05": float(np.percentile(pqs, 5)), "subset_p95": float(np.percentile(pqs, 95))}
    summary["paired"] = {}
    for x, y in itertools.combinations(names, 2):
        d = np.array([pooled_pq(arrays[y][i]) - pooled_pq(arrays[x][i]) for i in idx])
        summary["paired"][f"{y} - {x}"] = {"mean": float(d.mean()), "sd": float(d.std()),
                                          "p_negative": float((d < 0).mean())}
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--probs", nargs="+", required=True, help="val_probs.npz files or glob patterns")
    ap.add_argument("--post", nargs="+", type=Path, required=True)
    ap.add_argument("--gt-cache", type=Path, default=None)
    ap.add_argument("--data", default=None, help="competition data (when there is no --gt-cache)")
    ap.add_argument("--subset", type=int, default=90)
    ap.add_argument("--draws", type=int, default=4000)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    cfgs = {p.stem: PostConfig(**json.loads(p.read_text())) for p in args.post}
    counts = per_image_counts(expand(args.probs), args.gt_cache, args.data, cfgs, args.workers)
    summary = subset_stats(counts, args.subset, args.draws)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"summary": summary, "per_image": counts}))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
