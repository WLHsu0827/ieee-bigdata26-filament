"""Score validation probability maps with one post-processing config and print diagnostics.

Prints the official pooled PQ, alternative aggregations (mean per annotator entry /
per image), PQ by number of annotators per image, and the error analysis.

Usage:
    python scripts/eval_probs.py --probs run/val_probs.npz --post run/post.json --gt-cache gt_crops.pkl
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analyze_oof import load_probs  # noqa: E402

from filseg import analysis  # noqa: E402
from filseg.metrics import evaluate  # noqa: E402
from filseg.postprocess import PostConfig  # noqa: E402
from filseg.tune import Scorer  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--probs", nargs="+", type=Path, required=True)
    ap.add_argument("--post", type=Path, required=True)
    ap.add_argument("--gt-cache", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()

    probs = load_probs(args.probs)
    with open(args.gt_cache, "rb") as f:
        gt = {k: v for k, v in pickle.load(f).items() if k.split("-", 1)[1] in probs}
    cfg = PostConfig(**json.loads(args.post.read_text()))
    s = Scorer(probs, gt, args.workers)
    try:
        preds = s.predict(cfg)
    finally:
        s.close()
    res, odf = evaluate(gt, preds)
    print(f"{len(probs)} images, {len(gt)} entries | {cfg}\n{res}")
    for k, v in analysis.aggregation_breakdown(odf).items():
        print(f"{k}: {v}")
    summ = analysis.summary(gt, preds, odf)
    print("GT by size:\n", summ["gt_by_size"].to_string(index=False))
    print("predictions by size:\n", summ["pred_by_size"].to_string(index=False))
    print("GT degrees (1:n):", summ["gt_degrees"], "| prediction degrees (n:1):", summ["pred_degrees"])


if __name__ == "__main__":
    main()
