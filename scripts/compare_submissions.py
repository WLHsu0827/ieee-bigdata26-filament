"""Agreement of two submission files (e.g. a re-run of the pipeline notebook vs. the submitted file).

Scores file B against file A as if A were the ground truth (official PQ), and reports how many
instances of A have a counterpart in B with IoU > 0.5 / 0.9 / 0.99. ``--b-images-only`` compares only the
images that have at least one instance in B: for a partial re-run, e.g. the notebook with
``FILSEG_LIMIT_TEST=2`` on a machine without a GPU (images without any instance in B are then skipped).

Usage:
    python scripts/compare_submissions.py a.csv b.csv
    python scripts/compare_submissions.py submissions/submission.csv submission.csv
    python scripts/compare_submissions.py --b-images-only submissions/submission.csv submission.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from filseg.metrics import evaluate, frame_to_crops  # noqa: E402

LEVELS = (0.5, 0.9, 0.99)


def agreement(a: pd.DataFrame, b: pd.DataFrame, b_images_only: bool = False) -> dict:
    """PQ of submission frame ``b`` scored against submission frame ``a`` as ground truth, and the fraction
    of the instances of ``a`` that have a counterpart in ``b`` with IoU above each of ``LEVELS``."""
    ca, cb = frame_to_crops(a), frame_to_crops(b)
    images = sorted(set(cb) if b_images_only else set(ca) | set(cb))
    res, odf = evaluate({f"000000-{i}": ca.get(i, []) for i in images}, {i: cb.get(i, []) for i in images})
    best = np.concatenate([m.max(axis=1) for m in odf["iou_matrix"] if m.size] or [np.zeros(0)])
    n_a = sum(len(ca.get(i, [])) for i in images)
    return {"images": len(images), "n_a": n_a, "n_b": sum(len(cb.get(i, [])) for i in images),
            "pq": res.pq, "sq": res.sq, "tp": res.tp, "fp": res.fp, "fn": res.fn,
            "matched": {t: float((best > t).sum() / max(n_a, 1)) for t in LEVELS}}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("a", type=Path, help="reference submission, scored as ground truth")
    ap.add_argument("b", type=Path, help="submission to check")
    ap.add_argument("--b-images-only", action="store_true",
                    help="compare only the images with at least one instance in B (partial re-run)")
    args = ap.parse_args()
    r = agreement(pd.read_csv(args.a), pd.read_csv(args.b), args.b_images_only)
    print(f"{r['images']} images compared | A: {r['n_a']} instances | B: {r['n_b']} instances")
    print(f"PQ(B vs A) = {r['pq']:.4f} (SQ {r['sq']:.4f}, TP {r['tp']}, FP {r['fp']}, FN {r['fn']})")
    for t, frac in r["matched"].items():
        print(f"instances of A with a counterpart in B at IoU > {t}: {frac:.4f}")


if __name__ == "__main__":
    main()
