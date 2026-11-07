"""Agreement of two submission files (e.g. a re-run of the pipeline notebook vs. the submitted file).

Scores file B against file A as if A were the ground truth (official PQ), and reports how many
instances of A have a counterpart in B with IoU > 0.5 / 0.9 / 0.99.

Usage:
    python scripts/compare_submissions.py a.csv b.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from filseg.metrics import evaluate, frame_to_crops  # noqa: E402


def main() -> None:
    a, b = (frame_to_crops(pd.read_csv(p)) for p in sys.argv[1:3])
    images = sorted(set(a) | set(b))
    res, odf = evaluate({f"000000-{i}": a.get(i, []) for i in images}, {i: b.get(i, []) for i in images})
    best = np.concatenate([m.max(axis=1) for m in odf["iou_matrix"] if m.size] or [np.zeros(0)])
    n_a = sum(len(v) for v in a.values())
    print(f"A: {n_a} instances on {len(a)} images | B: {sum(len(v) for v in b.values())} on {len(b)} images")
    print(f"PQ(B vs A) = {res.pq:.4f} (SQ {res.sq:.4f}, TP {res.tp}, FP {res.fp}, FN {res.fn})")
    for t in (0.5, 0.9, 0.99):
        print(f"instances of A with a counterpart in B at IoU > {t}: {(best > t).sum() / max(n_a, 1):.4f}")


if __name__ == "__main__":
    main()
