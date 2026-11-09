"""Uncertainty of the nested-CV estimates of the two final configurations (rule A, learned filter B).

Uses the out-of-fold candidate table of the final detection shape (``scripts/extent_levels.py build``):
reproduces the nested decisions of the three-parameter rule and of the learned filter (objective PQ_r,
r = 0.77), then reports for each a 95 % bootstrap interval over images, the spread of the per-fold PQ
over the five outer folds, and the paired difference B - A.

Usage:
    python scripts/nested_uncertainty.py runs/levels.pkl --out docs/nested_uncertainty.json
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from extent_levels import Levels, nested, per_image_counts  # noqa: E402

from filseg.components import FEATURES, ComponentSet, nested_select  # noqa: E402
from filseg.postprocess import PostConfig  # noqa: E402


def rule_decisions(lv: Levels, r: float) -> tuple[np.ndarray, np.ndarray, dict]:
    """Keep mask of the nested three-parameter rule (outline 0.5) and the nested_select result."""
    t = lv.t.copy()
    t["v"], t["n_matched"] = t["v@0.50"], t["nm@0.50"]
    t["n_entries"] = np.where(t["a@0.50"].to_numpy() == 0, 0, t["n_entries"])
    cs = ComponentSet(t, sum(lv.gt.values()), PostConfig(threshold=0.25), dict(lv.gt))
    out = nested_select({"shape": cs}, lv.fold_of_image, r=r)
    keep = np.zeros(len(t), bool)
    for ch in out["fold_choices"]:
        rows = np.isin(lv.img, [i for i, f in lv.fold_of_image.items() if f == ch["fold"]])
        keep |= rows & cs.rule(**ch["rule"])
    return keep, np.full(len(t), lv.i05), out


def pq(x: np.ndarray) -> float:
    return float(x[0] / max(x[1] + 0.5 * x[2] + 0.5 * x[3], 1e-9))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("table", type=Path)
    ap.add_argument("--r", type=float, default=0.77)
    ap.add_argument("--draws", type=int, default=4000)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    with open(args.table, "rb") as fh:
        lv = Levels(pickle.load(fh))
    keep_a, lev_a, _ = rule_decisions(lv, args.r)
    learned = nested(lv, FEATURES, False, args.r, with_decisions=True)
    per = {"A_rule": per_image_counts(lv, keep_a, lev_a),
           "B_learned": per_image_counts(lv, learned["keep"], learned["lev"])}
    images = list(per["A_rule"].index)
    arrays = {k: v.loc[images, ["s", "tp", "fp", "fn"]].to_numpy(float) for k, v in per.items()}
    folds = np.array([lv.fold_of_image[i] for i in images])
    rng = np.random.default_rng(0)
    idx = [rng.integers(0, len(images), len(images)) for _ in range(args.draws)]
    res = {"images": len(images), "r": args.r, "draws": args.draws}
    for k, a in arrays.items():
        boot = np.array([pq(a[i].sum(0)) for i in idx])
        per_fold = [pq(a[folds == f].sum(0)) for f in sorted(set(folds))]
        res[k] = {"nested_pq": pq(a.sum(0)), "bootstrap_sd": float(boot.std()),
                  "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
                  "per_fold": [round(x, 4) for x in per_fold], "fold_mean": float(np.mean(per_fold)),
                  "fold_sd": float(np.std(per_fold, ddof=1))}
    d = np.array([pq(arrays["B_learned"][i].sum(0)) - pq(arrays["A_rule"][i].sum(0)) for i in idx])
    res["B_minus_A"] = {"diff": res["B_learned"]["nested_pq"] - res["A_rule"]["nested_pq"], "sd": float(d.std()),
                        "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
                        "p_le_0": float((d <= 0).mean())}
    print(json.dumps(res, indent=1))
    if args.out:
        args.out.write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    main()
