"""Learned component filter: a gradient-boosted tree ensemble stored as plain JSON.

The model predicts, for every candidate component, its expected matched IoU per annotator
entry (see :mod:`filseg.components`); components scoring above a threshold are kept. It is
trained with scikit-learn's ``HistGradientBoostingRegressor`` and exported to a dictionary of
node arrays, so inference needs only numpy and does not depend on the scikit-learn version.
"""
from __future__ import annotations

import numpy as np

NODE_FIELDS = ("value", "feature_idx", "num_threshold", "missing_go_to_left", "left", "right", "is_leaf")


def export_hgb(model, features: list[str]) -> dict:
    """``HistGradientBoostingRegressor`` -> JSON-serialisable dictionary."""
    if getattr(model, "_preprocessor", None) is not None or getattr(model, "is_categorical_", None) is not None:
        raise ValueError("categorical features are not supported")
    trees = []
    for predictors in model._predictors:  # one tree per boosting iteration (regression)
        nodes = predictors[0].nodes
        trees.append({k: nodes[k].tolist() for k in NODE_FIELDS})
    return {"kind": "hgb_regressor", "features": list(features),
            "baseline": float(np.ravel(model._baseline_prediction)[0]), "trees": trees}


def predict(spec: dict, x: np.ndarray) -> np.ndarray:
    """Evaluate an exported tree ensemble on ``x`` (n_samples, n_features in ``spec["features"]`` order)."""
    x = np.asarray(x, np.float64)
    out = np.full(len(x), spec["baseline"], np.float64)
    if len(x) == 0:
        return out
    rows = np.arange(len(x))
    for t in spec["trees"]:
        value = np.asarray(t["value"], np.float64)
        feat = np.asarray(t["feature_idx"], np.int64)
        thr = np.asarray(t["num_threshold"], np.float64)
        miss_left = np.asarray(t["missing_go_to_left"], bool)
        left, right = np.asarray(t["left"], np.int64), np.asarray(t["right"], np.int64)
        leaf = np.asarray(t["is_leaf"], bool)
        node = np.zeros(len(x), np.int64)
        active = ~leaf[node]
        while active.any():
            n = node[active]
            xv = x[rows[active], feat[n]]
            go_left = (xv <= thr[n]) | (np.isnan(xv) & miss_left[n])
            node[active] = np.where(go_left, left[n], right[n])
            active = ~leaf[node]
        out += value[node]
    return out
