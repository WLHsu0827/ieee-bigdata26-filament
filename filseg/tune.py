"""Validation-time tuning of the post-processing on predicted probability maps."""
from __future__ import annotations

import itertools
import multiprocessing as mp
import os
from dataclasses import asdict, replace

import numpy as np

from .metrics import Crop, PQResult, evaluate
from .model import upsample
from .postprocess import PostConfig, instances_from_prob, label_to_crops

_SHARED: dict = {}  # probability maps shared with forked workers (copy-on-write)


def _instances(job):
    image_id, cfg = job
    lab, _ = instances_from_prob(upsample(_SHARED["probs"][image_id]), cfg)
    return image_id, label_to_crops(lab)


class Scorer:
    """Scores post-processing configs on fixed probability maps, optionally in parallel."""

    def __init__(self, probs: dict[str, np.ndarray], gt: dict[str, list[Crop]], workers: int | None = None):
        ids = set(probs)
        self.probs = probs
        self.gt = {k: v for k, v in gt.items() if k.split("-", 1)[1] in ids}
        workers = workers if workers is not None else (os.cpu_count() or 1)
        self.pool = None
        if workers > 1 and "fork" in mp.get_all_start_methods():
            _SHARED["probs"] = probs
            self.pool = mp.get_context("fork").Pool(workers)

    def predict(self, cfg: PostConfig) -> dict[str, list[Crop]]:
        jobs = [(k, cfg) for k in self.probs]
        if self.pool is not None:
            return dict(self.pool.map(_instances, jobs, chunksize=4))
        _SHARED["probs"] = self.probs
        return dict(map(_instances, jobs))

    def __call__(self, cfg: PostConfig) -> PQResult:
        res, _ = evaluate(self.gt, self.predict(cfg))
        return res

    def close(self):
        if self.pool is not None:
            self.pool.close()
            self.pool.join()
            self.pool = None


def score(probs: dict[str, np.ndarray], gt: dict[str, list[Crop]], cfg: PostConfig,
          workers: int | None = None) -> PQResult:
    """PQ of post-processed ``probs`` (image_id -> prob map) on the GT entries of those images."""
    s = Scorer(probs, gt, workers)
    try:
        return s(cfg)
    finally:
        s.close()


def grid_search(probs: dict[str, np.ndarray], gt: dict[str, list[Crop]], base: PostConfig = PostConfig(),
                grid: dict[str, list] | None = None, log=print,
                workers: int | None = None) -> tuple[PostConfig, PQResult, list]:
    grid = grid or {"threshold": [0.35, 0.45, 0.55], "min_area": [100, 250, 500], "close_radius": [0, 4]}
    keys = list(grid)
    best_cfg, best_res, table = base, None, []
    s = Scorer(probs, gt, workers)
    try:
        for values in itertools.product(*(grid[k] for k in keys)):
            cfg = replace(base, **dict(zip(keys, values)))
            res = s(cfg)
            table.append((asdict(cfg), res.pq, res.sq, res.rq, res.tp, res.fp, res.fn))
            log(f"  {dict(zip(keys, values))} -> {res}")
            if best_res is None or res.pq > best_res.pq:
                best_cfg, best_res = cfg, res
    finally:
        s.close()
    return best_cfg, best_res, table
