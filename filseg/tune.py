"""Validation-time tuning of the post-processing on predicted probability maps."""
from __future__ import annotations

import itertools
import multiprocessing as mp
import os
from dataclasses import asdict, replace

import numpy as np

from .metrics import Crop, PQResult, evaluate
from .postprocess import PostConfig, instances_from_prob, label_to_crops, upsample


def _instances(job):
    image_id, prob, cfg = job
    lab, _ = instances_from_prob(upsample(prob), cfg)
    return image_id, label_to_crops(lab)


def _pool(workers: int):
    """Worker pool started from a fresh process (never forks a CUDA / multi-threaded parent)."""
    method = "forkserver" if "forkserver" in mp.get_all_start_methods() else "spawn"
    return mp.get_context(method).Pool(workers)


class Scorer:
    """Scores post-processing configs on fixed probability maps, optionally in parallel."""

    def __init__(self, probs: dict[str, np.ndarray], gt: dict[str, list[Crop]], workers: int | None = None):
        ids = set(probs)
        self.probs = probs
        self.gt = {k: v for k, v in gt.items() if k.split("-", 1)[1] in ids}
        workers = workers if workers is not None else (os.cpu_count() or 1)
        self.pool = _pool(workers) if workers > 1 and len(probs) > 1 else None

    def predict(self, cfg: PostConfig) -> dict[str, list[Crop]]:
        jobs = [(k, p, cfg) for k, p in self.probs.items()]
        if self.pool is not None:
            return dict(self.pool.imap_unordered(_instances, jobs, chunksize=2))
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
