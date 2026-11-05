"""Traditional (non-learning) baseline: relative-darkness thresholding.

Tunes the darkness threshold and the post-processing on a subset of training
images with the official PQ definition, then writes a test submission.

Usage:
    python scripts/classical_baseline.py --data <MAGFiLO dir> --out <dir> [--n-tune 80]
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from filseg import classical, data, rle  # noqa: E402
from filseg.metrics import Crop  # noqa: E402
from filseg.postprocess import PostConfig, instances_from_prob, label_to_masks  # noqa: E402
from filseg.tune import grid_search  # noqa: E402


def _dark(path):
    return classical.relative_darkness(data.read_image(path)).astype(np.float16)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--n-tune", type=int, default=80)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data) if args.data else next(Path("/kaggle/input").rglob(data.TRAIN_JSON)).parent.parent

    groups = data.group_by_image(data.load_entries(data_dir / "train" / data.TRAIN_JSON))
    folds = data.image_folds(list(groups))
    tune_ids = [i for i in groups if folds[i] == 0][:args.n_tune]
    gt = {e.entry_id: [Crop.from_full(m) for m in e.masks()] for i in tune_ids for e in groups[i]}
    paths = [data_dir / "train" / "train_images" / groups[i][0].file_name for i in tune_ids]
    with Pool(args.workers) as pool:
        darks = dict(zip(tune_ids, pool.map(_dark, paths)))

    grid = {"threshold": [0.06, 0.08, 0.10, 0.12], "min_area": [150, 400, 800], "close_radius": [0, 5]}
    cfg, best, _ = grid_search(darks, gt, PostConfig(), grid)
    print(f"best {asdict(cfg)} -> {best}", flush=True)
    (args.out / "post_classical.json").write_text(json.dumps(asdict(cfg), indent=1))

    files = sorted((data_dir / "test" / "test_images").glob("*.jp*g"))
    rows = []
    with Pool(args.workers) as pool:
        for f, d in zip(files, pool.imap(_dark, files, chunksize=4)):
            lab, _ = instances_from_prob(d.astype(np.float32), cfg)
            rows += [(f"{f.stem}_{k}", rle.encode(m)) for k, m in enumerate(label_to_masks(lab), start=1)]
    pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"]).to_csv(args.out / "submission.csv", index=False)
    print(f"test: {len(files)} images, {len(rows)} filaments", flush=True)


if __name__ == "__main__":
    main()
