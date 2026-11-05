"""Inference: load trained checkpoints, predict test images, write the submission.

Usage:
    python -m filseg.infer --data /path/MAGFiLO_1.0_Kaggle_2026 \
        --weights weights/model_a.pt weights/model_b.pt --post weights/post.json \
        --out submission.csv

Each checkpoint stores its architecture, encoder and input resolution; models
of different resolutions are ensembled by averaging full-resolution
(2048 x 2048) probability maps.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import data, rle
from .model import load_checkpoint, predict_images, upsample
from .postprocess import PostConfig, instances_from_prob, label_to_masks


def load_models(paths: list[Path], device: torch.device) -> list[tuple[torch.nn.Module, int]]:
    models = []
    for p in paths:
        m, meta = load_checkpoint(p, device)
        models.append((m, int(meta["res"])))
    return models


def predict_full(models: list[tuple[torch.nn.Module, int]], images: list[np.ndarray],
                 device: torch.device, tta: int = 4, batch: int = 4) -> list[np.ndarray]:
    """Average full-resolution probability maps of all models (grouped by resolution)."""
    by_res: dict[int, list[torch.nn.Module]] = {}
    for m, r in models:
        by_res.setdefault(r, []).append(m)
    acc = [np.zeros(im.shape, np.float32) for im in images]
    for r, ms in by_res.items():
        probs = predict_images(ms, images, r, device, tta=tta, batch=batch)
        for k, p in enumerate(probs):
            acc[k] += upsample(p) * len(ms)
    return [a / len(models) for a in acc]


def make_submission(models, data_dir: Path, cfg: PostConfig, device: torch.device,
                    tta: int = 4, batch: int = 4, chunk: int = 8) -> pd.DataFrame:
    files = sorted((data_dir / "test" / "test_images").glob("*.jp*g"))
    rows = []
    for s in range(0, len(files), chunk):
        part = files[s:s + chunk]
        probs = predict_full(models, [data.read_image(f) for f in part], device, tta, batch)
        for f, p in zip(part, probs):
            lab, _ = instances_from_prob(p, cfg)
            for k, m in enumerate(label_to_masks(lab), start=1):
                rows.append((f"{f.stem}_{k}", rle.encode(m)))
    return pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--weights", nargs="+", required=True, type=Path)
    ap.add_argument("--post", required=True, type=Path)
    ap.add_argument("--out", default=Path("submission.csv"), type=Path)
    ap.add_argument("--tta", type=int, default=4)
    ap.add_argument("--batch", type=int, default=4)
    args = ap.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = PostConfig(**json.loads(args.post.read_text()))
    models = load_models(args.weights, device)
    df = make_submission(models, args.data, cfg, device, args.tta, args.batch)
    df.to_csv(args.out, index=False)
    print(f"{df['filament_id'].str.rsplit('_', n=1).str[0].nunique()} images, {len(df)} filaments -> {args.out}")


if __name__ == "__main__":
    main()
