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

from . import data, submission
from .model import load_checkpoint, predict_images, upsample
from .postprocess import PostConfig, instances_from_prob


def load_models(paths: list[Path], device: torch.device) -> list[tuple[torch.nn.Module, int, str]]:
    """Checkpoints -> ``(network, input resolution, normalisation)`` tuples."""
    models = []
    for p in paths:
        m, meta = load_checkpoint(p, device)
        models.append((m, int(meta["res"]), meta.get("norm", "global")))
    return models


def predict_full(models: list[tuple[torch.nn.Module, int, str]], images: list[np.ndarray],
                 device: torch.device, tta: int = 4, batch: int = 4) -> list[np.ndarray]:
    """Average full-resolution probability maps of all models (grouped by resolution and normalisation)."""
    groups: dict[tuple[int, str], list[torch.nn.Module]] = {}
    for m, r, norm in models:
        groups.setdefault((r, norm), []).append(m)
    acc = [np.zeros(im.shape, np.float32) for im in images]
    for (r, norm), ms in groups.items():
        probs = predict_images(ms, images, r, device, tta=tta, batch=batch, norm=norm)
        for k, p in enumerate(probs):
            acc[k] += upsample(p) * len(ms)
    return [a / len(models) for a in acc]


def make_submission(models, data_dir: Path, cfg: PostConfig, device: torch.device,
                    tta: int = 4, batch: int = 4, chunk: int = 8, save_probs: Path | None = None,
                    timing: dict | None = None) -> pd.DataFrame:
    """Predict all test images and return the submission frame.

    ``save_probs``: also store the ensemble probability maps (uint8, x255, 2048 px) as an
    ``.npz`` file so other post-processing rules can be applied without re-running the networks
    (``scripts/ensemble_submit.py``). ``timing``: filled with seconds per image (network,
    post-processing).
    """
    import time

    files = sorted((data_dir / "test" / "test_images").glob("*.jp*g"))
    rows, maps = [], {}
    t_net = t_post = 0.0
    for s in range(0, len(files), chunk):
        part = files[s:s + chunk]
        t0 = time.time()
        probs = predict_full(models, [data.read_image(f) for f in part], device, tta, batch)
        t1 = time.time()
        for f, p in zip(part, probs):
            if save_probs is not None:
                maps[f.stem] = np.round(np.clip(p, 0, 1) * 255).astype(np.uint8)
            lab, _ = instances_from_prob(p, cfg)
            rows += submission.rows_from_label(f.stem, lab)
        t_net, t_post = t_net + t1 - t0, t_post + time.time() - t1
    if save_probs is not None:
        np.savez_compressed(save_probs, **maps)
    if timing is not None:
        n = max(len(files), 1)
        timing.update(device=str(device), images=len(files), models=len(models), tta=tta,
                      network_s_per_image=t_net / n, post_s_per_image=t_post / n)
    return pd.DataFrame(rows, columns=submission.COLUMNS)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=None, type=Path, help="MAGFiLO folder (default: found under /kaggle/input)")
    ap.add_argument("--weights", nargs="+", required=True, type=Path)
    ap.add_argument("--post", required=True, type=Path)
    ap.add_argument("--out", default=Path("submission.csv"), type=Path)
    ap.add_argument("--tta", type=int, default=4)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--save-probs", type=Path, default=None, help="also save the test probability maps (.npz)")
    args = ap.parse_args(argv)

    from .train import find_data_dir

    args.data = find_data_dir(args.data)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = PostConfig(**json.loads(args.post.read_text()))
    models = load_models(args.weights, device)
    timing: dict = {}
    df = make_submission(models, args.data, cfg, device, args.tta, args.batch, save_probs=args.save_probs,
                         timing=timing)
    test_ids = {p.stem for p in (args.data / "test" / "test_images").glob("*.jp*g")}
    df, report = submission.write(list(df.itertuples(index=False, name=None)), args.out, test_ids)
    print(f"{report['images']} images, {len(df)} filaments -> {args.out}; check: {report}")
    print(f"time per image: network {timing['network_s_per_image']:.2f} s + post-processing "
          f"{timing['post_s_per_image']:.2f} s ({timing['device']}, {timing['models']} model(s), TTA {args.tta})")
    (args.out.parent / "timing.json").write_text(json.dumps(timing, indent=1))


if __name__ == "__main__":
    main()
