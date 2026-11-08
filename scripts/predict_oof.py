"""Re-predict the out-of-fold probability maps of the committed networks with flip TTA (x4) and
flip x rot90 TTA (x8) in one pass, for a paired comparison of the two.

For every fold k, each network of ``manifest.json`` that was trained with fold k held out predicts
the fold's validation images (the training split: ``filseg.data.image_folds`` with the manifest's
number of folds and seed). The four rot90 passes of TTA x8 are computed on the rotated image and
rotated back, so the x4 maps are a by-product of the x8 maps and both come from the same weights.
Output, in the layout of the training runs (resumable per network):
``<out>/tta4/<encoder>_<res>/fold<k>/val_probs.npz`` and ``<out>/tta8/...`` (uint8, probability x 255).

Usage:
    python scripts/predict_oof.py --manifest weights/manifest.json --out runs/oof_tta [--folds 0 1] \
        [--data MAGFiLO_1.0_Kaggle_2026]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from cli_utils import expand  # noqa: E402

from filseg import data  # noqa: E402
from filseg.model import load_checkpoint, predict_images  # noqa: E402
from filseg.train import find_data_dir  # noqa: E402


def to_u8(p: np.ndarray) -> np.ndarray:
    """Probability map -> uint8 (x 255), the storage format of ``val_probs.npz``."""
    return np.round(np.clip(np.nan_to_num(p.astype(np.float32), nan=0.0), 0.0, 1.0) * 255).astype(np.uint8)


def predict_tta4_tta8(model: torch.nn.Module, images: list[np.ndarray], res: int, norm: str,
                      device: torch.device, batch: int) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Flip TTA (x4) maps and dihedral TTA (x8) maps of the same network (float32 at ``res``)."""
    p4 = predict_images([model], images, res, device, tta=4, batch=batch, norm=norm)
    rotated = [np.ascontiguousarray(np.rot90(im)) for im in images]
    pr = predict_images([model], rotated, res, device, tta=4, batch=batch, norm=norm)
    maps4 = [p.astype(np.float32) for p in p4]
    maps8 = [(a + np.rot90(b.astype(np.float32), -1)) / 2 for a, b in zip(maps4, pr, strict=True)]
    return maps4, maps8


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default=str(ROOT / "weights" / "manifest.json"), help="manifest.json (file or glob)")
    ap.add_argument("--data", default=None, help="competition data (found automatically on Kaggle)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--folds", type=int, nargs="*", default=None, help="default: all folds of the manifest")
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--chunk", type=int, default=16, help="images predicted (and held in memory) at a time")
    ap.add_argument("--limit", type=int, default=0, help="first n validation images per fold only (smoke tests)")
    args = ap.parse_args()

    manifest_path = expand([args.manifest])[0]
    manifest = json.loads(Path(manifest_path).read_text())
    data_dir = find_data_dir(args.data)
    groups = data.group_by_image(data.load_entries(data_dir / "train" / data.TRAIN_JSON))
    folds = data.image_folds(list(groups), int(manifest.get("nfolds", 5)), int(manifest.get("seed", 2026)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    todo = sorted({int(m["fold"]) for m in manifest["models"]}) if args.folds is None else args.folds
    for k in todo:
        ids = [i for i in groups if folds[i] == k]
        if args.limit:
            ids = ids[:args.limit]
        for entry in (m for m in manifest["models"] if int(m["fold"]) == k):
            outs = {t: args.out / t / f"{entry['encoder']}_{entry['res']}" / f"fold{k}" / "val_probs.npz"
                    for t in ("tta4", "tta8")}
            if all(p.exists() for p in outs.values()):
                print(f"fold {k} {entry['file']}: done", flush=True)
                continue
            model, meta = load_checkpoint(Path(manifest_path).parent / entry["file"], device)
            res, norm = int(meta["res"]), meta.get("norm", "global")
            maps = {"tta4": {}, "tta8": {}}
            t0 = time.time()
            for s in range(0, len(ids), args.chunk):
                part = ids[s:s + args.chunk]
                images = [data.read_image(data_dir / "train" / "train_images" / groups[i][0].file_name) for i in part]
                m4, m8 = predict_tta4_tta8(model, images, res, norm, device, args.batch)
                for i, a, b in zip(part, m4, m8, strict=True):
                    maps["tta4"][i], maps["tta8"][i] = to_u8(a), to_u8(b)
            for t, path in outs.items():
                path.parent.mkdir(parents=True, exist_ok=True)
                tmp = path.with_name("val_probs.tmp.npz")
                np.savez_compressed(tmp, **maps[t])
                os.replace(tmp, path)
            dt = time.time() - t0
            print(f"fold {k} {entry['file']}: {len(ids)} images, {dt / len(ids):.2f} s per image (TTA x8)", flush=True)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
