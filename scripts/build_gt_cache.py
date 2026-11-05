"""Build cached ground truth for local evaluation.

Writes to ``--out``:
  * ``gt_train.csv``   -- official GT format (filament_id, segmentation_rle)
  * ``gt_crops.pkl``   -- entry_id -> list[Crop] for fast scoring

Usage:
  python scripts/build_gt_cache.py --data <MAGFiLO_1.0_Kaggle_2026 dir> --out <work dir>
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

from filseg import data, rle
from filseg.metrics import Crop


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    entries = data.load_entries(args.data / "train" / data.TRAIN_JSON)
    crops: dict[str, list[Crop]] = {}
    rows = []
    for e in entries:
        cs = []
        for k, m in enumerate(e.masks(), start=1):
            rows.append((f"{e.entry_id}_{k}", rle.encode(m)))
            cs.append(Crop.from_full(m))
        crops[e.entry_id] = cs
    import pandas as pd
    pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"]).to_csv(args.out / "gt_train.csv", index=False)
    with open(args.out / "gt_crops.pkl", "wb") as f:
        pickle.dump(crops, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"entries={len(crops)} filaments={len(rows)}")


if __name__ == "__main__":
    main()
