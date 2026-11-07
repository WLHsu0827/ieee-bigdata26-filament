"""Teacher pseudo-labels for unlabelled images (semi-supervised training / distillation).

Averages the full-resolution probability maps of the given checkpoints (with flip TTA) and
writes one lossless 8-bit PNG per image (probability x 255) to ``--out``. Images can be split
over several processes (``--shard k/n``, e.g. one per GPU); with ``--wait`` a process returns
only when every image has its map, so a following training step sees the complete set.

To keep cross-validation honest, the teacher of a fold must be a model that did not train on
that fold's validation images (e.g. the fold's own CV models).

Usage:
    python scripts/pseudo_label.py --images "gong/images/*.jpeg" --weights "runs/*/fold0/model.pt" \
        --out pl/fold0 --tta 2 --shard 0/2 --wait
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cli_utils import expand  # noqa: E402

from filseg import data  # noqa: E402
from filseg.infer import load_models, predict_full  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", nargs="+", required=True, help="image files or glob patterns")
    ap.add_argument("--weights", nargs="+", required=True, help="checkpoints or glob patterns")
    ap.add_argument("--out", type=Path, required=True, help="output folder of <image stem>.png maps")
    ap.add_argument("--tta", type=int, default=2)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--shard", default="0/1", help="k/n: process every n-th image starting at k")
    ap.add_argument("--wait", action="store_true", help="return only when all images have a map")
    ap.add_argument("--wait-hours", type=float, default=3.0)
    args = ap.parse_args()

    files = sorted(set(expand(args.images)), key=lambda p: Path(p).name)
    weights = expand(args.weights)
    if not files or not weights:
        raise SystemExit(f"{len(files)} images, {len(weights)} checkpoints: nothing to do")
    k, n = (int(x) for x in args.shard.split("/"))
    mine = files[k::n]
    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = load_models([Path(w) for w in weights], device)
    print(f"shard {k}/{n}: {len(mine)} of {len(files)} images, {len(models)} model(s), TTA {args.tta}, {device}",
          flush=True)
    t0, done = time.time(), 0
    for s in range(0, len(mine), 8):
        part = [Path(f) for f in mine[s:s + 8] if not (args.out / f"{Path(f).stem}.png").exists()]  # resumable
        if not part:
            continue
        probs = predict_full(models, [data.read_image(f) for f in part], device, args.tta, args.batch)
        for f, p in zip(part, probs):
            tmp = args.out / f"{f.stem}.tmp.png"
            cv2.imwrite(str(tmp), np.round(np.clip(p, 0, 1) * 255).astype(np.uint8))
            tmp.replace(args.out / f"{f.stem}.png")
        done += len(part)
        if done % 80 < 8:
            print(f"{done}/{len(mine)} maps, {(time.time() - t0) / done:.2f} s/image", flush=True)
    print(f"shard {k}/{n} done: {done} new maps in {time.time() - t0:.0f}s", flush=True)
    if args.wait:
        want = {Path(f).stem for f in files}
        deadline = time.time() + args.wait_hours * 3600
        while True:
            have = {p.stem for p in args.out.glob("*.png") if not p.stem.endswith(".tmp")}
            if want <= have:
                break
            if time.time() > deadline:
                raise SystemExit(f"timed out waiting for {len(want - have)} maps of other shards")
            time.sleep(15)
        print(f"all {len(want)} maps present", flush=True)


if __name__ == "__main__":
    main()
