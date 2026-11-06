"""Build a submission by averaging saved test probability maps of several models.

Every training run with ``--predict-test`` stores ``test_probs.npz`` (uint8 maps at
the network resolution). This script averages them image by image (maps of
different resolutions are upsampled to 2048 px), applies one post-processing
configuration and writes a validated ``submission.csv``. It also reports the
end-to-end inference time per image of the ensemble from the runs' ``timing.json``.

Usage:
    python scripts/ensemble_submit.py --probs runs/*/test_probs.npz --post sel/post_selected.json \
        --test-dir <MAGFiLO>/test/test_images --out submission.csv
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from filseg import submission  # noqa: E402
from filseg.postprocess import PostConfig, instances_from_prob, upsample  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--probs", nargs="+", type=Path, required=True)
    ap.add_argument("--post", type=Path, required=True)
    ap.add_argument("--test-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    cfg = PostConfig(**json.loads(args.post.read_text()))
    files = [np.load(p) for p in args.probs]  # lazy: arrays are read per key
    test_ids = sorted(p.stem for p in args.test_dir.glob("*.jp*g"))
    missing = [str(p) for p, z in zip(args.probs, files) if set(test_ids) - set(z.files)]
    if missing:
        raise SystemExit(f"test images missing in {missing}")
    t0 = time.time()
    rows = []
    for image_id in test_ids:
        acc = np.zeros((2048, 2048), np.float32)
        for z in files:
            acc += upsample(z[image_id].astype(np.float32) / 255.0)
        lab, _ = instances_from_prob(acc / len(files), cfg)
        rows += submission.rows_from_label(image_id, lab)
    post_s = (time.time() - t0) / len(test_ids)
    df, report = submission.write(rows, args.out, set(test_ids))

    timing = []
    for p in args.probs:
        tj = p.parent / "timing.json"
        if tj.exists():
            timing.append(json.loads(tj.read_text()))
    model_s = sum(t["model_s_per_image"] for t in timing) if len(timing) == len(args.probs) else float("nan")
    print(f"{len(test_ids)} images, {len(df)} filaments -> {args.out}; check: {report}")
    print(f"ensemble of {len(args.probs)} models: network {model_s:.2f} s/image (sum of runs' timing.json, "
          f"{timing[0]['device'] if timing else '?'}), averaging + post-processing {post_s:.2f} s/image here")


if __name__ == "__main__":
    main()
