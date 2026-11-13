"""Human-level reference numbers from the multi-annotator training images.

Computes Panoptic Quality (official definition) for

* ``pairwise``         -- annotator A used as "prediction" for annotator B (all ordered pairs),
* ``consensus@t``      -- the pixel-wise annotator fraction of *all* annotators of an image,
                          thresholded at ``t`` and split into connected components, scored
                          against every annotator (an oracle that sees the scored annotator),
* ``loo_union`` / ``loo_intersection`` -- leave-one-out: union / intersection of the *other*
                          two annotators of a 3-annotator image scored against the held-out one
                          (the realistic target of a model trained on consensus).

Progress is checkpointed (``--state``) at least every ``--save-every`` seconds, so the
script can be interrupted and resumed.

Usage:
    python scripts/annotator_agreement.py --gt-cache <gt_crops.pkl> --out <dir>
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from filseg import IMAGE_SIZE  # noqa: E402
from filseg.metrics import Crop, overlap_matrices, pq_from_overlaps  # noqa: E402
from filseg.postprocess import PostConfig, instances_from_prob, label_to_crops  # noqa: E402

THRESHOLDS = (0.3, 0.45, 0.6, 0.9)


def union(crops: list[Crop]) -> np.ndarray:
    """Full-size boolean mask covered by ``crops`` (e.g. all filaments of one annotator)."""
    u = np.zeros((IMAGE_SIZE, IMAGE_SIZE), bool)
    for c in crops:
        h, w = c.mask.shape
        u[c.y0:c.y0 + h, c.x0:c.x0 + w] |= c.mask
    return u


def components(mask: np.ndarray) -> list[Crop]:
    """Instance crops of a binary mask: 8-connected components, holes filled, no minimum area."""
    lab, _ = instances_from_prob(mask.astype(np.float32), PostConfig(threshold=0.5, min_area=0))
    return label_to_crops(lab)


def image_rows(entries: dict[str, list[Crop]]) -> dict[str, list]:
    """Per-method list of ``(iou_matrix, n_gt, n_pred)`` for one image."""
    out: dict[str, list] = defaultdict(list)
    ids = sorted(entries)
    unions = {e: union(entries[e]) for e in ids}
    for a in ids:
        for b in ids:
            if a != b:
                out["pairwise"].append((overlap_matrices(entries[b], entries[a])[0], len(entries[b]), len(entries[a])))
    frac = sum(u.astype(np.float32) for u in unions.values()) / len(ids)
    for t in THRESHOLDS:
        pred = components(frac > t)
        for e in ids:
            out[f"consensus@{t}"].append((overlap_matrices(entries[e], pred)[0], len(entries[e]), len(pred)))
    if len(ids) == 3:
        for e in ids:
            o1, o2 = (unions[x] for x in ids if x != e)
            for name, m in (("loo_union", o1 | o2), ("loo_intersection", o1 & o2)):
                pred = components(m)
                out[name].append((overlap_matrices(entries[e], pred)[0], len(entries[e]), len(pred)))
    return dict(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt-cache", type=Path, required=True, help="entry_id -> list[Crop] pickle (build_gt_cache.py)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--save-every", type=float, default=60.0, help="checkpoint interval in seconds")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    state_path = args.out / "agreement_state.pkl"

    with open(args.gt_cache, "rb") as f:
        crops = pickle.load(f)
    by_image: dict[str, dict[str, list[Crop]]] = defaultdict(dict)
    for entry_id, cs in crops.items():
        by_image[entry_id.split("-", 1)[1]][entry_id] = cs
    multi = sorted(k for k, v in by_image.items() if len(v) >= 2)

    done: dict[str, dict] = {}
    if state_path.exists():
        with open(state_path, "rb") as f:
            done = pickle.load(f)
        print(f"resuming: {len(done)}/{len(multi)} images already processed", flush=True)

    def save():
        tmp = state_path.with_suffix(".tmp")
        with open(tmp, "wb") as f:
            pickle.dump(done, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, state_path)

    last = time.time()
    for n, iid in enumerate(multi):
        if iid in done:
            continue
        done[iid] = image_rows(by_image[iid])
        if time.time() - last > args.save_every:
            save()
            last = time.time()
            print(f"{len(done)}/{len(multi)} images", flush=True)
    save()

    methods = sorted({m for rows in done.values() for m in rows})
    result = {"images_multi": len(multi),
              "images_three": sum(len(by_image[i]) == 3 for i in multi)}
    for m in methods:
        res = pq_from_overlaps(r for rows in done.values() for r in rows.get(m, []))
        result[m] = {"pq": round(res.pq, 4), "sq": round(res.sq, 4), "rq": round(res.rq, 4),
                     "tp": res.tp, "fp": res.fp, "fn": res.fn}
        print(f"{m:>20}: {res}", flush=True)
    (args.out / "agreement.json").write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
