"""Label-free check of the ensemble-size shift between post-processing selection and submission.

The post-processing (rule or learned filter) is chosen on out-of-fold maps, and a training image is
out-of-fold for only two networks (the ResNet34 and EfficientNetV2-S models of its fold), whereas the
submission averages all ten. This script applies post-processing configurations to the ten-network
average and to the five two-network averages (one per fold) of the same test images and compares the
kept instances: number and area per image, and agreement as instances matched at IoU > 0.5 and a
PQ-style score (sum of matched IoU / (TP + FP/2 + FN/2)). As a reference, the two-network selections
of different folds are compared with each other.

Usage:
    python scripts/ensemble_shift.py <run_dir> <post.json> [<post.json> ...] --out shift.jsonl

``<run_dir>`` contains ``{r34,v2s}_2048/fold{0..4}/test_probs.npz`` (uint8 maps, probability x 255).
Per-image results are appended to ``--out`` (an interrupted run resumes); the summary is printed and
written next to it as ``<out>.summary.json``.
"""
from __future__ import annotations

import argparse
import itertools
import json
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from filseg.postprocess import PostConfig, instances_from_prob, upsample  # noqa: E402

ENCODERS = ("r34_2048", "v2s_2048")
FOLDS = range(5)
_STATE: dict = {}


def _init(run_dir: str, cfgs: dict) -> None:
    import cv2

    cv2.setNumThreads(1)
    _STATE["files"] = {(e, k): np.load(Path(run_dir) / e / f"fold{k}" / "test_probs.npz")
                       for e in ENCODERS for k in FOLDS}
    _STATE["cfgs"] = {name: PostConfig(**spec) for name, spec in cfgs.items()}


def average(maps: list[np.ndarray]) -> np.ndarray:
    acc = np.zeros(maps[0].shape, np.float32)
    for m in maps:
        acc += m
    return np.round(acc / len(maps)).astype(np.uint8)  # same rounding as scripts/analyze_oof.load_probs


def agreement(la: np.ndarray, lb: np.ndarray) -> tuple[int, float]:
    """Instances of two disjoint label maps matched at IoU > 0.5, and the sum of their IoU."""
    na, nb = int(la.max()), int(lb.max())
    if na == 0 or nb == 0:
        return 0, 0.0
    both = (la > 0) & (lb > 0)
    inter = np.bincount(la[both].astype(np.int64) * (nb + 1) + lb[both], minlength=(na + 1) * (nb + 1))
    inter = inter.reshape(na + 1, nb + 1)[1:, 1:].astype(np.float64)
    area_a = np.bincount(la.ravel(), minlength=na + 1)[1:].astype(np.float64)
    area_b = np.bincount(lb.ravel(), minlength=nb + 1)[1:].astype(np.float64)
    iou = inter / (area_a[:, None] + area_b[None, :] - inter)
    hit = iou > 0.5
    return int(hit.sum()), float(iou[hit].sum())


def _image(key: str) -> dict:
    maps = {ek: z[key] for ek, z in _STATE["files"].items()}
    ten = upsample(average(list(maps.values())))
    twos = [upsample(average([maps[(e, k)] for e in ENCODERS])) for k in FOLDS]
    out = {"image": key}
    for name, cfg in _STATE["cfgs"].items():
        lab10, _ = instances_from_prob(ten, cfg)
        labs = [instances_from_prob(p, cfg)[0] for p in twos]
        rec = {"n10": int(lab10.max()), "area10": int((lab10 > 0).sum()),
               "n2": [int(lab.max()) for lab in labs], "area2": [int((lab > 0).sum()) for lab in labs],
               "vs10": [agreement(lab10, lab) for lab in labs],
               "pairs": [agreement(labs[a], labs[b]) for a, b in itertools.combinations(FOLDS, 2)]}
        out[name] = rec
    return out


def _pq(tp: float, fp: float, fn: float, s: float) -> float:
    return s / max(tp + 0.5 * fp + 0.5 * fn, 1e-9)


def summarize(rows: list[dict], names: list[str]) -> dict:
    n_img = len(rows)
    summary = {"images": n_img}
    for name in names:
        r = [row[name] for row in rows]
        n10 = sum(x["n10"] for x in r)
        n2 = [sum(x["n2"][k] for x in r) for k in FOLDS]
        a10 = sum(x["area10"] for x in r)
        a2 = [sum(x["area2"][k] for x in r) for k in FOLDS]
        vs = []
        for k in FOLDS:
            tp = sum(x["vs10"][k][0] for x in r)
            s = sum(x["vs10"][k][1] for x in r)
            vs.append(dict(matched_of_ten=tp / max(n10, 1), matched_of_two=tp / max(n2[k], 1),
                           pq=_pq(tp, n10 - tp, n2[k] - tp, s)))
        pairs = []
        for j, (a, b) in enumerate(itertools.combinations(FOLDS, 2)):
            tp = sum(x["pairs"][j][0] for x in r)
            s = sum(x["pairs"][j][1] for x in r)
            pairs.append(_pq(tp, n2[a] - tp, n2[b] - tp, s))
        summary[name] = {
            "instances_per_image_ten": n10 / n_img,
            "instances_per_image_two": float(np.mean(n2)) / n_img,
            "instances_per_image_two_range": [min(n2) / n_img, max(n2) / n_img],
            "area_ratio_ten_vs_two": a10 / float(np.mean(a2)),
            "ten_vs_two": {"matched_of_ten": float(np.mean([v["matched_of_ten"] for v in vs])),
                           "matched_of_two": float(np.mean([v["matched_of_two"] for v in vs])),
                           "pq": float(np.mean([v["pq"] for v in vs])),
                           "pq_range": [min(v["pq"] for v in vs), max(v["pq"] for v in vs)]},
            "two_vs_two_pq": float(np.mean(pairs)),
            "two_vs_two_pq_range": [min(pairs), max(pairs)],
        }
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("post", nargs="+", type=Path, help="post-processing JSON files (configs/post_final_*.json)")
    ap.add_argument("--out", type=Path, required=True, help="per-image results (.jsonl, resumable)")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0, help="first n test images only (timing checks)")
    args = ap.parse_args()

    cfgs = {p.stem.replace("post_final_", ""): json.loads(p.read_text()) for p in args.post}
    with np.load(args.run_dir / ENCODERS[0] / "fold0" / "test_probs.npz") as z:
        keys = sorted(z.files)
    if args.limit:
        keys = keys[:args.limit]
    rows = []
    if args.out.exists():
        rows = [json.loads(line) for line in args.out.read_text().splitlines() if line.strip()]
        rows = [r for r in rows if r["image"] in set(keys) and all(n in r for n in cfgs)]
    todo = [k for k in keys if k not in {r["image"] for r in rows}]
    print(f"{len(keys)} test images, {len(rows)} done, {len(todo)} to do; configurations {list(cfgs)}", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with mp.Pool(args.workers, initializer=_init, initargs=(str(args.run_dir), cfgs)) as pool, \
            args.out.open("a") as f:
        for i, row in enumerate(pool.imap_unordered(_image, todo), start=1):
            f.write(json.dumps(row) + "\n")
            f.flush()
            rows.append(row)
            if i % 20 == 0:
                print(f"  {i}/{len(todo)}", flush=True)
    summary = summarize(rows, list(cfgs))
    args.out.with_suffix(".summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
