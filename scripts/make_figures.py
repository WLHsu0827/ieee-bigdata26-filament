"""Figures for the notebook / report.

Usage:
    python scripts/make_figures.py pipeline            # docs/figures/pipeline.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "docs" / "figures"
sys.path.insert(0, str(ROOT))


def pipeline_diagram(path: Path = FIG / "pipeline.png") -> Path:
    """System block diagram: inputs, processing blocks, key intermediate signals, outputs."""
    blocks = [
        ("H-alpha image\n2048 x 2048, 8-bit grey", "#dbe9f6"),
        ("Pre-processing\nresize to network res.\nnormalise", "#f2f2f2"),
        ("U-Net ensemble\n+ flip TTA", "#fde9d9"),
        ("Instance extraction\nthreshold, fill holes,\nconnected components,\nPQ-aware selection", "#f2f2f2"),
        ("submission.csv\none COCO RLE\nper filament", "#e2efda"),
    ]
    signals = ["image x\n(1 x R x R)", "logits -> P(filament)\n= expected fraction\nof annotators",
               "P upsampled\nto 2048 x 2048", "disjoint instance\nlabel map"]
    fig, ax = plt.subplots(figsize=(15, 3.2))
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 3.2)
    ax.axis("off")
    w, h, gap, y = 2.35, 1.7, 0.75, 0.9
    xs = [0.2 + i * (w + gap) for i in range(len(blocks))]
    for x, (text, color) in zip(xs, blocks):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05", fc=color, ec="#404040", lw=1.2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=10)
    for i, s in enumerate(signals):
        x0, x1 = xs[i] + w + 0.05, xs[i + 1] - 0.05
        ax.add_patch(FancyArrowPatch((x0, y + h / 2), (x1, y + h / 2), arrowstyle="-|>", mutation_scale=16,
                                     lw=1.4, color="#404040"))
        ax.text((x0 + x1) / 2, y + h + 0.12, s, ha="center", va="bottom", fontsize=8, color="#203864")
    ax.text(xs[2] + w / 2, 0.42, "target: per-pixel fraction\nof annotators (soft labels);\nBCE + soft Dice, EMA",
            ha="center", va="center", fontsize=8, color="#7f3f00")
    ax.text(xs[3] + w / 2, 0.42, "keep a component iff its\nexpected matched IoU per\nannotator > PQ / 2",
            ha="center", va="center", fontsize=8, color="#7f3f00")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def overlays(probs_npz: Path, post_json: Path, data_dir: Path, image_ids: list[str],
             path: Path = FIG / "overlays.png", zoom: int = 600) -> Path:
    """Validation examples: H-alpha image, every annotator's outline, our predicted instances.

    Each row shows one image: full disk with annotator outlines (one colour per
    annotator), full disk with predicted instances, and a zoom on the largest prediction.
    """
    import json

    import cv2
    import numpy as np

    from filseg import data
    from filseg.postprocess import PostConfig, instances_from_prob, upsample

    groups = data.group_by_image(data.load_entries(data_dir / "train" / data.TRAIN_JSON))
    probs = np.load(probs_npz)
    cfg = PostConfig(**json.loads(Path(post_json).read_text()))
    colors = [(255, 60, 60), (60, 220, 60), (60, 140, 255)]
    fig, axes = plt.subplots(len(image_ids), 3, figsize=(15, 5 * len(image_ids)))
    axes = np.atleast_2d(axes)
    for row, iid in zip(axes, image_ids):
        img = data.read_image(data_dir / "train" / "train_images" / groups[iid][0].file_name)
        gt_vis = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        for e, col in zip(groups[iid], colors):
            for poly in e.polygons:
                pts = np.round(np.array(poly).reshape(-1, 2)).astype(np.int32)
                cv2.polylines(gt_vis, [pts], True, col, 3)
        lab, _ = instances_from_prob(upsample(probs[iid].astype(np.float32) / 255.0), cfg)
        pred_vis = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        palette = [(255, 215, 0), (255, 0, 255), (0, 255, 255), (255, 128, 0), (0, 255, 0), (255, 64, 64)]
        for k in range(1, lab.max() + 1):
            m = (lab == k).astype(np.uint8)
            cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(pred_vis, cnts, -1, palette[(k - 1) % len(palette)], 4)
        row[0].imshow(gt_vis)
        row[0].set_title(f"{iid}: {len(groups[iid])} annotator(s)")
        row[1].imshow(pred_vis)
        row[1].set_title(f"prediction: {lab.max()} filaments")
        if lab.max():
            sizes = np.bincount(lab.ravel())[1:]
            ys, xs = np.nonzero(lab == 1 + int(np.argmax(sizes)))
            cy, cx = int(ys.mean()), int(xs.mean())
            y0, x0 = max(cy - zoom // 2, 0), max(cx - zoom // 2, 0)
            both = np.concatenate([gt_vis[y0:y0 + zoom, x0:x0 + zoom], pred_vis[y0:y0 + zoom, x0:x0 + zoom]], axis=1)
            row[2].imshow(both)
            row[2].set_title("zoom: annotators (left) vs prediction (right)")
        for ax in row:
            ax.axis("off")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def evaluation(probs_paths: list, post_json: Path, gt_cache: Path | None,
               path: Path = FIG / "evaluation.png") -> Path:
    """Rubric figure for out-of-fold predictions, in the style of the organisers' self-evaluation:
    IoU and Dice distributions of overlapping pairs, one-to-many / many-to-one matching degrees,
    and recall by filament size. Also writes the numbers to ``<path>.json``."""
    import json

    import numpy as np

    sys.path.insert(0, str(ROOT / "scripts"))
    from analyze_oof import load_probs

    from filseg import analysis
    from filseg.metrics import evaluate, matching_degrees
    from filseg.postprocess import PostConfig
    from filseg.tune import Scorer

    from types import SimpleNamespace

    from analyze_oof import load_gt
    from cli_utils import expand

    probs = load_probs(expand(probs_paths))
    gt = load_gt(SimpleNamespace(gt_cache=gt_cache, data=None), set(probs))
    cfg = PostConfig(**json.loads(Path(post_json).read_text()))
    scorer = Scorer(probs, gt, workers=1)
    try:
        preds = scorer.predict(cfg)
    finally:
        scorer.close()
    res, odf = evaluate(gt, preds)
    iou, dice = analysis.overlap_distributions(odf)
    gdeg, pdeg = matching_degrees(odf)
    by_size = analysis.by_size(analysis.gt_table(gt, odf))

    fig, ax = plt.subplots(1, 4, figsize=(22, 4.5))
    for a, vals, name in ((ax[0], iou, "IoU"), (ax[1], dice, "Dice")):
        a.hist(vals, bins=np.linspace(0, 1, 51), weights=np.ones(len(vals)) / max(len(vals), 1),
               color="navy", alpha=0.7, edgecolor="black")
        a.axvline(vals.mean(), color="crimson", ls="--", label=f"mean {vals.mean():.3f}")
        if name == "IoU":
            a.axvline(0.5, color="gray", ls=":", label="match threshold 0.5")
        a.set_xlabel(name)
        a.set_ylabel("fraction of overlapping pairs")
        a.set_title(f"{name} of overlapping GT / prediction pairs")
        a.legend(fontsize=8)
    ks = sorted(set(gdeg) | set(pdeg))
    ax[2].bar([k + 0.5 for k in ks], [gdeg.get(k, 0) for k in ks], width=0.8,
              color=["crimson" if k == 0 else "forestgreen" for k in ks], edgecolor="black", alpha=0.7)
    ax[2].bar([-(k + 0.5) for k in ks], [pdeg.get(k, 0) for k in ks], width=0.8,
              color=["darkorange" if k == 0 else "steelblue" for k in ks], edgecolor="black", alpha=0.7)
    ax[2].axvline(0, color="black", lw=0.8)
    ticks = [k + 0.5 for k in ks] + [-(k + 0.5) for k in ks]
    ax[2].set_xticks(ticks, [f"1:{k}" for k in ks] + [f"{k}:1" for k in ks], rotation=45, fontsize=8)
    ax[2].set_title("matching degrees (IoU > 0): GT 1:n right, prediction n:1 left")
    ax[2].set_ylabel("count")
    labels = [str(s).replace(".0", "") for s in by_size["size"]]
    ax[3].bar(range(len(labels)), by_size["match_rate"], color="navy", alpha=0.7, edgecolor="black")
    ax[3].set_xticks(range(len(labels)), labels, rotation=45, fontsize=8)
    ax[3].set_xlabel("GT filament area (px at 2048)")
    ax[3].set_ylabel("fraction matched (IoU > 0.5)")
    ax[3].set_title("recall by filament size")
    fig.suptitle(f"out-of-fold: {len(probs)} images, {len(gt)} annotator sets -- PQ {res.pq:.4f} "
                 f"(SQ {res.sq:.3f}, RQ {res.rq:.3f}; TP {res.tp}, FP {res.fp}, FN {res.fn})")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    summary = {"images": len(probs), "entries": len(gt), "post": json.loads(Path(post_json).read_text()),
               "pq": res.pq, "sq": res.sq, "rq": res.rq, "tp": res.tp, "fp": res.fp, "fn": res.fn,
               "mean_iou_overlapping": float(iou.mean()), "mean_dice_overlapping": float(dice.mean()),
               "gt_degrees": {int(k): int(v) for k, v in sorted(gdeg.items())},
               "pred_degrees": {int(k): int(v) for k, v in sorted(pdeg.items())},
               "recall_by_size": {str(s): round(float(r), 4) for s, r in zip(by_size["size"], by_size["match_rate"])},
               **{k: v for k, v in analysis.aggregation_breakdown(odf).items()}}
    path.with_suffix(".json").write_text(json.dumps(summary, indent=1))
    return path


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "pipeline"
    if what == "pipeline":
        print(pipeline_diagram())
    elif what == "overlays":  # overlays <val_probs.npz> <post.json> <data_dir> <image_id> [<image_id> ...]
        print(overlays(Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]), sys.argv[5:]))
    elif what == "evaluation":  # evaluation <post.json> <gt_crops.pkl | none> <out.png> <val_probs.npz | glob> [...]
        gt_arg = None if sys.argv[3].lower() == "none" else Path(sys.argv[3])
        print(evaluation(sys.argv[5:], Path(sys.argv[2]), gt_arg, Path(sys.argv[4])))
