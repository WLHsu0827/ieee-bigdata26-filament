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
        ("Pre-processing\nnormalise with the\nimage's statistics\ninside the solar disk", "#f2f2f2"),
        ("U-Net ensemble\n5 x ResNet34 +\n5 x EfficientNetV2-S\n+ flip TTA (x4)", "#fde9d9"),
        ("Instance extraction\nthreshold, components,\nlearned PQ-aware filter,\noutline at P > 0.5", "#f2f2f2"),
        ("submission.csv\none COCO RLE\nper filament", "#e2efda"),
    ]
    signals = ["image x\n(1 x 2048 x 2048)", "logits -> P(filament)\n= expected fraction\nof annotators",
               "mean P of the\n10 networks", "disjoint instance\nlabel map"]
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
    ax.text(xs[3] + w / 2, 0.42, "keep a candidate iff its predicted\nmatched IoU per annotator > ~PQ / 2\n(gradient-boosted trees, 12 features)",
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

    sys.path.insert(0, str(ROOT / "scripts"))
    from analyze_oof import load_probs

    groups = data.group_by_image(data.load_entries(data_dir / "train" / data.TRAIN_JSON))
    probs = load_probs([Path(p) for p in str(probs_npz).split(",")])  # several files: ensemble mean
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
        lab, _ = instances_from_prob(upsample(probs[iid]), cfg)
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
    tax = analysis.error_taxonomy(gt, preds, odf)

    recall = [(str(s).replace(".0", ""), float(r)) for s, r in zip(by_size["size"], by_size["match_rate"])]
    header = (f"out-of-fold: {len(probs)} images, {len(gt)} annotator sets -- PQ {res.pq:.4f} "
              f"(SQ {res.sq:.3f}, RQ {res.rq:.3f}; TP {res.tp}, FP {res.fp}, FN {res.fn})")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path.with_suffix(".npz"), iou=iou, dice=dice)  # for re-plotting without recomputation
    draw_evaluation(iou, dice, gdeg, pdeg, recall, tax, header, path, layout="row")
    stats = dict(images=len(probs), entries=len(gt), pq=res.pq, sq=res.sq, rq=res.rq, tp=res.tp, fp=res.fp, fn=res.fn)
    draw_evaluation(iou, dice, gdeg, pdeg, recall, tax, header, path.with_name(path.stem + "_grid.png"), layout="grid",
                    stats=stats)
    post = json.loads(Path(post_json).read_text())
    if post.get("scorer"):  # summarise the learned filter (the full trees live in the post-processing JSON)
        post["scorer"] = {"trees": len(post["scorer"]["trees"]), "features": post["scorer"]["features"]}
    summary = {"images": len(probs), "entries": len(gt), "post": post,
               "pq": res.pq, "sq": res.sq, "rq": res.rq, "tp": res.tp, "fp": res.fp, "fn": res.fn,
               "mean_iou_overlapping": float(iou.mean()), "mean_dice_overlapping": float(dice.mean()),
               "gt_degrees": {int(k): int(v) for k, v in sorted(gdeg.items())},
               "pred_degrees": {int(k): int(v) for k, v in sorted(pdeg.items())},
               "recall_by_size": {str(s): round(float(r), 4) for s, r in zip(by_size["size"], by_size["match_rate"])},
               "error_taxonomy": tax,
               **{k: v for k, v in analysis.aggregation_breakdown(odf).items()}}
    path.with_suffix(".json").write_text(json.dumps(summary, indent=1))
    return path


def morphology(probs_npz: Path, post_json: Path, data_dir: Path, image_ids: list[str],
               path: Path = FIG / "morphology.png", zoom: int = 520, dpi: int = 120) -> Path:
    """Compact report figure: one zoom per image around its largest predicted filament, annotator
    outlines (one colour per annotator) above and predicted instances below."""
    import json

    import cv2
    import numpy as np

    from filseg import data
    from filseg.postprocess import PostConfig, instances_from_prob, upsample

    sys.path.insert(0, str(ROOT / "scripts"))
    from analyze_oof import load_probs

    groups = data.group_by_image(data.load_entries(data_dir / "train" / data.TRAIN_JSON))
    probs = load_probs([Path(p) for p in str(probs_npz).split(",")])
    cfg = PostConfig(**json.loads(Path(post_json).read_text()))
    colors = [(255, 60, 60), (60, 220, 60), (60, 140, 255)]
    palette = [(255, 215, 0), (255, 0, 255), (0, 255, 255), (255, 128, 0), (0, 255, 0), (255, 64, 64)]
    fig, axes = plt.subplots(2, len(image_ids), figsize=(3.4 * len(image_ids) / 1.5, 4.6))
    axes = np.atleast_2d(axes).reshape(2, len(image_ids))
    for col, iid in enumerate(image_ids):
        img = data.read_image(data_dir / "train" / "train_images" / groups[iid][0].file_name)
        lab, _ = instances_from_prob(upsample(probs[iid]), cfg)
        gt_vis = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        for e, c in zip(groups[iid], colors):
            for poly in e.polygons:
                cv2.polylines(gt_vis, [np.round(np.array(poly).reshape(-1, 2)).astype(np.int32)], True, c, 3)
        pred_vis = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        for k in range(1, lab.max() + 1):
            cnts, _ = cv2.findContours((lab == k).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(pred_vis, cnts, -1, palette[(k - 1) % len(palette)], 3)
        sizes = np.bincount(lab.ravel())[1:]
        ys, xs = np.nonzero(lab == 1 + int(np.argmax(sizes))) if len(sizes) else (np.array([1024]), np.array([1024]))
        y0 = int(np.clip(ys.mean() - zoom // 2, 0, img.shape[0] - zoom))
        x0 = int(np.clip(xs.mean() - zoom // 2, 0, img.shape[1] - zoom))
        for row, vis, title in ((0, gt_vis, f"{len(groups[iid])} annotator(s)"), (1, pred_vis, "prediction")):
            axes[row, col].imshow(vis[y0:y0 + zoom, x0:x0 + zoom])
            axes[row, col].set_title(title, fontsize=8)
            axes[row, col].axis("off")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


def _short_size(label: str) -> str:
    """Compact size-bin label: '[300, 600)' -> '300-600', '[0, 300)' -> '<300', '[10000, inf)' -> '>=10k'."""
    import re

    lo, hi = (float(x) for x in re.findall(r"inf|\d+(?:\.\d+)?", label)[:2])

    def fmt(v: float) -> str:
        return f"{v / 1000:g}k" if v >= 1000 else f"{v:g}"

    if lo == 0:
        return f"<{fmt(hi)}"
    return f"\u2265{fmt(lo)}" if hi == float("inf") else f"{fmt(lo)}\u2013{fmt(hi)}"


def draw_evaluation(iou, dice, gdeg: dict, pdeg: dict, recall: list, tax: dict, header: str, path: Path,
                    layout: str = "row", stats: dict | None = None) -> Path:
    """Rubric figure: IoU / Dice of overlapping pairs, one-to-many / many-to-one degrees, recall by size
    and error causes. ``layout``: ``row`` (1 x 5 with long labels, notebook) or ``grid`` (2 x 3 with short
    labels and a summary panel, sized for a two-column-wide figure in the report)."""
    import numpy as np

    grid = layout == "grid"
    fs = 9 if grid else 8
    if grid:
        fig, axs = plt.subplots(2, 3, figsize=(9.6, 5.4))
        ax = [axs[0, 0], axs[0, 1], axs[0, 2], axs[1, 0], axs[1, 1]]
        for a in ax:
            a.tick_params(labelsize=fs - 1)
    else:
        fig, ax = plt.subplots(1, 5, figsize=(27, 4.5))
    for a, vals, name in ((ax[0], iou, "IoU"), (ax[1], dice, "Dice")):
        heights, _, _ = a.hist(vals, bins=np.linspace(0, 1, 51), weights=np.ones(len(vals)) / max(len(vals), 1),
                               color="navy", alpha=0.7, edgecolor="black")
        if grid and len(vals):
            a.set_ylim(0, heights.max() * 1.45)  # room for the legend above the bars
        a.axvline(vals.mean(), color="crimson", ls="--", label=f"mean {vals.mean():.3f}")
        if name == "IoU":
            a.axvline(0.5, color="gray", ls=":", label="match threshold 0.5")
        a.set_xlabel(name, fontsize=fs)
        a.set_ylabel("fraction of pairs" if grid else "fraction of overlapping pairs", fontsize=fs)
        a.set_title(f"{name} of overlapping pairs" if grid else f"{name} of overlapping GT / prediction pairs",
                    fontsize=fs + 1)
        a.legend(fontsize=fs - 1, loc="upper left" if grid else "best")
    ks = sorted({int(k) for k in gdeg} | {int(k) for k in pdeg})
    g = {int(k): v for k, v in gdeg.items()}
    p = {int(k): v for k, v in pdeg.items()}
    ax[2].bar([k + 0.5 for k in ks], [g.get(k, 0) for k in ks], width=0.8,
              color=["crimson" if k == 0 else "forestgreen" for k in ks], edgecolor="black", alpha=0.7)
    ax[2].bar([-(k + 0.5) for k in ks], [p.get(k, 0) for k in ks], width=0.8,
              color=["darkorange" if k == 0 else "steelblue" for k in ks], edgecolor="black", alpha=0.7)
    ax[2].axvline(0, color="black", lw=0.8)
    ax[2].set_xticks([k + 0.5 for k in ks] + [-(k + 0.5) for k in ks],
                     [f"1:{k}" for k in ks] + [f"{k}:1" for k in ks], rotation=45, fontsize=fs - 1)
    if grid:
        ax[2].set_title("matching degrees (IoU > 0)", fontsize=fs + 1)
        ax[2].set_xlabel("prediction n:1 (left), GT 1:n (right)", fontsize=fs)
    else:
        ax[2].set_title("matching degrees (IoU > 0): GT 1:n right, prediction n:1 left", fontsize=fs + 1)
    ax[2].set_ylabel("count", fontsize=fs)
    ax[3].bar(range(len(recall)), [r for _, r in recall], color="navy", alpha=0.7, edgecolor="black")
    if grid:
        ax[3].set_xticks(range(len(recall)), [_short_size(lbl) for lbl, _ in recall], rotation=45, ha="right",
                         rotation_mode="anchor", fontsize=fs - 1)
    else:
        ax[3].set_xticks(range(len(recall)), [lbl for lbl, _ in recall], rotation=45, fontsize=fs - 1)
    ax[3].set_xlabel("GT filament area (px)" if grid else "GT filament area (px at 2048)", fontsize=fs)
    ax[3].set_ylabel("matched (IoU > 0.5)" if grid else "fraction matched (IoU > 0.5)", fontsize=fs)
    ax[3].set_title("recall by filament size", fontsize=fs + 1)
    causes = [("missed", "no overlapping prediction"), ("partial", "covered < 50 %"),
              ("merged", "merged with a neighbour"), ("too_large", "prediction too large"),
              ("fragmented", "fragmented (union would match)")]
    vals = [tax["gt"][k] for k, _ in causes] + [tax["pred"]["spurious"], tax["pred"]["overlapping"]]
    if grid:  # the report caption spells the categories out
        names = [f"FN {k.replace('_', ' ')}" for k, _ in causes] + ["FP isolated", "FP overlapping"]
    else:
        names = [f"FN: {d}" for _, d in causes] + ["FP: no overlap with any GT", "FP: overlaps a GT (IoU <= 0.5)"]
    ax[4].barh(range(len(vals))[::-1], vals, color=["crimson"] * len(causes) + ["darkorange"] * 2, alpha=0.7,
               edgecolor="black")
    ax[4].set_yticks(range(len(vals))[::-1], names, fontsize=fs - 1)
    ax[4].set_xlabel("count (per annotator set)", fontsize=fs)
    ub = tax["upper_bounds"]
    ax[4].set_title("error causes" if grid else
                    f"error causes\nPQ upper bounds: fragments grouped {ub['group_fragments']:.3f},\n"
                    f"isolated false positives removed {ub['remove_spurious']:.3f}", fontsize=fs + 1)
    if grid:
        s = axs[1, 2]
        s.axis("off")
        st = stats or {}

        def num(k: str, spec: str = ",") -> str:
            return format(st[k], spec) if k in st else "?"

        lines = [f"out-of-fold: {num('images')} images,", f"{num('entries')} annotator sets", "",
                 f"PQ {num('pq', '.3f')}  (SQ {num('sq', '.3f')}, RQ {num('rq', '.3f')})",
                 f"TP {num('tp')}   FP {num('fp')}   FN {num('fn')}", "",
                 "overlapping pairs, mean:", f"IoU {iou.mean():.3f}, Dice {dice.mean():.3f}", "",
                 "PQ upper bounds:", f"  fragments grouped {ub['group_fragments']:.3f}",
                 f"  isolated FPs removed {ub['remove_spurious']:.3f}"]
        s.text(0.0, 1.0, "\n".join(lines), va="top", ha="left", fontsize=fs, linespacing=1.35,
               transform=s.transAxes)
    else:
        fig.suptitle(header)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=110 if not grid else 200)
    plt.close(fig)
    return path

if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "pipeline"
    if what == "pipeline":
        print(pipeline_diagram())
    elif what == "overlays":  # overlays <val_probs.npz> <post.json> <data_dir> <image_id> [<image_id> ...]
        print(overlays(Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]), sys.argv[5:]))
    elif what == "morphology":  # morphology <val_probs.npz[,more.npz]> <post.json> <data_dir> <image_id> [...]
        print(morphology(Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]), sys.argv[5:]))
    elif what == "evaluation-replot":  # evaluation-replot <arrays.npz> <evaluation.json> <out.png> [row|grid]
        import json

        import numpy as np

        arrays, ev = np.load(sys.argv[2]), json.loads(Path(sys.argv[3]).read_text())
        header = (f"out-of-fold: {ev['images']} images, {ev['entries']} annotator sets -- PQ {ev['pq']:.4f} "
                  f"(SQ {ev['sq']:.3f}, RQ {ev['rq']:.3f}; TP {ev['tp']}, FP {ev['fp']}, FN {ev['fn']})")
        stats = {k: ev[k] for k in ("images", "entries", "pq", "sq", "rq", "tp", "fp", "fn")}
        recall = [(k.replace(".0", ""), v) for k, v in ev["recall_by_size"].items()]
        print(draw_evaluation(arrays["iou"], arrays["dice"], ev["gt_degrees"], ev["pred_degrees"], recall,
                              ev["error_taxonomy"], header, Path(sys.argv[4]),
                              layout=sys.argv[5] if len(sys.argv) > 5 else "grid", stats=stats))
    elif what == "evaluation":  # evaluation <post.json> <gt_crops.pkl | none> <out.png> <val_probs.npz | glob> [...]
        gt_arg = None if sys.argv[3].lower() == "none" else Path(sys.argv[3])
        print(evaluation(sys.argv[5:], Path(sys.argv[2]), gt_arg, Path(sys.argv[4])))
