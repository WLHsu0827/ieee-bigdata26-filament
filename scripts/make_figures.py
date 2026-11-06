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


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "pipeline"
    if what == "pipeline":
        print(pipeline_diagram())
