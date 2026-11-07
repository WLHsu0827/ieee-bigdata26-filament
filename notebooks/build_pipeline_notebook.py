"""Generate ``notebooks/pipeline.ipynb``: the end-to-end, reproducible pipeline notebook.

The notebook loads the trained weights stored in ``weights/`` (no re-training),
predicts the test H-alpha images, writes ``submission.csv``, visualises the
predicted filaments, and (optionally, if the training annotations are present)
reproduces the hold-out evaluation: Panoptic Quality, IoU / Dice distributions
and one-to-many / many-to-one statistics.

Usage:
    python notebooks/build_pipeline_notebook.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CELLS = [
    ("markdown", """# Solar filament segmentation -- end-to-end pipeline

IEEE BigData Cup 2026, *Solar Filament Segmentation Challenge* (MAGFiLO, GONG H-alpha).

This notebook reproduces our submission **without re-training**: it loads the network
weights and the post-processing configuration stored in `weights/`, predicts every test
image, and writes `submission.csv` in the competition format.

![pipeline](../docs/figures/pipeline.png)

| stage | input | output | code |
|---|---|---|---|
| 1. pre-processing | 2048 x 2048 grey JPEG | resized, normalised image | `filseg.model.predict_images` |
| 2. segmentation network(s) | image | per-pixel filament probability (consensus of annotators) | `filseg.model` |
| 3. ensembling + TTA | probabilities of each model / flip | mean probability, 2048 x 2048 | `filseg.infer.predict_full` |
| 4. instance extraction | probability map | disjoint filament instances | `filseg.postprocess.instances_from_prob` |
| 5. encoding | instances | COCO RLE rows of `submission.csv` | `filseg.submission` |
"""),
    ("markdown", "## 1. Setup\n\nSet `DATA_DIR` to the unzipped competition data (`MAGFiLO_1.0_Kaggle_2026`). "
                 "Install the pinned packages with `pip install -r requirements.txt` if needed."),
    ("code", """import json, sys, time
from pathlib import Path

REPO = Path.cwd().resolve().parent if Path.cwd().name == "notebooks" else Path.cwd().resolve()
sys.path.insert(0, str(REPO))

CANDIDATES = [REPO / "MAGFiLO_1.0_Kaggle_2026", REPO.parent / "MAGFiLO_1.0_Kaggle_2026",
              Path("/kaggle/input/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026")]
DATA_DIR = next((p for p in CANDIDATES if (p / "test" / "test_images").exists()), CANDIDATES[0])
WEIGHTS_DIR = REPO / "weights"
OUT_CSV = REPO / "submission.csv"
print("repo:", REPO)
print("data:", DATA_DIR, "| exists:", DATA_DIR.exists())"""),
    ("code", """import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from filseg import data, infer, postprocess, submission
from filseg.postprocess import PostConfig

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("torch", torch.__version__, "| device:", device)"""),
    ("markdown", "## 2. Trained models and post-processing configuration"),
    ("code", """manifest = json.loads((WEIGHTS_DIR / "manifest.json").read_text())
cfg = PostConfig(**manifest["post"])
weight_files = [WEIGHTS_DIR / m["file"] for m in manifest["models"]]
models = infer.load_models(weight_files, device)
pd.DataFrame(manifest["models"])"""),
    ("code", "cfg"),
    ("markdown", "## 3. Inference on the test images -> `submission.csv`"),
    ("code", """test_files = sorted((DATA_DIR / "test" / "test_images").glob("*.jp*g"))
timing = {}
t0 = time.time()
df = infer.make_submission(models, DATA_DIR, cfg, device, tta=manifest.get("tta", 4), timing=timing)
elapsed = time.time() - t0
df, report = submission.write(list(df.itertuples(index=False, name=None)), OUT_CSV, {f.stem for f in test_files})
print(f"{len(test_files)} images, {len(df)} filaments -> {OUT_CSV}")
print(f"end-to-end {elapsed / len(test_files):.2f} s per image on {device}: networks "
      f"{timing['network_s_per_image']:.2f} s + post-processing {timing['post_s_per_image']:.2f} s "
      f"({len(models)} models, TTA x{manifest.get('tta', 4)})")
print(report)
df.head()"""),
    ("markdown", "## 4. Visual check of the predicted filaments"),
    ("code", """from filseg import rle
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for ax, f in zip(axes, test_files[:: max(len(test_files) // 3, 1)]):
    img = data.read_image(f)
    rows = df[df["filament_id"].str.rsplit("_", n=1).str[0] == f.stem]
    lab = np.zeros(img.shape, np.int32)
    for k, counts in enumerate(rows["segmentation_rle"], start=1):
        lab[rle.decode(counts)] = k
    ax.imshow(img, cmap="gray")
    ax.imshow(np.ma.masked_equal(lab, 0), cmap="tab20", alpha=0.6, interpolation="nearest")
    ax.set_title(f"{f.stem}: {len(rows)} filaments")
    ax.axis("off")
plt.tight_layout()"""),
    ("markdown", """## 5. Out-of-fold evaluation

The organisers' self-evaluation quantities (Panoptic Quality, IoU / Dice distributions,
one-to-many and many-to-one relations) on all 707 training images, each predicted only by
the models that did not see it during training, every annotator scored separately.
The figure and numbers below were produced by `scripts/make_figures.py evaluation` from the
saved out-of-fold probability maps; set `RUN_VALIDATION = True` to recompute them here from
the weights (about 15-30 min on one T4 GPU)."""),
    ("code", """from IPython.display import Image, display

eval_png, eval_json = REPO / "docs" / "figures" / "evaluation.png", REPO / "docs" / "evaluation.json"
if eval_png.exists():
    display(Image(filename=str(eval_png)))
if eval_json.exists():
    print(json.dumps(json.loads(eval_json.read_text()), indent=1)[:3000])"""),
    ("code", """from filseg import analysis, metrics
from filseg.metrics import Crop

RUN_VALIDATION = False  # set to True to recompute the out-of-fold evaluation from the weights
train_json = DATA_DIR / "train" / data.TRAIN_JSON
if RUN_VALIDATION and train_json.exists() and all("fold" in m for m in manifest["models"]):
    groups = data.group_by_image(data.load_entries(train_json))
    folds = data.image_folds(list(groups), manifest.get("nfolds", 5), manifest.get("seed", 2026))
    preds, gt = {}, {}
    for k in sorted({m["fold"] for m in manifest["models"]}):
        fold_models = [mm for mm, meta in zip(models, manifest["models"]) if meta["fold"] == k]
        ids = [i for i in groups if folds[i] == k]
        images = [data.read_image(DATA_DIR / "train" / "train_images" / groups[i][0].file_name) for i in ids]
        for i, p in zip(ids, infer.predict_full(fold_models, images, device, tta=manifest.get("tta", 4))):
            lab, _ = postprocess.instances_from_prob(p, cfg)
            preds[i] = postprocess.label_to_crops(lab)
            for e in groups[i]:
                gt[e.entry_id] = [Crop.from_full(m) for m in e.masks()]
    res, overlap_df = metrics.evaluate(gt, preds)
    print("out-of-fold:", res)
    iou, dice = analysis.overlap_distributions(overlap_df)
    gdeg, pdeg = metrics.matching_degrees(overlap_df)
    print(f"mean IoU / Dice of overlapping pairs: {iou.mean():.3f} / {dice.mean():.3f}")
    print("GT -> number of overlapping predictions:", dict(sorted(gdeg.items())))
    print("prediction -> number of overlapping GT:", dict(sorted(pdeg.items())))"""),
    ("markdown", """## 6. Re-training (reference only)

Each model in `weights/` was produced by `python -m filseg.train` with the arguments stored
in `weights/manifest.json` (field `train_args`). Training is not needed to reproduce the
submission."""),
    ("code", "pd.DataFrame([{\"file\": m[\"file\"], \"command\": \"python -m filseg.train \" + m.get(\"train_args\", \"\")} for m in manifest[\"models\"]])"),
]


def main() -> None:
    nb = {"nbformat": 4, "nbformat_minor": 5,
          "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"},
                       "language_info": {"name": "python"}},
          "cells": []}
    for kind, src in CELLS:
        cell = {"cell_type": kind, "metadata": {}, "source": src.splitlines(keepends=True)}
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        nb["cells"].append(cell)
    out = ROOT / "notebooks" / "pipeline.ipynb"
    out.write_text(json.dumps(nb, indent=1), encoding="utf-8", newline="\n")
    print("wrote", out)


if __name__ == "__main__":
    main()
