"""Generate ``notebooks/pipeline.ipynb``: the end-to-end, reproducible pipeline notebook.

The notebook loads the trained weights stored in ``weights/`` (no re-training),
predicts the test H-alpha images, writes ``submission.csv``, visualises the
predicted filaments, and (optionally, if the training annotations are present)
reproduces the hold-out evaluation: Panoptic Quality, IoU / Dice distributions
and one-to-many / many-to-one statistics.

The committed ``notebooks/pipeline.ipynb`` additionally contains the outputs of the reproducibility
run (``scripts/run_notebook.py`` on a Kaggle T4); running this builder resets it to an unexecuted copy.

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
| 4. instance extraction | probability map | candidate components -> learned filter -> outlines at p > 0.5 | `filseg.postprocess.instances_from_prob` |
| 5. encoding | instances | COCO RLE rows of `submission.csv` | `filseg.submission` |
"""),
    ("markdown", "## 1. Setup\n\nSet `DATA_DIR` to the unzipped competition data (`MAGFiLO_1.0_Kaggle_2026`; found "
                 "automatically next to the repository or under `/kaggle/input`). Install the pinned packages with "
                 "`pip install -r requirements.txt` if needed.\n\n"
                 "`PROFILE = \"full\"` reproduces the submission with all ten networks; `\"fast\"` uses the five "
                 "EfficientNetV2-S networks with their own post-processing (about half the inference time, "
                 "slightly lower out-of-fold PQ; see the table in section 2)."),
    ("code", """import json, os, sys, time
from pathlib import Path

PROFILE = os.environ.get("FILSEG_PROFILE", "full")      # "full" (submission) or "fast"
LIMIT = int(os.environ.get("FILSEG_LIMIT_TEST", "0"))    # > 0: only the first LIMIT test images (quick check)
SAVE_PROBS = os.environ.get("FILSEG_SAVE_PROBS")         # optional: also save the ensemble probability maps (.npz)

REPO = Path.cwd().resolve().parent if Path.cwd().name == "notebooks" else Path.cwd().resolve()
sys.path.insert(0, str(REPO))

CANDIDATES = [REPO / "MAGFiLO_1.0_Kaggle_2026", REPO.parent / "MAGFiLO_1.0_Kaggle_2026"]
DATA_DIR = next((p for p in CANDIDATES if (p / "test" / "test_images").exists()), None)
if DATA_DIR is None:  # $FILSEG_DATA or the competition data mounted under /kaggle/input
    from filseg.train import find_data_dir
    DATA_DIR = find_data_dir(None)
WEIGHTS_DIR = Path(os.environ.get("FILSEG_WEIGHTS", REPO / "weights"))
OUT_CSV = REPO / ("submission.csv" if PROFILE == "full" else f"submission_{PROFILE}.csv")
print("repo:", REPO)
print("data:", DATA_DIR, "| exists:", DATA_DIR.exists())
print("profile:", PROFILE, "| test images:", LIMIT or "all")"""),
    ("code", """import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from filseg import data, infer, postprocess, submission
from filseg.postprocess import PostConfig

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("torch", torch.__version__, "| device:", device)"""),
    ("markdown", """## 2. Trained models and post-processing configuration

| profile | networks | out-of-fold PQ (nested CV, 707 images) | T4 GPU time per image |
|---|---|---|---|
| `full` | 5 x U-Net ResNet34 + 5 x U-Net EfficientNetV2-S (one per CV fold) | 0.447 | ~12.8 s networks + ~0.5 s post-processing |
| `fast` | 5 x U-Net EfficientNetV2-S | 0.444 | ~7.1 s + ~0.5 s |

Post-processing: threshold, connected components, a learned component filter (gradient-boosted
trees on per-component features, stored as JSON in the manifest) and redrawing of each kept
instance at probability 0.5."""),
    ("code", """manifest = json.loads((WEIGHTS_DIR / "manifest.json").read_text())
if PROFILE == "fast":
    entries = [m for m in manifest["models"] if m["file"] in manifest["fast"]["models"]]
    post = manifest["fast"]["post"]
else:
    entries, post = manifest["models"], manifest["post"]
cfg = PostConfig(**post)
weight_files = [WEIGHTS_DIR / m["file"] for m in entries]
models = infer.load_models(weight_files, device)
pd.DataFrame(entries)[["file", "encoder", "res", "norm", "fold"]]"""),
    ("code", """summary = {k: v for k, v in post.items() if k != "scorer"}
if post.get("scorer"):
    summary["scorer"] = f"{len(post['scorer']['trees'])} trees on {post['scorer']['features']}"
summary"""),
    ("markdown", "## 3. Inference on the test images -> `submission.csv`\n\n"
                 "The `full` profile also writes `submission_rule.csv`: the same probability maps with the "
                 "three-parameter rule chosen by nested cross-validation (mean probability >= 0.85, peak >= 0.99) "
                 "instead of the learned filter."),
    ("code", """test_files = sorted((DATA_DIR / "test" / "test_images").glob("*.jp*g"))
if LIMIT:
    test_files = test_files[:LIMIT]
variants = {k: PostConfig(**v) for k, v in manifest.get("variants", {}).items()} if PROFILE == "full" else {}
timing, variant_frames = {}, {}
t0 = time.time()
df = infer.make_submission(models, DATA_DIR, cfg, device, tta=manifest.get("tta", 4), timing=timing, limit=LIMIT,
                           save_probs=Path(SAVE_PROBS) if SAVE_PROBS else None,
                           variants=variants, variant_frames=variant_frames)
elapsed = time.time() - t0
df, report = submission.write(list(df.itertuples(index=False, name=None)), OUT_CSV, {f.stem for f in test_files})
print(f"{len(test_files)} images, {len(df)} filaments -> {OUT_CSV}")
print(f"end-to-end {elapsed / len(test_files):.2f} s per image on {device}: networks "
      f"{timing['network_s_per_image']:.2f} s + post-processing {timing['post_s_per_image']:.2f} s "
      f"({len(models)} models, TTA x{manifest.get('tta', 4)})")
print(report)
for name, vdf in variant_frames.items():
    vcsv = OUT_CSV.with_name(f"submission_{name}.csv")
    vdf, vreport = submission.write(list(vdf.itertuples(index=False, name=None)), vcsv, {f.stem for f in test_files})
    print(f"variant {name!r}: {len(vdf)} filaments -> {vcsv}; {vreport}")
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
if RUN_VALIDATION and train_json.exists() and all("fold" in m for m in entries):
    groups = data.group_by_image(data.load_entries(train_json))
    folds = data.image_folds(list(groups), manifest.get("nfolds", 5), manifest.get("seed", 2026))
    preds, gt = {}, {}
    for k in sorted({m["fold"] for m in entries}):
        fold_models = [mm for mm, meta in zip(models, entries) if meta["fold"] == k]
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
    for i, (kind, src) in enumerate(CELLS):
        cell = {"cell_type": kind, "id": f"cell-{i:02d}", "metadata": {}, "source": src.splitlines(keepends=True)}
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        nb["cells"].append(cell)
    out = ROOT / "notebooks" / "pipeline.ipynb"
    out.write_text(json.dumps(nb, indent=1), encoding="utf-8", newline="\n")
    print("wrote", out)


if __name__ == "__main__":
    main()
