# Solar filament instance segmentation (IEEE BigData Cup 2026)

Code for the *Solar Filament Segmentation Challenge 2026*
(Kaggle community competition `filament-segmentation-2026`; data: MAGFiLO,
GONG H-alpha full-disk observations, 2048 x 2048 px).

The task is to output one mask per filament. Submissions are ranked by
Panoptic Quality (PQ: a predicted and a ground-truth filament match when their
IoU exceeds 0.5), computed against every annotator's set of polygons separately.

![pipeline](docs/figures/pipeline.png)

## Approach

1. **Label analysis.** Two experts annotating the same image agree at PQ 0.34,
   and the union / intersection of two annotators predicts a third no better
   (0.33 / 0.31). Annotator style dominates the metric
   (`scripts/annotator_agreement.py`, `docs/annotator_agreement.json`).
2. **Soft consensus target.** Up to three annotators labelled each image; the
   network learns the per-pixel fraction of annotators that mark a pixel.
3. **Segmentation network.** U-Net (segmentation-models-pytorch) on the single
   grey channel at the native 2048 px resolution: trained on 768 px crops
   (mostly centred on filaments) with flips, arbitrary rotations, scaling and
   photometric jitter; each image is normalised with its own statistics inside
   the solar disk; BCE + soft Dice; AdamW with warm-up and cosine decay; EMA of
   the weights. ResNet34 and EfficientNetV2-S encoders, 5-fold cross-validation.
4. **Instances = connected components.** Annotated filaments practically never
   touch (8,199 polygons form 8,219 connected components), so instances are
   the connected components of the thresholded (ensemble-averaged, flip-TTA)
   probability map, with hole filling.
5. **PQ-aware selection.** For disjoint instances PQ decomposes over
   components: keeping a candidate improves PQ iff its expected matched IoU per
   annotator exceeds PQ / 2 (`filseg/components.py`). Selection rules (minimum
   area, peak and mean probability) are chosen by nested cross-validation on
   out-of-fold predictions (`scripts/select_post.py`).

Experiments and numbers: `docs/experiments.md`. Report draft: `docs/report_draft.md`.

## Reproducing the submission

`notebooks/pipeline.ipynb` runs the whole inference pipeline from the weights
in `weights/` (no training needed): it predicts every test image, writes
`submission.csv`, shows predictions and, if the training annotations are
available, recomputes the hold-out PQ, IoU / Dice distributions and
one-to-many / many-to-one statistics.

```bash
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute notebooks/pipeline.ipynb   # or open and Run All
```

## Training from scratch

```bash
# one fold (fold k is held out); writes model.pt, val_probs.npz, test_probs.npz, logs
python -m filseg.train --data MAGFiLO_1.0_Kaggle_2026 --out runs/r34_f0 --fold 0 \
    --res 2048 --crop 768 --epoch-mult 4 --norm disk --encoder resnet34 --bs 8 --epochs 30 \
    --lr 3e-4 --ema 0.999 --tta 4 --predict-test --post configs/post_c1_nested.json
# several folds in sequence
python -m filseg.cv --folds 1 2 3 4 --out runs/r34 -- --data MAGFiLO_1.0_Kaggle_2026 --res 2048 ...
# choose the post-processing by nested CV on all out-of-fold maps
python scripts/select_post.py --probs runs/*/fold*/val_probs.npz --data MAGFiLO_1.0_Kaggle_2026 --out runs/sel
# ensemble submission from the saved test maps
python scripts/ensemble_submit.py --probs runs/*/fold*/test_probs.npz --post runs/sel/post_selected.json \
    --test-dir MAGFiLO_1.0_Kaggle_2026/test/test_images --out submission.csv
# collect the weights for the notebook
python scripts/make_weights.py --runs runs/*/fold* --post runs/sel/post_selected.json --out weights
```

## Repository layout

| path | content |
|------|---------|
| `filseg/data.py` | COCO annotation parsing, polygon rasterisation, soft targets, folds |
| `filseg/rle.py`, `filseg/submission.py` | COCO RLE and validated submission writing |
| `filseg/metrics.py` | Panoptic Quality identical to the organisers' self-evaluation notebook |
| `filseg/augment.py` | crop-first geometric and photometric augmentation |
| `filseg/model.py` | network, normalisation, fp16 checkpoints, flip / rotation TTA prediction |
| `filseg/train.py`, `filseg/cv.py` | training (EMA, OOM fallback, time budget), CV driver |
| `filseg/postprocess.py`, `filseg/tune.py` | probability map -> instances; post-processing scoring |
| `filseg/components.py` | per-component PQ decomposition, nested rule selection |
| `filseg/analysis.py` | error analysis (by size, aggregation, matching degrees) |
| `filseg/infer.py` | inference / ensembling from saved weights |
| `filseg/classical.py` | traditional (non-learning) darkness-threshold baseline |
| `scripts/` | label agreement, OOF analysis, rule selection, ensembling, figures, checks |
| `configs/` | post-processing configurations |
| `kaggle/` | builders for self-contained Kaggle kernels and Colab notebooks |
| `tests/` | unit tests (`python -m pytest tests`) |

## Data and licence

The MAGFiLO images and annotations are provided by the competition organisers
(GONG data courtesy of NSO/AURA/NSF) and are **not** redistributed here.
Training uses only the competition's training annotations; inference uses only
the test H-alpha images. Code is released under the MIT licence.
