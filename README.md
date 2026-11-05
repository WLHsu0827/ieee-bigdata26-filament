# Solar filament instance segmentation (IEEE BigData Cup 2026)

Code for the *Solar Filament Segmentation Challenge 2026*
(Kaggle community competition `filament-segmentation-2026`, data: MAGFiLO,
GONG H-alpha full-disk observations, 2048 x 2048).

The task is to output one mask per filament; submissions are ranked by
Panoptic Quality (PQ, IoU > 0.5 matching) against every annotator's set of
polygons.

## Approach (summary)

1. **Soft consensus target.** Up to three annotators labelled each image
   independently; we train on the per-pixel fraction of annotators that mark
   a pixel as filament.
2. **Semantic segmentation network.** U-Net (segmentation-models-pytorch)
   on the single grey channel, BCE + Dice loss, flip / rotation / photometric
   augmentation, flip test-time augmentation.
3. **Instances = connected components.** In the training annotations only 10 of
   1,154 annotator sets contain touching filaments (8,199 polygons vs 8,219
   connected components of their union), so instances are recovered from the
   thresholded probability map by connected components, hole filling and a
   minimum-area filter, all tuned for PQ on a hold-out fold.

See `docs/experiments.md` for the experiment log.

## Repository layout

| path | content |
|------|---------|
| `filseg/data.py` | COCO annotation parsing, polygon rasterisation, soft targets, folds |
| `filseg/rle.py` | COCO RLE encode / decode (submission format) |
| `filseg/metrics.py` | Panoptic Quality identical to the organisers' self-evaluation notebook |
| `filseg/postprocess.py` | probability map -> filament instances |
| `filseg/model.py` | network construction and TTA prediction |
| `filseg/train.py` | training, hold-out validation, post-processing tuning, submission |
| `filseg/tune.py` | grid search of post-processing parameters on validation PQ |
| `filseg/infer.py` | inference / ensembling from saved weights |
| `filseg/classical.py` | traditional (non-learning) darkness-threshold baseline |
| `scripts/` | GT cache builder, classical baseline |
| `kaggle/build_kernel.py` | packs the code into a self-contained Kaggle script kernel |

## Usage

```bash
pip install -r requirements.txt
# train on 4/5 of the images, tune post-processing on the hold-out fold, predict test
python -m filseg.train --data MAGFiLO_1.0_Kaggle_2026 --out runs/r34_512 --res 512 --predict-test
# inference with saved weights
python -m filseg.infer --data MAGFiLO_1.0_Kaggle_2026 --weights runs/r34_512/model.pt \
    --post runs/r34_512/post.json --out submission.csv
```

## Data and licence

The MAGFiLO images and annotations are provided by the competition organisers
(GONG data courtesy of NSO/AURA/NSF) and are **not** redistributed here.
Training uses only the competition's training annotations; inference uses only
the test H-alpha images. Code is released under the MIT licence.
