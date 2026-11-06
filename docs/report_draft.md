# Technical report draft (to be transferred to the official Overleaf template)

Working title: *Consensus-trained U-Nets with PQ-aware instance selection for solar
filament segmentation*

## 1. Data and label analysis

* 707 training observations (GONG H-alpha, 2048 x 2048, 6 sites, 2011-2022),
  1,154 annotator sets ("entries"), 8,199 filament polygons; 180 test observations.
  411 images have one annotator, 145 two and 151 three.
* Filament areas span 9 to 37,739 px (median 1,228 px, 10th percentile 409 px).
* Instances practically never touch: the 8,199 polygons form 8,219 connected
  components of their per-entry unions (10 of 1,154 entries contain touching
  filaments). Instance segmentation therefore reduces to semantic segmentation
  followed by connected components.
* **Annotator agreement (Panoptic Quality, official definition)** on the 296
  multi-annotator training images (`scripts/annotator_agreement.py`,
  `docs/annotator_agreement.json`):

  | prediction scored against annotator *e* | PQ | SQ | RQ |
  |---|---|---|---|
  | another single annotator (all ordered pairs) | 0.340 | 0.634 | 0.536 |
  | union of the two other annotators (3-annotator images) | 0.332 | 0.635 | 0.524 |
  | intersection of the two other annotators | 0.314 | 0.637 | 0.492 |
  | consensus of all annotators *including e*, fraction > 0.45 (oracle) | 0.648 | 0.804 | 0.806 |

  Human agreement is low (PQ 0.34) and a two-person consensus does not predict a
  third annotator better than one person does: annotator style (tight vs. loose
  outlines, inclusion of faint filaments, splitting of filament chains) dominates
  the metric. This matches the organisers' statement that PQ > 0.35 is valuable.

## 2. Metric and its consequences

* PQ = sum of IoU over matched pairs / (TP + FP/2 + FN/2); a pair matches if IoU > 0.5;
  every annotator entry is scored separately against the same predictions and
  the counts are pooled over the data set. Our implementation (`filseg/metrics.py`)
  reproduces the organisers' self-evaluation notebook exactly (difference 0.0 on
  a test set of perturbed masks) and is ~6x faster (bounding-box crops).
* **Decomposition.** For disjoint predicted instances, component *c* contributes
  v_c = sum over entries e of IoU_e(c)·[IoU_e(c) > 0.5] to the numerator and
  |E(c)|/2 to the denominator (it is either a TP or an FP for each entry). Hence
  PQ(K) = sum_{c in K} v_c / (0.5 sum_{c in K} |E(c)| + 0.5 G), and keeping a
  candidate improves PQ iff its average matched IoU per entry exceeds PQ/2.
  With PQ around 0.4 a candidate is worth keeping when it has roughly a 30 %
  chance of matching an annotator: the metric rewards recall more than a
  pixel-level F1 would suggest. `filseg/components.py` uses this identity to
  evaluate any selection rule exactly and instantly (unit-tested against the
  official computation).

## 3. Method (in progress)

* Input: single grey channel, full disk resized to the network resolution.
* Target: soft consensus = fraction of the image's annotators covering each pixel.
* Network: U-Net (segmentation-models-pytorch) with ImageNet-pretrained encoder;
  loss BCE + soft Dice; AdamW, cosine schedule with warm-up, EMA of weights.
* Augmentation: flips, arbitrary rotation, scale 0.9-1.1, gamma / contrast /
  blur / noise; crop-first sampling around filaments for 2048 px training.
* Post-processing: threshold, optional closing, hole filling, connected
  components, hysteresis-style selection (minimum area, peak and mean
  probability), optional grouping of nearby fragments; tuned for PQ on
  out-of-fold predictions.

## 4. Experiments (in progress)

See `docs/experiments.md`.
