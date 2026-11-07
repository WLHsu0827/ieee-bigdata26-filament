# Technical report draft (to be transferred to the official Overleaf template, 4 pages)

Working title: **Consensus-trained full-resolution U-Nets with PQ-aware instance
selection for solar filament segmentation**

Numbers marked `[TBD]` are filled in from `docs/experiments.md` when the final
cross-validation runs are complete.

## Abstract

We segment individual solar filaments in GONG H-alpha full-disk images
(MAGFiLO). An analysis of the multi-annotator training images shows that two
experts agree at a Panoptic Quality (PQ) of only 0.34, and that a two-person
consensus does not predict a third annotator better. We therefore (i) train
U-Nets at the native 2048 px resolution on the soft consensus of all
annotators, (ii) recover instances as connected components, which is lossless
for this data set because annotated filaments practically never touch, and
(iii) select instances with a rule derived from an exact per-component
decomposition of PQ and chosen by nested cross-validation. An ensemble of
ResNet34 and EfficientNetV2-S U-Nets reaches an out-of-fold PQ of `[TBD]`
(human agreement 0.34) at `[TBD]` s per image on one T4 GPU.

## 1. Data and label analysis

* 707 training observations (6 GONG sites, 2011-2022), 1,154 annotator sets
  ("entries"), 8,199 filament polygons; 180 test observations. 411 images have
  one annotator, 145 two and 151 three. Filament areas range from 9 to 37,739 px
  (median 1,228 px, 10th percentile 409 px).
* Instances practically never touch: the 8,199 polygons form 8,219 connected
  components of their per-entry unions (10 of 1,154 entries contain touching
  filaments). Instance segmentation therefore reduces to semantic segmentation
  followed by connected components.
* **Annotator agreement** (official PQ; 296 multi-annotator images;
  `scripts/annotator_agreement.py`):

  | "prediction" scored against annotator *e* | PQ | SQ | RQ |
  |---|---|---|---|
  | another single annotator (all ordered pairs) | 0.340 | 0.634 | 0.536 |
  | union of the two other annotators (3-annotator images) | 0.332 | 0.635 | 0.524 |
  | intersection of the two other annotators | 0.314 | 0.637 | 0.492 |
  | consensus of all annotators *including e*, fraction > 0.45 (oracle) | 0.648 | 0.804 | 0.806 |

  Annotator style (tight vs. loose outlines, inclusion of faint filaments,
  splitting of filament chains) dominates the metric.

## 2. The metric and what it rewards

* PQ = sum of IoU over matched pairs / (TP + FP/2 + FN/2); a pair matches if
  IoU > 0.5; every annotator entry is scored separately against the same
  predictions, and counts are pooled over the data set. Our implementation
  (`filseg/metrics.py`) reproduces the organisers' self-evaluation notebook
  exactly and is ~6x faster (bounding-box crops).
* **Per-component decomposition.** For disjoint predicted instances, component
  *c* adds v_c = sum_e IoU_e(c)·[IoU_e(c) > 0.5] to the numerator and |E(c)|/2
  to the denominator (it is a TP or an FP for each of the image's |E(c)|
  entries), so PQ(K) = sum_{c in K} v_c / (0.5 sum_{c in K} |E(c)| + 0.5 G),
  with G the number of GT filaments. Keeping a candidate improves PQ iff its
  average matched IoU per entry exceeds PQ/2: at PQ around 0.4 a candidate is
  worth keeping when it has roughly a 30 % chance of matching an annotator.
  `filseg/components.py` uses the identity to score any selection rule exactly
  and instantly, which makes nested cross-validation of the post-processing
  cheap (unit-tested against the official computation).
* Error profile of a 1024 px model (fold 0): GT filaments below 300 px are
  almost never matched (1 %), 300-600 px in 33 %, larger ones in 64-75 %;
  23 % of GT filaments are not touched by any prediction and ~18 % are touched
  but reach IoU <= 0.5. Boundary precision therefore matters, which motivates
  native-resolution prediction.

## 3. Method

* **Input.** One grey channel; each image is normalised by the mean and
  standard deviation of its pixels inside the solar disk (largest bright
  component, eroded), which removes exposure differences between sites.
* **Target.** Soft consensus: the fraction of an image's annotators that cover
  each pixel (polygons rasterised with pycocotools at full resolution).
* **Network.** U-Net (segmentation-models-pytorch) with ImageNet-pretrained
  ResNet34 or EfficientNetV2-S (`tf_efficientnetv2_s`, timm) encoders.
* **Training.** 768 px crops of the 2048 px image, 70 % centred on a filament
  pixel; a 1.6x larger window is rotated (any angle), scaled (0.9-1.1),
  flipped and centre-cropped, followed by gamma / contrast / blur / noise
  jitter. Loss BCE + soft Dice; AdamW (lr 3e-4, weight decay 1e-4), linear
  warm-up and cosine decay, mixed precision, 30 epochs of 4 crops per image,
  EMA of the weights (decay 0.999 with warm-up). 5-fold cross-validation
  grouped by image (all annotators of an image in one fold).
* **Inference.** Full 2048 px image, flip TTA (x4), average of the fold models
  of both encoders.
* **Instances.** Threshold the probability map, closing (radius 4 px), hole
  filling, 8-connected components.
* **Selection.** Keep a component if its area, mean and peak probability
  exceed thresholds chosen by nested CV (outer folds = CV folds; the inner
  choice maximises pooled PQ on the other four folds' out-of-fold maps).
* **Outline.** Detection and extent are decoupled: components are detected at
  a low threshold (0.3-0.35, which joins fragments and finds faint
  filaments), but each kept instance is redrawn where the probability exceeds
  0.5, i.e. where most annotators would draw it. Growing outlines below the
  detection threshold hurts; shrinking to 0.45-0.5 gains a little. Because
  shrinking acts on each component independently, the PQ decomposition stays
  exact and the outline level is part of the nested selection.
* **Robust selection objective.** Besides plain PQ we optimise the expected PQ
  when each ground-truth filament is labelled only with probability r
  (PQ_r = r S / (r TP + FP/2 + (1-r) TP/2 + r FN/2)); r < 1 penalises marginal
  detections more. On fold 0 the rule chosen with r = 0.77 is the same in 4 of
  5 outer folds and has a *higher* nested plain PQ (0.425 vs 0.417) than the
  rule chosen with plain PQ, i.e. the more conservative objective also acts as
  a regulariser against selection noise.

## 4. Experiments

| model | resolution | fold-0 PQ | out-of-fold PQ (nested) | public LB |
|---|---|---|---|---|
| classical darkness threshold | 2048 | 0.092 | - | 0.07 |
| U-Net ResNet34 | 512 (full disk) | 0.365 | - | 0.30 |
| U-Net ResNet34 | 1024 (full disk) | 0.416 (nested over 5 sub-folds: 0.417; robust objective 0.425) | - | 0.34 |
| U-Net ResNet34, disk norm | 2048 (crops) | [TBD] | [TBD] | [TBD] |
| U-Net EfficientNetV2-S, disk norm | 2048 (crops) | [TBD] | [TBD] | [TBD] |
| ensemble 2 x 5 folds | 2048 | - | [TBD] | [TBD] |

* Public LB vs validation: the model predicts as many filaments per test image
  as per validation image (6.1 vs 6.6), yet the public score is ~0.82 of the
  validation score for both the 512 px and the 1024 px model. Two explanations
  were checked against these two models: test labels sparser than training
  labels (each filament kept with probability r; r = 0.68 reproduces both
  public scores to the displayed precision), and test labels without the
  "Unidentifiable" class (34 % of the training filaments; predicts 0.367 / 0.311
  vs public 0.34 / 0.30, a worse fit). We do not tune on the public LB; the
  sparse-label view motivates the robust selection objective above.
* Efficiency: [TBD] s per image (network) + [TBD] s (post-processing) on one T4.

## 5. Conclusion

[TBD]

## Acknowledgement (required text, from the competition's Acknowledgment page)

This work utilizes GONG data obtained by the NSO Integrated Synoptic Program, managed by the National
Solar Observatory, which is operated by the Association of Universities for Research in Astronomy
(AURA), Inc. under a cooperative agreement with the National Science Foundation and with contribution
from the National Oceanic and Atmospheric Administration. The GONG network of instruments is hosted by
the Big Bear Solar Observatory, High Altitude Observatory, Learmonth Solar Observatory, Udaipur Solar
Observatory, Instituto de Astrofísica de Canarias, and Cerro Tololo Interamerican Observatory.

## References (to be converted to BibTeX)

* MAGFiLO: Manually Annotated GONG Filaments from H-Alpha Observations, Scientific Data (2024),
  doi:10.1038/s41597-024-03876-y.
* A. Kirillov, K. He, R. Girshick, C. Rother, P. Dollár. Panoptic Segmentation. CVPR 2019,
  doi:10.1109/CVPR.2019.00963.
* O. Ronneberger, P. Fischer, T. Brox. U-Net: Convolutional Networks for Biomedical Image
  Segmentation. MICCAI 2015.
* K. He, X. Zhang, S. Ren, J. Sun. Deep Residual Learning for Image Recognition. CVPR 2016.
* M. Tan, Q. V. Le. EfficientNetV2: Smaller Models and Faster Training. ICML 2021.
* Q. Xie, M.-T. Luong, E. Hovy, Q. V. Le. Self-training with Noisy Student Improves ImageNet
  Classification. CVPR 2020 (only if the semi-supervised students are used).
* G. Hinton, O. Vinyals, J. Dean. Distilling the Knowledge in a Neural Network. arXiv:1503.02531, 2015
  (only if used).
* J. W. Harvey et al. The Global Oscillation Network Group (GONG) Project. Science 272, 1284 (1996).
* P. Iakubovskii. Segmentation Models Pytorch. GitHub, 2019. R. Wightman. PyTorch Image Models. GitHub, 2019.