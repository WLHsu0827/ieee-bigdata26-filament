# Solar filament segmentation: extended technical description

Companion to the four-page report **Consensus-Trained Full-Resolution U-Nets with PQ-Aware
Instance Selection** (`docs/report/main.tex`, written in the competition's official template; build the
PDF with `python docs/report/build_pdf.py --out build/`). It describes the same method in more detail and
with intermediate numbers; every experiment, including the negative ones, is logged in
`docs/experiments.md`.

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
ResNet34 and EfficientNetV2-S U-Nets with a learned component filter reaches a
nested out-of-fold PQ of 0.447 (human agreement 0.34) at about 8 s per image on one
T4 GPU; five EfficientNetV2-S networks alone reach 0.444 at about 5 s per image.
The two final submissions use the ten networks with the robust nested-CV rule (public
leaderboard 0.39) and with the learned filter (0.38).

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
  warm-up and cosine decay, mixed precision, 4 crops per image and epoch for 30
  epochs (ResNet34, batch 8) or within a 1.5 h budget per fold (EfficientNetV2-S,
  batch 6, 21-24 epochs), EMA of the weights (decay 0.999 with warm-up). 5-fold
  cross-validation grouped by image (all annotators of an image in one fold);
  1.3-1.5 h per fold on one T4 GPU.
* **Inference.** Full 2048 px image, flip TTA (x4), average of the fold models
  of both encoders. Both encoders use the same five folds, so the out-of-fold
  map of a training image, on which all selection and evaluation below is done,
  averages the two networks that did not see it (one for the fast profile); the
  submission averages all ten. A label-free check on the 180 test images
  (`scripts/ensemble_shift.py`) shows that this transfer is benign: with the same
  post-processing the ten-network average keeps 3-4 % fewer instances than the
  two-network averages (5.95 vs 6.21 per image for the rule, 5.81 vs 6.02 for the
  learned filter; fold range 5.75-6.39) and agrees with each of them at a PQ-style
  score of 0.91, more than two-network selections of different folds agree with
  each other (0.87).
* **Instances.** Threshold the probability map (0.25 for the ten-network
  ensemble; 0.3 followed by a closing of radius 4 px for the five-network fast
  profile), fill holes, take 8-connected components.
* **Selection.** Keep a component if its area, mean and peak probability
  exceed thresholds chosen by nested CV (outer folds = CV folds; the inner
  choice maximises pooled PQ on the other four folds' out-of-fold maps). The
  rule of final submission A, chosen on all 707 images with the robust
  objective below (r = 0.77): mean probability >= 0.85 and peak >= 0.99, no
  area limit (`configs/post_final_rule.json`).
* **Outline.** Detection and extent are decoupled: components are detected at
  a low threshold (0.25-0.3, which joins fragments and finds faint
  filaments), but each kept instance is redrawn where the probability exceeds
  0.5, i.e. where most annotators would draw it. Growing outlines below the
  detection threshold hurts; shrinking to 0.45-0.5 gains a little. Because
  shrinking acts on each component independently, the PQ decomposition stays
  exact and the outline level is part of the nested selection.
* **Learned component filter.** The keep decision is finally made by gradient-boosted
  regression trees (scikit-learn, 200 trees of at most 15 leaves) that predict each
  candidate's per-entry matched IoU from 12 features (log area, mean / peak / 90th
  percentile / standard deviation of the probability, elongation, length, width, bounding-box
  fill ratio, distance from the disk centre, number of candidates in the image and area rank),
  trained on the out-of-fold candidates of all 707 training images; a candidate is kept if
  the prediction exceeds a threshold chosen on grouped out-of-fold predictions (the PQ/2
  criterion above, estimated: 0.28 for final submission B, `configs/post_final_full.json`, and
  0.25 for the fast profile). Nested PQ 0.447 vs 0.443 for the best three-parameter rule on
  the same candidates. The trees are exported to JSON (`filseg/scorer.py`), so inference needs
  only numpy.
* **Semi-supervised training (tested, not used).** We also tried noisy-student training with
  1,200 unlabelled GONG H-alpha images from the public NSO/NISP archive (the same JPEG product
  as the competition images, with the archive's text overlays removed by zeroing everything
  outside the inscribed circle), taken on days at least two days away from every competition
  train and test image (`docs/external_gong_manifest.csv` lists URL, date and SHA-256 of every
  file; `scripts/check_external.py` verifies the date gap and the byte-exact processing). For
  CV fold 0 the teacher was the fold's own pair of models; students of both architectures
  trained on labelled + pseudo-labelled crops (1:1) with the teachers' compute budget were
  slightly *worse* (nested fold-0 PQ 0.443 / 0.438 vs 0.445 / 0.442) and did not improve the
  ensemble, so the final models use the competition images only.
* **Robust selection objective.** Besides plain PQ we optimise the expected PQ
  when each ground-truth filament is labelled only with probability r
  (PQ_r = r S / (r TP + FP/2 + (1-r) TP/2 + r FN/2)); r < 1 penalises marginal
  detections more. On fold 0 the rule chosen with r = 0.77 is the same in 4 of
  5 outer folds and has a *higher* nested plain PQ (0.425 vs 0.417) than the
  rule chosen with plain PQ, i.e. the more conservative objective also acts as
  a regulariser against selection noise. For the ten-network ensemble the r = 0.77
  rule has nested plain PQ 0.4426 vs 0.4422 with r = 1, and it scored 0.39 vs 0.38
  on the public leaderboard.

## 4. Experiments

| model | resolution | fold-0 PQ | out-of-fold PQ (nested) | public LB |
|---|---|---|---|---|
| classical darkness threshold | 2048 | 0.092 | - | 0.07 |
| U-Net ResNet34 | 512 (full disk) | 0.365 | - | 0.30 |
| U-Net ResNet34 | 1024 (full disk) | 0.416 (nested over 5 sub-folds: 0.417; robust objective 0.425) | - | 0.34 (grid-tuned rule) / 0.35 (nested rules) |
| U-Net ResNet34, disk norm | 2048 (768 crops) | 0.428 (mean of 5 folds 0.422) | 5 folds: 0.432 | - |
| U-Net EfficientNetV2-S, disk norm | 2048 (768 crops) | 0.438 (mean of 5 folds 0.432) | 5 folds: 0.440 (rule) / 0.444 (learned filter) | 0.38 (learned filter) |
| ensemble 2 x 5 folds | 2048 | - | 0.442 (rule, plain PQ) / 0.443 (rule, r = 0.77) / **0.447** (learned filter) | 0.38 (rule, plain PQ) / 0.39 (rule, r = 0.77) / 0.38 (learned filter) |

(Fold-0 and fold-mean numbers use one fixed rule; "nested" numbers are pooled over the outer
folds of a nested cross-validation of the post-processing on all 707 out-of-fold images.)

* Error profile of the final configuration (ensemble + learned filter; 707 out-of-fold images,
  8,199 GT filaments of 1,154 annotator sets; in-sample for the filter): PQ 0.462, 62 % matched,
  SQ 0.68, 1,707 false positives (531 without any overlap). Of the unmatched GT filaments 64 % have no
  overlapping prediction (71 % of them < 1,200 px; 74 % contain probabilities >= 0.5, i.e. the
  network sees them but the filter drops them). On images with several annotators the ensemble
  (learned filter) matches 82 % of the filaments that another annotator also drew but only 32 %
  of those drawn by a single annotator, so the remaining error is largely annotator
  disagreement. 13 % of the unmatched GT filaments are covered by less than half, 10 % by a
  too-large and 9 % by a merged prediction, and 4 % are fragmented. Removing all isolated false
  positives would give PQ 0.479 and perfect fragment grouping 0.475 (upper bounds).

* Public LB vs validation: the model predicts as many filaments per test image
  as per validation image (6.1 vs 6.6), yet the public score is ~0.82 of the
  validation score for both the 512 px and the 1024 px model (0.85-0.88 for the
  2048 px models). Two explanations
  were checked against these two models: test labels sparser than training
  labels (each filament kept with probability r; r = 0.68 reproduces both
  public scores to the displayed precision), and test labels without the
  "Unidentifiable" class (34 % of the training filaments; predicts 0.367 / 0.311
  vs public 0.34 / 0.30, a worse fit). We do not tune on the public LB; the
  sparse-label view motivates the robust selection objective above. The public
  part has about 90 images: the rule and learned-filter submissions share 93 % of
  their instances, so their 0.01 difference rests on a few dozen filaments. Drawing
  random 90-image subsets of the 707 out-of-fold images, a pooled PQ scatters with
  SD 0.012 around the score on all images (`scripts/split_noise.py`), so the public
  part cannot resolve differences of a few thousandths.
* Efficiency (end to end, measured by executing the pipeline notebook on one T4 GPU for the
  180 test images): ten networks 7.3-8.0 s (flip TTA x4) + about 0.5 s (post-processing) per
  image, 8.0-8.5 s in total (three runs); five EfficientNetV2-S networks 4.7-4.8 s + 0.5 s
  (two runs); a single network 1.1-1.5 s + 0.5 s. Dihedral TTA x8 (flips and 90 degree
  rotations) instead of x4, re-predicted for all 707 out-of-fold images from the stored
  weights, raises the nested PQ by 0.001-0.003 (rule +0.003, learned filter +0.001 to +0.002)
  for twice the network time; the submission keeps x4.
* Further negative results (details and numbers in `docs/experiments.md`):
  * A larger encoder, EfficientNetV2-M (1.75x the training time of EfficientNetV2-S, 1.66 s
    instead of 1.31 s per image and network), looked promising on fold 0 (ensemble +0.005 to
    +0.011, nested within the fold), but on folds 0-2 (424 images) it did not improve the
    ensemble, neither added (-0.003 to +0.000) nor replacing ResNet34 (-0.004 to +0.001).
    A decision gate on a single fold of 142 images is too noisy.
  * Extra weight on small filaments and a 6 px band around every filament
    (`--small-weight 0.5`) did not improve the pair on fold 0 (-0.006 to +0.003).
  * Choosing each candidate's outline level (0.30-0.70) with a second regressor: no gain
    (nested 0.4478 / 0.4460 vs 0.4473 / 0.4472), although an oracle that knows each
    candidate's best outline would reach 0.538 vs 0.518.
  * Giving the learned filter each network's probabilities separately: +0.002 nested, but these
    features come from single networks out of fold and from averages of five in the submission,
    and on the test images the filter's decisions shift more between the two conditions than
    with the current features (`extent_levels.py test-shift`), so it was not adopted.

## 5. Conclusion

Native-resolution U-Nets trained on the soft consensus of the annotators, connected
components and a PQ-aware, cross-validated instance filter give an out-of-fold PQ of 0.447,
well above the agreement between two human annotators (0.34). The decisive steps were the
full 2048 px resolution (+0.05 over 512 px), exact component-level evaluation of the metric
(which turns post-processing into a cheap, honest nested-CV problem) and conservative
selection. Remaining errors are dominated by small, faint filaments on which annotators
disagree.

## Acknowledgement (required text, from the competition's Acknowledgment page)

This work utilizes GONG data obtained by the NSO Integrated Synoptic Program, managed by the National
Solar Observatory, which is operated by the Association of Universities for Research in Astronomy
(AURA), Inc. under a cooperative agreement with the National Science Foundation and with contribution
from the National Oceanic and Atmospheric Administration. The GONG network of instruments is hosted by
the Big Bear Solar Observatory, High Altitude Observatory, Learmonth Solar Observatory, Udaipur Solar
Observatory, Instituto de Astrofísica de Canarias, and Cerro Tololo Interamerican Observatory.

## References (BibTeX: `docs/report/main.bib`)

* MAGFiLO: Manually Annotated GONG Filaments from H-Alpha Observations, Scientific Data (2024),
  doi:10.1038/s41597-024-03876-y.
* A. Kirillov, K. He, R. Girshick, C. Rother, P. Dollár. Panoptic Segmentation. CVPR 2019,
  doi:10.1109/CVPR.2019.00963.
* O. Ronneberger, P. Fischer, T. Brox. U-Net: Convolutional Networks for Biomedical Image
  Segmentation. MICCAI 2015.
* K. He, X. Zhang, S. Ren, J. Sun. Deep Residual Learning for Image Recognition. CVPR 2016.
* M. Tan, Q. V. Le. EfficientNetV2: Smaller Models and Faster Training. ICML 2021.
* Q. Xie, M.-T. Luong, E. Hovy, Q. V. Le. Self-training with Noisy Student Improves ImageNet
  Classification. CVPR 2020.
* J. W. Harvey et al. The Global Oscillation Network Group (GONG) Project. Science 272, 1284 (1996).
* P. Iakubovskii. Segmentation Models Pytorch. GitHub, 2019. R. Wightman. PyTorch Image Models. GitHub, 2019.