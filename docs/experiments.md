# Experiment log

All validation numbers use the official Panoptic Quality definition
(`filseg/metrics.py`, checked against the organisers' self-evaluation
notebook) on hold-out fold 0 of a 5-fold image-level split
(`filseg.data.image_folds`, seed 2026): 141 images, every annotator's set
scored separately. LB = public leaderboard (~50 % of the test images).

| id | date | method | val PQ (SQ / RQ) | LB PQ | notes |
|----|------|--------|------------------|-------|-------|
| H0 | 11-06 | inter-annotator agreement: annotator A as "prediction" vs annotator B (all ordered pairs, 1,196 pairs) | 0.340 (0.634 / 0.536) | - | human reference on training images with >= 2 annotators |
| E0 | 11-06 | classical relative-darkness threshold (`scripts/classical_baseline.py`) | 0.092 (0.663 / 0.139) on 80 images | 0.07 | format check; dominated by false positives |
| H1 | 11-06 | "consensus oracle": per-pixel annotator fraction of the GT itself, thresholded, connected components (296 multi-annotator images, 743 entries) | 0.648 at t=0.45 (0.804 / 0.806); 0.619 at t=0.3; 0.584 at t=0.6 | - | ceiling for a perfect consensus predictor on multi-annotator images; with single-annotator images (ceiling 1.0) the training-set mix gives ~0.78 |
| H2 | 11-06 | leave-one-annotator-out on 3-annotator images (`scripts/annotator_agreement.py`): union / intersection of the *other two* annotators vs the held-out one | union 0.332 (0.635 / 0.524); intersection 0.314 (0.637 / 0.492) | - | a 2-person consensus predicts a third annotator no better than one annotator (pairwise 0.340): annotator style dominates; realistic honest PQ ceiling is far below the 0.648 oracle |
| E1 | 11-06 | U-Net ResNet34 (ImageNet), 1024 px full disk, soft consensus target, BCE + Dice, AdamW 5e-4 cosine, EMA 0.999 (no warm-up), 40 ep, bs 6, flips/rotation/scale/photometric aug; flip TTA x4; post-processing grid on fold 0: threshold 0.35, closing r=4, fill holes, min area 500 px | 0.4155 (0.671 / 0.619); TTA1 + default post 0.387 | 0.34 | Colab T4, 1.14 h; inference 0.65 s/image on T4 (0.21 s network, 0.42 s post-processing + RLE); EMA mid-run validations unusable (fixed later with EMA warm-up) |
| E0b | 11-06 | as E1 but 512 px, global norm, 25 epochs on Kaggle CPU (9.7 h), no EMA; post grid: threshold 0.55, closing 4, min area 500 | 0.365 (0.649 / 0.562) | 0.30 | resolution 512 -> 1024 = +0.05 PQ on fold 0 |

### Notes on E1 (11-06)

* Validation breakdown (`scripts/eval_probs.py`): pooled PQ 0.4155; mean per annotator set 0.404;
  mean per image 0.394; by annotators per image: 1 -> 0.397, 2 -> 0.447, 3 -> 0.413.
  GT recall by area: < 300 px 1 %, 300-600 px 33 %, 600-1200 px 65 %, > 2500 px 58-75 %.
  23 % of GT filaments are not touched by any prediction; ~18 % are touched but reach IoU <= 0.5.
* Public LB 0.34 vs validation 0.4155. Public LB context (1,403 teams, 11-06): 523 teams in
  0.35-0.40, 19 in 0.40-0.45, then a cluster of ~110 teams at 0.50-0.60 that coincides with a
  public notebook resubmitting a pre-computed prediction file (not a model). A public honest
  10-model ensemble reports nested OOF 0.450 vs public 0.39 (same direction of gap), explained
  there by sparser test labels.* E1 predicts 6.6 filaments per validation image and 6.1 per test image with the same rule, so the
  model does not behave differently on the test images; the validation-to-public gap (also seen by
  the public reference) is consistent with sparser test labels (training annotators draw 7.1
  filaments per image on average). Ratio public / validation: 0.82 for both E0b and E1.
* Nested post-processing selection on E1 (5 pseudo-folds of fold 0, `scripts/select_post.py`):
  nested PQ 0.4167 (in-sample best 0.4250); chosen rule threshold 0.35, closing 4, area >= 300,
  mean probability >= 0.8, peak >= 0.99 (`configs/post_c1_nested.json`). Optimising PQ under a
  label rate r = 0.77 picks a similar rule (threshold 0.30, area >= 400, mean >= 0.8).* Fragment grouping (merge nearby components, 10 / 20 px) on E1: nested PQ 0.4175 vs 0.4167 without
  (in-sample best 0.4268 with 10 px vs 0.4250) -- within noise; kept as an option for the final
  selection on the full out-of-fold set.* Why is the public LB ~0.82 x validation? Two hypotheses checked against E1 / E0b (11-07):
  (a) test labels sparser (each GT kept with probability r): r = 0.68 reproduces both public
  scores (E1 0.340 vs LB 0.34, E0b 0.296 vs 0.30); (b) test GT without the "Unidentifiable"
  class (34 % of training filaments, smaller: median 868 px vs ~1,500 px): E1 0.367 vs 0.34,
  E0b 0.311 vs 0.30 -- fits worse. Both favour stricter selection (fewer, larger instances) than
  the validation optimum; we do not tune on the public LB, but one of the two final submissions
  can use a rule chosen under hypothesis (a).