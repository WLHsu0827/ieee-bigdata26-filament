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
