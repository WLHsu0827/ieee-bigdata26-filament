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
