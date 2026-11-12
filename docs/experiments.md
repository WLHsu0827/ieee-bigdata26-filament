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
| E1c | 11-07 | E1 model, rule from nested CV with plain PQ (threshold 0.35, closing 4, area >= 300, mean >= 0.8, peak >= 0.99) | nested 0.417 | 0.35 | same network outputs as E1 (re-inferred on Kaggle CPU, TTA4: 14.8 s/image on 4 CPU cores) |
| E1d | 11-07 | E1 model, rule from nested CV with robust objective r = 0.77 (threshold 0.30, closing 4, area >= 400, mean >= 0.8, outline 0.5) | nested 0.425 | 0.35 | stricter selection transfers (+0.01 vs the grid-tuned rule, LB 0.34) |
| K1-r34 | 11-07 | U-Net ResNet34, native 2048 px (768 crops, 4 per image per epoch), disk normalisation, lr 3e-4, EMA 0.999 + warm-up, 30 epochs, TTA4; 5-fold CV on Kaggle T4 (GPU 0 of kernel fil-gpu-k1) | folds 0-4: 0.4276 / 0.4247 / 0.4163 / 0.4253 / 0.4181, mean **0.4224** with the fixed nested rule of E1 | - | 1.26-1.37 h per fold on one T4; inference 1.13-1.14 s/image (network, TTA4) + 0.49-0.55 s post |
| K1-v2s | 11-07 | same with EfficientNetV2-S encoder (timm tf_efficientnetv2_s), bs 6, 21-24 epochs (time budget 1.48 h per fold; GPU 1) | folds 0-4: 0.4375 / 0.4288 / 0.4208 / 0.4356 / 0.4375, mean **0.4320** | - | 1.38-1.53 h per fold; 1.39-1.48 s/image + 0.54-0.57 s post. K1 total 7.5 GPU h (T4 x2, 17:29-01:01) |
| K1-ens-fixed | 11-08 | ensemble of the 10 K1 models (mean probability, TTA4), fixed E1 nested rule (t 0.35, closing 4, area >= 300, mean >= 0.8, peak >= 0.99) | - (nested selection on all 707 OOF images: fil-ana) | **0.38** (rank 414 / 1,419; was 0.35 / 693) | 7.2 filaments per test image; 12.8 s/image network (10 models, TTA4, T4) + 0.4 s post |
| K1-ens-r1 | 11-08 | same ensemble; rule by nested CV on all 707 OOF images, plain PQ (t 0.30, no closing, area >= 200, mean >= 0.85, peak >= 0.99, outline 0.5; same rule in 4 / 5 outer folds) | nested **0.4422** (in-sample 0.4444) | 0.38 | 6.6 filaments per test image; learned filter on this shape: nested 0.4455 vs rule 0.4441 |
| K1-ens-r077 | 11-08 | same ensemble; rule by nested CV on all 707 OOF images with the robust objective r = 0.77 (t 0.25, no closing, no area minimum, mean >= 0.85, peak >= 0.99, outline 0.5) | nested **0.4426** (objective 0.389) | **0.39** (rank 306 / 1,419) | 6.1 filaments per test image; learned filter on this shape: nested 0.4472 vs rule 0.4432 |
| NB-full / NB-fast / NB-learned-r1 | 11-08 | pipeline notebook outputs from the committed weights: 10 networks + learned filter (r = 0.77); 5 EfficientNetV2-S + own learned filter; 10 networks + learned filter fitted with plain PQ | nested 0.4470 / 0.4440 / 0.4462 | 0.38 / 0.38 / 0.38 | the learned filter's CV gain (+0.004) does not show on the public part; 93 % of the instances are shared with the rule submission (LB 0.39), so the difference rests on a few dozen filaments of ~90 images |
| FINAL-A | 11-08 | pipeline notebook (committed weights), full profile, variant `rule` written from the same maps: 10 networks + robust nested-CV rule (t 0.25, mean >= 0.85, peak >= 0.99, outline 0.5) | nested 0.4426 | **0.39** | 1,071 filaments; 1,071 of 1,072 identical to K1-ens-r077; 8.1 s/image for both files (fil-nb-verify v3) |

### Notes on E1 (11-06)

* Validation breakdown (`scripts/eval_probs.py`): pooled PQ 0.4155; mean per annotator set 0.404;
  mean per image 0.394; by annotators per image: 1 -> 0.397, 2 -> 0.447, 3 -> 0.413.
  GT recall by area: < 300 px 1 %, 300-600 px 33 %, 600-1200 px 65 %, > 2500 px 58-75 %.
  23 % of GT filaments are not touched by any prediction; ~18 % are touched but reach IoU <= 0.5.
* Public LB 0.34 vs validation 0.4155. Public LB context (1,403 teams, 11-06): 523 teams in
  0.35-0.40, 19 in 0.40-0.45, then a cluster of ~110 teams at 0.50-0.60 that coincides with a
  public notebook resubmitting a pre-computed prediction file (not a model). A public honest
  10-model ensemble reports nested OOF 0.450 vs public 0.39 (same direction of gap), explained
  there by sparser test labels.
* E1 predicts 6.6 filaments per validation image and 6.1 per test image with the same rule, so the
  model does not behave differently on the test images; the validation-to-public gap (also seen by
  the public reference) is consistent with sparser test labels (training annotators draw 7.1
  filaments per image on average). Ratio public / validation: 0.82 for both E0b and E1.
* Nested post-processing selection on E1 (5 pseudo-folds of fold 0, `scripts/select_post.py`):
  nested PQ 0.4167 (in-sample best 0.4250); chosen rule threshold 0.35, closing 4, area >= 300,
  mean probability >= 0.8, peak >= 0.99 (`configs/post_c1_nested.json`). Optimising PQ under a
  label rate r = 0.77 picks a similar rule (threshold 0.30, area >= 400, mean >= 0.8).
* Fragment grouping (merge nearby components, 10 / 20 px) on E1: nested PQ 0.4175 vs 0.4167 without
  (in-sample best 0.4268 with 10 px vs 0.4250) -- within noise; kept as an option for the final
  selection on the full out-of-fold set.
* Why is the public LB ~0.82 x validation? Two hypotheses checked against E1 / E0b (11-07):
  (a) test labels sparser (each GT kept with probability r): r = 0.68 reproduces both public
  scores (E1 0.340 vs LB 0.34, E0b 0.296 vs 0.30); (b) test GT without the "Unidentifiable"
  class (34 % of training filaments, smaller: median 868 px vs ~1,500 px): E1 0.367 vs 0.34,
  E0b 0.311 vs 0.30 -- fits worse. Both favour stricter selection (fewer, larger instances) than
  the validation optimum; we do not tune on the public LB, but one of the two final submissions
  can use a rule chosen under hypothesis (a).
* Outline threshold (detect at t, redraw kept instances at p > t_mask; 11-07): growing hurts
  (t_mask 0.15-0.30: 0.403-0.423 vs 0.425), shrinking to 0.45-0.50 helps slightly (0.4269-0.4275,
  better in 3/5 pseudo-folds). Nested selection with outlines as an option: plain-PQ objective
  0.4167 (unchanged: more freedom = more selection noise on 142 images); the r = 0.77 objective
  picks the same rule in 4/5 outer folds -- threshold 0.30, closing 4, area >= 400, mean >= 0.8,
  outline 0.50 -- with nested official PQ 0.4247, the best honest estimate for E1 so far. The final
  selection is repeated on all 707 out-of-fold images of the K1 ensemble.
* Learned component filter (11-07, scratch prototype on the E1 component sets): gradient-boosted
  regression of each candidate's per-entry matched IoU from its features (area, mean / peak / p90
  probability, elongation, fill ratio, distance from disk centre, rank in image), threshold chosen
  on inner out-of-fold scores, nested over the same 5 pseudo-folds. Averaged over the 28 shape
  configurations it is *not* better than the three-parameter rule (plain-PQ objective: nested 0.419
  vs 0.420 on average, better in 9 / 28 shapes; best shapes 0.425 vs 0.427; r = 0.77 objective:
  0.418 vs 0.423). Negative on 142 images; re-checked on the 707 out-of-fold images of K1.
* Error taxonomy of E1 with the r = 0.77 rule (`analysis.error_taxonomy`, in-sample, PQ 0.427;
  per annotator entry): of 1,659 GT filaments 949 matched; of the 710 unmatched, 460 have no
  overlapping prediction, 107 are covered < 50 % (partial), 54 are merged with a neighbour into
  one prediction, 52 are covered by a too-large prediction, 37 are fragmented (>= 2 predictions
  whose union would match). Of 401 false positives 111 overlap no GT of that annotator.
  Upper bounds: perfect grouping of fragments 0.448 (+0.021), removing all non-overlapping false
  positives 0.443 (+0.016). Detection of unmarked / faint filaments (and annotator disagreement)
  dominates; fragment grouping is the largest *post-processing* headroom but distance-based merging
  realises almost none of it (+0.001).
* Lower detection thresholds (11-07; detect at 0.2 / 0.25, i.e. join fragments through weak bridges, and redraw
  kept instances at 0.45 / 0.5): nested PQ on E1 0.419 (plain-PQ objective; t = 0.25 chosen in 4 / 5 outer folds)
  and 0.424 (r = 0.77; same final rule as before: t 0.30, area >= 400, mean >= 0.8, outline 0.5) vs 0.417 / 0.425
  with thresholds >= 0.3 -- within noise. t = 0.25 is added to the K1 selection grid.
* What limits detection (E1, r = 0.77 rule; 460 GT filaments without any overlapping prediction): 286 of
  them (62 %) contain probabilities >= 0.5 (246 >= 0.8), i.e. the network marks them and the selection rule
  (area >= 400, mean >= 0.8) drops them; only 79 have a peak < 0.3. They are small (347 of 460 < 1,200 px).
  On multi-annotator images the network matches 74 % of the filaments that another annotator also drew
  (IoU > 0.5) but only 31 % of those drawn by one annotator only; 35 % of all filaments there are
  single-annotator. Selection of small components is therefore a trade-off against annotator-specific
  filaments, not a blind spot of the network; a better small-component discriminator is the remaining
  post-processing lever (learned filter re-tested on the 707 K1 out-of-fold images).
* K1 out-of-fold selection (fil-ana, 707 images, 1,154 annotator sets): two encoders 0.4422 nested (r = 1) /
  0.4426 (r = 0.77); EfficientNetV2-S folds alone 0.4401 / 0.4381 (rule t 0.30, closing 4, area >= 300,
  mean >= 0.85, outline 0.5; 7.1 s/image); ResNet34 folds alone 0.4318 / 0.4307 (mean >= 0.9). The selected
  rule is close to the public reference's (mean >= 0.85, peak >= 0.99, area >= 200). Evaluation of the r = 1
  rule (in-sample, `docs/evaluation.json`): PQ 0.4445, SQ 0.682, RQ 0.651; recall < 300 px 25 % (1024 px
  model: 1 %), 300-600 px 49 % (33 %); FN 3,038 = 1,751 missed + 481 partial + 348 too large + 304 merged +
  154 fragmented; FP 2,485 = 1,041 isolated + 1,444 overlapping. Upper bounds: isolated FPs removed 0.476,
  fragments grouped 0.461. Public LB: plain-PQ rule 0.38, robust r = 0.77 rule 0.39 -- the stricter selection
  transfers better again (third time, consistent with sparser test labels).
* Semi-supervised pilot (11-08, fil-gpu-st0 + fil-cmp; **negative**): 1,200 unlabelled GONG archive images
  (`docs/external_gong_manifest.csv`, >= 2 days from every competition image, verified by
  `scripts/check_external.py`), pseudo-labelled by the fold-0 teachers (K1 fold-0 ResNet34 + EfficientNetV2-S,
  flip TTA, 1.55 s/image); students of both architectures trained on fold 0's training images plus the
  pseudo-labelled images (1:1 crops per epoch) with the teachers' time budget (9-13 epochs instead of 21-30).
  Fold 0, 142 images, fixed rule / nested PQ: teacher ResNet34 0.4276 / 0.4423 vs student 0.4248 / 0.4382;
  teacher EfficientNetV2-S 0.4378 / 0.4447 vs student 0.4336 / 0.4432; teacher pair 0.4425 / 0.4455 vs
  student pair 0.4383 / 0.4439; all four 0.4405 / 0.4445. At equal compute the students are slightly worse
  and add nothing to the ensemble, so the direction was stopped (1.8 GPU h).
* Learned component filter on the K1 out-of-fold candidates (11-08, `scripts/fit_scorer.py`): gradient-boosted
  trees on 12 candidate features (area, probability statistics, shape, position, rank in image), keep threshold
  from grouped out-of-fold scores; stored as JSON in the post-processing configuration (`filseg/scorer.py`,
  no scikit-learn needed at inference). Nested PQ 0.4470 (r = 0.77 shape, threshold 0.28) and 0.4462 (plain-PQ
  shape, threshold 0.27) vs 0.4426 / 0.4422 for the rule selection (+0.004). End-to-end check: the written
  configuration run through the official evaluation reproduces the component-table PQ exactly on fold 0
  (features recomputed on another machine can differ by ~1e-7 and moved one of 1,933 candidates across a
  tree split; the check therefore rebuilds the candidates locally).
* Reproducibility run (11-08, kernel fil-nb-verify, `scripts/run_notebook.py`): `notebooks/pipeline.ipynb`
  executed with nbconvert in a fresh copy of the repository on a Kaggle T4, loading the fp16 weights of
  `weights/` (no training), all 180 test images. Profile `full` (10 networks, learned filter r = 0.77):
  26.5 min, **8.5 s/image** (8.0 s networks + 0.5 s post-processing), 1,076 filaments; profile `fast`
  (5 EfficientNetV2-S networks with their own learned filter): 17.0 min, **5.3 s/image**, 1,059 filaments.
  The `full` output agrees with the file built from the training-time probability maps at PQ 0.998
  (1,074 of 1,076 instances identical, two borderline candidates differ through fp16 rounding); the
  notebook outputs are therefore the files submitted from now on. Environment: Python 3.13.16, numpy 2.1.3,
  torch 2.11.0, segmentation-models-pytorch 0.6.0, timm 1.0.30 (pinned in `requirements.txt`).
* Final ensemble + learned filter (r = 0.77) on all 707 out-of-fold images (fold-wise, in-sample for the
  filter): 61.8 % of GT filaments matched; 24.6 % without any overlapping prediction, of which 74 % contain
  probabilities >= 0.5 (dropped by the filter) and 71 % are < 1,200 px. On multi-annotator images 62.8 %
  of the filaments are matched (IoU > 0.5) by another annotator; the ensemble matches 81.5 % of those vs
  32.2 % of the single-annotator ones. Recall by area: < 300 px 0.26, 300-600 0.46, 600-1200 0.64,
  1200-2500 0.69, 2500-5000 0.70, > 5000 0.68.
* Headroom of smarter instance formation (11-08, oracle on fold 0 of the final ensemble maps; candidates =
  connected components of p > t for several t, nested across levels, each scored exactly; oracle = selection
  with the true matched IoU, PQ ratio maximised by Dinkelbach iterations with a tree DP over the nesting):
  best single level 0.528 (t = 0.3); best antichain over levels 0.15-0.5 (all redrawn at 0.5, i.e. only
  *grouping* differs) 0.531 (+0.003); over levels 0.3-0.7 (outlines tighten above 0.5, i.e. the oracle
  picks each component's *extent*) 0.540 (+0.012). Even perfect fragment grouping gains little; the larger
  headroom is the per-component outline, which decides whether borderline matches cross IoU 0.5. A global
  tighter outline does not capture it (final configuration, all 707 OOF images: outline 0.45 / 0.5 / 0.55 /
  0.6 / 0.65 -> PQ 0.4612 / 0.4620 / 0.4623 / 0.4617 / 0.4614, in-sample for the filter); not pursued further.
* Final configuration evaluated on all 707 out-of-fold images (fil-eval2; in-sample for the learned filter):
  full ensemble PQ 0.4621 (SQ 0.683, RQ 0.677; TP 5,066, FP 1,707, FN 3,133); fast profile 0.4574. Compared
  with the plain rule, isolated false positives drop from 1,041 to 531. FN causes: 2,016 missed, 401 partial,
  310 too large, 295 merged, 111 fragmented. GT overlapping >= 2 predictions: 2.8 %; predictions overlapping
  >= 2 GT: 2.6 %. Mean IoU / Dice of overlapping pairs 0.606 / 0.736. `docs/evaluation*.json`, figures in
  `docs/figures/evaluation*.png` (the report uses the 2 x 3 `evaluation_grid.png`; re-run on Kaggle 11-08 with
  identical numbers; re-run 11-09, again with an identical `evaluation.json`, to add 95 % bootstrap intervals
  over images (2,000 resamples): PQ 0.453-0.471, recall per size bin as error bars, e.g. 0.21-0.31 below 300 px
  and 0.54-0.71 at >= 10k px, in `docs/evaluation_uncertainty.json`; both layouts draw the error bars, the grid
  also prints the PQ interval; the executed notebook still shows the earlier figure without them;
  `python scripts/make_figures.py evaluation-replot docs/evaluation.npz docs/evaluation.json
  <out.png> grid` redraws it from the stored IoU / Dice arrays and the intervals without recomputation).
* Ensemble-size shift (11-08, label-free, `scripts/ensemble_shift.py` on the K1 test maps; summary in
  `docs/ensemble_shift.json`): selection is tuned on out-of-fold maps that average only the two networks of a
  fold, the submission averages ten. On the 180 test images, with the same configuration, the ten-network
  average keeps 5.95 instances per image vs 6.21 for the five two-network averages (range 5.87-6.39; -4 %;
  area -3 %) for the rule of submission A, and 5.81 vs 6.02 (5.75-6.14; -3 %; area -2 %) for the learned filter
  of submission B. It agrees with each two-network selection at a PQ-style score of 0.91 (96 % of its instances
  matched at IoU > 0.5), more than two-network selections of different folds agree with each other (0.87). The
  transfer is therefore benign: the ten-network selection is a slightly more conservative consensus of the
  two-network ones, inside their fold-to-fold range; no correction applied.
* Offline check (11-08, `scripts/check_weights.py`): the ten weight files match the SHA-256 in the manifest, all
  networks load and run on the CPU with network access blocked (no pretrained or other downloads at inference;
  the verification kernels had internet enabled, so this was not covered before), and the three post-processing
  configurations each keep 3 of 4 synthetic ridges; 25 s on two CPU cores. The notebook's code cells make no
  network calls.
* Fresh-install check (11-11, Kaggle CPU kernel fil-fresh, `scripts/run_notebook.py --venv /tmp/fresh --limit 2`):
  the README steps in a new virtual environment (Python 3.13.18; `venv` lacks `ensurepip` in the Kaggle image, so
  `virtualenv` is used). `pip install -r requirements.txt` installs all 13 pins at the pinned version (3.4-4.8 min;
  `pip freeze` in the kernel output). First attempt, without a kernel step: nbconvert succeeded, but the notebook
  reported `torch 2.11.0+cpu`, the Kaggle image's own build; Jupyter had started the `python3` kernel of the base
  installation, so the cells did not use the new environment. The README therefore registers the environment's
  kernel (`python -m ipykernel install --sys-prefix`), and the check fails unless the notebook reports the new
  environment's torch build. Second attempt: kernel `/tmp/fresh/bin/python`, notebook `torch 2.11.0+cu130 |
  device: cpu`, 16 cells, no error (one harmless stderr warning: tqdm finds no ipywidgets), exit code 0. On the
  first two test images both submissions agree with the committed run (fil-nb-verify v3, T4 GPU) at PQ 0.9999
  (every instance matched at IoU > 0.99; fp32 on the CPU vs mixed precision on the GPU). Without a GPU the full
  profile needs 760 s per image (8 s on a T4).
* Reference files (11-12, `submissions/`): the two submitted files, downloaded back from Kaggle with the API
  (`competition_download_submission`): B, ref 58107878, 303,250 bytes, SHA-256 18d75e35...455c; A, ref 58109804,
  296,445 bytes, a6e10ecf...36bf. They equal the `submission.csv` of the notebook runs fil-nb-verify v2 and v3
  (byte-identical to each other) and the `submission_rule.csv` of v3, the run committed with the notebook. The
  first run v1, made before the fp16 NaN fallback in `predict_batch`, differs in bytes but agrees with v2 at
  PQ 1.0000 (1,076 of 1,076 instances, all at IoU > 0.99; 177 of the 180 test images have instances).
  `compare_submissions.py --b-images-only` on the two-image CPU re-run (fil-fresh v2): PQ 0.9999 against both
  files; without the option the 175 skipped images count as misses (PQ 0.0037).
* torch builds for older GPUs and drivers (11-12). `pip install -r requirements.txt` installs PyPI's torch 2.11.0,
  on Linux the CUDA 13.0 build (`2.11.0+cu130`), the same build as the committed run (`versions.json` of
  fil-nb-verify v3). PyPI's Windows wheel is the CPU build: its CUDA dependencies are marked Linux-only, and it
  has the size of the CPU build on PyTorch's index (109 MiB; the Windows CUDA 12.6 and 13.0 builds there have
  2,476 and 1,827 MiB, 506 MiB for PyPI's Linux wheel). CUDA 13.x needs driver r580 or newer and supports "all
  NVIDIA architectures from Turing through Grace Blackwell" (CUDA 13.0 release notes); CUDA 12.x builds run on
  drivers from 525 on (minor version compatibility, CUDA release notes table 3). PyTorch 2.11's build script
  (`.ci/manywheel/build_cuda.sh` at v2.11.0) compiles the CUDA 12.6 builds for compute capability 5.0, 6.0, 7.0,
  7.5, 8.0, 8.6 and 9.0, the CUDA 12.8 and 13.0 builds for 7.5, 8.0, 8.6, 9.0, 10.0 and 12.0. The CUDA 12.6 build
  installs cuDNN 9.10.2.21, whose CUDA 12 build supports compute capability 5.0 to 9.0 with driver 525.60.13
  (Linux) or 527.41 (Windows) and newer (cuDNN 9.10.2 support matrix). PyTorch's CUDA 12.6 index has torch
  2.11.0 and torchvision 0.26.0 for Linux and Windows, Python 3.11-3.13; the CUDA 12.8 and 13.0 indexes have them
  for Windows (checked for Python 3.12). A plain `pip install torch==2.11.0` does not replace an installed build
  (the local version label is ignored), so the README installs the alternative build first. Checked (Kaggle CPU
  kernel fil-fresh v3, `run_notebook.py --venv --pre-install`): requirements.txt kept `torch 2.11.0+cu126` and
  `torchvision 0.26.0+cu126` (no CUDA 13 packages left, all 13 pins satisfied), the notebook ran with that build
  (exit code 0, no error), and both submissions agree with the submitted files on the first two test images at
  PQ 0.9999. Not checked on an actual older GPU or on Windows.
* Entry points in a plain shell (11-12). All earlier local and Kaggle runs set `PYTHONPATH` to the repository
  (local runner script, Kaggle kernel template), which hides a script that does not put the repository root on
  `sys.path` itself. Run as a reader would, from the repository root without `PYTHONPATH`: `--help` of the 24
  command-line tools with an argument parser (21 in `scripts/`, `build_pdf.py`, `build_kernel.py`,
  `build_colab.py`) and of `python -m filseg.train` / `filseg.cv`, then `python scripts/check_weights.py` (all
  checks passed) and `python -m pytest tests` (41 passed). One failure: `scripts/build_gt_cache.py` (`No module
  named 'filseg'`), fixed; `test_scripts_find_the_package_without_pythonpath` now loads every script this way
  (it fails on the unfixed file). pytest and ruff are not in `requirements.txt` (not needed to reproduce the
  predictions); the README gives their versions.
* External-data hashes checkable without the processed images (11-12). `check_external.py --originals` used to
  only record the hashes; the byte-exact processing could be confirmed only against a local copy of the
  processed images (all 1,200 reproduced when the hashes were recorded, 11-08). It now compares every
  re-downloaded original, and its re-processed version, with the two SHA-256 in the manifest; a mismatch of
  either gives exit code 1 (test `test_check_external_verifies_the_recorded_hashes`). The images were fetched
  and processed on a Windows machine (access to the archive from Kaggle stalled) with OpenCV 5.0.0, which bundles
  libjpeg-turbo 3.1.2. Checked against the NSO archive on three images (first, middle and last manifest row),
  with that OpenCV and with the pinned `opencv-python-headless==4.14.0.94` and `numpy==2.1.3` (also
  libjpeg-turbo 3.1.2): originals unchanged and processing reproduced byte for byte; a manifest with one altered
  hash of either kind fails. Not checked on Linux; the full check (about 1,200 downloads) was not repeated.
* GPU memory of inference (11-13). The notebook keeps all ten networks on the GPU (233.5 M parameters, 0.87 GiB in
  fp32) and runs one network on four 2048 px test images per forward pass (`batch=4`). No GPU memory was recorded
  in the T4 runs. Activation peak of one forward pass, measured on the CPU in fp32 with the PyTorch profiler
  (allocations attributed to operators plus frees between operators, summed in time order; balanced to zero at
  the end): 382.0 MiB per 1024 x 1024 pixels for the ResNet34 U-Net and 331.5 MiB for the EfficientNetV2-S U-Net,
  the same at a quarter of that size and for 4 x 512 px vs 1 x 1024 px, so strictly proportional to the pixel
  count. Four 2048 px images: 6.0 and 5.2 GiB in fp32, about half under autocast; a pass repeated in fp32 after a
  half-precision overflow needs the fp32 amount. CUDA context, cuDNN workspace and allocator caching come on top
  and were not measured. `predict_images` now repeats a pass that raises `torch.cuda.OutOfMemoryError` with half
  as many images (down to one) and keeps the smaller batch for the remaining images (as training does for its
  first epoch). Without that error the output is byte-identical to the previous code (checked for 1, 2, 4 and 8
  images per pass, both normalisations, two random ResNet18 U-Nets on seven 96 px images); after it, the
  probabilities can differ slightly (rounding of other kernels for the smaller batch; the fp32 repeat of
  `predict_batch` then covers fewer images). Test
  `test_predict_images_repeats_a_pass_with_fewer_images_when_out_of_memory` simulates the error. Not tested on a
  GPU with less memory.
* Public-split noise (11-08, `scripts/split_noise.py` on the K1 out-of-fold maps, Kaggle CPU kernel fil-noise;
  `docs/split_noise.json`): per-image PQ counts of the final configurations, pooled PQ over 4,000 random
  90-image subsets of the 707 images. A score on 90 images scatters around the score on all images with
  SD 0.012 (5-95 %: 0.423-0.463 for rule A, whose 707-image PQ is 0.444), so the public leaderboard (about
  90 test images) cannot resolve differences of a few thousandths; the paired difference of two configurations
  on the same subsets has SD 0.004. Decisions therefore rest on nested CV over all 707 images.
* Uncertainty of the final estimates (11-09, `scripts/nested_uncertainty.py` on the fil-d1 candidate table of
  the final shape, r = 0.77; `docs/nested_uncertainty.json`): rule A nested 0.4432, 95 % bootstrap interval over
  images [0.434, 0.453] (SD 0.0048), per outer fold 0.450 / 0.440 / 0.437 / 0.443 / 0.445 (SD 0.005); learned
  filter B 0.4470 [0.438, 0.457] (SD 0.0048), per fold 0.454 / 0.445 / 0.438 / 0.451 / 0.449 (SD 0.006);
  paired B - A +0.0038 (SD 0.0013, 95 % [0.0012, 0.0065], <= 0 in 0.2 % of 4,000 resamples). Single networks
  with the fixed rule over five folds: ResNet34 0.422 +- 0.005, EfficientNetV2-S 0.432 +- 0.007 (SD).
* Direction 1 of the 11-08 push (`scripts/extent_levels.py`, Kaggle CPU kernel fil-d1; candidates of the final
  shape t 0.25, c0, all components, 9,394 on 707 images; nested CV over the CV folds, learned filter as in
  production): (i) **per-candidate outline level** (one regressor over candidate x level rows predicts the per-entry
  matched IoU at levels 0.30-0.70; a candidate leaves outline 0.5 only if another level is predicted better by a
  margin chosen on inner folds): nested 0.4478 / 0.4460 (r = 1 / 0.77) vs 0.4473 / 0.4472 for the current filter
  -- no gain, although the in-sample oracle that knows each candidate's best outline would reach 0.538 vs 0.518 at
  a fixed outline: the right outline depends on the annotators and is not predictable from the map. (ii) **extra
  evidence** for the filter (mean / peak probability under the ResNet34 and the EfficientNetV2-S network separately
  and their difference, distance to the nearest candidate, number of confident candidates): 0.4497 / 0.4488
  (+0.002); paired bootstrap over images (local re-run, sklearn 1.9): +0.0026, SD 0.0012 at r = 1 (98.5 % of
  resamples positive), +0.0012, SD 0.0014 at r = 0.77. The per-network evidence carries it (0.4497 / 0.4489 alone;
  context alone 0.4481 / 0.4478). Caveat before any adoption: out of fold the per-network features come from single
  networks, in the submission from averages of five per encoder. Rule on the same candidates: 0.4430 / 0.4432.
  Label-free test check of that caveat (`extent_levels.py test-shift`, Kaggle CPU kernel fil-shift2, 180 test images,
  filters fitted on all out-of-fold candidates, keep thresholds for r = 0.77): with ten networks the current filter
  keeps 5.65 instances per image vs 5.91 with the five two-network averages (-4 %, fold range 5.63-6.04); with the
  per-network evidence 5.82 vs 6.19 (-6 %, ten-network value below the fold range 5.94-6.32); agreement ten vs two
  0.91 for both (two vs two 0.87-0.88). The extra evidence therefore shifts more between out-of-fold and submission
  conditions than the current features, which puts its small nested gain at risk: **not adopted** (finals A / B
  unchanged).
* GPU pilot fil-gpu-p1 (11-08, pool, T4 x2, 2.60 h): fold 0 only, same recipe as K1 (2048 px, 768 crops, EMA,
  TTA4, fixed rule `post_c1_nested.json` for the single-model number).
  | model (fold 0) | epochs (budget) | val PQ, fixed rule | inference per image and network |
  |---|---|---|---|
  | ResNet34 (K1) | 30 (1.48 h) | 0.4276 | 1.13 s |
  | EfficientNetV2-S (K1) | 21 (1.48 h) | 0.4375 | 1.31-1.48 s |
  | EfficientNetV2-M, bs 4 | 24 (2.6 h) | 0.4383 | 1.66 s |
  | EfficientNetV2-S, `--small-weight 0.5` | 22 (1.48 h) | 0.4383 | 1.31 s |
  A larger encoder (+0.0008 at 1.75x the training time) and the small-filament / boundary weighting (+0.0008)
  are within noise as single models; ensemble-level gate below. GPU 1 first re-predicted all 707 OOF images from
  the committed weights with TTA x4 and x8 in one pass (2.5-3.0 s per image and network for x8).
* Gate analysis of the pilot (11-08, Kaggle CPU kernel fil-gate, `extent_levels.py build --subfolds 5` and
  `compare-tables`; fold 0 = 142 images, nested CV over 5 sub-folds; paired bootstrap over images, 4,000 draws).
  Differences to the current pair ResNet34 + EfficientNetV2-S (learned filter r = 1 / 0.77; rule r = 1 / 0.77):
  | fold-0 ensemble | learned filter | rule |
  |---|---|---|
  | R34 + V2-S (current; nested PQ) | 0.4383 / 0.4396 | 0.4452 / 0.4492 |
  | + V2-M (three networks) | +0.0077 / +0.0054 (P <= 0: 0.03 / 0.09) | +0.0091 / +0.0031 (0.01 / 0.15) |
  | V2-M replaces R34 | +0.0085 / +0.0068 (0.03 / 0.07) | +0.0110 / +0.0038 (0.00 / 0.14) |
  | V2-M replaces V2-S | +0.0077 / +0.0012 | +0.0029 / -0.0008 |
  | small-weight V2-S replaces V2-S | +0.0026 / 0.0000 | -0.0063 / -0.0045 |
  Single networks (nested): V2-M vs V2-S +0.004 to +0.016, small-weight V2-S vs V2-S +0.004 to +0.010 -- the
  single-network comparisons are noisy (the small-weight model looks better alone but not in the pair).
  **V2-M passes the pre-registered gate** (fold-0 ensemble >= +0.005 with the learned filter); the small-filament
  weighting does not. **TTA x8 vs x4** on all 707 images (same committed weights): learned filter +0.0024 / +0.0009
  (P <= 0: 0.03 / 0.26), rule +0.0031 / +0.0030 (0.02 / 0.01), for twice the network time.
* V2-M on more folds (11-09): folds 1 and 2 on Kaggle (fil-gpu-p2, 2.70 h, 23 epochs each, fixed-rule PQ 0.4329 /
  0.4137 vs V2-S 0.4288 / 0.4208), fold 3 on Colab (stopped after 21 of 24 epochs when the session ended). Interim
  check on folds 0-2 (424 images, nested CV with the three CV folds as outer folds, same fixed shape; kernel
  fil-gate3, `docs/gate3/`): adding V2-M +0.0001 / -0.0013 (learned filter, r = 1 / 0.77) and -0.0021 / -0.0034
  (rule); V2-M replacing R34 -0.0002 / +0.0006 and -0.0037 / -0.0024 (paired SD 0.002-0.003). The fold-0 gain
  (+0.005 to +0.011) did not repeat on folds 1-2 -- a single-fold gate on 142 images was too noisy. **Negative:**
  the larger encoder does not improve the ensemble; the V2-M track was stopped (folds 3-4 not completed).
* Half-precision overflow (11-08): the EfficientNetV2-S networks occasionally produce non-finite logits under
  fp16 autocast (numpy "invalid value encountered in cast" when saving maps; K1 fold 0 and the first notebook
  run). `predict_batch` now recomputes such a TTA pass in fp32. Second reproducibility run (fil-nb-verify v2):
  outputs identical to the first (100 % of instances at IoU > 0.99, i.e. the affected pixels never changed an
  instance), no warnings; full 7.95 s/image (7.40 + 0.53), fast 5.23 s/image (4.66 + 0.54). The executed
  notebook of this run is committed as `notebooks/pipeline.ipynb`; the 10 fp16 weights (452 MB, SHA-256 in
  `weights/manifest.json`) are committed in `weights/`.
* Final pair (agreed 11-08): A = 10 networks + robust rule (public 0.39, best on the public part) and
  B = 10 networks + learned filter (public 0.38, best in nested CV 0.447 vs 0.443; notebook default). Both
  files come from one execution of `notebooks/pipeline.ipynb` (committed with the outputs of that run,
  fil-nb-verify v3): `submission.csv` (B) and `submission_rule.csv` (A).
