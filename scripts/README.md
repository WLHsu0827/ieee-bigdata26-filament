# Command-line scripts

Run from the repository root (`python scripts/<name>.py --help` shows all options; every script
starts with a docstring and a usage example). The scripts import the `filseg` package. Most need
the competition data (`--data`, found automatically on Kaggle) or a cached ground truth
(`--gt-cache`, written by `build_gt_cache.py`). The Kaggle runs packed the sources into
self-contained kernels with `kaggle/build_kernel.py`.

## Reproducing the final submission

| script | purpose |
|---|---|
| `check_weights.py` | compares every file in `weights/` with the SHA-256 in `weights/manifest.json`, loads and runs all ten networks on the CPU with network access blocked, and applies every post-processing configuration to a synthetic map (about 30 s, no data needed) |
| `run_notebook.py` | executes `notebooks/pipeline.ipynb` in a fresh copy of the repository; used for the committed notebook and the end-to-end timing in the report; with `--venv`, follows the README steps in a new virtual environment and fails unless the notebook's cells ran with that environment's packages (checks the reproduction instructions; `--pre-install` for the alternative torch build) |
| `check_submission.py` | validates a submission CSV before uploading: header, unique ids, quote-free RLE, image ids of the test set, no empty and no overlapping masks |
| `compare_submissions.py` | scores one submission file against another, e.g. a re-run of the notebook against the submitted files in `submissions/`, or the notebook re-run against the file built from the training-time maps (PQ 0.998); `--b-images-only` for a partial re-run (`FILSEG_LIMIT_TEST`) |

## Building the final models and post-processing

Training itself is `python -m filseg.train` (one fold) or `python -m filseg.cv` (several folds); see
the top-level README.

| script | purpose |
|---|---|
| `select_post.py` | nested cross-validation of the post-processing (detection shape, rule, learned filter) on all out-of-fold maps: the "nested" numbers of the report; writes the candidate tables |
| `fit_scorer.py` | fits the learned component filter on a candidate table and writes a self-contained post-processing JSON (`configs/post_final_full.json`, `post_final_fast.json`) |
| `make_weights.py` | collects trained runs into `weights/` (fp16 checkpoints and `manifest.json`) |
| `ensemble_submit.py` | builds a submission by averaging the saved test probability maps of several runs |

## Analyses behind the report

| script | purpose |
|---|---|
| `annotator_agreement.py` | human reference: one annotator against another (PQ 0.340), leave-one-annotator-out (0.332 / 0.314) and the consensus oracle (0.648); `docs/annotator_agreement.json` |
| `classical_baseline.py` | non-learning darkness-threshold baseline (first row of Table I) |
| `make_figures.py` | all figures in `docs/figures/` (pipeline, out-of-fold evaluation with `docs/evaluation*.json`, overlays, morphology examples) |
| `ensemble_shift.py` | label-free check that a selection tuned on two-network out-of-fold maps transfers to the ten-network average of the submission; `docs/ensemble_shift.json` |
| `split_noise.py` | sampling noise of a score on 90 images, the size of the public split (SD 0.012); `docs/split_noise.json` |
| `nested_uncertainty.py` | uncertainty of the two final configurations in Table 1 of the report: 95 % bootstrap intervals over images, spread over the five folds and the paired difference of the learned filter and the rule; `docs/nested_uncertainty.json` |
| `compare_models.py` | ablation tables: models and ensembles scored on the same out-of-fold images with a fixed and a nested post-processing (used for the semi-supervised students against their teachers) |
| `eval_probs.py` | scores validation maps with one post-processing configuration, with a breakdown by annotator set and image |
| `analyze_oof.py` | early post-processing analysis of single-model maps; also the shared loaders `load_probs` and `load_gt` |
| `build_gt_cache.py` | caches the ground truth for fast local evaluation (`gt_crops.pkl`, and `gt_train.csv` in the official format) |

## Experiments that were not adopted

These produced the negative results of the report; the final pipeline does not use them.

| script | purpose |
|---|---|
| `fetch_gong.py` | semi-supervised experiment: fetches unlabelled GONG H-alpha archive images taken at least two days away from every competition image |
| `check_external.py` | re-checks that date gap against all train and test images and that the processing reproduces every external file byte for byte (`docs/external_gong_manifest.csv`) |
| `pseudo_label.py` | teacher probability maps of the unlabelled images |
| `predict_oof.py` | re-predicts all out-of-fold maps from the committed weights with test-time augmentation x4 and x8 in one pass (the TTA trade-off) |
| `extent_levels.py` | candidate tables with several outline levels and extra evidence; nested CV of a per-candidate outline and of extra filter features (`analyse`); paired bootstrap of two candidate tables (`compare-tables`: EfficientNetV2-M, small-filament weighting, TTA x8; `docs/gate/`, `docs/gate3/`); label-free check of a filter on the test images (`test-shift`) |
| `chain.py` | runs several commands one after another in one Kaggle worker (one GPU) |

`cli_utils.py` holds small helpers shared by the scripts (glob expansion, locating the test images).

Related tools outside this folder: `notebooks/build_pipeline_notebook.py` generates the pipeline
notebook, `docs/report/build_pdf.py` builds the report PDF, and `kaggle/build_kernel.py` /
`kaggle/build_colab.py` package the code for Kaggle kernels and Colab notebooks.
