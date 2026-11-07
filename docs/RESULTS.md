# Recorded experiments and validation study

## Corrected-run validation study (7 November 2026)

The user completed `corrected_single_seed42_20261106`: 100 epochs, selected
checkpoint at epoch 100. The compact architecture and patch remained unchanged;
the run used corrected training-only acquisition statistics and sampling
probabilities of 0.5 foreground, 0.4 anatomical background, and 0.1 uniform,
with hard-negative sampling disabled. Checkpoint SHA-256:

```text
8c9994b62c3ea2a2333c9d74304c5b814cc89aeaccb1fc39bc7ce82183e22aa4
```

Both selected checkpoints were compared on the same 83 validation patients with
the same bounded threshold grid. Each selected 0.98 without component removal.
The study used saved native-grid probabilities; it did not retrain either model,
execute new model inference, or use the test set.

| Validation metric | Previous compact v2 | Corrected run |
| --- | ---: | ---: |
| Mean patient Dice at 0.5 | 0.356317 | 0.377417 |
| Mean patient Dice at selected 0.98 | 0.464848 | 0.476109 |
| Precision at 0.98 | 0.538299 | 0.543753 |
| Recall at 0.98 | 0.474828 | 0.480031 |
| HD95 (mm, finite cases) at 0.98 | 106.190301 | 110.233027 |
| Surface Dice at 0.98 | 0.288232 | 0.279567 |
| Completely missed targets at 0.98 | 7 | 7 |

Paired mean Dice difference at 0.98: +0.011261; patient-bootstrap 95% interval
[-0.010362, +0.032668] (10,000 resamples, seed 42). Patients improved/worsened/tied:
40/37/6. This interval is conditional on validation-selected checkpoints and
operating points; it does not include selection uncertainty or establish held-out
improvement. Boundary metrics are slightly worse for the corrected run.

The corrected run completely misses five targets at 0.5 and seven at 0.98.
Keeping only the largest component at 0.98 raises mean Dice to 0.494660 but
increases completely missed targets to 23/83; it is not recommended as a default.
The component study was a second stage at each model's selected threshold, not
an exhaustive joint threshold/recipe search. No corrected-model test score is
reported, and no clinical usefulness is established.

Local patient-level artifacts remain excluded from Git:
`evaluations/validation_study_20261107/comparison.{json,md}` and the
saved-probability source evaluations. The summary helper is specific to this
local study and requires those artifacts. Only aggregate results are published.

## Historical compact v2 experiment

These numbers were read from existing artifacts during the repository review.
The review did not retrain the model or rerun the held-out test evaluation.

## Configuration

- Source implementation: `lung_ct_segmentation.py`, program version `1.0.0`.
- Local run: `hybrid_compact_v2_seed42`.
- Dataset: NSCLC-Radiomics; target GTV-1; 413 eligible cases.
- Patient/source grouped split: 247 train, 83 validation, 83 test; seed 42.
- Working voxel spacing (z/y/x): 2.5/1/1 mm; full volume, no tumour crop.
- Input: two HU windows (lung and soft tissue).
- Model: hybrid3d, compact; widths 16/32/64/128/192; 6,955,694 parameters.
- Patch (D/H/W): 32/96/96; batch 1; accumulation 2; foreground sampling 0.25.
- 100 epochs; 1,000 microbatches per epoch; validation every 10 epochs.
- AdamW: initial LR 0.00005; weight decay 0.0001; five warmup epochs; cosine decay.
- Deep supervision weight 1.0; boundary weight 0.1; gradient clipping 12.
- CUDA BF16; GPU recorded as NVIDIA GeForce RTX 5050 Laptop GPU.
- Gaussian sliding-window logit blending; overlap 0.5; no postprocessing.
- Checkpoint selection: highest mean patient full-volume validation Dice at 0.5.

Recorded environment: Python 3.12.10; torch 2.11.0+cu128; NumPy 2.5.4;
SciPy 1.18.1; SimpleITK 2.5.6; pydicom 3.0.2; rt-utils 1.2.7;
matplotlib 3.11.2; tqdm 4.70.1. These are artifact records, not independently
verified installation guarantees for every machine.

## Results and provenance

Best training validation: epoch 90, mean Dice 0.3563358974 at threshold 0.5.
Final epoch 100 validation Dice: 0.3268793035. The selected checkpoint was
evaluated separately on validation before threshold selection.

| Metric | Validation at 0.5 | Test at frozen 0.98 |
| --- | ---: | ---: |
| Attempted/successful cases | 83/83 | 83/83 |
| Technical failures | 0 | 0 |
| Mean patient Dice | 0.356317 | 0.421669 |
| Mean precision | 0.269122 | 0.492466 |
| Mean recall | 0.726417 | 0.458571 |
| HD95 (mm, finite cases) | 165.386273 | 110.004109 |
| Surface Dice (2 mm) | 0.225285 | 0.258422 |

The refined validation threshold search selected 0.98, with a validation Dice of
0.4648481538. The frozen operating point and test protocol refer to the same
checkpoint SHA-256:

```text
e8dc5f2b58a71e8a38cf899154aa2c5c5b6908df0e9db680213e3036b6600a9a
```

Local sources, intentionally excluded from Git: the run's `history.csv` and
`resolved_config.json`, `evaluations/hybrid_val/aggregates.json`,
`evaluations/hybrid_test/aggregates.json`, their protocols, and
`evaluations/hybrid_operating_point_refined.json`.

## Interpretation and limits

The test protocol records a frozen validation-selected threshold. That establishes
the recorded configuration; it cannot prove that nobody inspected test results
during earlier development. Keep subsequent tuning on development data.

The threshold-selection score is optimistic because it is selected on the same
validation cases. Validation at 0.5 and test at 0.98 are not a controlled
generalization comparison. False positives and missed target voxels remain, and
the surface distances are large. This experiment does not establish clinical
utility, architecture superiority, novelty, or state-of-the-art performance.
Matched baseline runs, repeated seeds, and independent data are still needed.

Metric surfaces are six-neighbour boundary voxel centres. HD95 pools distances
in both directions; Surface Dice is unweighted. Do not compare directly against
papers using a different convention. Refer to local aggregates for missing and
empty-surface counts when interpreting finite-only distance means.

The repository review fixes float32 accumulation of near-constant acquisition
descriptors for NEW training runs. Existing checkpoints restore their original
statistics unchanged. Fresh runs using corrected statistics may differ from this
historical run; no improved score is claimed from the code correction alone.
