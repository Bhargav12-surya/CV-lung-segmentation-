# Recorded compact v2 experiment

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
