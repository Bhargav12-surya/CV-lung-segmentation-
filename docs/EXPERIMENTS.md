# Reproducible development experiments

The maintained implementation is `lung_ct_segmentation.py`. Use the existing
`prepared/nsclc/manifest.json` and `splits/main.json`; none of the commands below
downloads data, recreates the split, or alters historical checkpoints. Run commands
from `D:\CV` in Windows PowerShell with the project's Python environment.

`runs/hybrid_compact_v2_seed42` is the currently available historical reference.
Its recorded configuration used 100 epochs, 1,000 microbatches/epoch, batch 1,
accumulation 2, LR 5e-5, foreground sampling 0.25, validation every 10 epochs, and
compact `(32,96,96)` patches. The selected epoch was 90. Confirm the audit's
checkpoint/configuration match before interpreting a newer local run as current.

## Diagnose before committing another long run

First run the geometry/cache audit and the short one/two-patient overfit mode
described in the repository README. Overfit mode disables augmentation and uses
training cases for the diagnostic evaluation; that score is not generalization.
Use a new run directory each time. Low diagnostic Dice after a short budget does
not establish a pipeline bug; inspect loss, gradients, positive-patch fraction,
native-grid overlays, and whether learning is still progressing.

The normalization correction computes new training-only descriptor statistics in
float64. A faithful resume restores the old statistics under the old weights.
The experiments below start fresh and keep historical checkpoints untouched.

## Prepare all recipes without starting training

```powershell
.\.venv\Scripts\python.exe scripts\plan_experiments.py --output outputs\experiment_plans\corrected_v3 --run-root runs\controlled_v3 --sampling mixed
```

This writes an immutable plan, a JSON configuration for each experiment, printable
commands, and a PowerShell launcher. It validates all commands using the actual
training parser and `model_config_from_args`, records source/config fingerprints,
checks the existing manifest/split, and measures parameter counts on CPU. It does
not allocate training tensors on the GPU or launch any training run.

By default, five model configurations are prepared for seeds 42, 43, and 44:

| Name | Change from full MSF-HSAT |
| --- | --- |
| `baseline` | The implemented residual `baseline3d`, without the proposed multiscale/transformer/wavelet/adaptive fusion mechanisms |
| `full` | Corrected full MSF-HSAT |
| `no_wavelet` | Only model-config `wavelet=False` |
| `no_adaptive_fusion` | Only model-config `adaptive_fusion=False` |
| `no_acquisition` | Only model-config `acquisition_conditioning=False` |

These are leave-one-out comparisons against the full model. They are separate
from the existing `ablate` command's cumulative module-addition sequence. The
planner tests reject unintended model-config differences in the three removals.
Removing wavelet skips also makes the conditioning inside those removed skips
inactive; nominal conditioning flags are retained so no extra module is disabled.

All recipes retain boundary and deep-supervision heads and share sampling, losses,
split, patch geometry, optimizer, epoch budget, checkpoint criterion, and threshold.
The baseline necessarily differs in architecture; it is the in-house residual
U-Net, not an official nnU-Net reproduction. Parameter counts differ and are logged.
The baseline command intentionally retains its boundary head to match supervision;
this differs from the older cumulative baseline ablation recipe.

CPU parameter counts measured while generating the compact plan (no accuracy
experiment was run): baseline 6,960,100; full 6,955,694; no wavelet 6,487,324;
no adaptive fusion 6,997,974; no acquisition conditioning 6,944,926. The residual
baseline and concatenation fusion can have more parameters than their adaptive
counterparts; describe the baseline as architecturally simpler, not necessarily
smaller. The generated plan records these numbers per configuration.

Early stopping is disabled in generated controlled plans so the planned budget
is exactly 50,000 optimizer updates per run. A failed/interrupted run must not be
presented as if it reached this budget. Use actual recorded steps, runtime, memory,
and parameters in comparisons. Identical budgets do not imply equal compute cost.

The proposed mixed sampling is **50% foreground, 40% anatomical background, 0%
hard negatives, 10% uniform context**. These are sampling-selection probabilities,
not guaranteed post-augmentation patch-label fractions, and are an explicit
development hypothesis rather than an established optimum. Inspect the actual
positive-patch fraction logged by training. Select proportions on validation.
Optional difficult-negative coordinates must come exclusively from training-set
predictions with the training-only mining command and a recorded index.

To isolate fresh normalization while retaining the historical sampling recipe:

```powershell
.\.venv\Scripts\python.exe scripts\plan_experiments.py --output outputs\experiment_plans\corrected_legacy_sampling --run-root runs\controlled_legacy_sampling --sampling legacy
```

Do not attribute a difference between `mixed` and the historical run solely to the
normalization correction: sampling and fixed-stop policy differ. Compare fresh
legacy and mixed runs with matched seeds/budgets to isolate sampling changes.

## Start exactly one selected experiment

List names without training:

```powershell
& .\outputs\experiment_plans\corrected_v3\run_one.ps1
```

Start the corrected full model, seed 42, after diagnostics and GPU preflight pass:

```powershell
& .\outputs\experiment_plans\corrected_v3\run_one.ps1 -Experiment full_seed42
```

For the corresponding baseline/removals, explicitly choose one of
`baseline_seed42`, `no_wavelet_seed42`, `no_adaptive_fusion_seed42`, or
`no_acquisition_seed42`. Repeat deliberately for seeds 43 and 44 only after the
first comparison is sound and the available compute budget is known. The launcher
uses an argument array rather than evaluating generated shell strings.

## Evaluate larger physical context separately

Compact patches cover approximately `80 x 96 x 96` mm. Balanced patches cover
`120 x 128 x 128` mm at the working spacing. Use measured training-set tumour
extents and the audit's fit fractions rather than guessing that compact context
is sufficient. A tumour bbox fitting in a patch is necessary for full-lesion
context, but not sufficient evidence of good segmentation.

```powershell
.\.venv\Scripts\python.exe scripts\plan_experiments.py --output outputs\experiment_plans\context_balanced --run-root runs\context_balanced --configs full --seeds 42 --preset balanced --sampling mixed --gradient-checkpointing
```

This only prepares a candidate recipe. Every training command includes preflight;
inspect measured real-data memory/throughput diagnostics on the available GPU
before a long run. Shipped compact/balanced patches obey the encoder stride lattice
`(8,16,16)`. Haar/window padding and interpolation handle mathematical grid details;
that is not a claim that every possible custom patch fits in VRAM. The planner
rejects custom controlled patches that violate this lattice. Gradient accumulation
reduces optimizer-update frequency, not the activation memory of one patch.

For a controlled context comparison, use exactly the same sampling, seed, other
training settings, and evaluation protocol as the compact counterpart. Record
increased voxel count, compute cost, and measured peak memory.

## Freeze development decisions before test assessment

Use `best.pt`, not automatically `last.pt`. Every run selects its checkpoint by
mean per-patient full-volume validation Dice at prespecified threshold 0.5. Patch
training loss/positive-patch statistics are diagnostics, not the main result.

For example, evaluate the new full seed-42 checkpoint on validation:

```powershell
.\.venv\Scripts\python.exe lung_ct_segmentation.py evaluate --manifest prepared\nsclc\manifest.json --splits splits\main.json --partition val --checkpoint runs\controlled_v3\full_seed42\checkpoints\best.pt --output evaluations\controlled_v3\full_seed42_val
.\.venv\Scripts\python.exe lung_ct_segmentation.py select-threshold --evaluation evaluations\controlled_v3\full_seed42_val --manifest prepared\nsclc\manifest.json --splits splits\main.json --output evaluations\controlled_v3\full_seed42_operating_point.json --thresholds 0.3 0.5 0.7 0.9 0.95 0.97 0.98 0.99
```

Use new output paths; output preservation checks reject reuse. Examine probability
distributions/calibration and precision-recall behavior before interpreting an
extreme selected threshold. A threshold-selected validation score is optimized on
that same development set; distinguish it from improvements after retraining.
Select postprocessing only on development data and freeze its provenance with the
checkpoint/threshold. Component removal has no guaranteed benefit and can remove
true targets. Do not use test outcomes to choose components or sampling.

The 83-case test set has already been evaluated in the historical project. Report
that history; further repeated development against its results compromises its
role as an untouched test. Use an independent cohort or a clearly prespecified
protocol for final claims after substantial additional development.

Compare compatible evaluation folders with the existing offline report command:

```powershell
.\.venv\Scripts\python.exe lung_ct_segmentation.py report --inputs evaluations\controlled_v3\baseline_seed42_val evaluations\controlled_v3\full_seed42_val --output outputs\comparisons\baseline_vs_full_seed42 --manifest prepared\nsclc\manifest.json --splits splits\main.json
```

Keep thresholds/metric convention/targets/partitions compatible for paired
comparisons; the report validates compatibility. Patient-level paired Dice
differences and patient bootstrap intervals are preferable to comparing isolated
means. Use all patients, explicitly report technical failures, empty predictions,
zero-overlap cases, and undefined surface distances, and retain representative
successes/failures. Synthetic tests are software verification, not accuracy results.
