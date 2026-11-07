# Correctness and training improvements (6 November 2026)

Status update, 7 November: all 413 caches were restored with original manifest
and split identities preserved. The restored-cache audit passed, including 12
raw geometry rechecks. The recovery commands and waiting states below describe
the historical recovery workflow, not a currently running recovery job.
The user subsequently completed one corrected 100-epoch run; its validation-only
comparison is recorded in [RESULTS.md](RESULTS.md#corrected-run-validation-study-7-november-2026).
No additional training is launched automatically.

Run commands from `D:\CV` in Windows PowerShell. No experiment suite is launched
automatically. The user has asked that no further model training be started.
Cache restoration and read-only verification may continue independently.

## Existing data and the relevant training session

The current complete, non-diagnostic training reference is
`runs/hybrid_compact_v2_seed42`. The audit compares its resolved model/configuration,
history, and both checkpoint metadata dictionaries rather than selecting a run
solely by its directory timestamp. `last.pt` records epoch 100; `best.pt` records
epoch 90. Historical results remain in [RESULTS.md](RESULTS.md).

Resolved project paths:

| Item | Existing location |
| --- | --- |
| Original raw CT/RTSTRUCT | `D:\CV\data\nsclc_radiomics` |
| Original manifest | `D:\CV\prepared\nsclc\manifest.json` |
| Working arrays and masks | `D:\CV\prepared\nsclc\cases` |
| Fixed patient split | `D:\CV\splits\main.json` |
| Historical checkpoints | `D:\CV\runs\hybrid_compact_v2_seed42\checkpoints` |
| Restorative preprocessing progress | `prepared/nsclc_recovery_all_20261106/recovery.json` |

The 413-case manifest and original 247/83/83 split remain unchanged. The user
confirmed deleting the processed case cache; raw data and trained checkpoints
were retained. Recovery rebuilds the same cache from those original raw sources.
It checks every original source fingerprint before restoration, stages one case
at a time, requires the regenerated case record to match, and restores only absent
cache directories. It never relabels a different source as an original case.
Existing valid caches, the original manifest/QC/split, and checkpoints are preserved.

Do not run a second recovery process while the first is active. If the recovery
process has stopped, resume with:

```powershell
.\.venv\Scripts\python.exe -u scripts\recover_cache.py --all --output prepared\nsclc_recovery_all_20261106 --resume
```

The progress file records verified/restored cases, remaining disk requirements,
memory waits, failures, and timing. Recovery uses one case at a time, no model,
no synthetic replacement data, and no download. Inspect `status`; do not assume
the whole cohort is ready because a single diagnostic case was restored.

A lightweight follow-up has also been started. It waits for successful recovery,
then audits every restored cache array and repeats 12 raw geometry checks. It
never starts training. Check its state with:

```powershell
Get-Content outputs\verification\dataset_audit_restored_20261106.job.json
```

`waiting_for_cache_recovery`, `running_read_only_audit`, and `passed` are distinct
states. A failed/interrupted recovery is reported without launching an audit or
claiming completion. Keep the computer on while these local processes run.
The complete raw geometry check for all 413 cases is already recorded separately.

## Verify data before another experiment

After recovery finishes, run this read-only audit:

```powershell
.\.venv\Scripts\python.exe scripts\audit_dataset.py --run runs\hybrid_compact_v2_seed42 --raw-limit 0 --full-hu --output outputs\verification\dataset_audit_restored_20261106.json
```

It validates manifest/split fingerprints, case/patient/source isolation, cache
completion markers, image/mask shapes, binary labels and emptiness, finite HU,
physical grids, and raw CT/contour geometry. Full-volume reconstructed overlays
remain important: header validity alone does not prove expert-label correctness.
Only training patients contribute acquisition statistics or tumour-extent/FOV
planning. The audit reports training tumour extents and physical patch-fit fractions;
it does not use test tumour size to choose a new patch.

## Sampling and faithful resume

The network and original Dice+BCE/auxiliary/boundary objectives are preserved.
The new controls are:

- `--foreground-probability`: foreground-centred selection probability.
- `--anatomical-background-probability`: tumour-free training patches centred in
  an image-derived region of lung-like HU and nearby soft tissue. The sparse HU
  region includes tissue within 24 mm of lung-like centres; it does not use the
  target to localize anatomy. Ground truth only excludes tumour-containing
  candidate background patches during training.
- `--hard-negative-probability` and `--hard-negative-index`: optional difficult
  negatives mined from training predictions only.
- The remaining probability samples uniform context.

Probabilities must sum to at most one. Candidate-negative rejection has a bounded
attempt count and explicitly logs uniform fallbacks. After augmentation, each
epoch records actual positive-patch fraction, mean foreground-voxel fraction,
selection counts, and fallbacks in history and `sampling/epoch_*.json`.
Selection probabilities are not guaranteed observed class fractions.

With both new probabilities at zero, the sampler preserves the old RNG stream,
crop, augmentation, windows, and descriptors. Legacy resume restores zero defaults
and refuses to introduce a new sampling recipe under old weights. Saved acquisition
statistics are restored verbatim; fresh statistics accumulate training-patient
observations in float64. Do not describe differences between a fresh mixed sampler
and the historical run as a normalization-only effect.

Hard-negative mining uses an explicitly selected non-diagnostic checkpoint and
the original training partition. It records manifest/split/checkpoint provenance
and bounded working-grid coordinates; the loader rejects validation/test cases,
wrong grids, malformed coordinates, and mutated fingerprints. Example for a future
development experiment (not launched during this update):

```powershell
.\.venv\Scripts\python.exe lung_ct_segmentation.py mine-negatives --manifest prepared\nsclc\manifest.json --splits splits\main.json --checkpoint runs\hybrid_compact_v2_seed42\checkpoints\best.pt --output outputs\hard_negatives\train_seed42.json --threshold 0.9 --centres-per-case 256
```

Per-slab candidate subsampling bounds memory and is documented in the index. A
high-confidence false-positive centre is not necessarily a tumour-free patch, so
the sampler checks the whole patch and logs fallback if no valid negative fits.

## Diagnostic and physical-context checks already performed

The completed raw-header/RTSTRUCT audit checked all 413 eligible cases and the
fixed split; raw geometry checks passed. Its initial cache snapshot had only the
first restored case, so its overall status correctly remained blocked on cache
recovery. Do not interpret that old missing-cache snapshot as a new data failure.

Using only the 247 training patients' verified physical contour bounds, median
tumour extents (z/y/x) were approximately 48.0/55.72/53.04 mm; 95th percentiles
were 108.0/105.21/99.39 mm. Compact physical FOV contained the bounding extents
in all axes for 190/247 cases, versus 237/247 for balanced. These are raw contour
bounds, not measurements of the reconstructed voxel masks or predictive accuracy.
They support testing larger context; extent fit does not guarantee a useful
anatomical margin or better segmentation.

Recomputing training descriptors from those verified raw headers gives unit
standard deviations for constant columns, and about 0.02313377 for the four
varying in-plane terms. Historical checkpoint values of roughly 1.4-2.1e-6 in
constant columns are the accumulation artifact; the old checkpoints still restore
those original values unchanged. Exact cached-metadata statistics are checked
again after recovery.

Before the later request to stop additional training, a separately named
one-training-patient diagnostic ran at `runs/diagnostic_corrected_20261106`.
It disabled augmentation, used the original train partition, and completed
10 epochs x 100 patches = 1,000 optimizer updates. It was not a full cohort run.
Full-volume training-case Dice rose from 0.120624 at step 500 to 0.332816 at
step 1,000. Final actual positive-patch fraction was 0.75; skipped optimizer updates
were zero. Losses and gradients remained finite. This is evidence of learning,
not successful complete memorization or a generalization score.

Axial/coronal/sagittal overlays were exported under the diagnostic run's
`diagnostic_evaluation/cases/` directory. Visual inspection of the restored case
showed correspondence between the reference target and learned prediction, with
additional false-positive tissue. This is one-case visual evidence only.
The diagnostic used the first lung-centred experimental sampler implementation;
the final anatomical sampler additionally includes nearby soft tissue to avoid
neglecting mediastinal background. Do not present its result as a validation
measurement of that later sampling refinement.

Two real-training-patch backward/AdamW allocation probes passed on this machine:

| Patch / physical field of view | Precision | Batch | Measured peak allocated CUDA |
| --- | --- | ---: | ---: |
| 32/96/96 voxels; 80/96/96 mm | BF16 | 1 | 0.8792 GiB |
| 48/128/128 voxels; 120/128/128 mm | BF16 | 1 | 2.1867 GiB |

The larger probe is saved in `runs/preflight_balanced_corrected_20261106`. It used
two optimizer steps and finite-gradient checks, and returned before training.
These are measured allocation probes on one training case, not full-cohort
throughput guarantees or accuracy improvements. CPU RAM remains a separate limit.
The supplied presets align with encoder strides; Haar/window padding and decoder
interpolation handle their internal grids.

`--preflight-only` on a new run path performs a real-patch probe and records its
configuration without creating trained checkpoints. No further probes/training
were launched after the user's later no-training instruction.

## Controlled comparisons and evaluation

See [EXPERIMENTS.md](EXPERIMENTS.md) for parser-validated baseline, full, no-wavelet,
no-adaptive-fusion, and no-acquisition recipes for seeds 42/43/44. Leave-one-out
config differences are tested. Heads, preprocessing, split, sampling, checkpoint
criterion, and planned update budgets are matched; parameter/compute costs differ.
The baseline is simpler in mechanisms but is not necessarily smaller in parameters.

See [EVALUATION.md](EVALUATION.md) for native full-volume patient metrics, empty
prediction versus nonempty-zero-overlap accounting, bootstrap confidence intervals,
validation-frozen thresholds/component recipes, and compatible paired comparisons.
Metrics, exported masks and overlays use the same filtered native prediction.
Validation tuning scores are reported separately from retraining results.

No new full-model, baseline, ablation, three-seed suite, hard-negative mining study,
postprocessing efficacy study, or test evaluation has been launched by this update.
Historical test results were not used to choose new settings. A new recipe is a
development hypothesis; no particular Dice or HD95 improvement is promised.
