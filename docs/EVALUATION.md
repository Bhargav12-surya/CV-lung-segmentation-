# Evaluation and operating-point integrity

Use full native CT volumes and one resampling unit per patient. Patch-training
Dice loss is a different measurement from full-volume validation Dice. Multiple
series from one patient are averaged before patient bootstrap resampling.

The checkpoint criterion remains the prespecified mean patient full-volume
validation Dice at the training threshold. Threshold and component filtering are
selected later on development data and recorded as a distinct operating-point
selection. A higher score after threshold tuning is not evidence of better trained
weights. Never use the already-evaluated test partition to pick a new recipe.

## Empty predictions, zero overlap, and undefined distances

`complete_miss` is retained for compatibility: positive reference with an entirely
empty prediction. It does not detect all missed targets. The additional fields are:

| Field | Meaning |
| --- | --- |
| `empty_prediction` | Prediction has no foreground, regardless of reference |
| `both_empty` | Both prediction and reference have no foreground |
| `zero_overlap` | Positive reference and no true-positive voxel; includes empty misses |
| `disjoint_nonempty_prediction` | Positive reference and nonempty prediction with no overlap |

Both-empty Dice/IoU/Surface Dice are 1. One-empty Dice and Surface Dice are 0.
HD95 and ASSD are null for both-empty or one-empty surfaces, with explicit status
and counts. Finite-only distance means must be read together with those counts.
Surface Dice uses the configured tolerance (default 2 mm), six-neighbour boundary
voxel centres, and unweighted pooled distances. It is not surfel-area weighted.

Technical failures remain in patient records and failures.csv. Available-case means
exclude them but explicitly report their count and policy; they are not substituted
with invented scores. Training rejects any validation technical failure during
checkpoint selection. Historical records can infer zero-overlap accounting from
Dice and predicted/reference volumes without rewriting old result files.

## Freeze threshold and component filtering using validation only

First evaluate the selected checkpoint on validation and retain probability files.
Then use a NEW output path for the operating point:

```powershell
.\.venv\Scripts\python.exe lung_ct_segmentation.py select-threshold --evaluation evaluations/NEW_val --manifest prepared/nsclc/manifest.json --splits splits/main.json --output evaluations/NEW_operating_point.json --thresholds 0.5 0.7 0.9 0.95 0.98 --postprocess-modes none largest min-volume --min-volumes-ml 0.1 1.0
```

The command scores threshold/recipe combinations on the same complete validation
case list, averages cases within patients, and chooses the highest patient mean
Dice. Exact ties prefer the simpler recipe, then a threshold closest to 0.5.
The output includes all candidate scores and failure counts, the selected recipe,
checkpoint hash, target, cohort/split fingerprints, and recorded overlap. It is a
development-selected score, with selection optimism, not independent performance.

Recipes operate on the native thresholded mask with six-neighbour components:

- `none`: preserve all predicted voxels.
- `largest`: retain the largest component; exact ties choose the first in array order.
- `min-volume`: retain every component meeting the physical cutoff in mL.

Neither the filtering nor inference uses a tumour reference to choose a crop or
component. Large-component filtering may discard true target tissue. No particular
HD95 or Dice improvement is guaranteed.

Pass the resulting `--operating-point` to `evaluate` or `predict`. Inference verifies
its fingerprint, checkpoint, cohort, split, target, and recorded overlap. Threshold
and recipe cannot be independently overridden at test time. Legacy operating points
with `postprocessing: "none"` remain compatible. Metrics, exported binary masks,
and qualitative contours use the same processed native mask. Continuous probability
exports remain unchanged for provenance and descriptive calibration.

## Probability and precision-recall interpretation

The existing validation reliability file is evidence of overconfidence: its top
probability bin had average predicted membership about 0.9734 and observed target
fraction about 0.5005. This supports investigating probability calibration and false
positives; it does not establish a unique cause. Foreground sampling, loss,
context and model fitting are possible contributors that require controlled studies.

Tiny voxel-pooled Brier/ECE values are dominated by abundant background. Updated
reports additionally record sampled class counts, estimated foreground prevalence,
class-conditional probability quantiles, and weighted raw precision/recall at fixed
descriptive thresholds. These use bounded saved voxel samples, not complete-volume
patient segmentation metrics. They do not select a new threshold or include
component filtering. Sparse positive samples can make class summaries noisy.

## Paired patient-level comparisons

Explicit evaluation outputs must share cohort, patient/case identities, partition,
target definition, metric convention/tolerance, and reconstruction overlap. Duplicate
case identities and technical failures are rejected. The report computes within-patient
mean Dice differences and patient-bootstrap confidence intervals.

Two supported operating policies are kept distinct:

- Common prespecified threshold and postprocessing across all models.
- Individually validation-frozen operating points, verified against each model's
  saved `operating_point.json`, checkpoint hash, and development provenance.

The second policy allows model-specific thresholds/recipes while using identical
selection rules and patient partitions. Reports label the policy and both recipes.
Do not mix prespecified and optimized models in one purported controlled paired
comparison. For multiple seeds, report per-seed results and an explicitly planned
aggregation; three seeds do not constitute independent patient cohorts.
