# MSF-HSAT: 3D lung tumour segmentation

A PyTorch research project for segmenting the annotated primary gross tumour
volume (GTV-1) in CT scans. The implementation includes a residual 3D U-Net
baseline and a hybrid model with multiscale convolutions, transformer context,
wavelet skip refinement, acquisition conditioning, and adaptive decoder fusion.

This is an experimental segmentation model. Its predictions do not establish a
patient-level cancer diagnosis or clinical suitability.

## Project files

| Path | Purpose |
| --- | --- |
| `lung_ct_segmentation.py` | Preparation, splits, model, training, inference, metrics, reports, self-tests |
| `requirements.txt` | Dependencies used by the current implementation |
| `tests/` | Regression tests for normalization and threshold-selection integrity |
| `docs/USER_GUIDE.md` | Detailed architecture explanations and command reference |
| `docs/RESULTS.md` | Recorded experiment configuration, results, and limitations |
| `docs/REVIEW.md` | Release review findings and verification scope |
| `docs/TRAINING_IMPROVEMENTS.md` | Existing paths, cache recovery, sampling controls, measured probes |
| `docs/EXPERIMENTS.md` | Controlled baseline/full/removal plans for three seeds |
| `docs/EVALUATION.md` | Failure accounting, frozen postprocessing, paired comparisons |
| `scripts/` | Read-only data audit, verified cache recovery, experiment planning |
| `.github/workflows/tests.yml` | CPU checks on Python 3.11 and 3.12 |

Local datasets, predictions, runs, checkpoints, and older notebooks in `CV works/`
are excluded from Git. Their contents are not required to import the program or
run synthetic tests. Preparation and training commands recreate experiment files.

## Install and check

Use Python 3.11 or 3.12 and a dedicated environment:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
```

Install a PyTorch build appropriate for your machine using the
[official installation selector](https://pytorch.org/get-started/locally/), then:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe lung_ct_segmentation.py doctor
.\.venv\Scripts\python.exe lung_ct_segmentation.py self-test
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

On Linux, use `.venv/bin/python` instead. GPU training requires a compatible CUDA
build. All synthetic tests run on CPU and need no downloaded patient data.
Dependencies are compatibility ranges, not a lockfile; retain exact versions
from `doctor` and each run's `environment.json` for reproducibility.

## Data and workflow

The deleted processed cache has been restored for all 413 cases from the existing
raw data, retaining the original manifest/split identities. The restored-cache
audit passed, including 12 raw geometry rechecks.
[Current improvements and commands](docs/TRAINING_IMPROVEMENTS.md) describe the
verified paths, sampling controls, and historical recovery workflow.
No further training should be started without the user's instruction.

Obtain [NSCLC-Radiomics](https://www.cancerimagingarchive.net/collection/nsclc-radiomics/)
separately and retain its citation and usage terms. Supported inputs are paired
NIfTI CT/masks or CT plus RTSTRUCT. DICOM SEG ingestion is not implemented.
The pipeline checks physical geometry and groups splits by patient and source.

The recorded cohort has 413 eligible cases: 247 train, 83 validation, and 83 test.
Eligibility depends on preprocessing QC; do not assume every nominal collection
subject is usable. Full-volume CTs are resampled to `(2.5, 1, 1)` mm in z/y/x order.
The model uses lung and soft-tissue windows and learns from 3D patches.

Examples below assume the data cache and patient split already exist. See the
[user guide](docs/USER_GUIDE.md) for `prepare` and `split` commands first.

```powershell
# A NEW training run with the recorded v2 budget (never reuse an existing run path).
.\.venv\Scripts\python.exe lung_ct_segmentation.py train --manifest prepared/nsclc/manifest.json --splits splits/main.json --run runs/hybrid_new_seed42 --model hybrid3d --preset compact --preflight --max-epochs 100 --steps-per-epoch 1000 --batch-size 1 --accumulation 2 --lr 5e-5 --foreground-probability 0.25 --val-every 10 --early-stopping-checks 5 --workers 0

# Full validation evaluation of the selected checkpoint.
.\.venv\Scripts\python.exe lung_ct_segmentation.py evaluate --manifest prepared/nsclc/manifest.json --splits splits/main.json --partition val --checkpoint runs/hybrid_new_seed42/checkpoints/best.pt --output evaluations/new_val

# Freeze an operating point using validation probabilities only.
.\.venv\Scripts\python.exe lung_ct_segmentation.py select-threshold --evaluation evaluations/new_val --manifest prepared/nsclc/manifest.json --splits splits/main.json --output evaluations/new_operating_point.json --thresholds 0.5 0.7 0.9 0.95 0.97 0.98 0.99

# Use the frozen checkpoint and threshold for final test assessment.
.\.venv\Scripts\python.exe lung_ct_segmentation.py evaluate --manifest prepared/nsclc/manifest.json --splits splits/main.json --partition test --checkpoint runs/hybrid_new_seed42/checkpoints/best.pt --operating-point evaluations/new_operating_point.json --output evaluations/new_test
```

Each output path must be new. Resume an existing training run with its original
`--run` and `--resume .../checkpoints/last.pt`. Scientific settings are restored
from the checkpoint. Ctrl+C preserves the last completed epoch; unfinished epochs
are replayed on resume. Existing checkpoints retain their saved normalization
statistics, including checkpoints produced before the normalization fix.

## Recorded results

The local compact v2 run completed 100 epochs. Checkpoint selection used full-volume
validation Dice at threshold 0.5; epoch 90 was selected.

| Evaluation | Cases | Threshold | Mean patient Dice |
| --- | ---: | ---: | ---: |
| Validation, selected checkpoint | 83 | 0.50 | 0.3563 |
| Validation threshold-selection score | 83 | 0.98 | 0.4648 |
| Test, frozen validation threshold | 83 | 0.98 | 0.4217 |

These rows use different operating points. The validation threshold-selection
score is an optimized development score, not an independent performance estimate.
Test HD95 was 110.00 mm and Surface Dice was 0.2584, so substantial segmentation
errors remain. No postprocessing was applied. See [results and limitations](docs/RESULTS.md).

The corrected 100-epoch run reached validation Dice 0.3774 at threshold 0.5 and
0.4761 at validation-selected threshold 0.98 without component removal. The matched
previous-model score at 0.98 is 0.4648; the paired confidence interval includes
zero, so this is not an established improvement. No corrected-model test result
is claimed. See the [matched validation study](docs/RESULTS.md#corrected-run-validation-study-7-november-2026).

Surface metrics use six-neighbour boundary voxel centres and unweighted pooled
distances, not surfel-area weighting. Empty-surface distances are reported as
null with explicit counts. Compare methods using the same metric convention.

## Upload to GitHub

The ignore rules exclude medical images, patient-level artifacts, environments,
checkpoints, credentials, and generated outputs. Inspect the staged list before
committing. Avoid `git add -f` for excluded artifacts.

```powershell
git status --short
git add README.md requirements.txt lung_ct_segmentation.py CODEX_PROJECT_KNOWLEDGE.md .gitignore .gitattributes docs tests scripts .github
git diff --cached --stat
git commit -m "Prepare MSF-HSAT research implementation and tests"
git remote -v
git push -u origin main
```

This local repository already has an `origin`; inspect it before pushing. For a
new clone without one, add your own empty GitHub repository as `origin`. Aggregate
results are documented without patient images, per-patient identifiers, or local
source paths. No code licence is assigned here; the owner should choose an
appropriate licence before granting reuse rights. Dataset terms remain separate.
