# Implementation review and verification

Pre-push update, 7 November 2026: the complete current synthetic regression suite
passed **69 tests**. This includes saved-probability rescoring/provenance checks
and patient-level paired-summary tests. The suite did not train a model or use
patient data; Windows temporary-file permissions required an approved run
outside the sandbox. The earlier verification records below are historical.

The 6 November 2026 changes preserve the main MSF-HSAT network and historical
checkpoints. Data restoration is tracked separately from code verification.

Changes include configurable and measured training sampling, training-only hard
negative provenance, faithful legacy resume, real-patch allocation preflight,
one-case diagnostic exports, explicit failure metrics, native component filtering
selected only on validation, model-comparison provenance, and controlled plans.
The core implementation remains `lung_ct_segmentation.py`.

New tests cover exact historical sampling compatibility, mixed sampling/fallbacks,
post-augmentation labels, hard-index integrity and train-only isolation, malformed
coordinates, resume recipe conflicts, component connectivity/physical volumes,
native output masks, operating-point validation, bootstrap units, and controlled
model differences. Mocked mining/preflight tests do not train a model.

Verified locally after the final sampler refinement: **58 regression tests passed**,
Python compilation passed, dependency checks reported no broken requirements,
and `git diff --check` passed. The newly completed synthetic self-test report
`outputs/verification/self_test_corrected_20261106.json` records **21 passes**,
including physical-grid reconstruction, raw DICOM/RTSTRUCT integration, Haar and
attention mathematics, model shapes/loss/backward, and checkpoint agreement.
That invocation completed before the later instruction not to train anything.
The separate older `self_test.json` is retained as a historical record.
Two additional non-training recovery-follow-up tests passed, for **60 regression
checks in total**. They verify that completion launches only the read-only audit
and recovery failure launches no command.

Actual checks and measured limitations are described in
[TRAINING_IMPROVEMENTS.md](TRAINING_IMPROVEMENTS.md). The synthetic/CLI checks do
not establish model accuracy. The diagnostic was a training-case measurement;
the historical held-out results remain clearly separated in [RESULTS.md](RESULTS.md).

Run the current regression suite without starting training:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m pip check
git diff --check
```

On the restricted Windows execution environment, temporary DICOM/NIfTI fixtures
needed an approved run outside the sandbox; initial permission failures were not
reported as passing assertions. Do not silently skip failed checks. Generated
artifacts and datasets remain ignored by Git. No remote push, clinical validation,
new full-training score, or GitHub CI result is claimed by this local review.
