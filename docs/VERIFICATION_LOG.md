# MSF-HSAT Optimization Verification Log

**Date:** 10 November 2026  
**Repository:** `D:\CV`  
**Execution Environment:** Windows 11, Python 3.12.10, PyTorch 2.11.0+cu128, NVIDIA GeForce RTX 5050 Laptop GPU (7.93 GiB VRAM)

---

## 1. Baseline Verification & Smoke Tests
- **Doctor CLI Check:** `lung_ct_segmentation.py doctor` ran cleanly. Verified CUDA 12.8, BF16 support, SimpleITK, NumPy, SciPy, and 49+ GiB disk space.
- **Regression Suite:** Executed `python -m unittest discover -s tests -v`. All 69 baseline tests passed.
- **Baseline Test Evidence:** Verified from `evaluations/hybrid_test/aggregates.json`:
  - Mean Dice: **0.4217**
  - Median Dice: **0.4403**
  - Mean IoU: **0.3047**
  - Mean Precision: **0.4925**
  - Mean Recall: **0.4586**
  - Mean HD95: **110.00 mm**
  - Technical Failures: **0 / 83**

---

## 2. Phase A: Zero-Retraining Threshold & Volume Postprocessing
- **Script:** `scripts/optimize_phase_a.py`
- **Unit Tests:** `tests/test_phase_a_postprocessing.py` (5/5 unit tests passed).
- **Execution:** 30 combinations on 83 validation cases from saved native-resolution probabilities.
- **Artifacts Saved:**
  - `evaluations/optimization_phase_a.csv`
  - `evaluations/optimization_phase_a.json`
  - `evaluations/selected_operating_point_phase_a.json`
- **Result:**
  - Baseline Validation Dice: 0.4761
  - Optimized Validation Dice: **0.4945** (Median: **0.5945**, Precision: **0.5905**) at threshold 0.98, `--min-volume-ml 2.0`.
  - Zero missed targets achieved at `--min-volume-ml 0.5` (Mean Dice 0.4815).

---

## 3. Phase B: Anatomical Lung-Region Safety Probes & Evaluation
- **Script:** `scripts/optimize_phase_b.py`
- **Safety Criterion:** $\ge 99\%$ ground-truth tumor voxel retention.
- **Probe Results:**
  - Tested margins: $5, 10, 15, 20, 25\text{ mm}$ across validation cases.
  - Margin $5.0\text{ mm}$ achieved **100.00% Mean Ground-Truth Retention** across all test cases.
  - Selected safe margin: $5.0\text{ mm}$.
- **83-Case Validation Results (`evaluations/optimization_phase_b.csv`):**
  - Baseline Thresh (0.98): Mean Dice = 0.4761, Prec = 0.5438, Rec = 0.4800
  - Thresh + Volume (2.0 mL): **Mean Dice = 0.4945**, **Prec = 0.5905**, Rec = 0.4750
  - Thresh + Anatomy (5.0 mm): Mean Dice = 0.4761, Prec = 0.5438, Rec = 0.4800
  - Thresh + Both (Vol + Anat): **Mean Dice = 0.4945**, **Prec = 0.5905**, Rec = 0.4750
- **Scientific Conclusion:** Predictions at operating threshold 0.98 are already confined to the thoracic cavity; physical volume filtering is the active mechanism eliminating small false positives.

---

## 4. Phase C: Receptive Field & Hardware VRAM Probes
- **Script:** `scripts/probe_memory.py`
- **Target Hardware:** NVIDIA GeForce RTX 5050 Laptop GPU (7.93 GiB Total VRAM)
- **Artifact Saved:** `evaluations/optimization_phase_c_preflight.json`
- **Allocations Measured:**
  - `compact` ($32 \times 96 \times 96$): Peak allocated **0.870 GiB**, Peak reserved **0.916 GiB** (7.02 GiB headroom).
  - `medium` ($48 \times 128 \times 128$): Peak allocated **2.127 GiB**, Peak reserved **2.273 GiB** (5.66 GiB headroom).
  - `large_64x128` ($64 \times 128 \times 128$): Peak allocated **2.798 GiB**, Peak reserved **2.973 GiB** (4.96 GiB headroom).
  - `large_48x160` ($48 \times 160 \times 160$): Peak allocated **3.286 GiB**, Peak reserved **3.525 GiB** (4.41 GiB headroom).
- **Preset Added:** Added `"medium": {"patch": (48, 128, 128), "widths": (16, 32, 64, 128, 192)}` to `lung_ct_segmentation.py`.

---

## 5. Phase D: Focal Tversky Loss Implementation
- **Mathematical Convention:**
  $$TI = \frac{TP + \epsilon}{TP + \alpha \cdot FP + \beta \cdot FN + \epsilon}$$
  $$\text{FocalTverskyLoss} = (1 - TI)^\gamma$$
  where $\alpha = 0.7$, $\beta = 0.3$, $\gamma = 4/3$, $\epsilon = 10^{-5}$.
- **Code Modifications:** Added `focal_tversky_loss`, updated `segmentation_loss` and `total_loss` in `lung_ct_segmentation.py`.
- **CLI Options Added:** `--loss`, `--loss-alpha`, `--loss-beta`, `--loss-gamma`.
- **Unit Tests:** `tests/test_phase_d_loss.py` (5/5 unit tests passed).

---

## 6. Phase F: 5-Fold Cross-Validation Setup
- **Script:** `scripts/train_kfold.py`
- **Split Artifact:** `splits/kfold_splits.json`
- **Patient Isolation:** 5 folds created from the 247 training patients exclusively. Zero overlap with the 83 dedicated validation or 83 held-out test patients.
- **Ensemble Pipeline:** Created `scripts/ensemble_eval.py` to average probability maps across trained fold checkpoints.

---

## 7. Full Regression Suite Status
Ran `python -m unittest discover -s tests -v`:
**Total Tests: 79 | Passed: 79 | Failed: 0 | Execution Time: 4.73s**
- Baseline Tests: 69/69 passed
- Phase A Postprocessing Tests: 5/5 passed
- Phase D Focal Tversky Tests: 5/5 passed

