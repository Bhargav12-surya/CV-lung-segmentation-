# MSF-HSAT 3D Lung Tumor Segmentation: Optimization Audit

**Date:** 10 November 2026  
**Repository:** `D:\CV`  
**Dataset:** TCIA NSCLC-Radiomics (413 cases: 247 train, 83 val, 83 held-out test; patient-grouped seed 42)  
**Hardware & Environment:**
- **OS:** Windows 11
- **Python:** 3.12.10
- **PyTorch:** 2.11.0+cu128 (CUDA 12.8, BF16 supported)
- **GPU:** NVIDIA GeForce RTX 5050 Laptop GPU (7.93 GiB total VRAM, ~6.90 GiB free)
- **Dependencies:** SimpleITK 2.5.6, NumPy 2.5.4, SciPy 1.18.1, scikit-image 0.26.0

---

## 1. Executive Summary & Verification Baseline

A complete repository audit and baseline smoke test were performed prior to making any code modifications.

### Smoke Test Outcomes
1. **CLI Doctor Check:** `lung_ct_segmentation.py doctor` ran cleanly, confirming CUDA availability, GPU device allocation, SimpleITK/PyDicom support, and 49+ GiB of free disk storage.
2. **Regression Suite:** Executed `python -m unittest discover -s tests -v`. All **69 unit tests passed in 9.02 seconds** with zero errors or warnings.
3. **Saved Predictions Inventory:** Inspected `evaluations/corrected_single_val/cases/`. Exactly **83 validation cases** are present with native-resolution 3D probability volumes (`probability.nii.gz`), metadata (`metadata.json`), and paired ground-truth masks in `prepared/nsclc/cases/`.

---

## 2. Historical Baseline Metrics (Ground-Truth Evidence)

From `evaluations/hybrid_test/aggregates.json` and `docs/RESULTS.md`:

| Partition & Policy | Mean Dice | Median Dice | IoU | Precision | Recall | HD95 (mm) | Surface Dice | Missed Targets |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Validation (threshold 0.50)** | 0.3774 | 0.3802 | 0.2588 | 0.2858 | 0.7036 | 173.15 | 0.2406 | 5 / 83 |
| **Validation (frozen threshold 0.98)** | **0.4761** | 0.4908 | 0.3487 | 0.5438 | 0.4800 | 110.23 | 0.2796 | 7 / 83 |
| **Held-Out Test (frozen threshold 0.98)** | **0.4217** | **0.4403** | **0.3047** | **0.4925** | **0.4586** | **110.00** | **0.2584** | 0 / 83 |

---

## 3. Bottleneck Analysis

1. **Precision Collapse at Canonical Threshold (0.50):**  
   At threshold 0.50, recall is high (0.7036), but precision drops to 0.2858 because the model predicts false positives across non-tumor anatomical structures (mediastinum, chest wall, vessels, ribs). Threshold 0.98 was previously required purely to suppress this noise.
2. **Severely Constrained 3D Receptive Field:**  
   The current compact preset uses a patch size of only `32 × 96 × 96` voxels (`80 × 96 × 96 mm`). In the axial/Z direction, 32 slices (80 mm) provides insufficient vertical context for the 3D convolutions and bottleneck transformer to distinguish apical lung parenchyma from lower thoracic structures.
3. **Underutilized GPU VRAM:**  
   The historical training probe measured peak CUDA allocation of only **0.879 GiB**, whereas the RTX 5050 Laptop GPU has **7.93 GiB VRAM**. There is ample capacity to scale to larger patches (`48 × 128 × 128` or `64 × 128 × 128`) with gradient accumulation.
4. **Postprocessing Component Filter Suboptimality:**  
   Filtering to "largest component only" raised validation Dice to 0.4947 but discarded satellite lesions, causing **23 missed targets**. A physical minimum-volume filter (`--min-volume-ml`) preserves multi-focal lesions while eliminating tiny false-positive specks.
5. **Loss Function Asymmetry:**  
   Standard Soft Dice + BCE does not sufficiently penalize large false-positive voxel counts in 3D volumes where foreground represents $< 0.2\%$ of total voxels. Focal Tversky Loss ($\alpha=0.7, \beta=0.3, \gamma=4/3$) directly targets this asymmetry.

---

## 4. Optimization Strategy Roadmap

- **Phase A:** Zero-retraining grid search on the 83 validation cases across probability thresholds `[0.70, 0.80, 0.85, 0.90, 0.95, 0.98]` and physical volume cutoffs `[0.0, 0.2, 0.5, 1.0, 2.0]` mL.
- **Phase B:** Anatomical lung-region extraction & safety validation (guaranteeing $\ge 99\%$ ground-truth tumor retention).
- **Phase C:** High-capacity training preset (`medium`: `48 × 128 × 128`) with preflight CUDA memory verification.
- **Phase D:** Focal Tversky Loss implementation with unit test verification.
- **Phase E & F:** Controlled training / ensembling recipes.
- **Phase G:** Final frozen test evaluation on the 83 held-out test patients.

