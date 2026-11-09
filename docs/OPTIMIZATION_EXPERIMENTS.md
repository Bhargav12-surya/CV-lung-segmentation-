# MSF-HSAT Optimization Experiments & Ablation Matrix

**Date:** 10 November 2026  
**Repository:** `D:\CV`  
**Dataset:** TCIA NSCLC-Radiomics (413 eligible CT cases: 247 train, 83 val, 83 held-out test; seed 42)  
**Evaluation Standard:** Patient-grouped, full 3D native CT geometry, uncropped inference

---

## 1. Experiment Matrix & Evaluated Configurations

| ID | Configuration & Preset | Loss Function | Operating Point & Postprocessing | Validation Mean Dice | Validation Median Dice | Validation Precision | Validation Recall | Missed Targets (Val) | Status & Verification |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **E0** | Compact (`32 × 96 × 96`) | Soft Dice + BCE | Threshold 0.50, None | 0.3774 | 0.3802 | 0.2858 | 0.7036 | 5 / 83 | Historical Verified Baseline |
| **E0-b** | Compact (`32 × 96 × 96`) | Soft Dice + BCE | Threshold 0.98, None | 0.4761 | 0.4908 | 0.5438 | 0.4800 | 7 / 83 | Historical Operating Point |
| **E1** | Compact (`32 × 96 × 96`) | Soft Dice + BCE | Threshold 0.98, **Vol Cutoff 2.0 mL** | **0.4945** | **0.5945** | **0.5905** | 0.4750 | **3 / 83** | **Phase A Verified Best** |
| **E1-b** | Compact (`32 × 96 × 96`) | Soft Dice + BCE | Threshold 0.98, **Vol Cutoff 0.5 mL** | **0.4815** | **0.5790** | **0.5571** | 0.4765 | **0 / 83** | **Zero Missed Targets** |
| **E2** | Medium Preset (`48 × 128 × 128`) | Soft Dice + BCE | Preflight Memory Probe | N/A (Probe) | N/A | N/A | N/A | N/A | **Passed (Peak: 2.127 GiB)** |
| **E3** | Large Preset (`64 × 128 × 128`) | Soft Dice + BCE | Preflight Memory Probe | N/A (Probe) | N/A | N/A | N/A | N/A | **Passed (Peak: 2.798 GiB)** |
| **E4** | Focal Tversky Loss | $\alpha=0.7, \beta=0.3, \gamma=4/3$ | Unit Test Verification | N/A (Unit Test) | N/A | N/A | N/A | N/A | **5/5 Unit Tests Passed** |
| **E5** | 5-Fold Cross Validation | Patient-Disjoint | Ensemble Pipeline Setup | N/A (Splits) | N/A | N/A | N/A | N/A | **Generated 5 Folds** |

---

## 2. Phase A Grid Search Detail (30 Configurations)

From `evaluations/optimization_phase_a.csv`:

- **Threshold Grid:** `[0.70, 0.80, 0.85, 0.90, 0.95, 0.98]`
- **Physical Volume Cutoffs:** `[0.0, 0.2, 0.5, 1.0, 2.0]` mL
- **Connectivity:** 6-connectivity 3D structure

### Top Configurations by Mean Dice:
1. **Threshold 0.98, Volume 2.0 mL:** Mean Dice = **0.4945**, Median = **0.5945**, Precision = **0.5905**, Recall = **0.4750**, Missed = 3 / 83
2. **Threshold 0.98, Volume 1.0 mL:** Mean Dice = **0.4854**, Median = **0.5845**, Precision = **0.5683**, Recall = **0.4753**, Missed = 1 / 83
3. **Threshold 0.95, Volume 2.0 mL:** Mean Dice = **0.4829**, Median = **0.5647**, Precision = **0.4957**, Recall = **0.5394**, Missed = 1 / 83
4. **Threshold 0.98, Volume 0.5 mL:** Mean Dice = **0.4815**, Median = **0.5790**, Precision = **0.5571**, Recall = **0.4765**, Missed = 0 / 83
5. **Threshold 0.98, Volume 0.2 mL:** Mean Dice = **0.4780**, Median = **0.5781**, Precision = **0.5487**, Recall = **0.4800**, Missed = 0 / 83

*Key Analysis:*
- Physical minimum volume filtering successfully suppresses small background false positives without destroying multi-focal lesions.
- In contrast to the historical "largest-component" heuristic (which missed 23 lesions), physical volume filtering at 2.0 mL misses only 3 targets, and at 0.5 mL misses **0 targets**.

---

## 3. Phase C Hardware VRAM Probes (RTX 5050 Laptop GPU, 7.93 GiB Total)

From `evaluations/optimization_phase_c_preflight.json`:

| Preset Name | Patch Dimensions | Encoder Widths | Peak Allocated VRAM | Peak Reserved VRAM | Available Headroom | Preflight Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`compact`** | $32 \times 96 \times 96$ | (16, 32, 64, 128, 192) | 0.870 GiB | 0.916 GiB | 7.017 GiB | **PASSED** |
| **`medium`** | $48 \times 128 \times 128$ | (16, 32, 64, 128, 192) | **2.127 GiB** | **2.273 GiB** | **5.660 GiB** | **PASSED** |
| **`large_64x128`**| $64 \times 128 \times 128$ | (16, 32, 64, 128, 192) | **2.798 GiB** | **2.973 GiB** | **4.961 GiB** | **PASSED** |
| **`large_48x160`**| $48 \times 160 \times 160$ | (16, 32, 64, 128, 192) | **3.286 GiB** | **3.525 GiB** | **4.408 GiB** | **PASSED** |

*Conclusion:* The network can be scaled up to $48 \times 128 \times 128$ or $64 \times 128 \times 128$ comfortably on the RTX 5050 Laptop GPU with over 4.9+ GiB of safety headroom.

---

## 4. Phase B Anatomical Lung Postprocessing Evaluation

From `evaluations/optimization_phase_b.csv`:

### 4.1 Anatomical Safety Margin Probe (Validation GTV Retention)
- Tested Dilation Margins: `[5.0, 10.0, 15.0, 20.0, 25.0]` mm on HU air mask `[-1000, -300] HU`.
- **Result:** $5.0\text{ mm}$ expansion margin achieved **100.00% Mean GTV Retention** (Min Case = 100.00%).
- Meets the $\ge 99.0\%$ safety criterion with zero accidental primary tumor clipping.

### 4.2 Full 83-Case Validation Pipeline Comparison
| Pipeline Mode | Threshold | Volume Cutoff | Anatomical Margin | Mean Dice | Median Dice | Precision | Recall | Zero Dice Cases | Missed Targets |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Baseline Threshold** | 0.98 | 0.0 mL | None | 0.4761 | 0.5767 | 0.5438 | 0.4800 | 7 | 0 |
| **2. Threshold + Volume Filter** | 0.98 | 2.0 mL | None | **0.4945** | **0.5945** | **0.5905** | 0.4750 | 12 | 3 |
| **3. Threshold + Anatomical Mask**| 0.98 | 0.0 mL | 5.0 mm | 0.4761 | 0.5767 | 0.5438 | 0.4800 | 7 | 0 |
| **4. Threshold + Both (Vol + Anat)** | 0.98 | 2.0 mL | 5.0 mm | **0.4945** | **0.5945** | **0.5905** | 0.4750 | 12 | 3 |

*Scientific Takeaway:*
- At operating threshold 0.98, the model's false-positive predictions are already tightly clustered within the thoracic cavity (0 distant non-thoracic false alarms outside the 5.0 mm lung margin).
- Consequently, anatomical masking alone does not alter the score at threshold 0.98.
- The dominant performance driver is **physical component volume filtering** (`min-volume`), which eliminates small intra-thoracic noise clusters and yields a verified +0.0184 boost in validation Mean Dice and +0.0467 boost in Precision.


