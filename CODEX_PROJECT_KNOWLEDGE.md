# CODEX KNOWLEDGE BASE: MSF-HSAT 3D LUNG TUMOR SEGMENTATION

> **Instructions for Codex / AI Assistants:**
> This document is an overview of the **MSF-HSAT** repository (`d:\CV`). The
> current source code, recorded configurations, and verified artifacts take
> precedence over prose summaries. See `docs/TRAINING_IMPROVEMENTS.md` and
> `docs/EVALUATION.md` for the corrected training/evaluation workflow.

---

## 1. Executive Summary & Clinical Task

- **Project Title:** Multi-Scale Feature Fusion and Hybrid Spectral Attention Transformer for 3D Thoracic CT Lung Cancer Tumor Segmentation (MSF-HSAT).
- **Core Research Task:** Voxel-wise binary segmentation of the expert-delineated primary gross tumor volume (**GTV-1**) from planning thoracic CT scans.
- **Dataset:** NSCLC-Radiomics (The Cancer Imaging Archive - TCIA). 413 eligible patient cases (247 train, 83 validation, 83 test).
- **Primary Model:** `hybrid3d` (MSF-HSAT) with a comparative baseline `baseline3d` (3D residual U-Net).
- **Clinical Scope & Boundaries:** 
  - Predicts **GTV-1 membership** within the annotated region.
  - **Does NOT** perform patient-level cancer diagnosis, benign vs. malignant nodule classification, metastasis detection, or clinical survival prediction.
  - False positives are strictly measured relative to the primary GTV-1 annotation ground truth.

---

## 2. Repository Map & File Inventory

```
d:\CV\
├── lung_ct_segmentation.py        # Core monolithic pipeline (data, models, train, eval, metrics, CLI)
├── requirements.txt               # PyTorch, SimpleITK, SciPy, NumPy, pydicom, rt-utils, etc.
├── README.md                      # GitHub root overview, quickstart, verification summary
├── doctor.json                    # Diagnostic system and dependency inspection cache
├── self_test.json                 # Synthetic mathematical and integration test results
│
├── docs/
│   ├── USER_GUIDE.md              # Detailed architecture, mathematical formulations, and manual
│   └── RESULTS.md                 # Verified empirical results, benchmark tables, and provenance
│
├── tests/
│   ├── test_integrity.py          # Regression tests for acquisition normalization & thresholding
│   └── __init__.py
│
├── outputs/
│   └── figures/                   # Academic publication figures
│       ├── generate_architecture.py   # Script rendering the detailed MSF-HSAT tensor flow diagram
│       ├── generate_methodology.py    # Script rendering the 5-stage clinical methodology workflow
│       ├── msf_hsat_architecture.png  # Rendered high-res architecture diagram
│       ├── msf_hsat_architecture.svg  # Rendered vector architecture diagram
│       ├── msf_hsat_methodology.png   # Rendered high-res methodology workflow
│       └── msf_hsat_methodology.svg   # Rendered vector methodology workflow
│
├── splits/
│   └── main.json                  # Immutable patient/source-grouped partitions (train/val/test)
│
├── runs/
│   └── hybrid_compact_v2_seed42/  # Primary 100-epoch training run
│       ├── checkpoints/
│       │   ├── best.pt            # Epoch 90 best validation checkpoint (SHA-256 verified)
│       │   └── last.pt            # Epoch 100 final checkpoint
│       ├── history.csv / .json    # Epoch-by-epoch loss, lr, grad norms, and validation metrics
│       ├── resolved_config.json   # Full resolved hyperparameters and model configuration
│       ├── protocol.json          # Run provenance and fingerprint record
│       ├── report.html            # 54-figure visual evaluation report with CT overlays
│       └── figures/               # 30+ generated training and validation charts (PNG)
│
├── evaluations/
│   ├── hybrid_val/                # Full-volume validation evaluation (83 cases)
│   ├── hybrid_test/               # Held-out test evaluation using frozen threshold (83 cases)
│   ├── hybrid_operating_point.json         # Coarse validation threshold search results
│   └── hybrid_operating_point_refined.json # Refined validation threshold (frozen at 0.98)
│
├── data/                          # Original DICOM / raw data archive
│   └── nsclc_radiomics/
│
├── prepared/                      # Resampled working arrays cache
│   └── nsclc/
│       ├── manifest.json          # Master dataset manifest with geometric fingerprints
│       ├── manifest.csv           # Tabular dataset summary
│       ├── qc.json / qc.csv       # Preprocessing quality control logs
│       └── qc/                    # QC visualization slices
│
└── CV works/                      # Legacy slides, reference prompts, and exploration notebooks
```

---

## 3. Data Ingestion, Physical Geometry & Preprocessing

### 3.1. Physical Space & Geometry
- **Axis Order Conventions:**
  - NumPy arrays: $(Z, Y, X)$
  - SimpleITK images: $(X, Y, Z)$
  - PyTorch Tensors: $(B, C, D, H, W)$ where $D=Z, H=Y, W=X$
- **Target Working Grid:** All volumes are resampled using SimpleITK to a fixed isotropic/anisotropic resolution of:
  $$\text{Target Spacing} = (2.5, 1.0, 1.0) \text{ mm in } (z, y, x)$$
- **Coordinate Transformations:** Physical origin, spacing, and 3D direction cosine matrices are preserved. Native grid predictions are resampled back to the original patient coordinate system during inference.

### 3.2. Dual-Window CT Normalization
Rather than raw scalar min-max scaling, the network accepts $C=2$ channels derived from aligned Hounsfield Unit (HU) windows:
1. **Lung Window ($C_0$):**
   $$\text{lung} = \frac{\text{clip}(\text{HU}, -1000, 400) + 1000}{1400}$$
   Captures fine parenchyma, bronchial trees, and pulmonary margins.
2. **Soft-Tissue / Mediastinal Window ($C_1$):**
   $$\text{soft} = \frac{\text{clip}(\text{HU}, -160, 240) + 160}{400}$$
   Resolves solid tumor cores, nodal involvement, and soft-tissue boundaries.

### 3.3. Target Masks & Boundary Ribbon
- **Ground Truth Target ($Y$):** Binary voxel mask ($0 = \text{background}, 1 = \text{GTV-1}$).
- **Boundary Distance Band ($Y_{\text{bound}}$):** A physical 2 mm two-sided surface band generated on the full working volume before cropping:
  - Computed via 6-neighbor surface voxel distance transforms.
  - Ensures patch boundaries do not manufacture artificial boundary artifacts.

### 3.4. Patient Partitioning
- Split generated by `split_command` with seed 42 into immutable grouped partitions:
  - **Train:** 247 patients (247 scans)
  - **Validation:** 83 patients (83 scans)
  - **Test:** 83 patients (83 scans)
- Grouping guarantees that patients with multiple scans or duplicate series identities are strictly confined to a single partition.

---

## 4. Model Architecture: MSF-HSAT (`hybrid3d`)

MSF-HSAT is a hybrid 3D convolutional-transformer-wavelet architecture designed to address anisotropic resolution, multiscale tumor morphology, and high-frequency edge loss.

```
Input Patch: (B, 2, D, H, W)
   │
[E0] ── 16 ch, (D, H, W) ── 1x3x3 Anisotropic Convs ────────────────────────┐
   │                                                                        │
[E1] ── 32 ch, (D, H/2, W/2) ── Stride (1,2,2) ──────────────────┐          │
   │                                                             │          │
[E2] ── 64 ch, (D/2, H/4, W/4) ── MultiScale Dilation (1,2,3) ───┼──┐       │
   │                                                             │  │       │
[E3] ── 128 ch, (D/4, H/8, W/8) ─ MultiScale Dilation (1,2,3) ───┼──┼──┐    │
   │                                                             │  │  │    │
[E4] ── 192 ch, (D/8, H/16, W/16) ── Dense Residual Block        │  │  │    │
   │                                                             │  │  │    │
[Bottleneck Transformer] ── 2x Pre-Norm Attention + 3D Pos Embed │  │  │    │
   │                                                             │  │  │    │
   ▼                                                             │  │  │    │
[D3 Adaptive Fusion] ◄── Conditioned Wavelet Skip (E3) ──────────┼──┼──┘    │
   │                  ◄── Aligned Transferred Skip (E2) ─────────┼──┘       │
   ▼                                                             │          │
[D2 Adaptive Fusion] ◄── Conditioned Wavelet Skip (E2) ──────────┼──────────┘
   │                  ◄── Aligned Transferred Skip (E1) ─────────┘
   ▼
[D1 Adaptive Fusion] ◄── Aligned Direct Skip (E1 & E0)
   │
[D0 Final Refinement] ◄── Direct Skip (E0)
   ├── Main Logits Head (B, 1, D, H, W)
   ├── Auxiliary Heads at D1, D2 (Deep Supervision)
   └── Boundary Head at D0 (2 mm Physical Band)
```

### 4.1. Detailed Stage Breakdown (Compact Preset: 6,955,694 params)

| Stage | Output Shape $(C, D, H, W)$ | Operation / Block Type | Key Mathematical Mechanism |
|---|---|---|---|
| **Input** | $(2, 32, 96, 96)$ | Dual-window CT patch | Normalized lung and soft-tissue windows |
| **E0** | $(16, 32, 96, 96)$ | Anisotropic Residual Block | $1 \times 3 \times 3$ convolutions; prevents premature through-plane slice mixing |
| **E1** | $(32, 32, 48, 48)$ | Downsampling Residual Block | Stride $(1, 2, 2)$; downsamples in-plane while preserving axial slice details |
| **E2** | $(64, 16, 24, 24)$ | `MultiScaleResidual` | Stride $(2, 2, 2)$; parallel depthwise $3\text{D}$ convs with dilations $d \in \{1, 2, 3\}$ |
| **E3** | $(128, 8, 12, 12)$ | `MultiScaleResidual` | Stride $(2, 2, 2)$; parallel depthwise $3\text{D}$ convs with dilations $d \in \{1, 2, 3\}$ |
| **E4** | $(192, 4, 6, 6)$ | Residual blocks | Stride $(2, 2, 2)$; deep semantic feature extraction |
| **T (Bottleneck)**| $(192, 4, 6, 6)$ | `BottleneckTransformer` | 2 blocks, 8 heads, 3D sinusoidal coordinate embeddings, scaled dot-product attention |
| **E3 Skips** | 8 bands $\times 128$ ch | `Haar3D` + Detail Gating | LLL passed to `WindowAttention3D`; 7 detail bands dynamically gated by $U_3$ and $q_3$ |
| **E2 Skips** | 8 bands $\times 64$ ch | `Haar3D` + Detail Gating | LLL passed to `WindowAttention3D`; 7 detail bands dynamically gated by $U_2$ and $q_2$ |
| **D3** | $(128, 8, 12, 12)$ | `AdaptiveFusion` | 3-way channel-wise Softmax source weighting (Deeper $T$, Wavelet $E_3$, Aligned $E_2$) |
| **D2** | $(64, 16, 24, 24)$ | `AdaptiveFusion` | 3-way channel-wise Softmax source weighting (Deeper $D_3$, Wavelet $E_2$, Aligned $E_1$) |
| **D1** | $(32, 32, 48, 48)$ | `AdaptiveFusion` | 3-way channel-wise Softmax source weighting (Deeper $D_2$, Skip $E_1$, Aligned $E_0$) |
| **D0** | $(16, 32, 96, 96)$ | Final Conv Refinement | Trilinear upsampling + concatenation with $E_0$ |
| **Heads** | $(1, 32, 96, 96)$ | $1 \times 1 \times 1$ Convs | Main logits, Aux 1, Aux 2, and Boundary Logits |

### 4.2. HSAT Wavelet-Conditioned Skip Details
1. **3D Haar Decomposition (`Haar3D`):**
   Fixed orthonormal filters $L = [1, 1]/\sqrt{2}$ and $H = [1, -1]/\sqrt{2}$ yield 8 directional bands:
   $$\{LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH\} \quad (\text{order: } z, y, x)$$
2. **Low-Frequency Attention (`WindowAttention3D`):**
   The $LLL$ coarse band passes through window multi-head self-attention with window size $(4, 6, 6)$ and 4 heads. Windows are not shifted.
3. **Conditioned High-Frequency Detail Gating:**
   For each detail band $H_b$:
   $$R_b = H_b + \delta_b(H_b)$$
   $$g_b = \sigma\Big(G_b\big(\text{concat}(|H_b|,\, \text{GAP}(U_i),\, \text{Embed}(q_i))\big)\Big)$$
   $$\widetilde{H}_b = g_b \cdot R_b$$
   - $U_i$: Top-down decoder context from deeper stage.
   - $q_i$: 9-dimensional normalized acquisition descriptor:
     $$q_i = \big[\log(\text{native\_spacing}),\, \log(\text{target\_spacing} / \text{native\_spacing}),\, \log(\text{target\_spacing}\cdot\text{stage\_stride}_i)\big]$$
4. **Inverse Haar Synthesis & Bounded Residual Injection:**
   After 3D inverse Haar transform:
   $$S_i = E_i + \sigma(\kappa_i) \cdot \text{Conv}_{1\times 1\times 1}(\text{Reconstructed} - E_i)$$
   where $\kappa_i$ is initialized such that $\sigma(\kappa_i) \approx 0.1$, guaranteeing gradient stability.

### 4.3. 3-Way Adaptive Decoder Fusion
At each decoder stage $D_i$, three aligned feature streams ($S_{\text{deep}}, S_{\text{skip}}, S_{\text{cross}}$) are projected to channel count $C$:
$$W = \text{Softmax}_{\text{sources}}\Big(\text{MLP}\big(\text{GAP}(S_1 + S_2 + S_3)\big)\Big) \in \mathbb{R}^{3 \times C}$$
$$F_{\text{fused}} = \sum_{m=1}^3 W_{m, c} \cdot S_{m, c}$$
This allows feature channels to independently select semantic context, wavelet-refined boundaries, or multi-scale features.

---

## 5. Loss Formulation & Optimization

### 5.1. Total Loss Function
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{main}} + \lambda_{\text{ds}}\Big(0.5 \mathcal{L}_{D1} + 0.25 \mathcal{L}_{D2}\Big) + \lambda_{\text{boundary}} \mathcal{L}_{\text{boundary}}$$
- Default weights: $\lambda_{\text{ds}} = 1.0$, $\lambda_{\text{boundary}} = 0.1$.
- **Main Segmentation Loss:**
  $$\mathcal{L}_{\text{main}} = \mathcal{L}_{\text{Dice}}(P_{\text{main}}, Y) + \mathcal{L}_{\text{BCE}}(Z_{\text{main}}, Y)$$
  $$\mathcal{L}_{\text{Dice}} = 1 - \frac{2 \sum (P \cdot Y) + 10^{-5}}{\sum P + \sum Y + 10^{-5}}$$
- **Auxiliary Losses:** $\mathcal{L}_{D1}, \mathcal{L}_{D2}$ evaluate trilinearly interpolated auxiliary logits against full-resolution ground truth.
- **Boundary Loss:** Binary cross-entropy between predicted boundary logits and the 2 mm distance band $Y_{\text{bound}}$.

### 5.2. Optimizer & Training Hyperparameters
- **Optimizer:** AdamW ($\beta_1 = 0.9, \beta_2 = 0.999$, weight decay $= 10^{-4}$).
- **Base Learning Rate:** $\eta = 5 \times 10^{-5}$.
- **Schedule:** 5 warmup epochs, followed by cosine annealing decay to $0.01 \times \eta$.
- **Batching & Accumulation:** Microbatch size $= 1$, gradient accumulation steps $= 2$ (effective batch size $= 2$).
- **Precision:** Mixed Precision AMP (`bf16` or `fp16` with `torch.amp.GradScaler`).
- **Gradient Clipping:** Max norm $= 12.0$.
- **Steps per Epoch:** 1,000 microbatches per epoch (100 epochs total $= 50,000$ optimizer steps).

---

## 6. Empirical Benchmarks & Verified Results

The local reference run (`hybrid_compact_v2_seed42`) completed all 100 epochs on an NVIDIA RTX 5050 Laptop GPU. Summed history components are about 4.12 hours training plus 7.35 hours validation; these are logged components, not a fresh elapsed-time benchmark.

### 6.1. Official Validation and Test Results

| Evaluation Split | Checkpoint | Threshold | Mean Dice | Median Dice | Precision | Recall | HD95 (mm) | Surface Dice (2mm) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Validation (83 cases)** | Epoch 90 (`best.pt`) | 0.50 (raw) | 0.3563 | 0.3639 | 0.2691 | **0.7264** | 165.39 | 0.2253 |
| **Validation (Threshold search)** | Epoch 90 (`best.pt`) | 0.98 (tuned) | **0.4648** | not recorded by threshold search | not recorded | not recorded | not recorded | not recorded |
| **Held-out Test (83 cases)** | Epoch 90 (`best.pt`) | **0.98 (frozen)** | **0.4217** | **0.4403** | **0.4925** | 0.4586 | **110.00** | **0.2584** |

### 6.2. Empirical Observations & Behavior
1. **Zero Empty-Prediction Misses:** The historical `complete_miss` count is 0. This only means positive-reference scans had nonempty predictions. A nonempty prediction can have zero overlap with the tumour. Updated evaluation reports empty predictions, zero-overlap cases, and nonempty disjoint cases separately.
2. **Prominent Tumor Accuracy:** On large, distinct lesions (e.g., `LUNG1-396`, `LUNG1-042`, `LUNG1-228`), the model achieves Dice scores between **0.75 and 0.78**.
3. **Threshold Sensitivity:** Validation at 0.5 had recall about 73% and precision about 27%. Validation-selected 0.98 gave development Dice 0.4648; test at that frozen threshold gave Dice 0.4217. These are operating-point changes, not newly trained weight improvements.
4. **HD95:** Test HD95 is about 110 mm. Determining whether distant islands, boundary errors, or missed targets drive it requires examining actual predictions; the aggregate alone does not establish the cause.

---

## 7. CLI Reference & Operational Workflows

All commands are executed via `lung_ct_segmentation.py` using the project's virtual environment:

### 7.1. Environment Diagnosis & Self-Tests
```powershell
.\.venv\Scripts\python.exe lung_ct_segmentation.py doctor
.\.venv\Scripts\python.exe lung_ct_segmentation.py self-test --output self_test.json
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

### 7.2. Training a New Model
```powershell
.\.venv\Scripts\python.exe lung_ct_segmentation.py train `
  --manifest prepared/nsclc/manifest.json `
  --splits splits/main.json `
  --run runs/hybrid_v3_seed42 `
  --model hybrid3d `
  --preset compact `
  --preflight `
  --max-epochs 100 `
  --steps-per-epoch 1000 `
  --batch-size 1 `
  --accumulation 2 `
  --lr 5e-5 `
  --foreground-probability 0.25 `
  --val-every 10 `
  --early-stopping-checks 5 `
  --workers 0
```

### 7.3. Resuming an Interrupted Run
```powershell
.\.venv\Scripts\python.exe lung_ct_segmentation.py train `
  --manifest prepared/nsclc/manifest.json `
  --splits splits/main.json `
  --run runs/hybrid_v3_seed42 `
  --resume runs/hybrid_v3_seed42/checkpoints/last.pt
```

### 7.4. Full-Volume Evaluation
```powershell
# Evaluate validation partition
.\.venv\Scripts\python.exe lung_ct_segmentation.py evaluate `
  --manifest prepared/nsclc/manifest.json `
  --splits splits/main.json `
  --partition val `
  --checkpoint runs/hybrid_compact_v2_seed42/checkpoints/best.pt `
  --output evaluations/hybrid_val

# Evaluate test partition using frozen operating point
.\.venv\Scripts\python.exe lung_ct_segmentation.py evaluate `
  --manifest prepared/nsclc/manifest.json `
  --splits splits/main.json `
  --partition test `
  --checkpoint runs/hybrid_compact_v2_seed42/checkpoints/best.pt `
  --operating-point evaluations/hybrid_operating_point_refined.json `
  --output evaluations/hybrid_test
```

### 7.5. Operating Threshold Selection
```powershell
.\.venv\Scripts\python.exe lung_ct_segmentation.py select-threshold `
  --evaluation evaluations/hybrid_val `
  --manifest prepared/nsclc/manifest.json `
  --splits splits/main.json `
  --output evaluations/hybrid_operating_point_refined.json `
  --thresholds 0.5 0.7 0.9 0.95 0.97 0.98 0.99
```

### 7.6. Standalone HTML Report Generation
```powershell
.\.venv\Scripts\python.exe lung_ct_segmentation.py report `
  --inputs evaluations/hybrid_test `
  --manifest prepared/nsclc/manifest.json `
  --splits splits/main.json `
  --output evaluations/hybrid_test/report
```

---

## 8. Coding Guidelines & Codex Directives

When writing code, refactoring, or generating modules for this repository, Codex **must** adhere to the following strict conventions:

1. **Preserve Monolithic Code Integrity:**
   - The primary pipeline is deliberately contained within [lung_ct_segmentation.py](file:///d:/CV/lung_ct_segmentation.py) to prevent import desynchronization across platforms.
   - Do not split `lung_ct_segmentation.py` into micro-packages unless specifically asked.
2. **Explicit Axis Labeling:**
   - Always state axis conventions in comments or variable names (e.g., `spacing_zyx`, `shape_xyz`, `tensor_bcdhw`).
   - SimpleITK expects $(X, Y, Z)$; NumPy arrays are $(Z, Y, X)$. Never invert without explicit transposition.
3. **Atomic File I/O:**
   - Never write directly to persistent output destinations. Use `atomic_json()` and `atomic_checkpoint()` (write to `.tmp-<uuid>` and replace) to prevent corruption during process interruptions.
4. **Deterministic Randomness:**
   - Use `indexed_rng(seed, epoch, index)` for data sampling and augmentation. Never rely on non-seeded global Python or NumPy random states.
5. **No Silent Fallbacks:**
   - Do not catch exceptions silently or replace `NaN` losses with zeros. Fail fast with informative `UserError` messages.
6. **Windows & PowerShell Compatibility:**
   - Always protect entry points with `multiprocessing.freeze_support()`.
   - Default `workers=0` on Windows DataLoader invocations to avoid inter-process handle locking issues.
7. **Suggested Immediate Enhancements:**
   - **Post-Processing:** `select-threshold` can select connected-component recipes on validation and freeze them with the checkpoint. Filtering can remove true targets and does not guarantee a Dice or HD95 improvement.
   - **Physical Context:** Compare `balanced` patches `(48,128,128)` only after a real-patch memory preflight. Change context independently of loss weights before drawing conclusions. See the controlled plans for matching model budgets and seeds.

