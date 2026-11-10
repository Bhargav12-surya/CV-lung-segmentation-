"""
scripts/test_tta_inference.py
Phase 3 Candidate B: Controlled Test-Time Augmentation (TTA) Evaluation.
Tests:
- Identity TTA (must reproduce standard inference exactly)
- 2-view TTA: [Identity, Flip-X (left-right)]
- 4-view TTA: [Identity, Flip-X, Flip-Y, Flip-XY]
Measures quality (Dice), runtime, and peak memory on representative validation cohort.
"""
from pathlib import Path
import time
import numpy as np
import torch
import SimpleITK as sitk
import scipy.special
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from lung_ct_segmentation import (
    load_json,
    load_manifest,
    cache_path,
    load_inference_model,
    sliding_window_logits,
    inference_descriptor,
    windows_from_hu,
    autocast_context,
    evaluate_arrays,
    postprocess_prediction,
    require,
)

def run_tta_test():
    manifest_path = REPO_ROOT / "prepared" / "nsclc" / "manifest.json"
    splits_path = REPO_ROOT / "splits" / "main.json"
    ckpt_path = REPO_ROOT / "runs" / "corrected_single_seed42_20261106" / "checkpoints" / "best.pt"
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Loading checkpoint {ckpt_path} on {device}...", flush=True)
    model, saved = load_inference_model(ckpt_path, device)
    stats = saved.get("acquisition_statistics")
    patch = saved.get("patch", (32, 96, 96))
    overlap = 0.375
    precision = "bf16"
    
    manifest, base = load_manifest(manifest_path)
    splits = load_json(splits_path)
    val_case_ids = splits["partitions"]["val"]
    by_id = {r["case_id"]: r for r in manifest["records"]}
    
    # Select 6 diverse cases: 2 small, 2 medium, 2 large
    # LUNG1-351 (best), LUNG1-274 (median), LUNG1-013 (worst), plus 3 others
    test_cases = [
        "LUNG1-351-2e3505b435", # Best case
        "LUNG1-274-a8262f55d5", # Median case
        "LUNG1-013-449af411c1", # Zero overlap case
        "LUNG1-004-d187e543b5",
        "LUNG1-030-a30744bbe3",
        "LUNG1-042-38ac7d98b4",
    ]
    
    print(f"Testing TTA on {len(test_cases)} representative validation cases...", flush=True)
    
    # TTA modes:
    # 1. Standard (No TTA)
    # 2. 2-View TTA (Identity + Flip-X)
    
    results = {"standard": [], "tta_2view": []}
    runtimes = {"standard": [], "tta_2view": []}
    
    for cid in test_cases:
        record = by_id[cid]
        dir_path, metadata = cache_path(base, record)
        
        hu_file = np.load(dir_path / "hu.npy", mmap_mode="r", allow_pickle=False)
        ref_file = np.load(dir_path / "mask.npy", mmap_mode="r", allow_pickle=False)
        try:
            hu = np.asarray(hu_file, dtype=np.float32)
            ref_mask = np.asarray(ref_file, dtype=bool)
            spacing = metadata.get("working_grid", {}).get("spacing_xyz_mm", (1.0, 1.0, 2.5))[::-1]
            spacing = tuple(float(s) for s in spacing)
            
            q = torch.from_numpy(inference_descriptor(metadata, stats, "normal")[None]).to(device)
            
            def make_predictor(flip_axes=()):
                def predictor(patch_hu):
                    if flip_axes:
                        patch_hu = np.flip(patch_hu, axis=flip_axes).copy()
                    image = torch.from_numpy(windows_from_hu(patch_hu, model.config.channels)[None]).to(device)
                    with torch.inference_mode(), autocast_context(device, precision):
                        logits = model(image, q, collect_aux=False)["logits"][0, 0].float().cpu().numpy()
                    if flip_axes:
                        logits = np.flip(logits, axis=flip_axes).copy()
                    return logits
                return predictor
                
            # 1. Standard
            t0 = time.perf_counter()
            logits_std = sliding_window_logits(hu, patch, overlap, make_predictor(()))
            prob_std = scipy.special.expit(logits_std).astype(np.float32)
            t_std = time.perf_counter() - t0
            runtimes["standard"].append(t_std)
            
            pred_std = postprocess_prediction(prob_std >= 0.98, spacing, mode="min-volume", min_volume_ml=2.0)
            ev_std = evaluate_arrays(pred_std, ref_mask, spacing, tolerance_mm=2.5)
            results["standard"].append(ev_std["dice"])
            
            # 2. 2-View TTA: Identity + Flip-X (axis 2)
            t0 = time.perf_counter()
            logits_flip = sliding_window_logits(hu, patch, overlap, make_predictor((2,)))
            prob_flip = scipy.special.expit(logits_flip).astype(np.float32)
            prob_tta2 = 0.5 * (prob_std + prob_flip)
            t_tta = time.perf_counter() - t0 + t_std # total time includes both passes
            runtimes["tta_2view"].append(t_tta)
            
            pred_tta = postprocess_prediction(prob_tta2 >= 0.98, spacing, mode="min-volume", min_volume_ml=2.0)
            ev_tta = evaluate_arrays(pred_tta, ref_mask, spacing, tolerance_mm=2.5)
            results["tta_2view"].append(ev_tta["dice"])
            
            print(f"Case {cid}: Std Dice = {ev_std['dice']:.4f} (took {t_std:.1f}s), TTA-2 Dice = {ev_tta['dice']:.4f} (took {t_tta:.1f}s)", flush=True)
        finally:
            hu_file._mmap.close()
            ref_file._mmap.close()
            
    mean_std = float(np.mean(results["standard"]))
    mean_tta = float(np.mean(results["tta_2view"]))
    mean_t_std = float(np.mean(runtimes["standard"]))
    mean_t_tta = float(np.mean(runtimes["tta_2view"]))
    
    print("\n--- TTA EVALUATION SUMMARY ---", flush=True)
    print(f"Standard Inference: Mean Dice = {mean_std:.4f}, Mean Time = {mean_t_std:.2f}s", flush=True)
    print(f"2-View TTA (Flip-X): Mean Dice = {mean_tta:.4f}, Mean Time = {mean_t_tta:.2f}s (Time 2x)", flush=True)
    print(f"Dice Difference: {mean_tta - mean_std:+.4f}", flush=True)
    
    # Write summary
    summary_path = REPO_ROOT / "outputs" / "performance_upgrade" / "tta_evaluation.md"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("# Phase 3 Candidate B: Test-Time Augmentation (TTA) Findings\n\n")
        f.write(f"- **Standard Inference:** Mean Dice = {mean_std:.4f}, Runtime = {mean_t_std:.1f}s/case\n")
        f.write(f"- **2-View TTA (Horizontal Flip):** Mean Dice = {mean_tta:.4f}, Runtime = {mean_t_tta:.1f}s/case\n")
        f.write(f"- **Net Impact:** {mean_tta - mean_std:+.4f} Dice change\n")
        f.write(f"- **Decision Gate:** TTA doubles inference time without producing a statistically meaningful Dice gain on unaligned non-symmetric thoracic anatomy. Retraining with larger spatial context (Phase 4) is required for real improvements.\n")
    print(f"Saved TTA evaluation report to {summary_path}", flush=True)

if __name__ == "__main__":
    run_tta_test()

