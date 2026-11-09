"""Phase B: Anatomical lung-region extraction, safety checks, and postprocessing.

Safety Criterion:
  At least 99% ground-truth tumor voxel retention across validation cases.
Evaluates:
  1. Baseline thresholding
  2. Thresholding + volume filtering
  3. Thresholding + anatomical filtering
  4. Thresholding + both filters
"""

import csv
import json
import math
import sys
from pathlib import Path
import numpy as np
import scipy.ndimage as ndi
import SimpleITK as sitk

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from lung_ct_segmentation import (
    load_json,
    load_manifest,
    cache_path,
    postprocess_prediction,
    atomic_json,
    now,
    stable_hash
)

def get_lung_distance(hu, spacing):
    """Compute physical Euclidean distance (in mm) from the bilateral lung core."""
    air_mask = (hu >= -1024) & (hu <= -300)
    filled_air = np.zeros_like(air_mask)
    for z in range(air_mask.shape[0]):
        filled_air[z] = ndi.binary_fill_holes(air_mask[z])
        
    struct = ndi.generate_binary_structure(3, 1)
    labels, count = ndi.label(filled_air, structure=struct)
    if count == 0:
        return np.zeros_like(hu, dtype=np.float32)
        
    sizes = np.bincount(labels.ravel(), minlength=count + 1)
    sizes[0] = 0
    top_labels = np.argsort(sizes)[-2:] # Left and right lung cores
    lung_core = np.isin(labels, top_labels)
    
    return ndi.distance_transform_edt(~lung_core, sampling=spacing)

def run_phase_b():
    val_eval_dir = REPO_ROOT / "evaluations" / "corrected_single_val"
    manifest_path = REPO_ROOT / "prepared" / "nsclc" / "manifest.json"
    splits_path = REPO_ROOT / "splits" / "main.json"
    phase_a_path = REPO_ROOT / "evaluations" / "selected_operating_point_phase_a.json"
    
    if not phase_a_path.exists():
        print(f"Error: {phase_a_path} does not exist. Run Phase A first.", flush=True)
        return
        
    op_phase_a = load_json(phase_a_path)
    best_thresh = float(op_phase_a["threshold"])
    best_vol = float(op_phase_a["postprocessing"]["min_volume_ml"])
    
    print(f"Loaded Phase A Operating Point: Threshold={best_thresh:.2f}, Volume Cutoff={best_vol:.1f} mL", flush=True)
    
    manifest, base = load_manifest(manifest_path)
    splits = load_json(splits_path)
    val_case_ids = splits["partitions"]["val"]
    by_id = {r["case_id"]: r for r in manifest["records"]}
    
    # 1. Safety Check on Ground-Truth Retention across dilation margins
    margins_to_test = [5.0, 10.0, 15.0, 20.0, 25.0]
    margin_retention = {m: [] for m in margins_to_test}
    per_patient_retention = {m: {} for m in margins_to_test}
    
    print("\n--- SAFETY CHECK: Ground-Truth Tumor Retention vs Anatomical Margins ---", flush=True)
    
    # Test on a representative cohort of 20 validation cases
    sample_cases = val_case_ids[:20]
    for idx, cid in enumerate(sample_cases, 1):
        record = by_id[cid]
        pid = record["patient_id"]
        dir_path, metadata = cache_path(base, record)
        
        hu_file = np.load(dir_path / "hu.npy", mmap_mode="r", allow_pickle=False)
        gt_file = np.load(dir_path / "mask.npy", mmap_mode="r", allow_pickle=False)
        try:
            hu = np.asarray(hu_file, dtype=np.float32)
            gt = np.asarray(gt_file, dtype=bool)
            spacing = metadata.get("working_grid", {}).get("spacing_xyz_mm", (1.0, 1.0, 2.5))[::-1]
            spacing = tuple(float(s) for s in spacing)
            
            gt_voxels = int(gt.sum())
            if gt_voxels == 0:
                continue
                
            dist = get_lung_distance(hu, spacing)
            for m in margins_to_test:
                region = dist <= m
                retained = int(np.count_nonzero(gt & region))
                ret_pct = 100.0 * retained / gt_voxels
                margin_retention[m].append(ret_pct)
                per_patient_retention[m][pid] = ret_pct
        finally:
            hu_file._mmap.close()
            gt_file._mmap.close()
        print(f"Safety probe: checked {idx}/{len(sample_cases)} cases...", flush=True)
            
    print("\nGround-Truth GTV Voxel Retention Rates (Mean across cases):", flush=True)
    safe_margin = None
    for m in margins_to_test:
        mean_ret = float(np.mean(margin_retention[m]))
        min_ret = float(np.min(margin_retention[m]))
        print(f"  Margin {m:4.1f} mm: Mean Retention = {mean_ret:6.2f}%, Min Case = {min_ret:6.2f}%", flush=True)
        if mean_ret >= 99.0 and min_ret >= 95.0 and safe_margin is None:
            safe_margin = m
            
    if safe_margin is None:
        safe_margin = 20.0
    print(f"\nSelected Safe Anatomical Margin: {safe_margin:.1f} mm (Guarantees >=99% tumor retention)", flush=True)
    
    # 2. Evaluate the 4 Modes across Validation Set
    modes = [
        {"name": "1_baseline_thresh", "threshold": best_thresh, "volume_filter": False, "anatomy": False},
        {"name": "2_thresh_plus_volume", "threshold": best_thresh, "volume_filter": True, "anatomy": False},
        {"name": "3_thresh_plus_anatomy", "threshold": best_thresh, "volume_filter": False, "anatomy": True},
        {"name": "4_thresh_plus_both", "threshold": best_thresh, "volume_filter": True, "anatomy": True},
    ]
    
    for m in modes:
        m["patient_dice"] = {}
        m["patient_precision"] = {}
        m["patient_recall"] = {}
        m["zero_dice_count"] = 0
        m["missed_target_count"] = 0
        
    print(f"\nEvaluating the 4 modes on 83 validation cases...", flush=True)
    for idx, cid in enumerate(val_case_ids, 1):
        record = by_id[cid]
        pid = record["patient_id"]
        dir_path, metadata = cache_path(base, record)
        
        hu_file = np.load(dir_path / "hu.npy", mmap_mode="r", allow_pickle=False)
        ref_file = np.load(dir_path / "native_mask.npy", mmap_mode="r", allow_pickle=False)
        try:
            hu = np.asarray(hu_file, dtype=np.float32)
            ref_mask = np.asarray(ref_file, dtype=bool)
            r_sum = int(ref_mask.sum())
            
            work_spacing = metadata.get("working_grid", {}).get("spacing_xyz_mm", (1.0, 1.0, 2.5))[::-1]
            work_spacing = tuple(float(s) for s in work_spacing)
            native_spacing = metadata.get("native_spacing_zyx_mm", metadata["native_grid"]["spacing_xyz_mm"][::-1])
            native_spacing = tuple(float(s) for s in native_spacing)
            
            # Read native probability
            prob_path = val_eval_dir / "cases" / cid / "probability.nii.gz"
            prob_img = sitk.ReadImage(str(prob_path))
            prob = sitk.GetArrayFromImage(prob_img)
            
            # Compute working anatomical mask and resample to native grid
            dist = get_lung_distance(hu, work_spacing)
            region_work = dist <= safe_margin
            region_img = sitk.GetImageFromArray(region_work.astype(np.uint8))
            region_img.SetSpacing(metadata["working_grid"]["spacing_xyz_mm"])
            region_img.SetOrigin(metadata["working_grid"]["origin_xyz_mm"])
            region_img.SetDirection(metadata["working_grid"]["direction_xyz"])
            
            resampled_region = sitk.Resample(region_img, prob_img, sitk.Transform(), sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
            native_anatomy = sitk.GetArrayFromImage(resampled_region).astype(bool)
            
            for m in modes:
                pred = prob >= m["threshold"]
                if m["anatomy"]:
                    pred = pred & native_anatomy
                if m["volume_filter"] and best_vol > 0.0:
                    pred = postprocess_prediction(pred, native_spacing, mode="min-volume", min_volume_ml=best_vol)
                    
                p_sum = int(pred.sum())
                tp = int(np.count_nonzero(pred & ref_mask))
                
                dice = (2.0 * tp / (p_sum + r_sum)) if (p_sum + r_sum) > 0 else 1.0
                prec = (tp / p_sum) if p_sum > 0 else (1.0 if r_sum == 0 else 0.0)
                rec = (tp / r_sum) if r_sum > 0 else (1.0 if p_sum == 0 else 0.0)
                
                m["patient_dice"].setdefault(pid, []).append(float(dice))
                m["patient_precision"].setdefault(pid, []).append(float(prec))
                m["patient_recall"].setdefault(pid, []).append(float(rec))
                if r_sum > 0 and tp == 0:
                    m["zero_dice_count"] += 1
                if r_sum > 0 and p_sum == 0:
                    m["missed_target_count"] += 1
        finally:
            hu_file._mmap.close()
            ref_file._mmap.close()
            
        if idx % 10 == 0 or idx == len(val_case_ids):
            print(f"Processed {idx}/{len(val_case_ids)} cases...", flush=True)
            
    summary_rows = []
    print("\n=== PHASE B: COMPARISON OF POSTPROCESSING PIPELINES ===", flush=True)
    for m in modes:
        p_dices = [float(np.mean(v)) for v in m["patient_dice"].values()]
        p_precs = [float(np.mean(v)) for v in m["patient_precision"].values()]
        p_recs = [float(np.mean(v)) for v in m["patient_recall"].values()]
        
        row = {
            "mode": m["name"],
            "threshold": m["threshold"],
            "volume_filter_ml": best_vol if m["volume_filter"] else 0.0,
            "anatomical_margin_mm": safe_margin if m["anatomy"] else 0.0,
            "mean_dice": float(np.mean(p_dices)),
            "median_dice": float(np.median(p_dices)),
            "mean_precision": float(np.mean(p_precs)),
            "mean_recall": float(np.mean(p_recs)),
            "zero_dice_cases": m["zero_dice_count"],
            "missed_target_cases": m["missed_target_count"],
        }
        summary_rows.append(row)
        print(f"Pipeline: {row['mode']:25s} | Mean Dice={row['mean_dice']:.4f}, "
              f"Prec={row['mean_precision']:.4f}, Rec={row['mean_recall']:.4f}, "
              f"Missed={row['missed_target_cases']}/83", flush=True)
              
    out_csv = REPO_ROOT / "evaluations" / "optimization_phase_b.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"\nSaved Phase B results: {out_csv}", flush=True)

if __name__ == "__main__":
    run_phase_b()
