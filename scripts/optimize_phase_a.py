"""Phase A: Zero-retraining threshold and component volume optimization.

Evaluates 30 combinations on the 83 validation cases from saved native probabilities.
Grid:
  thresholds: [0.70, 0.80, 0.85, 0.90, 0.95, 0.98]
  min_volume_ml: [0.0, 0.2, 0.5, 1.0, 2.0]
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
    evaluate_arrays,
    atomic_json,
    now,
    stable_hash
)

def run_phase_a():
    val_eval_dir = REPO_ROOT / "evaluations" / "corrected_single_val"
    manifest_path = REPO_ROOT / "prepared" / "nsclc" / "manifest.json"
    splits_path = REPO_ROOT / "splits" / "main.json"
    
    protocol = load_json(val_eval_dir / "protocol.json")
    manifest, base = load_manifest(manifest_path)
    splits = load_json(splits_path)
    
    val_case_ids = splits["partitions"]["val"]
    by_id = {r["case_id"]: r for r in manifest["records"]}
    
    thresholds = [0.70, 0.80, 0.85, 0.90, 0.95, 0.98]
    volumes_ml = [0.0, 0.2, 0.5, 1.0, 2.0]
    
    combos = {}
    for t in thresholds:
        for v in volumes_ml:
            mode = "none" if v == 0.0 else "min-volume"
            combos[(t, v)] = {
                "threshold": float(t),
                "min_volume_ml": float(v),
                "mode": mode,
                "patient_dice": {},
                "patient_iou": {},
                "patient_precision": {},
                "patient_recall": {},
                "patient_pred_vol": {},
                "patient_ref_vol": {},
                "zero_dice_count": 0,
                "missed_target_count": 0,
            }
            
    print(f"Starting Phase A evaluation: 83 validation cases across {len(combos)} configurations...", flush=True)
    
    structure = ndi.generate_binary_structure(3, 1) # 6-connectivity standard
    
    # Process one case at a time to keep RAM bounded
    for idx, cid in enumerate(val_case_ids, 1):
        record = by_id[cid]
        pid = record["patient_id"]
        dir_path, metadata = cache_path(base, record)
        ref_file = np.load(dir_path / "native_mask.npy", mmap_mode="r", allow_pickle=False)
        try:
            ref_mask = np.asarray(ref_file, dtype=bool)
            prob_path = val_eval_dir / "cases" / cid / "probability.nii.gz"
            prob_img = sitk.ReadImage(str(prob_path))
            prob = sitk.GetArrayFromImage(prob_img)
            spacing = metadata.get("native_spacing_zyx_mm", metadata["native_grid"]["spacing_xyz_mm"][::-1])
            spacing = tuple(float(s) for s in spacing)
            voxel_ml = float(np.prod(spacing) / 1000.0)
            
            r_sum = int(ref_mask.sum())
            ref_vol = r_sum * voxel_ml
            
            for t in thresholds:
                pred_bin = prob >= t
                if not pred_bin.any():
                    labels, count = None, 0
                    sizes = None
                else:
                    labels, count = ndi.label(pred_bin, structure=structure)
                    sizes = np.bincount(labels.ravel(), minlength=count + 1)
                    sizes[0] = 0
                
                for v in volumes_ml:
                    c = combos[(t, v)]
                    if v == 0.0:
                        pred_proc = pred_bin
                    else:
                        if count == 0:
                            pred_proc = pred_bin
                        else:
                            keep = (sizes * voxel_ml) >= v
                            keep[0] = False
                            pred_proc = keep[labels]
                    
                    p_sum = int(pred_proc.sum())
                    tp = int(np.count_nonzero(pred_proc & ref_mask))
                    
                    dice = (2.0 * tp / (p_sum + r_sum)) if (p_sum + r_sum) > 0 else 1.0
                    iou = (tp / (p_sum + r_sum - tp)) if (p_sum + r_sum - tp) > 0 else 1.0
                    prec = (tp / p_sum) if p_sum > 0 else (1.0 if r_sum == 0 else 0.0)
                    rec = (tp / r_sum) if r_sum > 0 else (1.0 if p_sum == 0 else 0.0)
                    pred_vol = p_sum * voxel_ml
                    
                    c["patient_dice"].setdefault(pid, []).append(float(dice))
                    c["patient_iou"].setdefault(pid, []).append(float(iou))
                    c["patient_precision"].setdefault(pid, []).append(float(prec))
                    c["patient_recall"].setdefault(pid, []).append(float(rec))
                    c["patient_pred_vol"].setdefault(pid, []).append(float(pred_vol))
                    c["patient_ref_vol"].setdefault(pid, []).append(float(ref_vol))
                    
                    if r_sum > 0 and tp == 0:
                        c["zero_dice_count"] += 1
                    if r_sum > 0 and p_sum == 0:
                        c["missed_target_count"] += 1
        finally:
            ref_file._mmap.close()
            
        if idx % 10 == 0 or idx == len(val_case_ids):
            print(f"Processed {idx}/{len(val_case_ids)} cases...", flush=True)

    results = []
    for (t, v), c in combos.items():
        # Patient-level aggregation first (mean per patient)
        p_dices = [float(np.mean(vals)) for vals in c["patient_dice"].values()]
        p_ious = [float(np.mean(vals)) for vals in c["patient_iou"].values()]
        p_precs = [float(np.mean(vals)) for vals in c["patient_precision"].values()]
        p_recs = [float(np.mean(vals)) for vals in c["patient_recall"].values()]
        p_pvols = [float(np.mean(vals)) for vals in c["patient_pred_vol"].values()]
        p_rvols = [float(np.mean(vals)) for vals in c["patient_ref_vol"].values()]
        
        row = {
            "threshold": c["threshold"],
            "min_volume_ml": c["min_volume_ml"],
            "mode": c["mode"],
            "mean_dice": float(np.mean(p_dices)),
            "median_dice": float(np.median(p_dices)),
            "std_dice": float(np.std(p_dices, ddof=1)),
            "mean_iou": float(np.mean(p_ious)),
            "mean_precision": float(np.mean(p_precs)),
            "mean_recall": float(np.mean(p_recs)),
            "zero_dice_cases": c["zero_dice_count"],
            "missed_target_cases": c["missed_target_count"],
            "mean_predicted_volume_ml": float(np.mean(p_pvols)),
            "mean_reference_volume_ml": float(np.mean(p_rvols)),
        }
        results.append(row)

    # Sort results
    # Tie-breaking preference: highest mean_dice, fewer missed targets, better precision-recall balance
    best = max(results, key=lambda r: (
        r["mean_dice"],
        -r["missed_target_cases"],
        -abs(r["mean_precision"] - r["mean_recall"])
    ))
    
    print("\n=== PHASE A: TOP 5 CONFIGURATIONS BY MEAN DICE ===", flush=True)
    sorted_res = sorted(results, key=lambda r: r["mean_dice"], reverse=True)
    for r in sorted_res[:5]:
        print(f"Thresh={r['threshold']:.2f}, VolCutoff={r['min_volume_ml']:.1f}mL | "
              f"Mean Dice={r['mean_dice']:.4f}, Median={r['median_dice']:.4f}, "
              f"Prec={r['mean_precision']:.4f}, Rec={r['mean_recall']:.4f}, "
              f"Missed={r['missed_target_cases']}/83", flush=True)
              
    print(f"\nSELECTED VALIDATION CONFIGURATION:", flush=True)
    print(json.dumps(best, indent=2), flush=True)
    
    # Save CSV
    csv_path = REPO_ROOT / "evaluations" / "optimization_phase_a.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"Saved: {csv_path}", flush=True)
    
    # Save JSON
    json_path = REPO_ROOT / "evaluations" / "optimization_phase_a.json"
    atomic_json(json_path, {
        "version": "1.0.0",
        "date": now(),
        "total_configurations": len(results),
        "selected_configuration": best,
        "configurations": results
    })
    print(f"Saved: {json_path}", flush=True)
    
    # Save Selected Operating Point JSON
    op_path = REPO_ROOT / "evaluations" / "selected_operating_point_phase_a.json"
    op_data = {
        "version": 2,
        "threshold": best["threshold"],
        "postprocessing": {
            "mode": best["mode"],
            "min_volume_ml": best["min_volume_ml"],
            "connectivity": 6,
            "grid": "native"
        },
        "source_partition": "val",
        "selected_validation_dice": best["mean_dice"],
        "selected_validation_median_dice": best["median_dice"],
        "selected_validation_precision": best["mean_precision"],
        "selected_validation_recall": best["mean_recall"],
        "missed_target_cases": best["missed_target_cases"],
        "created": now(),
        "checkpoint_sha256": protocol["checkpoint_sha256"],
        "manifest_fingerprint": manifest["fingerprint"],
        "split_fingerprint": protocol["split_fingerprint"],
    }
    op_data["fingerprint"] = stable_hash(op_data)
    atomic_json(op_path, op_data)
    print(f"Saved: {op_path}", flush=True)

if __name__ == "__main__":
    run_phase_a()
