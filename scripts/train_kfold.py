"""Phase F: Patient-level 5-Fold Cross-Validation Setup & Training Runner.

Rules:
  - 5 folds drawn EXCLUSIVELY from the 247 training patients.
  - Dedicated validation (83 patients) and held-out test (83 patients) are strictly excluded from fold training.
  - Grouped by patient ID to eliminate data leakage.
"""

import json
import math
import sys
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from lung_ct_segmentation import (
    load_json,
    load_manifest,
    atomic_json,
    now,
    stable_hash
)

def create_kfold_splits():
    splits_path = REPO_ROOT / "splits" / "main.json"
    manifest_path = REPO_ROOT / "prepared" / "nsclc" / "manifest.json"
    
    splits = load_json(splits_path)
    manifest, _ = load_manifest(manifest_path)
    
    train_case_ids = splits["partitions"]["train"]
    by_id = {r["case_id"]: r for r in manifest["records"]}
    
    # Group case IDs by patient
    patient_to_cases = {}
    for cid in train_case_ids:
        pid = by_id[cid]["patient_id"]
        patient_to_cases.setdefault(pid, []).append(cid)
        
    unique_patients = sorted(patient_to_cases.keys())
    n_patients = len(unique_patients)
    print(f"Total training patients for K-Fold: {n_patients} ({len(train_case_ids)} cases)", flush=True)
    
    # Deterministic K-Fold split (5 folds, seed 42)
    rng = np.random.default_rng(42)
    shuffled_patients = np.array(unique_patients)
    rng.shuffle(shuffled_patients)
    
    fold_splits = []
    fold_size = math.ceil(n_patients / 5)
    
    for fold in range(5):
        val_start = fold * fold_size
        val_end = min((fold + 1) * fold_size, n_patients)
        
        val_pids = set(shuffled_patients[val_start:val_end])
        train_pids = set(unique_patients) - val_pids
        
        fold_train_cases = []
        for pid in sorted(train_pids):
            fold_train_cases.extend(patient_to_cases[pid])
            
        fold_val_cases = []
        for pid in sorted(val_pids):
            fold_val_cases.extend(patient_to_cases[pid])
            
        # Verify strict patient disjointness
        assert train_pids.isdisjoint(val_pids), f"Fold {fold}: Patient overlap between train and val!"
        
        fold_splits.append({
            "fold": fold,
            "train_patients_count": len(train_pids),
            "val_patients_count": len(val_pids),
            "train_cases": fold_train_cases,
            "val_cases": fold_val_cases,
            "dedicated_validation_untouched": len(splits["partitions"]["val"]),
            "dedicated_test_untouched": len(splits["partitions"]["test"]),
        })
        print(f"  Fold {fold}: {len(train_pids)} train patients ({len(fold_train_cases)} cases) | "
              f"{len(val_pids)} val patients ({len(fold_val_cases)} cases)", flush=True)
              
    out_path = REPO_ROOT / "splits" / "kfold_splits.json"
    kfold_manifest = {
        "schema": "msf-hsat-kfold-v1",
        "date": now(),
        "seed": 42,
        "n_folds": 5,
        "folds": fold_splits,
    }
    kfold_manifest["fingerprint"] = stable_hash(kfold_manifest)
    atomic_json(out_path, kfold_manifest)
    print(f"\nSaved 5-Fold patient splits to: {out_path}", flush=True)
    return kfold_manifest

if __name__ == "__main__":
    create_kfold_splits()

