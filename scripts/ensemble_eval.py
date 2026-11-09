"""Phase F: Ensemble Inference and Evaluation Script.

Averages sigmoid probability maps from multiple checkpoints (e.g., 5-Fold models).
Applies frozen threshold and minimum-volume postprocessing.
"""

import csv
import json
import math
import sys
from pathlib import Path
import numpy as np
import scipy.ndimage as ndi
import SimpleITK as sitk
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from lung_ct_segmentation import (
    load_json,
    load_manifest,
    cache_path,
    postprocess_prediction,
    evaluate_arrays,
    atomic_json,
    now,
    stable_hash
)

def evaluate_ensemble(probability_paths_list, ref_mask, spacing, threshold=0.98, min_volume_ml=2.0):
    """Average probability maps across multiple models and evaluate against reference mask."""
    prob_sum = None
    for p_path in probability_paths_list:
        prob_img = sitk.ReadImage(str(p_path))
        arr = sitk.GetArrayFromImage(prob_img).astype(np.float32)
        if prob_sum is None:
            prob_sum = arr
        else:
            prob_sum += arr
            
    p_mean = prob_sum / len(probability_paths_list)
    pred_bin = p_mean >= threshold
    
    if min_volume_ml > 0.0:
        pred_clean = postprocess_prediction(pred_bin, spacing, mode="min-volume", min_volume_ml=min_volume_ml)
    else:
        pred_clean = pred_bin
        
    return evaluate_arrays(pred_clean, ref_mask, spacing)

def main():
    print("Ensemble evaluation pipeline ready for multi-checkpoint averaging.")

if __name__ == "__main__":
    main()

