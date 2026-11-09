"""Unit tests for Phase A postprocessing and physical component volume filtering."""

import unittest
import numpy as np
import scipy.ndimage as ndi
from lung_ct_segmentation import (
    postprocess_prediction,
    postprocessing_recipe,
    evaluate_arrays
)

class PhaseAPostprocessingTests(unittest.TestCase):
    def test_physical_volume_calculation_anisotropic(self):
        # Spacing: z=2.5mm, y=1.0mm, x=1.0mm -> voxel_ml = 2.5 * 1.0 * 1.0 / 1000 = 0.0025 mL
        spacing = (2.5, 1.0, 1.0)
        # Create a cube of 10x10x10 voxels = 1000 voxels = 2.5 mL
        mask = np.zeros((30, 30, 30), dtype=bool)
        mask[5:15, 5:15, 5:15] = True
        
        # Test cutoff 2.0 mL: should be kept
        kept_2 = postprocess_prediction(mask, spacing, mode="min-volume", min_volume_ml=2.0)
        self.assertEqual(int(kept_2.sum()), 1000)
        
        # Test cutoff 3.0 mL: should be removed
        kept_3 = postprocess_prediction(mask, spacing, mode="min-volume", min_volume_ml=3.0)
        self.assertEqual(int(kept_3.sum()), 0)

    def test_multiple_valid_components_preserved(self):
        # Two real tumor nodules: 400 voxels (1.0 mL) and 800 voxels (2.0 mL)
        spacing = (2.5, 1.0, 1.0) # 0.0025 mL per voxel
        mask = np.zeros((40, 40, 40), dtype=bool)
        # Component 1: 4 x 10 x 10 = 400 voxels = 1.0 mL
        mask[2:6, 5:15, 5:15] = True
        # Component 2: 8 x 10 x 10 = 800 voxels = 2.0 mL
        mask[20:28, 20:30, 20:30] = True
        
        # Cutoff 0.5 mL: both should be preserved! (unlike largest-only which deletes component 1)
        filtered = postprocess_prediction(mask, spacing, mode="min-volume", min_volume_ml=0.5)
        self.assertEqual(int(filtered.sum()), 1200)

    def test_tiny_false_positive_components_filtered(self):
        # 1 real tumor (1000 voxels = 2.5 mL) + 5 tiny noise specks (5 voxels each = 0.0125 mL)
        spacing = (2.5, 1.0, 1.0)
        mask = np.zeros((50, 50, 50), dtype=bool)
        mask[10:20, 10:20, 10:20] = True # real tumor
        
        # Add 5 tiny specks
        mask[2, 2, 2] = True
        mask[45, 45, 45] = True
        mask[30, 5, 5] = True
        mask[5, 40, 40] = True
        mask[40, 10, 40] = True
        
        self.assertEqual(int(mask.sum()), 1005)
        
        filtered = postprocess_prediction(mask, spacing, mode="min-volume", min_volume_ml=0.2)
        # The 5 tiny specks should be removed, leaving exactly 1000 voxels
        self.assertEqual(int(filtered.sum()), 1000)

    def test_empty_prediction_mask(self):
        spacing = (2.5, 1.0, 1.0)
        mask = np.zeros((20, 20, 20), dtype=bool)
        filtered = postprocess_prediction(mask, spacing, mode="min-volume", min_volume_ml=0.5)
        self.assertEqual(int(filtered.sum()), 0)
        self.assertFalse(filtered.any())

    def test_evaluate_arrays_with_filtered_mask(self):
        spacing = (2.5, 1.0, 1.0)
        ref = np.zeros((30, 30, 30), dtype=bool)
        ref[10:20, 10:20, 10:20] = True # 1000 voxels
        
        pred = np.zeros((30, 30, 30), dtype=bool)
        pred[10:20, 10:20, 10:20] = True # perfect overlap
        pred[2, 2, 2] = True # noise speck
        
        metrics_before = evaluate_arrays(pred, ref, spacing)
        # Precision before < 1.0 due to noise speck
        self.assertLess(metrics_before["precision"], 1.0)
        
        pred_clean = postprocess_prediction(pred, spacing, mode="min-volume", min_volume_ml=0.1)
        metrics_after = evaluate_arrays(pred_clean, ref, spacing)
        # Precision after is 1.0 and Dice is 1.0
        self.assertEqual(metrics_after["precision"], 1.0)
        self.assertEqual(metrics_after["dice"], 1.0)

if __name__ == "__main__":
    unittest.main()

