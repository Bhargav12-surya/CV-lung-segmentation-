"""Evaluation integrity: empty targets, disjoint predictions, frozen recipes."""
import argparse
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import SimpleITK as sitk

import lung_ct_segmentation as pipeline


class PatientMetricTests(unittest.TestCase):
    def test_nonempty_zero_overlap_is_not_an_empty_prediction(self):
        pred = np.zeros((5, 5, 5), bool)
        ref = pred.copy()
        pred[1, 1, 1], ref[2, 1, 1] = True, True
        result = pipeline.evaluate_arrays(pred, ref, (3, 1, 1))
        self.assertEqual(result["dice"], 0.)
        self.assertEqual(result["hd95_mm"], 3.)
        self.assertEqual(result["complete_miss"], 0)
        self.assertEqual(result["empty_prediction"], 0)
        self.assertEqual(result["zero_overlap"], 1)
        self.assertEqual(result["disjoint_nonempty_prediction"], 1)

    def test_empty_positive_reference_has_explicit_missing_surface(self):
        pred = np.zeros((3, 3, 3), bool)
        ref = pred.copy()
        ref[1, 1, 1] = True
        result = pipeline.evaluate_arrays(pred, ref, (2.5, 1, 1))
        self.assertEqual(result["empty_prediction"], 1)
        self.assertEqual(result["zero_overlap"], 1)
        self.assertEqual(result["complete_miss"], 1)
        self.assertEqual(result["disjoint_nonempty_prediction"], 0)
        self.assertIsNone(result["hd95_mm"])
        self.assertEqual(result["surface_dice"], 0.)

    def test_both_empty_is_not_a_positive_target_miss(self):
        empty = np.zeros((3, 3, 3), bool)
        result = pipeline.evaluate_arrays(empty, empty, (1, 1, 1))
        self.assertEqual(result["dice"], 1.)
        self.assertEqual(result["both_empty"], 1)
        self.assertEqual(result["empty_prediction"], 1)
        self.assertEqual(result["zero_overlap"], 0)
        self.assertIsNone(result["hd95_mm"])

    def test_summary_counts_historical_disjoint_case_and_technical_failure(self):
        rows = [{"case_id": "a", "patient_id": "p", "status": "success", "dice": 0.,
                 "predicted_volume_ml": 1., "reference_volume_ml": 2., "surface_status": "finite"},
                {"case_id": "b", "patient_id": "q", "status": "failed", "reason": "IO failure"}]
        result = pipeline.bootstrap_summary(rows, repeats=20)
        self.assertEqual(result["attempted_cases"], 2)
        self.assertEqual(result["technical_failures"], 1)
        self.assertEqual(result["zero_overlap_cases"], 1)
        self.assertEqual(result["disjoint_nonempty_prediction_cases"], 1)
        self.assertEqual(result["nonfinite_surface_cases"], 0)
        self.assertEqual(result["metrics"]["dice"]["mean"], 0.)

    def test_patient_bootstrap_does_not_weight_multi_series_patient_twice(self):
        rows = [{"case_id": "a", "patient_id": "p", "dice": 0.},
                {"case_id": "b", "patient_id": "p", "dice": 1.},
                {"case_id": "c", "patient_id": "q", "dice": 1.}]
        result = pipeline.bootstrap_summary(rows, repeats=20)
        self.assertEqual(result["metrics"]["dice"]["n_patients"], 2)
        self.assertEqual(result["metrics"]["dice"]["mean"], .75)
        self.assertEqual(result["nonfinite_surface_cases"], 0)
        self.assertEqual(result["unknown_surface_status_cases"], 3)


class ComponentFilteringTests(unittest.TestCase):
    def setUp(self):
        self.mask = np.zeros((6, 6, 6), bool)
        self.mask[1:3, 1:3, 1:3] = True
        self.mask[5, 5, 5] = True

    def test_largest_preserves_big_component_without_reference(self):
        result = pipeline.postprocess_prediction(self.mask, (2.5, 1, 1), "largest")
        self.assertEqual(int(result.sum()), 8)
        self.assertFalse(result[5, 5, 5])
        self.assertEqual(int(self.mask.sum()), 9)

    def test_minimum_volume_uses_physical_voxel_size(self):
        small = pipeline.postprocess_prediction(self.mask, (1, 1, 1), "min-volume", .01)
        thick = pipeline.postprocess_prediction(self.mask, (2.5, 1, 1), "min-volume", .01)
        self.assertEqual(int(small.sum()), 0)  # Eight 1 mm^3 voxels are 0.008 mL.
        self.assertEqual(int(thick.sum()), 8)  # Eight 2.5 mm^3 voxels are 0.020 mL.

    def test_largest_tie_is_deterministic_and_empty_is_preserved(self):
        mask = np.zeros((4, 4, 4), bool)
        mask[0, 0, 0] = mask[3, 3, 3] = True
        result = pipeline.postprocess_prediction(mask, (1, 1, 1), "largest")
        self.assertTrue(result[0, 0, 0])
        self.assertFalse(result[3, 3, 3])
        self.assertFalse(pipeline.postprocess_prediction(mask & False, (1, 1, 1), "largest").any())

    def test_legacy_none_recipe_and_invalid_volume(self):
        self.assertEqual(pipeline.postprocessing_recipe("none")["mode"], "none")
        with self.assertRaises(pipeline.UserError):
            pipeline.postprocess_prediction(self.mask, (1, 1, 1), "min-volume", 0)


class FrozenOperatingPointTests(unittest.TestCase):
    def setUp(self):
        temporary_root = Path(__file__).resolve().parents[1] / "outputs" / "verification"
        temporary_root.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="evaluation-test-", dir=temporary_root)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.cache = self.root / "cache"
        self.cache.mkdir()
        self.reference = np.zeros((6, 6, 6), np.uint8)
        self.reference[1:3, 1:3, 1:3] = 1
        self.probability = self.reference.astype(np.float32) * .8
        self.probability[5, 5, 5] = .8
        self.image = sitk.GetImageFromArray(self.probability)
        self.image.SetSpacing((1, 1, 2.5))
        self.image.SetOrigin((10, 20, 30))
        grid = pipeline.grid_dict(self.image)
        np.save(self.cache / "native_mask.npy", self.reference, allow_pickle=False)
        pipeline.atomic_json(self.cache / "metadata.json", {"fingerprint": "test", "native_grid": grid})
        pipeline.atomic_json(self.cache / "complete.json", {"fingerprint": "test", "files": {
            "native_mask.npy": (self.cache / "native_mask.npy").stat().st_size}})
        self.record = {"case_id": "case", "patient_id": "patient", "source_identity": "source",
                       "annotation_status": "positive", "cache_dir": "cache", "fingerprint": "test"}
        manifest = {"schema": pipeline.CACHE_SCHEMA, "settings": {"target_roi": "test"}, "records": [self.record]}
        manifest["fingerprint"] = pipeline.stable_hash({"settings": manifest["settings"], "records": manifest["records"]})
        pipeline.atomic_json(self.root / "manifest.json", manifest)
        self.evaluation = self.root / "evaluation"
        folder = self.evaluation / "cases" / "case"
        folder.mkdir(parents=True)
        sitk.WriteImage(self.image, str(folder / "probability.nii.gz"))
        self.protocol = {"partition": "val", "case_ids": ["case"], "checkpoint_sha256": "hash",
                         "target": "test", "manifest_fingerprint": manifest["fingerprint"],
                         "split_fingerprint": "split", "overlap": .5}
        pipeline.atomic_json(self.evaluation / "protocol.json", self.protocol)
        pipeline.atomic_json(self.evaluation / "patient_metrics.json", [{"case_id": "case", "patient_id": "patient", "status": "success"}])
        self.args = argparse.Namespace(evaluation=self.evaluation, manifest=self.root / "manifest.json",
            output=self.root / "point.json", splits=None, thresholds=[.5, .9],
            postprocess_modes=["none", "largest"], min_volumes_ml=[.01])

    def test_joint_selection_freezes_recipe_checkpoint_and_reconstruction(self):
        pipeline.select_threshold_command(self.args)
        result = pipeline.load_json(self.args.output)
        self.assertEqual(result["threshold"], .5)
        self.assertEqual(result["postprocessing"]["mode"], "largest")
        self.assertEqual(result["selected_validation_dice"], 1.)
        self.assertEqual(result["checkpoint_sha256"], "hash")
        self.assertEqual(result["overlap"], .5)
        self.assertEqual(len(result["candidates"]), 4)
        self.assertEqual(result["fingerprint"], pipeline.stable_hash({k: v for k, v in result.items() if k != "fingerprint"}))

    def test_export_uses_processed_mask_and_preserves_probabilities(self):
        pred = pipeline.postprocess_prediction(self.probability >= .5, (2.5, 1, 1), "largest")
        output = self.root / "export"
        pipeline.export_prediction(output, self.image, .5, pred)
        stored = sitk.ReadImage(str(output / "prediction.nii.gz"))
        np.testing.assert_array_equal(sitk.GetArrayFromImage(stored), pred.astype(np.uint8))
        self.assertTrue(pipeline.grids_equal(stored, self.image))
        stored_probability = sitk.GetArrayFromImage(sitk.ReadImage(str(output / "probability.nii.gz")))
        np.testing.assert_allclose(stored_probability, self.probability)

    def test_postprocessing_selection_never_accepts_test_partition(self):
        self.protocol["partition"] = "test"
        pipeline.atomic_json(self.evaluation / "protocol.json", self.protocol)
        with self.assertRaisesRegex(pipeline.UserError, "never test"):
            pipeline.select_threshold_command(self.args)


class PairedComparisonTests(unittest.TestCase):
    def compare(self, protocols, rows, points=None):
        roots = [Path("synthetic") / str(i) for i in range(len(protocols))]
        def load(path):
            i = int(path.parent.name)
            return points[i] if path.name == "operating_point.json" else protocols[i]
        with patch.object(Path, "exists", return_value=True), \
             patch.object(pipeline, "load_json", side_effect=load), \
             patch.object(pipeline, "recorded_metrics", side_effect=[(r, root / "patient_metrics.json", "explicit evaluation") for r, root in zip(rows, roots)]):
            return pipeline.compatible_comparisons(roots)

    def setUp(self):
        self.protocol = {"manifest_fingerprint": "m", "split_fingerprint": "s", "partition": "val",
            "target": "GTV-1", "threshold": .5, "postprocessing": "none", "surface_tolerance_mm": 2.,
            "surface_convention": "same", "overlap": .5,
            "threshold_source": "prespecified before training, not test-tuned"}
        self.rows = [{"case_id": "c", "patient_id": "p", "status": "success", "dice": .5}]

    def test_prespecified_comparison_requires_common_threshold(self):
        other = dict(self.protocol, threshold=.9)
        result, reason = self.compare([self.protocol, other], [self.rows, self.rows])
        self.assertFalse(result)
        self.assertIn("common threshold", reason)

    def test_duplicate_and_technical_failure_are_rejected(self):
        for rows in ([self.rows[0], self.rows[0]], [dict(self.rows[0], status="failed")]):
            result, reason = self.compare([self.protocol, self.protocol], [self.rows, rows])
            self.assertFalse(result)
            self.assertTrue(reason)

    def frozen_protocols(self):
        protocols, points = [], []
        for index, threshold in enumerate((.5, .9)):
            point = {"source_partition": "val", "threshold": threshold, "postprocessing": "none",
                     "checkpoint_sha256": f"checkpoint-{index}", "target": "GTV-1",
                     "manifest_fingerprint": "m", "split_fingerprint": "s"}
            point["fingerprint"] = pipeline.stable_hash(point)
            points.append(point)
            protocols.append(dict(self.protocol, threshold=threshold,
                checkpoint_sha256=point["checkpoint_sha256"],
                threshold_source="frozen validation selection: " + point["fingerprint"]))
        return protocols, points

    def test_individually_validation_frozen_thresholds_can_differ(self):
        protocols, points = self.frozen_protocols()
        result, reason = self.compare(protocols, [self.rows, self.rows], points)
        self.assertEqual(len(result), 2)
        self.assertEqual(reason, "")

    def test_foreign_checkpoint_operating_point_is_rejected(self):
        protocols, points = self.frozen_protocols()
        protocols[1]["checkpoint_sha256"] = "another-checkpoint"
        result, reason = self.compare(protocols, [self.rows, self.rows], points)
        self.assertFalse(result)
        self.assertIn("compatible frozen validation", reason)


if __name__ == "__main__":
    unittest.main()
