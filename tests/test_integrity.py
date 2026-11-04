"""Synthetic regressions; no patient data, trained weights, or GPU required."""
import argparse
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import SimpleITK as sitk

import lung_ct_segmentation as pipeline


class AcquisitionStatisticsTests(unittest.TestCase):
    def test_many_identical_columns_get_unit_std(self):
        records = [{"patient_id": f"synthetic-{i}"} for i in range(247)]
        metadata = {"native_spacing_zyx_mm": [3, .9765625, .9765625],
                    "target_spacing_zyx_mm": [2.5, 1, 1]}
        with patch.object(pipeline, "cache_path", return_value=(Path("unused"), metadata)):
            stats = pipeline.acquisition_statistics(records, Path("unused"))
        np.testing.assert_array_equal(stats["std"], np.ones(9))
        np.testing.assert_allclose(pipeline.normalized_descriptors(metadata, stats), 0, atol=1e-6)

    def test_statistics_average_series_within_patient_first(self):
        records = [{"patient_id": "a", "spacing": [3, .8, .8]},
                   {"patient_id": "a", "spacing": [3, .8, .8]},
                   {"patient_id": "b", "spacing": [3, 1.2, 1.2]}]
        def fake_cache(base, record):
            return Path("unused"), {"native_spacing_zyx_mm": record["spacing"],
                                    "target_spacing_zyx_mm": [2.5, 1, 1]}
        with patch.object(pipeline, "cache_path", side_effect=fake_cache):
            stats = pipeline.acquisition_statistics(records, Path("unused"))
        expected = np.mean([pipeline.raw_descriptor(s, [2.5, 1, 1], (2, 4, 4))
                            for s in ([3, .8, .8], [3, 1.2, 1.2])], axis=0)
        np.testing.assert_allclose(stats["mean2"], expected, atol=1e-7)
        self.assertEqual(stats["training_patients"], 2)

    def test_saved_legacy_statistics_are_not_mutated(self):
        metadata = {"native_spacing_zyx_mm": [3, 1, 1], "target_spacing_zyx_mm": [2.5, 1, 1]}
        raw = pipeline.raw_descriptor([3, 1, 1], [2.5, 1, 1], (2, 4, 4))
        stats = {"mean2": raw.tolist(), "mean3": raw.tolist(), "std": [1.5e-6] * 9}
        original = copy.deepcopy(stats)
        pipeline.normalized_descriptors(metadata, stats)
        self.assertEqual(stats, original)


class ThresholdSelectionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="msf-hsat-regression-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.evaluation = self.root / "evaluation"
        self.reference = np.zeros((3, 4, 5), np.uint8)
        self.reference[1, 2, 3] = 1
        self.image = sitk.GetImageFromArray(self.reference.astype(np.float32))
        self.image.SetSpacing((1., 1., 2.5))
        self.image.SetOrigin((10., 20., 30.))
        self.grid = pipeline.grid_dict(self.image)
        self.metadata = {"fingerprint": "synthetic", "native_grid": self.grid}
        self.record = {"case_id": "synthetic-case", "patient_id": "synthetic-patient",
                       "source_identity": "synthetic-source", "annotation_status": "positive",
                       "cache_dir": "cache", "fingerprint": "synthetic"}
        cache = self.root / "cache"
        cache.mkdir()
        np.save(cache / "native_mask.npy", self.reference, allow_pickle=False)
        pipeline.atomic_json(cache / "metadata.json", self.metadata)
        pipeline.atomic_json(cache / "complete.json", {"fingerprint": "synthetic", "files": {
            "native_mask.npy": (cache / "native_mask.npy").stat().st_size}})
        self.manifest = {"schema": pipeline.CACHE_SCHEMA, "settings": {"target_roi": "synthetic"},
                         "records": [self.record]}
        self.manifest["fingerprint"] = pipeline.stable_hash({
            "settings": self.manifest["settings"], "records": self.manifest["records"]})
        pipeline.atomic_json(self.root / "manifest.json", self.manifest)
        self.split = {"manifest_fingerprint": self.manifest["fingerprint"], "partitions": {
            "train": [], "val": ["synthetic-case"], "test": [], "external": []}}
        self.split["fingerprint"] = pipeline.stable_hash(self.split)
        pipeline.atomic_json(self.root / "splits.json", self.split)
        self.protocol = {"partition": "val", "case_ids": ["synthetic-case"],
                         "checkpoint_sha256": "synthetic", "target": "synthetic",
                         "manifest_fingerprint": self.manifest["fingerprint"],
                         "split_fingerprint": self.split["fingerprint"]}
        self.rows = [{"case_id": "synthetic-case", "patient_id": "synthetic-patient", "status": "success"}]
        self.probability_path = self.evaluation / "cases" / "synthetic-case" / "probability.nii.gz"
        self.probability_path.parent.mkdir(parents=True)
        sitk.WriteImage(self.image, str(self.probability_path))
        self.args = argparse.Namespace(evaluation=self.evaluation, manifest=self.root / "manifest.json",
                                       output=self.root / "point.json", thresholds=[.5, .9], splits=None)

    def select(self):
        pipeline.atomic_json(self.evaluation / "protocol.json", self.protocol)
        pipeline.atomic_json(self.evaluation / "patient_metrics.json", self.rows)
        pipeline.select_threshold_command(self.args)

    def test_complete_validation_freezes_correct_threshold_and_hash(self):
        self.select()
        point = pipeline.load_json(self.args.output)
        self.assertEqual(point["threshold"], .5)  # Equal Dice: closest to 0.5 wins.
        self.assertEqual(point["scores"]["0.5"], 1.)
        self.assertEqual(point["fingerprint"], pipeline.stable_hash({
            k: v for k, v in point.items() if k != "fingerprint"}))

    def test_incomplete_validation_cannot_select_threshold(self):
        self.protocol["case_ids"].append("missing-case")
        with self.assertRaisesRegex(pipeline.UserError, "every expected validation case"):
            self.select()
        self.assertFalse(self.args.output.exists())

    def test_duplicate_case_cannot_select_threshold(self):
        self.rows.append(dict(self.rows[0]))
        with self.assertRaisesRegex(pipeline.UserError, "every expected validation case"):
            self.select()

    def test_legacy_evaluation_requires_verified_split(self):
        del self.protocol["case_ids"]
        with self.assertRaisesRegex(pipeline.UserError, "Supply --splits"):
            self.select()
        self.args.splits = self.root / "splits.json"
        self.select()

    def test_test_set_threshold_tuning_is_rejected(self):
        self.protocol["partition"] = "test"
        with self.assertRaisesRegex(pipeline.UserError, "never test"):
            self.select()

    def test_shifted_probability_grid_is_rejected(self):
        self.image.SetOrigin((11., 20., 30.))
        sitk.WriteImage(self.image, str(self.probability_path))
        with self.assertRaisesRegex(pipeline.UserError, "geometry differs"):
            self.select()

    def test_nonfinite_probability_is_rejected(self):
        array = self.reference.astype(np.float32)
        array[0, 0, 0] = 1.5
        image = sitk.GetImageFromArray(array)
        image.CopyInformation(self.image)
        sitk.WriteImage(image, str(self.probability_path))
        with self.assertRaisesRegex(pipeline.UserError, "finite and in"):
            self.select()

    def test_empty_threshold_grid_is_rejected(self):
        self.args.thresholds = []
        with self.assertRaisesRegex(pipeline.UserError, "candidate thresholds"):
            self.select()


if __name__ == "__main__":
    unittest.main()
