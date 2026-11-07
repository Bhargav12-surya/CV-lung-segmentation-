"""Saved-validation rescoring retains native geometry and frozen provenance."""
import argparse
import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np
import SimpleITK as sitk

import lung_ct_segmentation as pipeline

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "rescore_saved_validation.py"
SPEC = importlib.util.spec_from_file_location("rescore_saved_validation", SCRIPT)
rescore = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(rescore)


class SavedValidationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="saved-validation-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "validation"
        self.cache = self.root / "cache"
        self.cache.mkdir()
        self.ref = np.zeros((6, 6, 6), np.uint8)
        self.ref[1:3, 1:3, 1:3] = 1
        probability = self.ref.astype(np.float32) * .8
        probability[5, 5, 5] = .8
        self.image = sitk.GetImageFromArray(probability)
        self.image.SetOrigin((10., 20., 30.))
        self.image.SetSpacing((1., 1., 2.5))
        self.metadata = {"fingerprint": "cache-id", "native_grid": pipeline.grid_dict(self.image),
                         "native_spacing_zyx_mm": [2.5, 1., 1.]}
        pipeline.atomic_json(self.cache / "metadata.json", self.metadata)
        np.save(self.cache / "native_mask.npy", self.ref, allow_pickle=False)
        pipeline.atomic_json(self.cache / "complete.json", {"fingerprint": "cache-id", "files": {
            "native_mask.npy": (self.cache / "native_mask.npy").stat().st_size}})
        record = {"case_id": "case", "patient_id": "patient", "source_identity": "source",
                  "annotation_status": "positive", "cache_dir": "cache", "fingerprint": "cache-id"}
        self.manifest = {"schema": pipeline.CACHE_SCHEMA, "settings": {"target_roi": "GTV-1"}, "records": [record]}
        self.manifest["fingerprint"] = pipeline.stable_hash({"settings": self.manifest["settings"], "records": [record]})
        pipeline.atomic_json(self.root / "manifest.json", self.manifest)
        self.split = {"manifest_fingerprint": self.manifest["fingerprint"], "partitions": {
            "train": [], "val": ["case"], "test": [], "external": []}}
        self.split["fingerprint"] = pipeline.stable_hash(self.split)
        pipeline.atomic_json(self.root / "splits.json", self.split)
        self.protocol = {"partition": "val", "case_ids": ["case"], "target": "GTV-1", "overlap": .5,
            "manifest_fingerprint": self.manifest["fingerprint"], "split_fingerprint": self.split["fingerprint"],
            "checkpoint_sha256": "a" * 64, "threshold": .5, "postprocessing": "none",
            "threshold_source": "prespecified before training, not test-tuned", "precision": "bf16",
            "surface_tolerance_mm": 2., "surface_convention": "6-neighbor boundary voxel centers; pooled unweighted distances"}
        pipeline.atomic_json(self.source / "protocol.json", self.protocol)
        self.source_rows = [{"case_id": "case", "patient_id": "patient", "status": "success", "dice": .5,
                             "inference_seconds": 123., "peak_cuda_gib": 1.}]
        pipeline.atomic_json(self.source / "patient_metrics.json", self.source_rows)
        probability_dir = self.source / "cases" / "case"
        probability_dir.mkdir(parents=True)
        self.probability_path = probability_dir / "probability.nii.gz"
        sitk.WriteImage(self.image, str(self.probability_path))
        self.point = {"source_partition": "val", "case_ids": ["case"], "threshold": .5, "target": "GTV-1",
            "postprocessing": pipeline.postprocessing_recipe("largest"), "overlap": .5,
            "checkpoint_sha256": "a" * 64, "manifest_fingerprint": self.manifest["fingerprint"],
            "split_fingerprint": self.split["fingerprint"]}
        self.save_point()
        self.args = argparse.Namespace(evaluation=self.source, operating_point=self.root / "point.json",
            manifest=self.root / "manifest.json", splits=self.root / "splits.json", output=self.root / "rescored",
            bootstrap_repeats=20, seed=1234)

    def save_point(self):
        self.point["fingerprint"] = pipeline.stable_hash({k: v for k, v in self.point.items() if k != "fingerprint"})
        pipeline.atomic_json(self.root / "point.json", self.point)

    def test_rescore_matches_metric_helper_and_records_no_inference(self):
        source_hash = pipeline.file_digest(self.probability_path)
        rows = rescore.rescore_saved_validation(self.args)
        self.assertEqual(len(rows), 1)
        expected = pipeline.evaluate_arrays(self.ref > 0, self.ref, (2.5, 1, 1), 2.)
        self.assertEqual(rows[0]["dice"], expected["dice"])
        self.assertEqual(rows[0]["hd95_mm"], expected["hd95_mm"])
        self.assertNotIn("inference_seconds", rows[0])
        self.assertNotIn("peak_cuda_gib", rows[0])
        self.assertGreaterEqual(rows[0]["metric_rescore_seconds"], 0)
        protocol = pipeline.load_json(self.args.output / "protocol.json")
        self.assertEqual(protocol["threshold_source"], "frozen validation selection: " + self.point["fingerprint"])
        self.assertFalse(protocol["model_executed"])
        self.assertEqual(protocol["checkpoint_sha256"], self.protocol["checkpoint_sha256"])
        self.assertEqual(pipeline.load_json(self.args.output / "operating_point.json"), self.point)
        self.assertEqual(pipeline.load_json(self.args.output / "aggregates.json")["metrics"]["dice"]["mean"], 1.)
        self.assertEqual(pipeline.load_json(self.args.output / "progress.json")["completed_cases"], 1)
        self.assertEqual(pipeline.file_digest(self.probability_path), source_hash)
        self.assertFalse((self.args.output / "cases").exists())

    def test_test_source_is_rejected_before_output_creation(self):
        self.protocol["partition"] = "test"
        pipeline.atomic_json(self.source / "protocol.json", self.protocol)
        with self.assertRaisesRegex(pipeline.UserError, "never test"):
            rescore.rescore_saved_validation(self.args)
        self.assertFalse(self.args.output.exists())

    def test_foreign_checkpoint_and_bad_fingerprint_are_rejected(self):
        self.point["checkpoint_sha256"] = "b" * 64
        self.save_point()
        with self.assertRaisesRegex(pipeline.UserError, "checkpoint_sha256"):
            rescore.rescore_saved_validation(self.args)
        self.point["fingerprint"] = "invalid"
        pipeline.atomic_json(self.root / "point.json", self.point)
        with self.assertRaisesRegex(pipeline.UserError, "fingerprint"):
            rescore.rescore_saved_validation(self.args)

    def test_frozen_reconstruction_overlap_must_match(self):
        self.point["overlap"] = .25
        self.save_point()
        with self.assertRaisesRegex(pipeline.UserError, "overlap"):
            rescore.rescore_saved_validation(self.args)

    def test_complete_case_and_patient_identity_is_required(self):
        self.source_rows[0]["patient_id"] = "another-patient"
        pipeline.atomic_json(self.source / "patient_metrics.json", self.source_rows)
        with self.assertRaisesRegex(pipeline.UserError, "identities"):
            rescore.rescore_saved_validation(self.args)
        self.source_rows.append(dict(self.source_rows[0]))
        pipeline.atomic_json(self.source / "patient_metrics.json", self.source_rows)
        with self.assertRaisesRegex(pipeline.UserError, "without duplicates"):
            rescore.rescore_saved_validation(self.args)

    def test_shifted_native_probability_is_counted_as_failure(self):
        self.image.SetOrigin((11., 20., 30.))
        sitk.WriteImage(self.image, str(self.probability_path))
        with self.assertRaisesRegex(pipeline.UserError, "technical failures"):
            rescore.rescore_saved_validation(self.args)
        rows = pipeline.load_json(self.args.output / "patient_metrics.json")
        self.assertEqual(rows[0]["status"], "failed")
        self.assertIn("geometry differs", rows[0]["reason"])
        self.assertEqual(pipeline.load_json(self.args.output / "aggregates.json")["technical_failures"], 1)

    def test_existing_output_is_never_overwritten(self):
        self.args.output.mkdir()
        with self.assertRaisesRegex(pipeline.UserError, "fresh output"):
            rescore.rescore_saved_validation(self.args)


if __name__ == "__main__":
    unittest.main()
