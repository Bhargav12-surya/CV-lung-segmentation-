"""CPU-only provenance/bounds checks for mining and preflight-only control flow."""
import argparse
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

import lung_ct_segmentation as pipeline


class MiningFixture(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="msf-hsat-mining-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.shape = (17, 8, 8)  # Exercises multiple bounded eight-slice slabs.
        self.records = []
        self.references = {}
        for i in range(4):
            case = f"synthetic-{i}"
            directory = self.root / case
            directory.mkdir()
            mask = np.zeros(self.shape, np.uint8)
            mask[9, 3, 4] = 1
            self.references[case] = mask
            arrays = {"hu": np.full(self.shape, -600., np.float32), "mask": mask}
            for name, array in arrays.items():
                np.save(directory / f"{name}.npy", array, allow_pickle=False)
            metadata = {"fingerprint": case, "case_id": case, "working_shape_zyx": list(self.shape),
                        "native_spacing_zyx_mm": [3., 1., 1.], "target_spacing_zyx_mm": [2.5, 1., 1.]}
            pipeline.atomic_json(directory / "metadata.json", metadata)
            pipeline.atomic_json(directory / "complete.json", {"fingerprint": case, "files": {
                f"{name}.npy": (directory / f"{name}.npy").stat().st_size for name in arrays}})
            self.records.append({"case_id": case, "patient_id": f"patient-{i}", "source_identity": f"source-{i}",
                                 "annotation_status": "positive", "cache_dir": case, "fingerprint": case})
        self.manifest = {"schema": pipeline.CACHE_SCHEMA,
                         "settings": {"spacing_zyx_mm": [2.5, 1., 1.], "target_roi": "synthetic"},
                         "records": self.records}
        self.manifest["fingerprint"] = pipeline.stable_hash({
            "settings": self.manifest["settings"], "records": self.records})
        self.split = {"manifest_fingerprint": self.manifest["fingerprint"], "partitions": {
            "train": ["synthetic-0", "synthetic-1"], "val": ["synthetic-2"],
            "test": ["synthetic-3"], "external": []}}
        self.split["fingerprint"] = pipeline.stable_hash(self.split)
        pipeline.atomic_json(self.root / "split.json", self.split)
        self.saved = {"manifest_fingerprint": self.manifest["fingerprint"],
                      "split_fingerprint": self.split["fingerprint"], "overfit_cases": 0,
                      "acquisition_stats": {}, "patch": [8, 16, 16], "metadata_control": "normal"}
        self.args = argparse.Namespace(manifest="unused", splits=self.root / "split.json", checkpoint="unused.pt",
                                       output=self.root / "negative-index.json", threshold=.9,
                                       centres_per_case=5, max_cases=0, device="cpu", precision="fp32", overlap=.5, seed=42)

    def invoke(self):
        predictions = []
        def predict(model, hu, metadata, stats, patch_shape, overlap, device, precision, control):
            predictions.append(metadata["case_id"])
            return np.ones(self.shape, np.float32), {}
        with patch.object(pipeline, "load_manifest", return_value=(self.manifest, self.root)), \
                patch.object(pipeline, "load_inference_model", return_value=(object(), self.saved)), \
                patch.object(pipeline, "file_digest", return_value="synthetic-checkpoint-hash"), \
                patch.object(pipeline, "predict_working_volume", side_effect=predict):
            pipeline.mine_negatives_command(self.args)
        return predictions, pipeline.load_json(self.args.output)


class MiningTests(MiningFixture):
    def test_only_training_cases_are_predicted_and_bounded_indices_round_trip(self):
        predictions, result = self.invoke()
        self.assertEqual(predictions, ["synthetic-0", "synthetic-1"])
        self.assertEqual(set(result["cases"]), set(self.split["partitions"]["train"]))
        self.assertEqual(result["source_partition"], "train")
        self.assertEqual(result["prediction_checkpoint_sha256"], "synthetic-checkpoint-hash")
        self.assertEqual(result["coordinate_grid"], "working_zyx")
        self.assertEqual(result["fingerprint"], pipeline.stable_hash({
            k: v for k, v in result.items() if k != "fingerprint"}))
        for case, record in result["cases"].items():
            centres = np.asarray(record["centres_zyx"])
            self.assertEqual(record["shape_zyx"], list(self.shape))
            self.assertEqual(record["false_positive_voxels"], np.prod(self.shape) - 1)
            self.assertLessEqual(len(centres), self.args.centres_per_case)
            self.assertTrue((centres >= 0).all())
            self.assertTrue((centres < self.shape).all())
            self.assertFalse(self.references[case][tuple(centres.T)].any())
        loaded, fingerprint = pipeline.load_hard_negative_index(self.args.output, self.manifest, self.split,
                                                                self.records[:2], self.root)
        self.assertEqual(set(loaded), {"synthetic-0", "synthetic-1"})
        self.assertEqual(fingerprint, result["fingerprint"])

    def test_max_cases_still_selects_only_training_and_never_validation(self):
        self.args.max_cases = 1
        predictions, result = self.invoke()
        self.assertEqual(predictions, ["synthetic-0"])
        self.assertEqual(set(result["cases"]), {"synthetic-0"})

    def test_diagnostic_and_mismatched_checkpoints_are_rejected_before_prediction(self):
        for field, value in (("overfit_cases", 1), ("manifest_fingerprint", "wrong"),
                             ("split_fingerprint", "wrong")):
            with self.subTest(field=field):
                original = copy.deepcopy(self.saved)
                self.saved[field] = value
                with self.assertRaisesRegex(pipeline.UserError, "non-diagnostic checkpoint"):
                    self.invoke()
                self.assertFalse(self.args.output.exists())
                self.saved = original

    def test_existing_output_is_preserved(self):
        pipeline.atomic_json(self.args.output, {"preserve": True})
        with self.assertRaisesRegex(pipeline.UserError, "already exists"):
            self.invoke()
        self.assertEqual(pipeline.load_json(self.args.output), {"preserve": True})

    def test_invalid_threshold_and_centre_budgets_are_rejected(self):
        for field, value in (("threshold", float("nan")), ("threshold", 1.),
                             ("centres_per_case", 0), ("max_cases", -1)):
            with self.subTest(field=field):
                original = getattr(self.args, field)
                setattr(self.args, field, value)
                with self.assertRaises(pipeline.UserError):
                    self.invoke()
                setattr(self.args, field, original)


class PreflightOnlyTests(MiningFixture):
    def test_preflight_only_returns_without_checkpoint_or_training(self):
        class TinyModel(torch.nn.Module):
            def __init__(self, config):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(1))
        class ProbeData:
            def __init__(self, *args, **kwargs):
                pass
            def __getitem__(self, index):
                return {"image": torch.zeros(2, 8, 16, 16), "mask": torch.ones(1, 8, 16, 16),
                        "boundary": torch.zeros(1, 8, 16, 16), "q": torch.zeros(2, 9)}
            def close(self):
                pass
        args = pipeline.build_parser().parse_args(["train", "--manifest", "unused",
            "--splits", str(self.root / "split.json"), "--run", str(self.root / "preflight"),
            "--device", "cpu", "--precision", "fp32", "--preset", "compact", "--patch", "8", "16", "16",
            "--preflight-only"])
        args._provided = ["--preflight-only"]
        stats = {"mean2": [0.] * 9, "mean3": [0.] * 9, "std": [1.] * 9}
        with patch.object(pipeline, "load_manifest", return_value=(self.manifest, self.root)), \
                patch.object(pipeline, "acquisition_statistics", return_value=stats), \
                patch.object(pipeline, "PatchDataset", ProbeData), \
                patch.object(pipeline, "SegmentationModel", TinyModel), \
                patch.object(pipeline, "preflight_model", return_value={"status": "passed"}) as preflight, \
                patch.object(pipeline, "atomic_checkpoint") as checkpoint, \
                patch.object(pipeline, "report_build") as report, \
                patch.object(pipeline, "environment", return_value={}), \
                patch.object(pipeline, "DataLoader") as loader:
            self.assertEqual(pipeline.train_command(args), 0)
            preflight.assert_called_once()
            checkpoint.assert_not_called()
            report.assert_not_called()
            loader.assert_not_called()
        result = pipeline.load_json(Path(args.run) / "preflight_only.json")
        self.assertEqual(result["status"], "passed")
        self.assertFalse((Path(args.run) / "checkpoints").exists())


if __name__ == "__main__":
    unittest.main()
