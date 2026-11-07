"""Sampling regressions using tiny mathematical cache fixtures, never patient data."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import lung_ct_segmentation as pipeline


class SamplingFixture(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="msf-hsat-sampling-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.shape = (12, 24, 24)
        self.patch = (6, 8, 8)
        self.records, self.arrays, self.metadata = [], [], []
        for i in range(4):
            case = f"synthetic-{i}"
            directory = self.root / case
            directory.mkdir()
            z, y, x = np.indices(self.shape)
            hu = (-650 + 2 * z + y + .5 * x + 20 * i).astype(np.float32)
            mask = np.zeros(self.shape, np.uint8)
            mask[4:8, 10:14, 10:14] = 1
            boundary = mask.copy()  # Sampling correspondence, not a boundary-metric test.
            fg = np.flatnonzero(mask).astype(np.int64)
            arrays = {"hu": hu, "mask": mask, "boundary": boundary, "fg_indices": fg}
            for name, value in arrays.items():
                np.save(directory / f"{name}.npy", value, allow_pickle=False)
            metadata = {"fingerprint": case, "native_spacing_zyx_mm": [3., .8 + .1 * i, .8 + .1 * i],
                        "target_spacing_zyx_mm": [2.5, 1., 1.], "working_shape_zyx": list(self.shape)}
            pipeline.atomic_json(directory / "metadata.json", metadata)
            pipeline.atomic_json(directory / "complete.json", {"fingerprint": case,
                "files": {f"{name}.npy": (directory / f"{name}.npy").stat().st_size for name in arrays}})
            self.records.append({"case_id": case, "patient_id": f"synthetic-patient-{i}",
                                 "source_identity": f"synthetic-source-{i}", "annotation_status": "positive",
                                 "fingerprint": case, "cache_dir": case})
            self.arrays.append(arrays)
            self.metadata.append(metadata)
        self.stats = {"mean2": [0.] * 9, "mean3": [0.] * 9, "std": [1.] * 9}
        self.manifest = {"schema": pipeline.CACHE_SCHEMA, "settings": {}, "records": self.records}
        self.manifest["fingerprint"] = pipeline.stable_hash({"settings": {}, "records": self.records})
        self.split = {"manifest_fingerprint": self.manifest["fingerprint"], "partitions": {
            "train": [r["case_id"] for r in self.records[:2]],
            "val": [self.records[2]["case_id"]], "test": [self.records[3]["case_id"]], "external": []}}
        self.split["fingerprint"] = pipeline.stable_hash(self.split)
        pipeline.validate_split(self.manifest, self.split)

    def dataset(self, records=None, **kwargs):
        options = dict(foreground_probability=.5, augment=False, max_open_cases=1)
        options.update(kwargs)
        dataset = pipeline.PatchDataset(records or self.records[:2], self.root, self.stats,
                                        self.patch, 2, 64, 42, 3, **options)
        self.addCleanup(dataset.close)  # Runs before temporary-directory cleanup.
        return dataset

    def independent_legacy_sample(self, index, augment=False, metadata_control="normal"):
        """The pre-change selection stream, independent crop/window/q calculations."""
        rng = np.random.default_rng(np.random.SeedSequence([42, 3, index]))
        selected = int(rng.integers(2))
        arrays = self.arrays[selected]
        if len(arrays["fg_indices"]) and rng.random() < .5:
            center = np.asarray(np.unravel_index(int(rng.choice(arrays["fg_indices"])), self.shape))
        else:
            center = np.asarray([rng.integers(n) for n in self.shape])
        start = center - np.asarray(self.patch) // 2

        def crop(array, fill):
            result = np.full(self.patch, fill, dtype=array.dtype)
            for z in range(self.patch[0]):
                for y in range(self.patch[1]):
                    for x in range(self.patch[2]):
                        source = start + (z, y, x)
                        if np.all(source >= 0) and np.all(source < self.shape):
                            result[z, y, x] = array[tuple(source)]
            return result

        hu, mask, band = crop(arrays["hu"], -1024.), crop(arrays["mask"], 0), crop(arrays["boundary"], 0)
        # Reuse the unchanged augmentation function only after independently
        # consuming the historical sampling RNG; detects any extra sampler draw.
        hu, mask, band = pipeline.augment_patch(hu, mask, band, rng, augment)
        image = np.stack(((np.clip(hu, -1000, 400) + 1000) / 1400,
                          (np.clip(hu, -160, 240) + 160) / 400)).astype(np.float32)
        q_selected = selected
        if metadata_control == "shuffled":
            q_selected = int(np.random.default_rng(42 + 712).permutation(2)[selected])
        native = np.asarray(self.metadata[q_selected]["native_spacing_zyx_mm"], float)
        target = np.asarray([2.5, 1., 1.], float)
        q = np.stack([np.concatenate((np.log(native), np.log(target / native),
                                      np.log(target * np.asarray(stage)))).astype(np.float32)
                      for stage in ((2, 4, 4), (4, 8, 8))])
        if metadata_control == "constant":
            q.fill(0)
        return image, mask[None].astype(np.float32), band[None].astype(np.float32), q


class PatchSamplingTests(SamplingFixture):
    def test_disabled_new_probabilities_preserve_exact_legacy_sampling_and_rng(self):
        for augment, control in ((False, "normal"), (True, "normal"),
                                 (False, "constant"), (True, "shuffled")):
            with self.subTest(augmentation=augment, metadata_control=control):
                dataset = self.dataset(augment=augment, metadata_control=control,
                                       anatomical_background_probability=0., hard_negative_probability=0.)
                for index in range(12):
                    actual = dataset[index]
                    expected = self.independent_legacy_sample(index, augment, control)
                    for name, value in zip(("image", "mask", "boundary", "q"), expected):
                        np.testing.assert_array_equal(actual[name].numpy(), value, err_msg=f"{name}, sample {index}")
                dataset.close()

    def test_mixed_sampling_is_deterministic_and_negatives_are_tumour_free(self):
        left = self.dataset(foreground_probability=.4, anatomical_background_probability=.5, augment=True)
        right = self.dataset(foreground_probability=.4, anatomical_background_probability=.5, augment=True)
        background_count = 0
        for index in range(32):
            a, b = left[index], right[index]
            self.assertEqual(a["sampling_kind"], b["sampling_kind"])
            self.assertEqual(a["sampling_fallback"], b["sampling_fallback"])
            np.testing.assert_array_equal(a["image"].numpy(), b["image"].numpy())
            np.testing.assert_array_equal(a["mask"].numpy(), b["mask"].numpy())
            self.assertEqual(a["contains_foreground"], bool(a["mask"].any()))
            self.assertAlmostEqual(a["foreground_fraction"], float(a["mask"].numpy().mean()))
            if a["sampling_kind"] == "anatomical_background":
                background_count += 1
                self.assertFalse(a["contains_foreground"])
                self.assertFalse(a["sampling_fallback"])
        self.assertGreater(background_count, 0)

    def test_unavailable_anatomical_candidates_log_uniform_fallback(self):
        # Test fixture has no lung-like HU; no genuine data is modified.
        np.save(self.root / "synthetic-0" / "hu.npy", np.full(self.shape, -1024., np.float32), allow_pickle=False)
        dataset = self.dataset(records=self.records[:1], foreground_probability=0., anatomical_background_probability=1.)
        sample = dataset[0]
        self.assertEqual(sample["sampling_kind"], "uniform_fallback")
        self.assertTrue(sample["sampling_fallback"])

    def test_positive_patch_log_is_measured_after_augmentation(self):
        dataset = self.dataset(foreground_probability=1., augment=True)
        def erased_by_transform(hu, mask, band, rng, enabled):
            self.assertTrue(mask.any())
            return hu, np.zeros_like(mask), np.zeros_like(band)
        with patch.object(pipeline, "augment_patch", side_effect=erased_by_transform):
            sample = dataset[0]
        self.assertEqual(sample["sampling_kind"], "foreground")
        self.assertFalse(sample["contains_foreground"])
        self.assertEqual(sample["foreground_fraction"], 0.)

    def test_anatomical_candidates_include_near_lung_soft_tissue(self):
        hu = np.full(self.shape, -1024., np.float32)
        hu[:, 8, 8] = -650.  # Lung-like sparse-grid centres.
        hu[:, 8, 16] = 0.   # Nearby soft tissue must also be represented.
        np.save(self.root / "synthetic-0" / "hu.npy", hu, allow_pickle=False)
        dataset = self.dataset(records=self.records[:1], foreground_probability=0.,
                               anatomical_background_probability=1.)
        dataset[0]
        candidates = dataset._open[0]["anatomical_centres"]
        self.assertTrue(np.any((candidates[:, 1] == 8) & (candidates[:, 2] == 16)))

    def test_hard_negative_sampling_checks_the_whole_patch(self):
        dataset = self.dataset(records=self.records[:1], foreground_probability=0., hard_negative_probability=1.,
                               hard_negatives={"synthetic-0": np.asarray([[0, 0, 0], [6, 12, 12]])})
        for index in range(12):
            sample = dataset[index]
            if not sample["sampling_fallback"]:
                self.assertEqual(sample["sampling_kind"], "hard_negative")
                self.assertFalse(sample["contains_foreground"])


class HardNegativeIndexTests(SamplingFixture):
    def save_index(self, **changes):
        payload = {"source_partition": "train", "manifest_fingerprint": self.manifest["fingerprint"],
                   "split_fingerprint": self.split["fingerprint"], "coordinate_grid": "working_zyx",
                   "cases": {"synthetic-0": {"shape_zyx": list(self.shape), "centres_zyx": [[0, 0, 0]]}}}
        payload.update(changes)
        payload["fingerprint"] = pipeline.stable_hash(payload)
        path = self.root / "hard-negatives.json"
        pipeline.atomic_json(path, payload)
        return path, payload

    def load(self, path):
        return pipeline.load_hard_negative_index(path, self.manifest, self.split, self.records[:2], self.root)

    def test_training_only_index_is_loaded_with_fingerprint(self):
        path, payload = self.save_index()
        indices, fingerprint = self.load(path)
        np.testing.assert_array_equal(indices["synthetic-0"], [[0, 0, 0]])
        self.assertEqual(fingerprint, payload["fingerprint"])

    def test_validation_and_test_prediction_sources_are_rejected(self):
        for partition in ("val", "test"):
            with self.subTest(partition=partition):
                path, _ = self.save_index(source_partition=partition)
                with self.assertRaisesRegex(pipeline.UserError, "training-only"):
                    self.load(path)

    def test_nontraining_case_is_rejected_even_when_source_claims_train(self):
        for name in ("synthetic-2", "synthetic-3"):
            with self.subTest(case=name):
                path, _ = self.save_index(cases={name: {"shape_zyx": list(self.shape), "centres_zyx": [[0, 0, 0]]}})
                with self.assertRaisesRegex(pipeline.UserError, "non-training"):
                    self.load(path)

    def test_wrong_grid_shape_is_rejected(self):
        path, _ = self.save_index(cases={"synthetic-0": {"shape_zyx": [13, 24, 24], "centres_zyx": [[0, 0, 0]]}})
        with self.assertRaisesRegex(pipeline.UserError, "different working grid"):
            self.load(path)

    def test_invalid_coordinate_values_are_rejected(self):
        for centre in ([12, 0, 0], [-1, 0, 0], [.5, 0, 0], [0, 0]):
            with self.subTest(centre=centre):
                path, _ = self.save_index(cases={"synthetic-0": {"shape_zyx": list(self.shape), "centres_zyx": [centre]}})
                with self.assertRaisesRegex(pipeline.UserError, "Invalid hard-negative"):
                    self.load(path)

    def test_coordinate_mutation_without_rehash_is_rejected(self):
        path, payload = self.save_index()
        payload["cases"]["synthetic-0"]["centres_zyx"][0][0] = 1
        pipeline.atomic_json(path, payload)
        with self.assertRaisesRegex(pipeline.UserError, "fingerprinted"):
            self.load(path)


class SamplingConfigurationTests(SamplingFixture):
    def train_args(self, flags=()):
        args = pipeline.build_parser().parse_args(["train", "--manifest", "unused-manifest", "--splits", "unused-split",
                                                   "--run", str(self.root / "run"), *flags])
        args._provided = [token for token in flags if token.startswith("--")]
        return args

    def test_probability_sum_and_finite_values_are_validated(self):
        for flags in (("--foreground-probability", ".7", "--anatomical-background-probability", ".4"),
                      ("--anatomical-background-probability", "nan"),
                      ("--anatomical-background-probability", "-.1")):
            with self.subTest(flags=flags), self.assertRaises(pipeline.UserError):
                pipeline.validate_train_args(self.train_args(flags))

    def test_hard_negative_probability_requires_an_index(self):
        with self.assertRaisesRegex(pipeline.UserError, "requires a train-only"):
            pipeline.validate_train_args(self.train_args(("--hard-negative-probability", ".1")))

    def test_legacy_resume_cannot_adopt_new_sampling_flags(self):
        args = self.train_args(("--resume", "unused-checkpoint", "--anatomical-background-probability", ".2"))
        saved = {"manifest_fingerprint": self.manifest["fingerprint"], "split_fingerprint": self.split["fingerprint"],
                 "run_path": str(Path(args.run).resolve()), "train_args": {"lr": 5e-5}}
        with patch.object(pipeline, "load_manifest", return_value=(self.manifest, self.root)), \
                patch.object(pipeline, "load_json", return_value=self.split), \
                patch.object(pipeline, "safe_load_checkpoint", return_value=saved):
            with self.assertRaisesRegex(pipeline.UserError, "Legacy resume cannot change"):
                pipeline.train_command(args)

    def test_legacy_resume_restores_new_sampling_defaults_before_loading_model(self):
        class StopBeforeModel(Exception):
            pass
        args = self.train_args(("--resume", "unused-checkpoint"))
        args.anatomical_background_probability = .2  # Not an explicit user override.
        args.hard_negative_probability = .1
        args.hard_negative_index = "should-not-survive"
        saved = {"manifest_fingerprint": self.manifest["fingerprint"], "split_fingerprint": self.split["fingerprint"],
                 "run_path": str(Path(args.run).resolve()), "train_args": {"lr": 5e-5}}
        with patch.object(pipeline, "load_manifest", return_value=(self.manifest, self.root)), \
                patch.object(pipeline, "load_json", return_value=self.split), \
                patch.object(pipeline, "safe_load_checkpoint", return_value=saved), \
                patch.object(pipeline, "select_device", side_effect=StopBeforeModel):
            with self.assertRaises(StopBeforeModel):
                pipeline.train_command(args)
        self.assertEqual(args.anatomical_background_probability, 0.)
        self.assertEqual(args.hard_negative_probability, 0.)
        self.assertIsNone(args.hard_negative_index)


if __name__ == "__main__":
    unittest.main()
