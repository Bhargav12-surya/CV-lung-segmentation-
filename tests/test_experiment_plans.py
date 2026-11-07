"""Controlled leave-one-out recipes must differ only in intended settings."""
import unittest

from scripts.plan_experiments import build_plan, powershell_quote


class ExperimentPlanTests(unittest.TestCase):
    def setUp(self):
        self.reference = {"preset": "compact", "train_args": {
            "max_epochs": 100, "steps_per_epoch": 1000, "batch_size": 1,
            "accumulation": 2, "lr": 5e-5, "foreground_probability": .25, "val_every": 10}}

    def plan(self, **kwargs):
        return build_plan(self.reference, "prepared/nsclc/manifest.json", "splits/main.json",
                          "runs/synthetic-plan-only", count_parameters=False, **kwargs)

    def test_all_five_models_and_three_seeds_are_prepared_without_launching(self):
        plan = self.plan()
        self.assertEqual(len(plan["experiments"]), 15)
        self.assertEqual({e["seed"] for e in plan["experiments"]}, {42, 43, 44})
        self.assertEqual(len({e["run"] for e in plan["experiments"]}), 15)
        self.assertTrue(all(e["status"] == "planned_not_run" for e in plan["experiments"]))

    def test_removals_change_exactly_the_intended_model_field(self):
        experiments = {e["variant"]: e for e in self.plan(seeds=[42])["experiments"]}
        full = experiments["full"]["model_config"]
        for name, intended in [("no_wavelet", "wavelet"), ("no_adaptive_fusion", "adaptive_fusion"),
                                ("no_acquisition", "acquisition_conditioning")]:
            config = experiments[name]["model_config"]
            differences = {k for k in full if full[k] != config[k]}
            self.assertEqual(differences, {intended})
            self.assertFalse(config[intended])
            self.assertTrue(config["boundary"])
            self.assertTrue(config["deep_supervision"])

    def test_baseline_retains_shared_supervision_heads(self):
        baseline = self.plan(variants=["baseline"], seeds=[42])["experiments"][0]["model_config"]
        self.assertEqual(baseline["name"], "baseline3d")
        for key in ("multiscale", "transformer", "wavelet", "adaptive_fusion",
                    "decoder_conditioning", "acquisition_conditioning"):
            self.assertFalse(baseline[key])
        self.assertTrue(baseline["boundary"])
        self.assertTrue(baseline["deep_supervision"])

    def test_fixed_training_budget_and_patient_split_are_shared(self):
        plan = self.plan()
        for experiment in plan["experiments"]:
            self.assertEqual(experiment["planned_optimizer_updates"], 50000)
            self.assertEqual(experiment["planned_microbatches"], 100000)
            self.assertEqual(experiment["patch"], [32, 96, 96])
            argv = experiment["argv"]
            self.assertEqual(argv[argv.index("--manifest") + 1], "prepared/nsclc/manifest.json")
            self.assertEqual(argv[argv.index("--splits") + 1], "splits/main.json")
            self.assertEqual(argv[argv.index("--early-stopping-checks") + 1], "0")

    def test_balanced_context_plan_uses_one_shared_larger_patch(self):
        for experiment in self.plan(preset="balanced")["experiments"]:
            self.assertEqual(experiment["patch"], [48, 128, 128])

    def test_mixed_sampling_is_explicit_and_legacy_sampling_is_preserved(self):
        mixed = self.plan()["shared_train_args"]
        self.assertEqual(mixed["foreground_probability"], .5)
        self.assertEqual(mixed["anatomical_background_probability"], .4)
        self.assertEqual(mixed["hard_negative_probability"], 0.)
        legacy = self.plan(sampling="legacy")["shared_train_args"]
        self.assertEqual(legacy["foreground_probability"], .25)
        self.assertEqual(legacy["anatomical_background_probability"], 0.)

    def test_nondivisible_controlled_patch_is_rejected(self):
        with self.assertRaises(ValueError):
            self.plan(overrides={"patch": [31, 96, 96]})

    def test_duplicate_seeds_and_variants_are_rejected(self):
        with self.assertRaises(ValueError):
            self.plan(seeds=[42, 42])
        with self.assertRaises(ValueError):
            self.plan(variants=["full", "full"])

    def test_literal_powershell_quoting_handles_apostrophes_and_dollars(self):
        self.assertEqual(powershell_quote("D:/User's/$data"), "'D:/User''s/$data'")


if __name__ == "__main__":
    unittest.main()
