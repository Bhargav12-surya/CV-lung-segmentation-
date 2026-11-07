import unittest
from scripts.summarize_validation_study import paired_dice


class PairedValidationSummaryTests(unittest.TestCase):
    def test_bootstrap_unit_is_patient_not_series(self):
        old = [{"case_id": "a", "patient_id": "p", "status": "success", "dice": .1},
               {"case_id": "b", "patient_id": "p", "status": "success", "dice": .3},
               {"case_id": "c", "patient_id": "q", "status": "success", "dice": .6}]
        new = [dict(row, dice=value) for row, value in zip(old, (.3, .5, .4))]
        result = paired_dice(old, new, repeats=100)
        self.assertEqual(result["patients"], 2)
        self.assertAlmostEqual(result["mean_delta"], 0.)
        self.assertEqual((result["improved"], result["worsened"]), (1, 1))

    def test_different_cases_or_failures_are_rejected(self):
        row = {"case_id": "a", "patient_id": "p", "status": "success", "dice": .5}
        for other in (dict(row, case_id="b"), dict(row, status="failed")):
            with self.assertRaises(ValueError):
                paired_dice([row], [other], repeats=10)


if __name__ == "__main__":
    unittest.main()
