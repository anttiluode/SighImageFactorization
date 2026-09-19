import unittest

from gate19_predictive_cue_portability import run_reference


class PredictiveCuePortabilityTests(unittest.TestCase):
    def test_frozen_cue_tracks_contingency_strength(self):
        report = run_reference(
            gate18_train_pairs=16,
            calibration_pairs=12,
            test_pairs=20,
            gate18_train_seed=191,
            scenario_seed=192,
        )
        scenarios = report["scenarios"]

        self.assertGreater(
            scenarios["preserved_covariate_shift"][
                "frozen_balanced_accuracy"
            ],
            scenarios["broken_contingency"][
                "frozen_balanced_accuracy"
            ],
        )
        self.assertLess(
            scenarios["reversed_contingency"][
                "frozen_balanced_accuracy"
            ],
            0.45,
        )

    def test_recalibration_can_flip_but_not_create_information(self):
        report = run_reference(
            gate18_train_pairs=20,
            calibration_pairs=16,
            test_pairs=24,
            gate18_train_seed=201,
            scenario_seed=202,
        )
        scenarios = report["scenarios"]

        self.assertGreater(
            scenarios["reversed_contingency"][
                "recalibrated_balanced_accuracy"
            ],
            scenarios["reversed_contingency"][
                "frozen_balanced_accuracy"
            ],
        )
        self.assertLess(
            scenarios["broken_contingency"][
                "recalibrated_balanced_accuracy"
            ],
            0.75,
        )


if __name__ == "__main__":
    unittest.main()
