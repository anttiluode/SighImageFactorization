import unittest

from gate14_rgb_predictive_relation_admission import (
    generate_rgb_dataset,
    run_reference,
)
from gate13_predictive_relation_admission import learn_residual_threshold


class RGBPredictiveRelationAdmissionTests(unittest.TestCase):
    def test_rgb_history_contains_common_fate_signal(self):
        train = generate_rgb_dataset(40, seed=141)
        threshold, _ = learn_residual_threshold(
            train["rgb_residual"],
            train["labels"],
        )
        test = generate_rgb_dataset(60, seed=142)
        prediction = test["rgb_residual"] <= threshold
        accuracy = (prediction == (test["labels"] == 1)).mean()
        self.assertGreater(accuracy, 0.75)
        self.assertGreater(test["valid_fraction"].mean(), 0.85)

    def test_rgb_cue_routes_fast_clock_without_fast_clock_collapse(self):
        report = run_reference(
            train_trials=40,
            test_trials=60,
            dim=8,
            gamma=0.20,
            pre_steps=240,
            train_seed=151,
            test_seed=152,
            address_seed=153,
            shuffle_seed=154,
        )
        predictive = report["policies"]["rgb_predictive_fast_or_cautious"]
        always_fast = report["policies"]["always_fast_tau16"]

        self.assertGreater(predictive["genuine_merge_fraction"], 0.55)
        self.assertLess(
            predictive["accidental_false_merge_fraction"],
            0.25,
        )
        self.assertGreater(
            always_fast["accidental_false_merge_fraction"],
            0.60,
        )


if __name__ == "__main__":
    unittest.main()
