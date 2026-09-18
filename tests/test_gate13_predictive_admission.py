import unittest

import numpy as np

from gate11_contact_persistence import prepare_addresses
from gate13_predictive_relation_admission import (
    CAUTIOUS_TAU,
    FAST_TAU,
    balanced_error,
    evaluate_policy,
    generate_motion_histories,
    learn_residual_threshold,
)


class PredictiveRelationAdmissionTests(unittest.TestCase):
    def test_precontact_common_fate_is_learnable_with_noise(self):
        train_labels, train_residual = generate_motion_histories(
            1200,
            seed=11,
        )
        threshold, _ = learn_residual_threshold(
            train_residual,
            train_labels,
        )

        labels, residual = generate_motion_histories(
            1600,
            seed=12,
        )
        prediction = residual <= threshold
        error, fnr, fpr = balanced_error(labels, prediction)

        self.assertLess(error, 0.03)
        self.assertLess(fnr, 0.03)
        self.assertLess(fpr, 0.04)

    def test_predictive_clock_beats_age_ambiguity_at_equal_duration(self):
        labels, residual = generate_motion_histories(
            1800,
            seed=21,
        )
        train_labels, train_residual = generate_motion_histories(
            1400,
            seed=20,
        )
        threshold, _ = learn_residual_threshold(
            train_residual,
            train_labels,
        )
        prediction = residual <= threshold

        base = prepare_addresses(
            trials=len(labels),
            dim=8,
            seed=22,
            gamma=0.20,
            pre_steps=240,
        )

        predictive = evaluate_policy(
            base,
            labels,
            np.where(
                prediction,
                FAST_TAU,
                CAUTIOUS_TAU,
            ),
            gamma=0.20,
        )
        self.assertGreater(
            predictive["genuine_merge_fraction"],
            0.70,
        )
        self.assertLess(
            predictive["accidental_false_merge_fraction"],
            0.05,
        )
        self.assertGreater(
            predictive["balanced_relation_accuracy"],
            0.82,
        )


if __name__ == "__main__":
    unittest.main()
