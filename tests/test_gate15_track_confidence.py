import unittest

from gate15_track_confidence_admission import (
    generate_tracking_dataset,
    learn_confidence_threshold,
    run_reference,
)
from gate13_predictive_relation_admission import (
    learn_residual_threshold,
)


class TrackConfidenceAdmissionTests(unittest.TestCase):
    def test_confidence_recovers_relation_signal_lost_by_bad_tracks(self):
        train = generate_tracking_dataset(80, seed=151)
        residual_threshold, _ = learn_residual_threshold(
            train["rgb_residual"],
            train["labels"],
        )
        confidence_threshold, _ = learn_confidence_threshold(
            train["rgb_residual"],
            train["min_confidence"],
            train["labels"],
            residual_threshold,
        )

        test = generate_tracking_dataset(120, seed=152)
        residual_only = (
            test["rgb_residual"] <= residual_threshold
        )
        gated = (
            residual_only
            & (test["min_confidence"] >= confidence_threshold)
        )

        residual_accuracy = (
            residual_only == (test["labels"] == 1)
        ).mean()
        gated_accuracy = (
            gated == (test["labels"] == 1)
        ).mean()

        self.assertLess(residual_accuracy, 0.65)
        self.assertGreater(gated_accuracy, 0.82)

    def test_confidence_gate_preserves_fast_genuine_binding(self):
        report = run_reference(
            train_trials=80,
            test_trials=120,
            dim=8,
            gamma=0.20,
            pre_steps=240,
            train_seed=161,
            test_seed=162,
            address_seed=163,
            shuffle_seed=164,
        )

        gated = report["policies"][
            "confidence_gated_fast_or_cautious"
        ]
        residual = report["policies"][
            "residual_only_fast_or_cautious"
        ]
        cautious = report["policies"][
            "global_cautious_tau96"
        ]

        self.assertGreater(
            gated["genuine_merge_fraction"],
            0.60,
        )
        self.assertLess(
            gated["accidental_false_merge_fraction"],
            0.20,
        )
        self.assertGreater(
            residual["accidental_false_merge_fraction"],
            0.55,
        )
        self.assertLess(
            cautious["genuine_merge_fraction"],
            0.05,
        )


if __name__ == "__main__":
    unittest.main()
