import unittest

from gate18_motion_surprise_predictive_cue import (
    generate_dataset,
    run_reference,
)


class MotionSurprisePredictiveCueTests(unittest.TestCase):
    def test_old_admission_bit_is_exactly_matched(self):
        data = generate_dataset(
            40,
            seed=181,
        )
        self.assertTrue(data["ordinary_bit"].all())
        self.assertLessEqual(
            data["max_common_fate_residual"].max(),
            0.0,
        )
        self.assertGreater(
            data["minimum_confidence"].min(),
            0.95,
        )

    def test_motion_surprise_escapes_matched_old_evidence(self):
        report = run_reference(
            train_trials=40,
            test_trials=80,
            dim=8,
            gamma=0.20,
            pre_steps=240,
            train_seed=191,
            test_seed=192,
            shuffle_seed=193,
            address_seed=194,
        )
        ordinary = report["held_out_cues"][
            "ordinary_matched_admission"
        ]
        predictive = report["held_out_cues"][
            "rgb_motion_surprise"
        ]

        self.assertAlmostEqual(
            ordinary["accuracy"],
            0.5,
            places=12,
        )
        self.assertGreater(
            predictive["accuracy"],
            0.75,
        )


if __name__ == "__main__":
    unittest.main()
