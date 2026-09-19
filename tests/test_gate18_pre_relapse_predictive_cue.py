import unittest

from gate18_pre_relapse_predictive_cue import (
    make_pairs,
    paired_dataset,
    run_reference,
)


class PreRelapsePredictiveCueTests(unittest.TestCase):
    def test_recovery_keeps_ordinary_confidence_high(self):
        pairs = make_pairs(12, seed=181)
        features, _ = paired_dataset(pairs)
        stable = features["mean_confidence"][:12]
        relapse = features["mean_confidence"][12:]

        self.assertGreater(stable.mean(), 0.95)
        self.assertGreater(relapse.mean(), 0.95)

    def test_new_cue_beats_old_temporal_state(self):
        report = run_reference(
            train_pairs=16,
            validation_pairs=12,
            test_pairs=24,
            train_seed=191,
            validation_seed=192,
            test_seed=193,
            shuffle_seed=194,
        )

        selected = report["selected_predictive_cue"]
        self.assertGreater(
            report["held_out_balanced_accuracy"][selected],
            0.70,
        )
        self.assertLess(
            max(
                report["held_out_balanced_accuracy"][name]
                for name in (
                    "raw_state",
                    "cumulative_state",
                    "symmetric_state",
                    "asymmetric_state",
                )
            ),
            0.70,
        )


if __name__ == "__main__":
    unittest.main()
