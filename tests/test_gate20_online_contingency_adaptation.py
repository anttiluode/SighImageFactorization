import unittest

from gate20_online_contingency_adaptation import run_reference


class OnlineContingencyAdaptationTests(unittest.TestCase):
    def test_recent_policy_flips_after_reversal(self):
        report = run_reference(
            gate18_train_pairs=16,
            pairs_per_segment=24,
            gate18_train_seed=211,
            stream_seed=212,
            outcome_delay=6,
        )
        reversed_segment = report["segments"]["reversed_contingency"]

        frozen = reversed_segment["policies"]["frozen"]["tail_half"]
        recent = reversed_segment["policies"]["recent_window"]["tail_half"]

        self.assertGreater(
            recent["balanced_relation_accuracy"],
            frozen["balanced_relation_accuracy"],
        )
        self.assertGreater(
            recent["high_polarity_fraction"],
            frozen["high_polarity_fraction"],
        )

    def test_guard_can_withdraw_authority_when_cue_breaks(self):
        report = run_reference(
            gate18_train_pairs=20,
            pairs_per_segment=28,
            gate18_train_seed=221,
            stream_seed=222,
            outcome_delay=6,
        )
        broken = report["segments"]["broken_contingency"]["policies"]
        guarded = broken["reliability_gated_recent"]["tail_half"]

        self.assertLess(
            guarded["active_fraction"],
            0.75,
        )


if __name__ == "__main__":
    unittest.main()
