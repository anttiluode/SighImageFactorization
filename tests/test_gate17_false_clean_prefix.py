import unittest

from gate17_false_clean_identical_prefix import (
    make_pairs,
    run_reference,
    state_at_split,
)


class FalseCleanIdenticalPrefixTests(unittest.TestCase):
    def test_paired_worlds_have_exactly_identical_causal_state(self):
        pairs = make_pairs(12, seed=171)
        for pair in pairs:
            split = pair["split_index"]
            self.assertTrue(
                (
                    pair["stable_target"][: split + 1]
                    == pair["relapse_target"][: split + 1]
                ).all()
            )
            for family in (
                "raw",
                "cumulative_mean",
                "symmetric_ema",
                "asymmetric_eligibility",
            ):
                stable = state_at_split(
                    pair["stable_target"][: split + 1],
                    family,
                )
                relapse = state_at_split(
                    pair["relapse_target"][: split + 1],
                    family,
                )
                self.assertEqual(stable, relapse)

    def test_new_observation_breaks_the_future_ambiguity(self):
        report = run_reference(
            train_pairs=16,
            test_pairs=24,
            train_seed=181,
            test_seed=182,
        )
        for result in report["split_point_family_results"].values():
            self.assertAlmostEqual(
                result["best_balanced_accuracy"],
                0.5,
                places=12,
            )
        self.assertGreater(
            report["first_post_split_observation"][
                "held_out_balanced_accuracy"
            ],
            0.80,
        )


if __name__ == "__main__":
    unittest.main()
