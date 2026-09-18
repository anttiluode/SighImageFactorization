import unittest

from gate11_contact_persistence import prepare_addresses
from gate12_learned_relation_timescale import (
    CANDIDATE_TAUS,
    learn_tau,
    paired_prefix_attack,
    score_tau,
)


class LearnedRelationTimescaleTests(unittest.TestCase):
    def test_training_selects_slow_timescale_and_generalizes(self):
        train = prepare_addresses(
            trials=1000,
            dim=8,
            seed=1,
            gamma=0.20,
            pre_steps=240,
        )
        tau, _ = learn_tau(train, CANDIDATE_TAUS, gamma=0.20)
        self.assertEqual(tau, 64.0)

        test = prepare_addresses(
            trials=1200,
            dim=8,
            seed=2,
            gamma=0.20,
            pre_steps=240,
        )
        held_out = score_tau(test, tau=tau, gamma=0.20)
        self.assertLess(held_out["false_merge"], 0.01)
        self.assertLess(held_out["false_nonmerge"], 0.01)

    def test_identical_prefix_cannot_reveal_future_relation(self):
        base = prepare_addresses(
            trials=800,
            dim=8,
            seed=3,
            gamma=0.20,
            pre_steps=240,
        )
        attack = paired_prefix_attack(base, tau=64.0, gamma=0.20)
        self.assertEqual(attack["paired_state_difference"], 0.0)
        self.assertEqual(
            attack["balanced_future_label_accuracy_from_age_only"],
            0.5,
        )
        self.assertAlmostEqual(
            attack["transient_false_merge_if_contact_ends_now"],
            attack["persistent_early_merge_success_if_contact_continues"],
            places=15,
        )


if __name__ == "__main__":
    unittest.main()
