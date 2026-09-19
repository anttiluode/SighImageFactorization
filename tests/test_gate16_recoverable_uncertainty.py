import unittest

from gate16_recoverable_uncertainty import (
    run_reference,
)


class RecoverableUncertaintyTests(unittest.TestCase):
    def test_asymmetric_state_attacks_then_recovers(self):
        report = run_reference(
            train_trials=30,
            test_trials=40,
            dim=8,
            gamma=0.20,
            pre_steps=240,
            threshold_train_trials=30,
            threshold_seed=160,
            train_seed=161,
            test_seed=162,
            address_seed=163,
        )

        selected = report["authority_policies"][
            "symmetric_ema"
        ]
        pre = selected["checkpoints"]["pre_attack"]
        attack = selected["checkpoints"]["under_ambiguity"]
        early = selected["checkpoints"]["one_clean_after_attack"]
        final = selected["checkpoints"]["full_recovery"]

        self.assertGreater(pre["genuine_fast_fraction"], 0.65)
        self.assertLess(attack["genuine_fast_fraction"], 0.35)
        self.assertLess(early["genuine_fast_fraction"], 0.50)
        self.assertGreater(final["genuine_fast_fraction"], 0.60)

    def test_recovery_beats_permanent_veto_without_naive_attack_failure(self):
        report = run_reference(
            train_trials=30,
            test_trials=40,
            dim=8,
            gamma=0.20,
            pre_steps=240,
            threshold_train_trials=30,
            threshold_seed=170,
            train_seed=171,
            test_seed=172,
            address_seed=173,
        )

        policies = report["authority_policies"]
        symmetric = policies["symmetric_ema"]
        asymmetric = policies["asymmetric_eligibility"]
        permanent = policies["permanent_veto"]
        cumulative = policies["cumulative_mean"]

        self.assertLess(
            symmetric["balanced_checkpoint_error"],
            permanent["balanced_checkpoint_error"],
        )
        self.assertLessEqual(
            symmetric["balanced_checkpoint_error"],
            cumulative["balanced_checkpoint_error"],
        )
        self.assertLessEqual(
            symmetric["balanced_checkpoint_error"],
            asymmetric["balanced_checkpoint_error"],
        )


if __name__ == "__main__":
    unittest.main()
