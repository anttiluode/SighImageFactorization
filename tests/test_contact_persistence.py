import unittest

from gate11_contact_persistence import run_sweep


class ContactPersistenceTests(unittest.TestCase):
    def test_persistence_gate_rejects_moderate_contact(self):
        report = run_sweep(
            trials=512,
            dim=8,
            tau=64.0,
            gamma=0.20,
            pre_steps=240,
            post_separation_steps=120,
            seed=123,
        )
        d32 = report["duration_sweep"]["32"]
        self.assertGreater(
            d32["instantaneous"]["collision_fraction"],
            0.95,
        )
        self.assertLess(
            d32["persistence_gated"]["collision_fraction"],
            0.02,
        )

    def test_long_contact_eventually_overrides_old_identity(self):
        report = run_sweep(
            trials=512,
            dim=8,
            tau=64.0,
            gamma=0.20,
            pre_steps=240,
            post_separation_steps=120,
            seed=321,
        )
        self.assertGreater(
            report["duration_sweep"]["64"]["persistence_gated"][
                "collision_fraction"
            ],
            0.90,
        )


if __name__ == "__main__":
    unittest.main()
