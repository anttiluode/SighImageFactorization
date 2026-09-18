import unittest
import numpy as np

from gate9_emergent_phase_address import (
    coupling_phase_receipt,
    kuramoto_relax,
)


class EmergentPhaseTests(unittest.TestCase):
    def test_two_attractive_pairs_separate_under_repulsion(self):
        coupling = np.full((4, 4), -0.2, dtype=np.float64)
        np.fill_diagonal(coupling, 0.0)
        coupling[0, 1] = coupling[1, 0] = 1.0
        coupling[2, 3] = coupling[3, 2] = 1.0

        theta0 = np.array([-2.1, -0.4, 0.7, 2.2], dtype=np.float64)
        theta = kuramoto_relax(coupling, theta0)
        receipt = coupling_phase_receipt(coupling, theta)

        self.assertGreater(
            receipt["minimum_attractive_edge_cosine"],
            0.99,
        )
        self.assertLess(
            receipt["maximum_repulsive_edge_cosine"],
            -0.99,
        )

    def test_exactly_collapsed_phase_is_symmetry_fixed_point(self):
        coupling = np.full((4, 4), -0.2, dtype=np.float64)
        np.fill_diagonal(coupling, 0.0)
        coupling[0, 1] = coupling[1, 0] = 1.0
        coupling[2, 3] = coupling[3, 2] = 1.0

        theta0 = np.zeros(4, dtype=np.float64)
        theta = kuramoto_relax(coupling, theta0)
        np.testing.assert_allclose(theta, theta0, atol=1e-15, rtol=0.0)


if __name__ == "__main__":
    unittest.main()
