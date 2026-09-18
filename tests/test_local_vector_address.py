import unittest
import numpy as np

from gate10_local_vector_address import (
    capacity_sweep,
    projected_vector_relax,
)


class LocalVectorAddressTests(unittest.TestCase):
    def test_disconnected_pairs_synchronize_locally(self):
        rng = np.random.default_rng(7)
        initial = rng.normal(size=(4, 8))
        edges = [(0, 1), (2, 3)]
        relaxed = projected_vector_relax(initial, edges)

        self.assertGreater(float(np.dot(relaxed[0], relaxed[1])), 0.999)
        self.assertGreater(float(np.dot(relaxed[2], relaxed[3])), 0.999)

    def test_higher_dimension_reduces_address_collisions(self):
        sweep = capacity_sweep(20000, 0.99, seed=11)
        self.assertGreater(
            sweep["2"]["collision_fraction"],
            sweep["4"]["collision_fraction"],
        )
        self.assertGreater(
            sweep["4"]["collision_fraction"],
            sweep["8"]["collision_fraction"],
        )


if __name__ == "__main__":
    unittest.main()
