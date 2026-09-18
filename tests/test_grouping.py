import unittest
import numpy as np

from grouping import adjusted_rand_index, kmeans, normalize_rows
from gate4_common_fate_write import (
    coherent_flow,
    learn_feature_compatibility,
    make_scene,
    part_shuffled_flow,
)


class GroupingTests(unittest.TestCase):
    def test_adjusted_rand_identity_and_permutation(self):
        truth = np.array([0, 0, 1, 1, 2, 2])
        permuted = np.array([2, 2, 0, 0, 1, 1])
        self.assertAlmostEqual(adjusted_rand_index(truth, truth), 1.0)
        self.assertAlmostEqual(adjusted_rand_index(truth, permuted), 1.0)

    def test_kmeans_separates_obvious_clouds(self):
        x = np.array([
            [-2.0, -2.0],
            [-2.1, -1.9],
            [2.0, 2.0],
            [2.1, 1.9],
        ])
        labels = kmeans(x, 2, seed=3)
        self.assertAlmostEqual(
            adjusted_rand_index(np.array([0, 0, 1, 1]), labels),
            1.0,
        )

    def test_normalize_rows(self):
        x = np.array([[3.0, 4.0], [5.0, 12.0]])
        y = normalize_rows(x)
        np.testing.assert_allclose(np.linalg.norm(y, axis=1), 1.0)

    def test_common_fate_learns_relation_not_part_shuffle(self):
        types, labels = make_scene()
        good, _ = learn_feature_compatibility(types, coherent_flow(labels))
        bad, _ = learn_feature_compatibility(
            types, part_shuffled_flow(labels, types)
        )
        self.assertGreater(good[1, 2], 0.99)
        self.assertGreater(good[1, 3], 0.99)
        self.assertLess(bad[1, 2], 1e-6)
        self.assertLess(bad[1, 3], 1e-6)


if __name__ == "__main__":
    unittest.main()
