import unittest
import numpy as np

from factorization import best_assignment_score, pca_fastica


class FactorizationTests(unittest.TestCase):
    def test_fastica_recovers_simple_independent_mixture(self):
        rng = np.random.default_rng(5)
        s = rng.laplace(size=(1200, 3))
        mixing = np.array([
            [1.0, 0.3, -0.2],
            [0.2, 1.2, 0.4],
            [-0.5, 0.1, 0.9],
            [0.7, -0.6, 0.2],
        ])
        x = s @ mixing.T
        fit = pca_fastica(x, 3, seed=6)
        score, _, _ = best_assignment_score(s, fit.activations)
        self.assertTrue(fit.converged)
        self.assertGreater(score, 0.97)


if __name__ == "__main__":
    unittest.main()
