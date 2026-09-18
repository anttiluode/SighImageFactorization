import unittest
import numpy as np

from residue import build_sigh_filter, encode_residues, encoder_matrix


class ResidueTests(unittest.TestCase):
    def test_exact_reconstruction_batch(self):
        rng = np.random.default_rng(3)
        x = rng.standard_normal((5, 16, 16))
        stack = encode_residues(x, build_sigh_filter(16), depth=7)
        np.testing.assert_allclose(stack.reconstruct(), x, rtol=0.0, atol=2e-14)

    def test_matrix_matches_encoder(self):
        rng = np.random.default_rng(4)
        x = rng.standard_normal((3, 8, 8))
        filt = build_sigh_filter(8)
        t = encoder_matrix(8, 4, filt)
        direct = encode_residues(x, filt, 4).flatten()
        via_matrix = x.reshape(3, -1) @ t
        np.testing.assert_allclose(via_matrix, direct, rtol=1e-12, atol=1e-12)

    def test_original_sigh_nyquist_bin_is_unique_maximum(self):
        filt = build_sigh_filter(16)
        maxima = np.argwhere(np.isclose(filt, 1.0, atol=1e-15, rtol=0.0))
        self.assertEqual(maxima.tolist(), [[8, 8]])


if __name__ == "__main__":
    unittest.main()
