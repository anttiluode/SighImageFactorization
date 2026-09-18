"""Exact residue bookkeeping for Sigh-style iterative spectral operators."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

HIGH_PASS_GAINS = np.array(
    [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0],
    dtype=np.float64,
)


def build_sigh_filter(n: int = 32, num_points: int = 256) -> np.ndarray:
    """Match SighImageSuper's radial high-pass lookup on an even n x n grid."""
    if n <= 0 or n % 2:
        raise ValueError("n must be a positive even integer")
    k = np.fft.fftfreq(n, d=1 / n)
    ky, kx = np.meshgrid(k, k, indexing="ij")
    k2 = kx * kx + ky * ky
    k2 = k2 / k2.max()
    centers = np.linspace(0.0, 1.0, len(HIGH_PASS_GAINS))
    lookup_x = np.linspace(0.0, 1.0, num_points)
    lookup = np.interp(lookup_x, centers, HIGH_PASS_GAINS)
    idx = np.floor(k2 * (num_points - 1)).astype(np.int64)
    return lookup[np.clip(idx, 0, num_points - 1)]


def spectral_step(x: np.ndarray, filt: np.ndarray) -> np.ndarray:
    """Apply one real-valued diagonal Fourier step to (..., n, n) arrays."""
    return np.fft.ifft2(np.fft.fft2(x, axes=(-2, -1)) * filt, axes=(-2, -1)).real


@dataclass(frozen=True)
class ResidueStack:
    residuals: np.ndarray
    terminal: np.ndarray

    def reconstruct(self) -> np.ndarray:
        return self.residuals.sum(axis=-3) + self.terminal

    def flatten(self) -> np.ndarray:
        """Concatenate residual images and terminal image along the feature axis."""
        shape = self.terminal.shape
        lead = shape[:-2]
        n = shape[-1]
        return np.concatenate(
            [self.residuals.reshape(*lead, -1), self.terminal.reshape(*lead, n * n)],
            axis=-1,
        )


def encode_residues(x0: np.ndarray, filt: np.ndarray, depth: int = 8) -> ResidueStack:
    """Save every departure r_k=x_k-x_{k+1}; reconstruction is telescopically exact."""
    if depth < 1:
        raise ValueError("depth must be >= 1")
    x = np.asarray(x0, dtype=np.float64).copy()
    residuals = []
    for _ in range(depth):
        nxt = spectral_step(x, filt)
        residuals.append(x - nxt)
        x = nxt
    return ResidueStack(np.stack(residuals, axis=-3), x)


def encoder_matrix(n: int, depth: int, filt: np.ndarray | None = None) -> np.ndarray:
    """Return T with row-vectors encoded as x @ T."""
    filt = build_sigh_filter(n) if filt is None else filt
    basis = np.eye(n * n, dtype=np.float64).reshape(n * n, n, n)
    return encode_residues(basis, filt, depth).flatten()
