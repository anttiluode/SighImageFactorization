"""Tiny NumPy PCA/FastICA helpers used by the falsification gates."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
import numpy as np


@dataclass(frozen=True)
class ICAResult:
    activations: np.ndarray
    component_maps: np.ndarray
    mean: np.ndarray
    converged: bool
    iterations: int


def _sym_decorrelation(w: np.ndarray) -> np.ndarray:
    vals, vecs = np.linalg.eigh(w @ w.T)
    vals = np.maximum(vals, 1e-12)
    return (vecs * (1.0 / np.sqrt(vals))) @ vecs.T @ w


def pca_fastica(
    x: np.ndarray,
    n_components: int,
    seed: int = 0,
    max_iter: int = 1000,
    tol: float = 1e-7,
) -> ICAResult:
    """PCA-whiten then symmetric FastICA (tanh nonlinearity), NumPy only."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("x must be samples x features")
    if not (1 <= n_components <= min(x.shape)):
        raise ValueError("invalid n_components")

    mean = x.mean(axis=0)
    xc = x - mean
    u, s, _ = np.linalg.svd(xc, full_matrices=False)
    if s[n_components - 1] <= 1e-12:
        raise ValueError("requested component count exceeds numerical data rank")

    z = u[:, :n_components] * np.sqrt(x.shape[0])
    rng = np.random.default_rng(seed)
    w = _sym_decorrelation(rng.standard_normal((n_components, n_components)))

    converged = False
    iterations = max_iter
    for it in range(1, max_iter + 1):
        wz = z @ w.T
        gwz = np.tanh(wz)
        gp = 1.0 - gwz * gwz
        w_new = (gwz.T @ z) / z.shape[0] - gp.mean(axis=0)[:, None] * w
        w_new = _sym_decorrelation(w_new)
        lim = np.max(np.abs(np.abs(np.diag(w_new @ w.T)) - 1.0))
        w = w_new
        if lim < tol:
            converged = True
            iterations = it
            break

    activations = z @ w.T
    component_maps, *_ = np.linalg.lstsq(activations, xc, rcond=None)
    return ICAResult(activations, component_maps, mean, converged, iterations)


def abs_corr_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aa = a - a.mean(axis=0, keepdims=True)
    bb = b - b.mean(axis=0, keepdims=True)
    aa /= np.linalg.norm(aa, axis=0, keepdims=True) + 1e-30
    bb /= np.linalg.norm(bb, axis=0, keepdims=True) + 1e-30
    return np.abs(aa.T @ bb)


def best_assignment_score(
    true_latents: np.ndarray, estimated: np.ndarray
) -> tuple[float, list[int], np.ndarray]:
    c = abs_corr_matrix(true_latents, estimated)
    p = c.shape[0]
    if estimated.shape[1] != p or p > 8:
        raise ValueError("assignment helper expects equal component counts <= 8")
    best_score = -1.0
    best_perm: tuple[int, ...] | None = None
    for perm in permutations(range(p)):
        score = float(np.mean([c[i, perm[i]] for i in range(p)]))
        if score > best_score:
            best_score = score
            best_perm = perm
    assert best_perm is not None
    return best_score, list(best_perm), c
