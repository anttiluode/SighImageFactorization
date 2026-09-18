"""Small NumPy utilities for controlled grouping experiments."""
from __future__ import annotations

import math
import numpy as np


def normalize_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-30)


def kmeans(
    x: np.ndarray,
    k: int,
    seed: int = 0,
    restarts: int = 12,
    max_iter: int = 80,
) -> np.ndarray:
    """Deterministic-seeded k-means++ style clustering, NumPy only."""
    x = np.asarray(x, dtype=np.float64)
    rng = np.random.default_rng(seed)
    best_inertia = float("inf")
    best_labels = None

    for _ in range(restarts):
        centers = [x[rng.integers(len(x))].copy()]
        for _ in range(1, k):
            dist = np.min(
                np.stack(
                    [np.sum((x - c) ** 2, axis=1) for c in centers],
                    axis=1,
                ),
                axis=1,
            )
            total = float(dist.sum())
            if total <= 1e-30:
                centers.append(x[rng.integers(len(x))].copy())
            else:
                centers.append(x[rng.choice(len(x), p=dist / total)].copy())

        c = np.stack(centers)
        labels = None
        for _ in range(max_iter):
            dist = np.sum((x[:, None, :] - c[None, :, :]) ** 2, axis=-1)
            new_labels = dist.argmin(axis=1)
            if labels is not None and np.array_equal(new_labels, labels):
                break
            labels = new_labels
            for j in range(k):
                sel = labels == j
                c[j] = x[sel].mean(axis=0) if np.any(sel) else x[rng.integers(len(x))]

        inertia = float(np.sum((x - c[labels]) ** 2))
        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels.copy()

    assert best_labels is not None
    return best_labels


def adjusted_rand_index(true_labels: np.ndarray, pred_labels: np.ndarray) -> float:
    """Adjusted Rand Index without scikit-learn."""
    a = np.asarray(true_labels).ravel()
    b = np.asarray(pred_labels).ravel()
    if len(a) != len(b):
        raise ValueError("label arrays must have equal length")

    _, ai = np.unique(a, return_inverse=True)
    _, bi = np.unique(b, return_inverse=True)
    table = np.zeros((int(ai.max()) + 1, int(bi.max()) + 1), dtype=np.int64)
    np.add.at(table, (ai, bi), 1)

    def pairs(v: np.ndarray) -> float:
        vv = np.asarray(v, dtype=np.float64)
        return float(np.sum(vv * (vv - 1.0) / 2.0))

    nij = pairs(table)
    rows = pairs(table.sum(axis=1))
    cols = pairs(table.sum(axis=0))
    n = len(a)
    total = n * (n - 1.0) / 2.0
    expected = rows * cols / (total + 1e-30)
    max_index = 0.5 * (rows + cols)
    return float((nij - expected) / (max_index - expected + 1e-30))


def grid_rgb_affinity(
    image: np.ndarray,
    sigma: float = 0.12,
    epsilon: float = 1e-4,
) -> np.ndarray:
    """4-neighbour symmetric affinity from local RGB similarity."""
    image = np.asarray(image, dtype=np.float64)
    n, m, channels = image.shape
    if n != m or channels != 3:
        raise ValueError("expected square RGB image")
    count = n * n
    w = np.zeros((count, count), dtype=np.float64)

    for y in range(n):
        for x in range(n):
            i = y * n + x
            for dy, dx in ((1, 0), (0, 1)):
                yy, xx = y + dy, x + dx
                if yy >= n or xx >= n:
                    continue
                j = yy * n + xx
                d2 = float(np.sum((image[y, x] - image[yy, xx]) ** 2))
                weight = epsilon + math.exp(-d2 / (2.0 * sigma * sigma))
                w[i, j] = w[j, i] = weight
    return w


def row_stochastic(w: np.ndarray) -> np.ndarray:
    d = np.asarray(w, dtype=np.float64).sum(axis=1, keepdims=True)
    return np.asarray(w, dtype=np.float64) / (d + 1e-30)


def spectral_embedding(w: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Top algebraic modes of the symmetric normalized affinity."""
    w = np.asarray(w, dtype=np.float64)
    d = w.sum(axis=1)
    s = w / np.sqrt((d[:, None] + 1e-30) * (d[None, :] + 1e-30))
    values, vectors = np.linalg.eigh(s)
    return normalize_rows(vectors[:, -k:]), values[-k:]


def spectral_cluster(w: np.ndarray, k: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    embedding, values = spectral_embedding(w, k)
    return kmeans(embedding, k, seed=seed, restarts=30), values
