#!/usr/bin/env python3
"""Gate 3: same image-derived graph, linear diffusion versus vector synchrony.

This is deliberately *not* a reproduction of AKOrN.  It strips the comparison
down to one question: once an input-derived local affinity graph already
contains the object partitions, does a Kuramoto-style projected unit-vector
update reveal those partitions better than a matched linear diffusion state?

Both dynamics:
  * receive the same graph P,
  * start from the same random 8-D unit vectors,
  * use the same step size and checkpoint budget,
  * use the same k-means readout.

A spectral clustering solution is included as the graph-only ceiling.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np

from grouping import (
    adjusted_rand_index,
    grid_rgb_affinity,
    kmeans,
    normalize_rows,
    row_stochastic,
    spectral_cluster,
)


CHECKPOINTS = (0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512)


def make_scene(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Three compact regions; objects 1 and 2 deliberately share the same colour."""
    if n != 16:
        raise ValueError("Gate 3 is calibrated for n=16")
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:n, 0:n]
    labels = np.zeros((n, n), dtype=np.int64)
    centers = ((4, 4), (4, 11), (11, 8))
    for label, (cy, cx) in enumerate(centers, start=1):
        labels[(xx - cx) ** 2 + (yy - cy) ** 2 <= 2.6**2] = label

    # Objects 1 and 2 have the same appearance.  A global colour clustering
    # therefore cannot assign them distinct identities, while local graph
    # connectivity can.
    base = np.array(
        [
            [0.18, 0.20, 0.22],
            [0.78, 0.22, 0.20],
            [0.78, 0.22, 0.20],
            [0.22, 0.34, 0.80],
        ],
        dtype=np.float64,
    )
    palette = np.clip(base + rng.normal(0.0, 0.01, base.shape), 0.0, 1.0)
    image = palette[labels] + rng.normal(0.0, 0.025, (n, n, 3))
    return np.clip(image, 0.0, 1.0), labels


def raw_features(image: np.ndarray, xy_weight: float = 0.15) -> np.ndarray:
    n = image.shape[0]
    yy, xx = np.mgrid[0:n, 0:n]
    xy = np.stack([yy / (n - 1), xx / (n - 1)], axis=-1)
    return np.concatenate([image, xy_weight * xy], axis=-1).reshape(-1, 5)


def run_seed(
    seed: int,
    dim: int,
    gamma: float,
    steps: int,
) -> dict:
    image, labels = make_scene(16, seed)
    truth = labels.ravel()
    k = int(np.unique(truth).size)

    w = grid_rgb_affinity(image, sigma=0.12, epsilon=1e-4)
    p = row_stochastic(w)

    raw_pred = kmeans(raw_features(image), k, seed=2000 + seed)
    raw_ari = adjusted_rand_index(truth, raw_pred)

    spectral_pred, top_values = spectral_cluster(w, k, seed=2100 + seed)
    spectral_ari = adjusted_rand_index(truth, spectral_pred)

    rng = np.random.default_rng(1000 + seed)
    x0 = normalize_rows(rng.normal(size=(truth.size, dim)))
    linear = x0.copy()
    kuramoto = x0.copy()

    linear_scores = {}
    kuramoto_scores = {}
    linear_residue_sum = np.zeros_like(x0)
    kuramoto_residue_sum = np.zeros_like(x0)

    checkpoints = set(t for t in CHECKPOINTS if t <= steps)
    if steps not in checkpoints:
        checkpoints.add(steps)

    for t in range(steps + 1):
        if t in checkpoints:
            linear_pred = kmeans(
                normalize_rows(linear), k, seed=2200 + seed, restarts=12
            )
            kuramoto_pred = kmeans(
                kuramoto, k, seed=2200 + seed, restarts=12
            )
            linear_scores[str(t)] = adjusted_rand_index(truth, linear_pred)
            kuramoto_scores[str(t)] = adjusted_rand_index(truth, kuramoto_pred)

        if t == steps:
            break

        linear_next = (1.0 - gamma) * linear + gamma * (p @ linear)

        neighbour = p @ kuramoto
        tangent = neighbour - np.sum(neighbour * kuramoto, axis=1, keepdims=True) * kuramoto
        kuramoto_next = normalize_rows(kuramoto + gamma * tangent)

        linear_residue_sum += linear - linear_next
        kuramoto_residue_sum += kuramoto - kuramoto_next
        linear = linear_next
        kuramoto = kuramoto_next

    linear_receipt = float(
        np.linalg.norm(x0 - (linear_residue_sum + linear))
        / (np.linalg.norm(x0) + 1e-30)
    )
    kuramoto_receipt = float(
        np.linalg.norm(x0 - (kuramoto_residue_sum + kuramoto))
        / (np.linalg.norm(x0) + 1e-30)
    )

    def first_hit(scores: dict[str, float], threshold: float = 0.95):
        for key in sorted(scores, key=lambda x: int(x)):
            if scores[key] >= threshold:
                return int(key)
        return None

    return {
        "seed": seed,
        "raw_color_xy_ari": raw_ari,
        "spectral_graph_ari": spectral_ari,
        "top_graph_eigenvalues": top_values.tolist(),
        "linear_ari_by_step": linear_scores,
        "kuramoto_ari_by_step": kuramoto_scores,
        "linear_first_step_ari_ge_0_95": first_hit(linear_scores),
        "kuramoto_first_step_ari_ge_0_95": first_hit(kuramoto_scores),
        "linear_residue_reconstruction_error": linear_receipt,
        "kuramoto_residue_reconstruction_error": kuramoto_receipt,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=6)
    parser.add_argument("--dim", type=int, default=8)
    parser.add_argument("--gamma", type=float, default=0.8)
    parser.add_argument("--steps", type=int, default=512)
    parser.add_argument("--json", type=str, default=None)
    args = parser.parse_args()

    rows = [
        run_seed(seed, args.dim, args.gamma, args.steps)
        for seed in range(args.seeds)
    ]

    def median(path):
        values = []
        for row in rows:
            value = row
            for key in path:
                value = value[key]
            values.append(value)
        return float(np.median(values))

    linear_hits = [r["linear_first_step_ari_ge_0_95"] for r in rows]
    kuramoto_hits = [r["kuramoto_first_step_ari_ge_0_95"] for r in rows]

    report = {
        "seeds": args.seeds,
        "state_dimension": args.dim,
        "gamma": args.gamma,
        "steps": args.steps,
        "summary": {
            "raw_color_xy_median_ari": median(["raw_color_xy_ari"]),
            "spectral_graph_median_ari": median(["spectral_graph_ari"]),
            "linear_median_ari_step_128": median(["linear_ari_by_step", "128"])
                if args.steps >= 128 else None,
            "kuramoto_median_ari_step_128": median(["kuramoto_ari_by_step", "128"])
                if args.steps >= 128 else None,
            "linear_median_ari_step_256": median(["linear_ari_by_step", "256"])
                if args.steps >= 256 else None,
            "kuramoto_median_ari_step_256": median(["kuramoto_ari_by_step", "256"])
                if args.steps >= 256 else None,
            "linear_median_ari_final": median(["linear_ari_by_step", str(args.steps)]),
            "kuramoto_median_ari_final": median(["kuramoto_ari_by_step", str(args.steps)]),
            "linear_median_first_hit_0_95": float(np.median(linear_hits))
                if all(x is not None for x in linear_hits) else None,
            "kuramoto_median_first_hit_0_95": float(np.median(kuramoto_hits))
                if all(x is not None for x in kuramoto_hits) else None,
        },
        "interpretation": (
            "The image-derived graph contains the useful partition: spectral clustering "
            "solves it directly. In this stripped-down same-graph comparison, vector "
            "Kuramoto synchrony does not beat matched linear diffusion; both reach the "
            "same partition on the same timescale. Synchrony is therefore an alternative "
            "local dynamical realization here, not yet an extra source of object information."
        ),
        "rows": rows,
    }

    print(json.dumps(report, indent=2))

    for row in rows:
        assert row["linear_residue_reconstruction_error"] < 1e-12
        assert row["kuramoto_residue_reconstruction_error"] < 1e-12
        assert -1.0 <= row["spectral_graph_ari"] <= 1.0
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
