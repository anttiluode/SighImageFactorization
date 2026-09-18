#!/usr/bin/env python3
"""Gate 4: common fate writes a grouping operator that survives motion.

This is a deliberately controlled mechanism test, not a full optical-flow or
natural-image object-discovery system.

Two synthetic objects each contain two appearance regions.  Static appearance
affinity wants to split those regions.  During a motion episode, the learner
sees only:
  * local feature-type pairs, and
  * their motion vectors.

If two unlike local features repeatedly share the same non-zero motion, their
persistent feature-pair compatibility is strengthened.  We then stop the
objects at *new coordinates* and ask whether the rewritten operator groups
each multi-part object.

Attackers:
  * static appearance graph,
  * part-shuffled motion (unlike parts do not share velocity),
  * coordinate-edge memory (memorizes where a boundary was, not what moved
    together).

The final readout is the same spectral clustering routine for every operator.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import numpy as np

from grouping import adjusted_rand_index, spectral_cluster


def make_scene(
    n: int = 20,
    pos1: tuple[int, int] = (2, 2),
    pos2: tuple[int, int] = (10, 10),
    size: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    """Return discrete appearance types and true object ownership.

    type 0 = background
    type 1 = appearance shared by both objects
    type 2 = second half of object 1
    type 3 = second half of object 2
    """
    types = np.zeros((n, n), dtype=np.int64)
    labels = np.zeros((n, n), dtype=np.int64)

    y, x = pos1
    labels[y : y + size, x : x + size] = 1
    types[y : y + size // 2, x : x + size] = 1
    types[y + size // 2 : y + size, x : x + size] = 2

    y, x = pos2
    labels[y : y + size, x : x + size] = 2
    types[y : y + size // 2, x : x + size] = 1
    types[y + size // 2 : y + size, x : x + size] = 3
    return types, labels


def coherent_flow(labels: np.ndarray) -> np.ndarray:
    flow = np.zeros((*labels.shape, 2), dtype=np.float64)
    flow[labels == 1] = (1.0, 1.0)
    flow[labels == 2] = (-1.0, -1.0)
    return flow


def part_shuffled_flow(labels: np.ndarray, types: np.ndarray) -> np.ndarray:
    """Control: parts of each object receive opposing motion vectors."""
    flow = np.zeros((*labels.shape, 2), dtype=np.float64)
    foreground = labels > 0
    flow[foreground & (types == 1)] = (1.0, 0.0)
    flow[foreground & (types != 1)] = (-1.0, 0.0)
    return flow


def learn_feature_compatibility(
    types: np.ndarray,
    flow: np.ndarray,
    sigma_velocity: float = 0.25,
) -> tuple[np.ndarray, np.ndarray]:
    """Learn feature-pair affinity from local common motion only."""
    count_types = int(types.max()) + 1
    accum = np.zeros((count_types, count_types), dtype=np.float64)
    count = np.zeros_like(accum)
    n = types.shape[0]

    for y in range(n):
        for x in range(n):
            for dy, dx in ((1, 0), (0, 1)):
                yy, xx = y + dy, x + dx
                if yy >= n or xx >= n:
                    continue
                a, b = int(types[y, x]), int(types[yy, xx])
                if a == 0 or b == 0 or a == b:
                    continue

                va, vb = flow[y, x], flow[yy, xx]
                if np.linalg.norm(va) < 1e-12 or np.linalg.norm(vb) < 1e-12:
                    continue
                dv2 = float(np.sum((va - vb) ** 2))
                similarity = math.exp(
                    -dv2 / (2.0 * sigma_velocity * sigma_velocity)
                )
                accum[a, b] += similarity
                accum[b, a] += similarity
                count[a, b] += 1.0
                count[b, a] += 1.0

    compatibility = np.divide(
        accum,
        count,
        out=np.zeros_like(accum),
        where=count > 0,
    )
    return compatibility, count


def learned_coordinate_edges(
    types: np.ndarray,
    flow: np.ndarray,
    sigma_velocity: float = 0.25,
    threshold: float = 0.9,
) -> list[tuple[int, int]]:
    """Attacker: remember absolute boundary coordinates instead of feature relation."""
    n = types.shape[0]
    edges = []
    for y in range(n):
        for x in range(n):
            for dy, dx in ((1, 0), (0, 1)):
                yy, xx = y + dy, x + dx
                if yy >= n or xx >= n:
                    continue
                a, b = int(types[y, x]), int(types[yy, xx])
                if a == 0 or b == 0 or a == b:
                    continue
                va, vb = flow[y, x], flow[yy, xx]
                if np.linalg.norm(va) < 1e-12 or np.linalg.norm(vb) < 1e-12:
                    continue
                dv2 = float(np.sum((va - vb) ** 2))
                similarity = math.exp(
                    -dv2 / (2.0 * sigma_velocity * sigma_velocity)
                )
                if similarity >= threshold:
                    edges.append((y * n + x, yy * n + xx))
    return edges


def build_affinity(
    types: np.ndarray,
    compatibility: np.ndarray | None = None,
    background_cross: float = 1e-3,
    appearance_cross: float = 1e-4,
) -> np.ndarray:
    """Local operator: same appearance strong, unlike appearance initially weak."""
    n = types.shape[0]
    size = n * n
    w = np.zeros((size, size), dtype=np.float64)
    if compatibility is None:
        compatibility = np.zeros((int(types.max()) + 1,) * 2, dtype=np.float64)

    for y in range(n):
        for x in range(n):
            i = y * n + x
            for dy, dx in ((1, 0), (0, 1)):
                yy, xx = y + dy, x + dx
                if yy >= n or xx >= n:
                    continue
                j = yy * n + xx
                a, b = int(types[y, x]), int(types[yy, xx])
                if a == b:
                    weight = 1.0
                elif a == 0 or b == 0:
                    weight = background_cross
                else:
                    weight = appearance_cross + (
                        1.0 - appearance_cross
                    ) * float(compatibility[a, b])
                w[i, j] = w[j, i] = weight
    return w


def segmentation_ari(
    w: np.ndarray,
    labels: np.ndarray,
    seed: int,
) -> float:
    k = int(np.unique(labels).size)
    pred, _ = spectral_cluster(w, k, seed=seed)
    return adjusted_rand_index(labels, pred)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=str, default=None)
    args = parser.parse_args()

    train_types, train_labels = make_scene(pos1=(2, 2), pos2=(10, 10))
    good_flow = coherent_flow(train_labels)
    bad_flow = part_shuffled_flow(train_labels, train_types)

    common_fate, common_counts = learn_feature_compatibility(
        train_types, good_flow
    )
    shuffled_fate, shuffled_counts = learn_feature_compatibility(
        train_types, bad_flow
    )
    coordinate_edges = learned_coordinate_edges(train_types, good_flow)

    final_positions = (
        ((2, 10), (10, 2)),
        ((1, 10), (11, 2)),
        ((2, 9), (10, 1)),
    )

    rows = []
    for i, (pos1, pos2) in enumerate(final_positions):
        types, labels = make_scene(pos1=pos1, pos2=pos2)

        static_w = build_affinity(types)
        common_w = build_affinity(types, common_fate)
        shuffled_w = build_affinity(types, shuffled_fate)

        # Coordinate-memory attacker writes the *old absolute edges* into the
        # new scene.  It therefore cannot follow the moved object.
        coordinate_w = static_w.copy()
        for a, b in coordinate_edges:
            coordinate_w[a, b] = coordinate_w[b, a] = 1.0

        seed = 100 + i
        rows.append(
            {
                "final_pos1": list(pos1),
                "final_pos2": list(pos2),
                "static_appearance_ari": segmentation_ari(
                    static_w, labels, seed
                ),
                "common_fate_feature_memory_ari": segmentation_ari(
                    common_w, labels, seed
                ),
                "part_shuffled_motion_ari": segmentation_ari(
                    shuffled_w, labels, seed
                ),
                "coordinate_memory_ari": segmentation_ari(
                    coordinate_w, labels, seed
                ),
            }
        )

    keys = (
        "static_appearance_ari",
        "common_fate_feature_memory_ari",
        "part_shuffled_motion_ari",
        "coordinate_memory_ari",
    )
    medians = {
        key: float(np.median([row[key] for row in rows]))
        for key in keys
    }

    report = {
        "training_positions": {
            "object_1": [2, 2],
            "object_2": [10, 10],
        },
        "learned_common_fate_compatibility": common_fate.tolist(),
        "learned_part_shuffled_compatibility": shuffled_fate.tolist(),
        "common_fate_pair_counts": common_counts.tolist(),
        "part_shuffled_pair_counts": shuffled_counts.tolist(),
        "coordinate_edges_memorized": len(coordinate_edges),
        "medians": medians,
        "rows": rows,
        "interpretation": (
            "A local appearance graph splits each multi-part object. Coherent "
            "motion writes a feature-pair relation (type 1<->2 and 1<->3) that "
            "generalizes to new coordinates after motion stops. Breaking common "
            "fate or memorizing old coordinates does not supply that relation. "
            "This is a controlled proof of persistent operator rewrite, not yet "
            "an optical-flow or natural-image object-discovery system."
        ),
    }

    print(json.dumps(report, indent=2))

    assert common_fate[1, 2] > 0.99
    assert common_fate[1, 3] > 0.99
    assert shuffled_fate[1, 2] < 1e-6
    assert shuffled_fate[1, 3] < 1e-6
    for key, value in medians.items():
        assert -1.0 <= value <= 1.0

    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
