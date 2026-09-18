#!/usr/bin/env python3
"""Gate 5: remove the discrete feature-ID lookup from common-fate memory.

Gate 4 used a tiny table saying feature type 1 co-moved with types 2 and 3.
This gate gives the learner only continuous noisy RGB pairs plus oracle motion.

During three motion frames, each local edge becomes a training example:
  * same non-zero motion -> target affinity 1
  * different motion, or moving versus stationary -> target affinity 0
  * stationary versus stationary -> no evidence

The memory stores continuous RGB-pair examples.  At a later static frame, a
local RGB pair may override appearance affinity only when it falls inside a
radius calibrated *only from training-pair nearest-neighbour distances*.

The readout is deliberately local: threshold the learned affinities at 0.5 and
take connected components.  This exposes fragmentation directly rather than
letting a global clusterer merge disconnected pieces after the fact.

This is still a controlled gate.  Motion is oracle-provided and the image
generator has simple coloured parts.  The learner itself never receives the
part IDs or object labels.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import numpy as np

from grouping import adjusted_rand_index


PALETTE = np.array(
    [
        [0.55, 0.20, 0.20],  # background: deliberately close to feature 1
        [0.75, 0.20, 0.20],  # feature shared by both objects
        [0.20, 0.25, 0.80],  # object-1 second part
        [0.20, 0.75, 0.25],  # object-2 second part
    ],
    dtype=np.float64,
)


def make_scene(
    pos1: tuple[int, int],
    pos2: tuple[int, int],
    seed: int,
    n: int = 20,
    size: int = 8,
    noise: float = 0.02,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """RGB image, true object ownership, and generator-only part IDs."""
    rng = np.random.default_rng(seed)
    part = np.zeros((n, n), dtype=np.int64)
    labels = np.zeros((n, n), dtype=np.int64)

    y, x = pos1
    labels[y : y + size, x : x + size] = 1
    part[y : y + size // 2, x : x + size] = 1
    part[y + size // 2 : y + size, x : x + size] = 2

    y, x = pos2
    labels[y : y + size, x : x + size] = 2
    part[y : y + size // 2, x : x + size] = 1
    part[y + size // 2 : y + size, x : x + size] = 3

    image = PALETTE[part] + rng.normal(0.0, noise, (n, n, 3))
    return np.clip(image, 0.0, 1.0), labels, part


def coherent_flow(labels: np.ndarray) -> np.ndarray:
    flow = np.zeros((*labels.shape, 2), dtype=np.float64)
    flow[labels == 1] = (0.0, 1.0)
    flow[labels == 2] = (0.0, -1.0)
    return flow


def part_split_flow(labels: np.ndarray, part: np.ndarray) -> np.ndarray:
    """Attacker: each object's two appearance parts move oppositely."""
    flow = np.zeros((*labels.shape, 2), dtype=np.float64)
    foreground = labels > 0
    flow[foreground & (part == 1)] = (1.0, 0.0)
    flow[foreground & (part != 1)] = (-1.0, 0.0)
    return flow


def collect_motion_examples(
    image: np.ndarray,
    flow: np.ndarray,
    velocity_tolerance: float = 0.1,
) -> list[tuple[np.ndarray, np.ndarray, float]]:
    """Collect symmetric local RGB-pair examples without seeing object labels."""
    n = image.shape[0]
    examples = []
    for y in range(n):
        for x in range(n):
            for dy, dx in ((1, 0), (0, 1)):
                yy, xx = y + dy, x + dx
                if yy >= n or xx >= n:
                    continue
                va, vb = flow[y, x], flow[yy, xx]
                na, nb = np.linalg.norm(va), np.linalg.norm(vb)

                # Two stationary pixels tell us nothing about object common fate.
                if na < 1e-12 and nb < 1e-12:
                    continue

                same_motion = (
                    na > 1e-12
                    and nb > 1e-12
                    and np.linalg.norm(va - vb) < velocity_tolerance
                )
                examples.append(
                    (
                        image[y, x].copy(),
                        image[yy, xx].copy(),
                        1.0 if same_motion else 0.0,
                    )
                )
    return examples


def symmetric_pair_distance(
    a: tuple[np.ndarray, np.ndarray, float],
    b: tuple[np.ndarray, np.ndarray, float],
) -> float:
    a0, a1, _ = a
    b0, b1, _ = b
    direct = np.sum((a0 - b0) ** 2) + np.sum((a1 - b1) ** 2)
    swapped = np.sum((a0 - b1) ** 2) + np.sum((a1 - b0) ** 2)
    return float(min(direct, swapped))


def calibrate_radius(
    examples: list[tuple[np.ndarray, np.ndarray, float]],
    quantile: float = 0.99,
    multiplier: float = 3.0,
) -> float:
    """Calibrate pair-memory radius from training examples only."""
    nearest = []
    for i, example in enumerate(examples):
        same_target = [
            symmetric_pair_distance(example, other)
            for j, other in enumerate(examples)
            if j != i and other[2] == example[2]
        ]
        if same_target:
            nearest.append(min(same_target))
    if not nearest:
        raise ValueError("not enough examples to calibrate pair memory")
    return multiplier * float(np.quantile(nearest, quantile))


def prepare_memory(
    examples: list[tuple[np.ndarray, np.ndarray, float]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.stack([e[0] for e in examples]),
        np.stack([e[1] for e in examples]),
        np.asarray([e[2] for e in examples], dtype=np.float64),
    )


def build_affinity(
    image: np.ndarray,
    memory: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    radius: float | None = None,
    appearance_sigma: float = 0.18,
    epsilon: float = 1e-4,
) -> np.ndarray:
    """Appearance graph optionally overridden by nearby learned RGB-pair memory."""
    n = image.shape[0]
    w = np.zeros((n * n, n * n), dtype=np.float64)

    for y in range(n):
        for x in range(n):
            i = y * n + x
            for dy, dx in ((1, 0), (0, 1)):
                yy, xx = y + dy, x + dx
                if yy >= n or xx >= n:
                    continue
                j = yy * n + xx
                a, b = image[y, x], image[yy, xx]
                d2 = float(np.sum((a - b) ** 2))
                weight = epsilon + math.exp(
                    -d2 / (2.0 * appearance_sigma * appearance_sigma)
                )

                if memory is not None:
                    if radius is None:
                        raise ValueError("memory requires a calibrated radius")
                    p, q, target = memory
                    direct = np.sum((p - a) ** 2, axis=1) + np.sum(
                        (q - b) ** 2, axis=1
                    )
                    swapped = np.sum((p - b) ** 2, axis=1) + np.sum(
                        (q - a) ** 2, axis=1
                    )
                    distances = np.minimum(direct, swapped)
                    match = int(np.argmin(distances))
                    if float(distances[match]) <= radius:
                        weight = epsilon + (
                            1.0 - epsilon
                        ) * float(target[match])

                w[i, j] = w[j, i] = weight
    return w


def threshold_components(
    w: np.ndarray,
    threshold: float = 0.5,
) -> np.ndarray:
    """Connected components of graph edges whose learned affinity crosses threshold."""
    count = w.shape[0]
    parent = np.arange(count)

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = int(parent[a])
        return a

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    aa, bb = np.where(np.triu(w, 1) >= threshold)
    for a, b in zip(aa, bb):
        union(int(a), int(b))

    roots = np.asarray([find(i) for i in range(count)])
    _, labels = np.unique(roots, return_inverse=True)
    return labels


def mean_object_fragmentation(
    true_labels: np.ndarray,
    predicted: np.ndarray,
) -> float:
    truth = np.asarray(true_labels).ravel()
    pred = np.asarray(predicted).ravel()
    objects = [x for x in np.unique(truth) if x != 0]
    return float(
        np.mean(
            [len(np.unique(pred[truth == obj])) for obj in objects]
        )
    )


def evaluate(
    image: np.ndarray,
    truth: np.ndarray,
    memory,
    radius,
) -> dict:
    w = build_affinity(image, memory, radius)
    pred = threshold_components(w, threshold=0.5)
    return {
        "ari": adjusted_rand_index(truth, pred),
        "component_count": int(np.unique(pred).size),
        "mean_object_fragmentation": mean_object_fragmentation(truth, pred),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=12)
    parser.add_argument("--json", type=str, default=None)
    args = parser.parse_args()

    training_positions = (
        ((2, 1), (10, 11)),
        ((2, 2), (10, 10)),
        ((2, 3), (10, 9)),
    )

    coherent_examples = []
    split_examples = []
    for seed, (pos1, pos2) in enumerate(training_positions):
        image, labels, part = make_scene(pos1, pos2, seed=seed)
        coherent_examples.extend(
            collect_motion_examples(image, coherent_flow(labels))
        )
        split_examples.extend(
            collect_motion_examples(image, part_split_flow(labels, part))
        )

    coherent_radius = calibrate_radius(coherent_examples)
    split_radius = calibrate_radius(split_examples)
    coherent_memory = prepare_memory(coherent_examples)
    split_memory = prepare_memory(split_examples)

    final_positions = (
        ((10, 2), (2, 10)),
        ((9, 2), (1, 10)),
        ((10, 3), (2, 9)),
    )

    rows = []
    for sample in range(args.samples):
        pos1, pos2 = final_positions[sample % len(final_positions)]
        image, truth, _ = make_scene(
            pos1,
            pos2,
            seed=100 + sample,
        )
        rows.append(
            {
                "sample": sample,
                "pos1": list(pos1),
                "pos2": list(pos2),
                "static": evaluate(image, truth, None, None),
                "coherent_common_fate": evaluate(
                    image,
                    truth,
                    coherent_memory,
                    coherent_radius,
                ),
                "part_split_motion": evaluate(
                    image,
                    truth,
                    split_memory,
                    split_radius,
                ),
            }
        )

    def summary(name: str) -> dict:
        ari = [row[name]["ari"] for row in rows]
        count = [row[name]["component_count"] for row in rows]
        frag = [row[name]["mean_object_fragmentation"] for row in rows]
        return {
            "median_ari": float(np.median(ari)),
            "min_ari": float(np.min(ari)),
            "median_component_count": float(np.median(count)),
            "median_object_fragmentation": float(np.median(frag)),
            "max_object_fragmentation": float(np.max(frag)),
        }

    report = {
        "training_frames": len(training_positions),
        "coherent_examples": len(coherent_examples),
        "part_split_examples": len(split_examples),
        "coherent_memory_radius": coherent_radius,
        "part_split_memory_radius": split_radius,
        "radius_rule": "3 * 99th percentile leave-one-out nearest same-target pair distance",
        "threshold": 0.5,
        "summary": {
            "static": summary("static"),
            "coherent_common_fate": summary("coherent_common_fate"),
            "part_split_motion": summary("part_split_motion"),
        },
        "interpretation": (
            "Continuous RGB-pair memory transfers the common-fate relation to "
            "new noisy appearances and locations without discrete feature IDs. "
            "Differential motion alone can make ARI look high by separating "
            "foreground from background, but it leaves each true object split. "
            "Object fragmentation is therefore a necessary attacker metric: only "
            "coherent common fate binds the two appearance parts into one component."
        ),
        "rows": rows,
    }

    print(json.dumps(report, indent=2))

    assert coherent_radius > 0.0
    assert split_radius > 0.0
    for row in rows:
        for key in ("static", "coherent_common_fate", "part_split_motion"):
            assert -1.0 <= row[key]["ari"] <= 1.0
            assert row[key]["component_count"] >= 1

    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
