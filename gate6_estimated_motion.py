#!/usr/bin/env python3
"""Gate 6: estimated motion writes a persistent grouping operator.

Each run gets exactly two consecutive noisy RGB frames.

No motion field, part ID, or object label is given to the learner.  Motion is
estimated by local patch correspondence with:
  * a 5x5 patch search,
  * strong center-pixel colour consistency,
  * forward/backward consistency,
  * confidence and photometric-cost rejection.

The estimated flow writes the same kind of persistent RGB-pair relation used in
Gate 5.  The object is then stopped at several *new* locations.

Matched controls:
  * static RGB appearance only;
  * oracle flow on the same first frame (upper bound for motion estimation);
  * time-shuffled frame pairing, which preserves plausible individual frames
    but destroys temporal correspondence.

This is still synthetic.  It removes the oracle flow from the learning path,
not the synthetic image generator.
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
        [0.55, 0.20, 0.20],  # background: deliberately close to shared red part
        [0.75, 0.20, 0.20],  # shared part in both objects
        [0.20, 0.25, 0.80],  # object-1 second part
        [0.20, 0.75, 0.25],  # object-2 second part
    ],
    dtype=np.float64,
)

N = 20
SIZE = 8
TRAIN_POS1 = (2, 1)
TRAIN_POS2 = (10, 11)
FLOW1 = (0, 1)
FLOW2 = (0, -1)

TEST_POSITIONS = (
    ((10, 1), (2, 11)),
    ((10, 2), (2, 10)),
    ((11, 3), (1, 9)),
    ((9, 1), (1, 11)),
    ((11, 2), (1, 10)),
)

# A frame from this unrelated episode is paired with the real first frame for
# the time-shuffled attacker.  Its objects are far outside the local search
# radius, so correct local correspondence has been destroyed.
SHUFFLE_POS1 = (10, 1)
SHUFFLE_POS2 = (2, 11)


def _part_templates(size: int = SIZE) -> tuple[np.ndarray, np.ndarray]:
    p1 = np.zeros((size, size), dtype=np.int64)
    p2 = np.zeros((size, size), dtype=np.int64)
    p1[: size // 2] = 1
    p1[size // 2 :] = 2
    p2[: size // 2] = 1
    p2[size // 2 :] = 3
    return p1, p2


def make_motion_pair(
    pos1: tuple[int, int],
    pos2: tuple[int, int],
    seed: int,
    d1: tuple[int, int] = FLOW1,
    d2: tuple[int, int] = FLOW2,
    texture: float = 0.02,
    sensor_noise: float = 0.003,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Two RGB frames with texture carried by the translating objects."""
    rng = np.random.default_rng(seed)
    p1_parts, p2_parts = _part_templates()

    background = np.clip(
        PALETTE[0] + rng.normal(0.0, texture, (N, N, 3)),
        0.0,
        1.0,
    )
    object1 = np.clip(
        PALETTE[p1_parts] + rng.normal(0.0, texture, (SIZE, SIZE, 3)),
        0.0,
        1.0,
    )
    object2 = np.clip(
        PALETTE[p2_parts] + rng.normal(0.0, texture, (SIZE, SIZE, 3)),
        0.0,
        1.0,
    )

    def render(
        q1: tuple[int, int],
        q2: tuple[int, int],
        noise_seed: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        image = background.copy()
        labels = np.zeros((N, N), dtype=np.int64)
        parts = np.zeros((N, N), dtype=np.int64)

        y, x = q1
        image[y : y + SIZE, x : x + SIZE] = object1
        labels[y : y + SIZE, x : x + SIZE] = 1
        parts[y : y + SIZE, x : x + SIZE] = p1_parts

        y, x = q2
        image[y : y + SIZE, x : x + SIZE] = object2
        labels[y : y + SIZE, x : x + SIZE] = 2
        parts[y : y + SIZE, x : x + SIZE] = p2_parts

        nrng = np.random.default_rng(noise_seed)
        image = np.clip(
            image + nrng.normal(0.0, sensor_noise, image.shape),
            0.0,
            1.0,
        )
        return image, labels, parts

    frame0, labels0, parts0 = render(pos1, pos2, seed + 1000)
    end1 = (pos1[0] + d1[0], pos1[1] + d1[1])
    end2 = (pos2[0] + d2[0], pos2[1] + d2[1])
    frame1, _, _ = render(end1, end2, seed + 2000)

    oracle = np.zeros((N, N, 2), dtype=np.float64)
    oracle[labels0 == 1] = d1
    oracle[labels0 == 2] = d2
    return frame0, frame1, labels0, parts0, oracle


def make_static_scene(
    pos1: tuple[int, int],
    pos2: tuple[int, int],
    seed: int,
    noise: float = 0.02,
) -> tuple[np.ndarray, np.ndarray]:
    """Fresh static scene used only after the motion evidence is gone."""
    rng = np.random.default_rng(seed)
    p1_parts, p2_parts = _part_templates()
    parts = np.zeros((N, N), dtype=np.int64)
    labels = np.zeros((N, N), dtype=np.int64)

    y, x = pos1
    labels[y : y + SIZE, x : x + SIZE] = 1
    parts[y : y + SIZE, x : x + SIZE] = p1_parts

    y, x = pos2
    labels[y : y + SIZE, x : x + SIZE] = 2
    parts[y : y + SIZE, x : x + SIZE] = p2_parts

    image = np.clip(
        PALETTE[parts] + rng.normal(0.0, noise, (N, N, 3)),
        0.0,
        1.0,
    )
    return image, labels


def _one_way_patch_match(
    frame0: np.ndarray,
    frame1: np.ndarray,
    search_radius: int = 2,
    patch_radius: int = 2,
    center_weight: float = 256.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Small brute-force flow estimator; deliberately inspectable, not fast."""
    n = frame0.shape[0]
    flow = np.zeros((n, n, 2), dtype=np.int64)
    confidence = np.zeros((n, n), dtype=np.float64)
    best_cost = np.full((n, n), np.inf, dtype=np.float64)

    for y in range(n):
        for x in range(n):
            y0, y1 = max(0, y - patch_radius), min(n, y + patch_radius + 1)
            x0, x1 = max(0, x - patch_radius), min(n, x + patch_radius + 1)
            source_patch = frame0[y0:y1, x0:x1]
            candidates = []

            for dy in range(-search_radius, search_radius + 1):
                for dx in range(-search_radius, search_radius + 1):
                    yy0, yy1 = y0 + dy, y1 + dy
                    xx0, xx1 = x0 + dx, x1 + dx
                    if yy0 < 0 or xx0 < 0 or yy1 > n or xx1 > n:
                        continue

                    target_patch = frame1[yy0:yy1, xx0:xx1]
                    patch_cost = float(
                        np.mean((source_patch - target_patch) ** 2)
                    )
                    center_cost = float(
                        np.mean(
                            (
                                frame0[y, x]
                                - frame1[y + dy, x + dx]
                            )
                            ** 2
                        )
                    )
                    cost = patch_cost + center_weight * center_cost
                    candidates.append((cost, dy, dx))

            candidates.sort()
            c0, dy, dx = candidates[0]
            c1 = candidates[1][0]
            flow[y, x] = (dy, dx)
            best_cost[y, x] = c0
            confidence[y, x] = (c1 - c0) / (c1 + 1e-30)

    return flow, confidence, best_cost


def estimate_flow(
    frame0: np.ndarray,
    frame1: np.ndarray,
    min_confidence: float = 0.03,
    max_cost: float = 0.02,
) -> tuple[np.ndarray, np.ndarray]:
    """Forward/backward-consistent local patch correspondence."""
    forward, confidence, cost = _one_way_patch_match(frame0, frame1)
    backward, _, _ = _one_way_patch_match(frame1, frame0)

    n = frame0.shape[0]
    fb_error = np.full((n, n), 99, dtype=np.int64)
    for y in range(n):
        for x in range(n):
            dy, dx = forward[y, x]
            yy, xx = y + int(dy), x + int(dx)
            if 0 <= yy < n and 0 <= xx < n:
                by, bx = backward[yy, xx]
                fb_error[y, x] = (
                    abs(int(dy) + int(by))
                    + abs(int(dx) + int(bx))
                )

    valid = (
        (confidence >= min_confidence)
        & (cost <= max_cost)
        & (fb_error == 0)
    )
    return forward.astype(np.float64), valid


def pair_feature(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Swap-invariant continuous feature for one local RGB pair."""
    return np.concatenate([(a + b) / 2.0, np.abs(a - b)])


def collect_motion_examples(
    image: np.ndarray,
    flow: np.ndarray,
    valid: np.ndarray | None = None,
    min_colour_distance2: float = 0.01,
) -> tuple[np.ndarray, np.ndarray]:
    """Write examples only at visible appearance boundaries.

    No object labels or part IDs are consulted.
    """
    if valid is None:
        valid = np.ones(image.shape[:2], dtype=bool)

    features = []
    targets = []
    n = image.shape[0]

    for y in range(n):
        for x in range(n):
            for dy, dx in ((1, 0), (0, 1)):
                yy, xx = y + dy, x + dx
                if yy >= n or xx >= n:
                    continue
                if not (valid[y, x] and valid[yy, xx]):
                    continue

                a, b = image[y, x], image[yy, xx]
                if float(np.sum((a - b) ** 2)) < min_colour_distance2:
                    continue

                va, vb = flow[y, x], flow[yy, xx]
                na, nb = np.linalg.norm(va), np.linalg.norm(vb)
                if na < 1e-12 and nb < 1e-12:
                    continue

                same_motion = (
                    na > 1e-12
                    and nb > 1e-12
                    and np.linalg.norm(va - vb) < 0.1
                )
                features.append(pair_feature(a, b))
                targets.append(1.0 if same_motion else 0.0)

    if not features:
        return (
            np.empty((0, 6), dtype=np.float64),
            np.empty((0,), dtype=np.float64),
        )
    return np.stack(features), np.asarray(targets, dtype=np.float64)


def support_radius(features: np.ndarray) -> float:
    """Training-only support radius for continuous pair memory."""
    if len(features) < 2:
        return 0.0
    dist = np.sum(
        (features[:, None, :] - features[None, :, :]) ** 2,
        axis=-1,
    )
    np.fill_diagonal(dist, np.inf)
    nearest = np.min(dist, axis=1)
    return 4.0 * float(np.quantile(nearest, 0.99))


def prepare_memory(
    features: np.ndarray,
    targets: np.ndarray,
) -> tuple[tuple[np.ndarray, np.ndarray] | None, float | None]:
    # Require both positive and negative evidence.  A correspondence failure is
    # not silently converted into a useful memory.
    if len(features) < 2 or len(np.unique(targets)) < 2:
        return None, None
    return (features, targets), support_radius(features)


def memory_vote(
    memory: tuple[np.ndarray, np.ndarray],
    query: np.ndarray,
    k: int = 9,
) -> tuple[float, float]:
    features, targets = memory
    dist = np.sum((features - query) ** 2, axis=1)
    kk = min(k, len(dist))
    idx = np.argpartition(dist, kk - 1)[:kk]
    weight = 1.0 / (dist[idx] + 1e-5)
    vote = float(np.sum(weight * targets[idx]) / np.sum(weight))
    return vote, float(np.min(dist[idx]))


def build_affinity(
    image: np.ndarray,
    memory: tuple[np.ndarray, np.ndarray] | None = None,
    radius: float | None = None,
    sigma: float = 0.18,
    epsilon: float = 1e-4,
    min_colour_distance2: float = 0.01,
) -> np.ndarray:
    """Static appearance graph, optionally rewritten by learned pair memory."""
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
                colour_d2 = float(np.sum((a - b) ** 2))
                weight = epsilon + math.exp(
                    -colour_d2 / (2.0 * sigma * sigma)
                )

                if (
                    memory is not None
                    and radius is not None
                    and colour_d2 >= min_colour_distance2
                ):
                    vote, distance = memory_vote(
                        memory,
                        pair_feature(a, b),
                    )
                    if distance <= radius:
                        weight = epsilon + (1.0 - epsilon) * vote

                w[i, j] = w[j, i] = weight
    return w


def threshold_components(
    w: np.ndarray,
    threshold: float = 0.5,
) -> np.ndarray:
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


def object_fragmentation(
    truth: np.ndarray,
    prediction: np.ndarray,
) -> float:
    t = truth.ravel()
    p = prediction.ravel()
    objects = [x for x in np.unique(t) if x != 0]
    return float(
        np.mean(
            [len(np.unique(p[t == obj])) for obj in objects]
        )
    )


def evaluate(
    image: np.ndarray,
    truth: np.ndarray,
    memory: tuple[np.ndarray, np.ndarray] | None,
    radius: float | None,
) -> dict:
    pred = threshold_components(build_affinity(image, memory, radius))
    score = adjusted_rand_index(truth, pred)
    return {
        "ari": score,
        "component_count": int(np.unique(pred).size),
        "object_fragmentation": object_fragmentation(truth, pred),
        "exact_partition": bool(score > 1.0 - 1e-12),
    }


def summarize_method(rows: list[dict], name: str) -> dict:
    # First summarize across test locations inside each independent two-frame
    # training run.  Then summarize those run-level medians across seeds.
    by_seed = {}
    for row in rows:
        by_seed.setdefault(row["seed"], []).append(row[name])

    run_summaries = []
    for seed in sorted(by_seed):
        values = by_seed[seed]
        run_summaries.append(
            {
                "seed": seed,
                "median_ari": float(
                    np.median([v["ari"] for v in values])
                ),
                "median_component_count": float(
                    np.median([v["component_count"] for v in values])
                ),
                "median_object_fragmentation": float(
                    np.median(
                        [v["object_fragmentation"] for v in values]
                    )
                ),
            }
        )

    return {
        "median_of_run_median_ari": float(
            np.median([x["median_ari"] for x in run_summaries])
        ),
        "min_run_median_ari": float(
            np.min([x["median_ari"] for x in run_summaries])
        ),
        "exact_run_fraction": float(
            np.mean(
                [x["median_ari"] > 1.0 - 1e-12 for x in run_summaries]
            )
        ),
        "median_component_count": float(
            np.median(
                [x["median_component_count"] for x in run_summaries]
            )
        ),
        "median_object_fragmentation": float(
            np.median(
                [x["median_object_fragmentation"] for x in run_summaries]
            )
        ),
        "max_run_median_object_fragmentation": float(
            np.max(
                [x["median_object_fragmentation"] for x in run_summaries]
            )
        ),
        "scene_exact_fraction": float(
            np.mean([row[name]["exact_partition"] for row in rows])
        ),
        "run_summaries": run_summaries,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--tests-per-seed", type=int, default=5)
    parser.add_argument("--json", type=str, default=None)
    args = parser.parse_args()

    rows = []
    flow_stats = []
    example_stats = []

    for seed in range(args.seeds):
        frame0, frame1, train_truth, _, oracle_flow = make_motion_pair(
            TRAIN_POS1,
            TRAIN_POS2,
            seed=10000 + seed,
        )
        estimated_flow, valid = estimate_flow(frame0, frame1)

        estimated_x, estimated_y = collect_motion_examples(
            frame0,
            estimated_flow,
            valid,
        )
        oracle_x, oracle_y = collect_motion_examples(
            frame0,
            oracle_flow,
        )
        estimated_memory, estimated_radius = prepare_memory(
            estimated_x,
            estimated_y,
        )
        oracle_memory, oracle_radius = prepare_memory(
            oracle_x,
            oracle_y,
        )

        # Pair frame0 with a frame from a different episode.  The individual
        # images remain plausible, but there is no local temporal identity.
        _, shuffled_future, _, _, _ = make_motion_pair(
            SHUFFLE_POS1,
            SHUFFLE_POS2,
            seed=50000 + seed,
        )
        shuffled_flow, shuffled_valid = estimate_flow(
            frame0,
            shuffled_future,
        )
        shuffled_x, shuffled_y = collect_motion_examples(
            frame0,
            shuffled_flow,
            shuffled_valid,
        )
        shuffled_memory, shuffled_radius = prepare_memory(
            shuffled_x,
            shuffled_y,
        )

        moving = train_truth > 0
        valid_moving = moving & valid
        flow_accuracy = (
            float(
                np.mean(
                    np.all(
                        estimated_flow[valid_moving]
                        == oracle_flow[valid_moving],
                        axis=1,
                    )
                )
            )
            if np.any(valid_moving)
            else float("nan")
        )
        flow_stats.append(
            {
                "seed": seed,
                "moving_pixel_coverage": float(np.mean(valid[moving])),
                "valid_moving_pixel_flow_accuracy": flow_accuracy,
                "overall_valid_fraction": float(np.mean(valid)),
                "shuffled_overall_valid_fraction": float(
                    np.mean(shuffled_valid)
                ),
            }
        )
        example_stats.append(
            {
                "seed": seed,
                "estimated_examples": int(len(estimated_y)),
                "estimated_positive_fraction": (
                    float(np.mean(estimated_y))
                    if len(estimated_y)
                    else None
                ),
                "oracle_examples": int(len(oracle_y)),
                "oracle_positive_fraction": float(np.mean(oracle_y)),
                "shuffled_examples": int(len(shuffled_y)),
                "shuffled_memory_available": shuffled_memory is not None,
            }
        )

        for test_index in range(args.tests_per_seed):
            pos1, pos2 = TEST_POSITIONS[
                test_index % len(TEST_POSITIONS)
            ]
            image, truth = make_static_scene(
                pos1,
                pos2,
                seed=20000 + seed * 10 + test_index,
            )
            rows.append(
                {
                    "seed": seed,
                    "test_index": test_index,
                    "pos1": list(pos1),
                    "pos2": list(pos2),
                    "static": evaluate(image, truth, None, None),
                    "estimated_motion": evaluate(
                        image,
                        truth,
                        estimated_memory,
                        estimated_radius,
                    ),
                    "oracle_motion": evaluate(
                        image,
                        truth,
                        oracle_memory,
                        oracle_radius,
                    ),
                    "time_shuffled": evaluate(
                        image,
                        truth,
                        shuffled_memory,
                        shuffled_radius,
                    ),
                }
            )

    methods = (
        "static",
        "estimated_motion",
        "oracle_motion",
        "time_shuffled",
    )
    summary = {
        name: summarize_method(rows, name)
        for name in methods
    }

    coverage = [
        x["moving_pixel_coverage"]
        for x in flow_stats
    ]
    accuracy = [
        x["valid_moving_pixel_flow_accuracy"]
        for x in flow_stats
        if np.isfinite(x["valid_moving_pixel_flow_accuracy"])
    ]
    report = {
        "training_frames_per_run": 2,
        "seeds": args.seeds,
        "tests_per_seed": args.tests_per_seed,
        "flow_estimator": {
            "search_radius": 2,
            "patch_size": [5, 5],
            "center_colour_weight": 256.0,
            "forward_backward_consistency": True,
            "median_moving_pixel_coverage": float(np.median(coverage)),
            "median_valid_moving_pixel_accuracy": float(
                np.median(accuracy)
            ),
        },
        "examples": {
            "median_estimated_examples": float(
                np.median(
                    [x["estimated_examples"] for x in example_stats]
                )
            ),
            "median_oracle_examples": float(
                np.median(
                    [x["oracle_examples"] for x in example_stats]
                )
            ),
            "total_time_shuffled_examples": int(
                sum(x["shuffled_examples"] for x in example_stats)
            ),
            "time_shuffled_memory_available_fraction": float(
                np.mean(
                    [
                        x["shuffled_memory_available"]
                        for x in example_stats
                    ]
                )
            ),
        },
        "summary": summary,
        "interpretation": (
            "One pair of consecutive RGB frames is enough for the local patch "
            "estimator to write a feature-pair grouping relation that usually "
            "survives relocation after motion stops. Oracle flow closes the "
            "remaining gap, localizing residual failures to correspondence / "
            "memory noise rather than the common-fate mechanism. Time-shuffling "
            "destroys correspondence and falls back to static fragmentation."
        ),
        "flow_rows": flow_stats,
        "example_rows": example_stats,
        "rows": rows,
    }

    print(json.dumps(report, indent=2))

    assert summary["static"]["median_object_fragmentation"] > 1.5
    assert (
        summary["estimated_motion"]["median_object_fragmentation"]
        <= 1.0 + 1e-12
    )
    assert (
        summary["oracle_motion"]["median_object_fragmentation"]
        <= 1.0 + 1e-12
    )
    assert summary["time_shuffled"]["median_object_fragmentation"] > 1.5
    assert report["flow_estimator"]["median_valid_moving_pixel_accuracy"] > 0.95

    if args.json:
        Path(args.json).write_text(
            json.dumps(report, indent=2) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
