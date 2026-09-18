#!/usr/bin/env python3
"""Gate 6: remove oracle motion with local RGB correspondence.

Gate 5 still received a perfect motion field. This gate receives only two
consecutive noisy RGB frames.

A deliberately small local estimator searches +/-2 pixels. Correspondence
cost is center-anchored: center RGB carries almost all of the cost and a 3x3
patch is only a tie-breaker. This avoids the motion-boundary failure where a
plain patch matcher can drag a strip of stationary background with an object.

We keep only forward/backward-consistent high-confidence matches. If two
neighbouring, visibly different pixels share the same non-zero estimated
displacement, their continuous RGB-pair feature is saved as a positive binding
memory.

At test time the objects are static, at new coordinates, with new texture and
sensor noise. A remembered pair can raise a weak appearance edge to a strong
edge. Thresholded connected components are the readout.

Controls:
  * static appearance only
  * oracle-flow upper bound
  * estimated flow
  * time-shuffled second frames

This remains a controlled synthetic gate. It is not optical flow for natural
video and does not claim semantic object discovery.
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
        [0.10, 0.10, 0.10],  # background
        [0.78, 0.18, 0.18],  # feature shared by both objects
        [0.18, 0.25, 0.82],  # object-1 second part
        [0.18, 0.78, 0.24],  # object-2 second part
    ],
    dtype=np.float64,
)

TRAINING_EPISODES = (
    ((3, 2), (9, 10), (0, 1), (0, -1), 11),
    ((2, 3), (10, 9), (1, 0), (-1, 0), 12),
    ((4, 2), (8, 10), (0, 1), (0, -1), 13),
)

TEST_POSITIONS = (
    ((10, 2), (2, 10)),
    ((9, 2), (1, 10)),
    ((10, 3), (2, 9)),
)


def render_scene(
    pos1: tuple[int, int],
    pos2: tuple[int, int],
    texture_seed: int,
    sensor_seed: int,
    n: int = 20,
    size: int = 8,
    texture_amplitude: float = 0.03,
    sensor_noise: float = 0.004,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Render two two-part objects with object-attached microtexture."""
    sensor_rng = np.random.default_rng(sensor_seed)
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

    image = PALETTE[part].copy()

    # Stationary texture is tied to world coordinates.
    bg_rng = np.random.default_rng(999)
    image += (
        bg_rng.normal(0.0, texture_amplitude, (n, n, 3))
        * (part == 0)[..., None]
    )

    # Object texture is tied to local object coordinates and therefore moves.
    for object_id, (y, x) in enumerate((pos1, pos2), start=1):
        object_rng = np.random.default_rng(texture_seed * 10 + object_id)
        texture = object_rng.normal(
            0.0, texture_amplitude, (size, size, 3)
        )
        image[y : y + size, x : x + size] += texture

    image += sensor_rng.normal(0.0, sensor_noise, image.shape)
    return np.clip(image, 0.0, 1.0), labels, part


def true_flow(
    labels: np.ndarray,
    shifts: tuple[tuple[int, int], tuple[int, int]],
) -> np.ndarray:
    flow = np.zeros((*labels.shape, 2), dtype=np.float64)
    for object_id, shift in enumerate(shifts, start=1):
        flow[labels == object_id] = shift
    return flow


def estimate_center_anchored_patch_flow(
    frame0: np.ndarray,
    frame1: np.ndarray,
    patch_radius: int = 1,
    search_radius: int = 2,
    patch_weight: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray]:
    """Search local motion; use a 3x3 patch only to break center-RGB ties."""
    n = frame0.shape[0]
    flow = np.zeros((n, n, 2), dtype=np.float64)
    confidence = np.zeros((n, n), dtype=np.float64)
    pr = patch_radius

    for y in range(pr, n - pr):
        for x in range(pr, n - pr):
            patch0 = frame0[
                y - pr : y + pr + 1,
                x - pr : x + pr + 1,
            ]
            center0 = frame0[y, x]
            candidates = []

            for dy in range(-search_radius, search_radius + 1):
                for dx in range(-search_radius, search_radius + 1):
                    yy, xx = y + dy, x + dx
                    if (
                        yy - pr < 0
                        or yy + pr >= n
                        or xx - pr < 0
                        or xx + pr >= n
                    ):
                        continue

                    patch1 = frame1[
                        yy - pr : yy + pr + 1,
                        xx - pr : xx + pr + 1,
                    ]
                    center_cost = float(
                        np.mean((center0 - frame1[yy, xx]) ** 2)
                    )
                    patch_cost = float(np.mean((patch0 - patch1) ** 2))
                    cost = center_cost + patch_weight * patch_cost
                    candidates.append((cost, dy, dx))

            candidates.sort()
            best, second = candidates[0], candidates[1]
            flow[y, x] = best[1:]
            confidence[y, x] = (
                second[0] - best[0]
            ) / (second[0] + 1e-12)

    return flow, confidence


def estimate_bidirectional_flow(
    frame0: np.ndarray,
    frame1: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    forward, confidence = estimate_center_anchored_patch_flow(
        frame0, frame1
    )
    backward, _ = estimate_center_anchored_patch_flow(frame1, frame0)

    n = frame0.shape[0]
    fb_error = np.full((n, n), np.inf, dtype=np.float64)
    for y in range(n):
        for x in range(n):
            dy, dx = np.rint(forward[y, x]).astype(int)
            yy, xx = y + dy, x + dx
            if 0 <= yy < n and 0 <= xx < n:
                fb_error[y, x] = np.linalg.norm(
                    forward[y, x] + backward[yy, xx]
                )
    return forward, confidence, fb_error


def pair_feature(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Order-invariant continuous RGB-pair feature."""
    return np.concatenate([(a + b) / 2.0, np.abs(a - b)])


def collect_binding_examples(
    image: np.ndarray,
    flow: np.ndarray,
    confidence: np.ndarray | None = None,
    fb_error: np.ndarray | None = None,
    confidence_threshold: float = 0.3,
    fb_threshold: float = 0.1,
    min_color_distance: float = 0.25,
) -> list[np.ndarray]:
    """Collect unlike-appearance boundaries with the same nonzero motion."""
    n = image.shape[0]
    examples: list[np.ndarray] = []

    for y in range(n):
        for x in range(n):
            for dy, dx in ((1, 0), (0, 1)):
                yy, xx = y + dy, x + dx
                if yy >= n or xx >= n:
                    continue

                if confidence is not None and (
                    confidence[y, x] < confidence_threshold
                    or confidence[yy, xx] < confidence_threshold
                ):
                    continue
                if fb_error is not None and (
                    fb_error[y, x] > fb_threshold
                    or fb_error[yy, xx] > fb_threshold
                ):
                    continue

                a, b = image[y, x], image[yy, xx]
                if np.linalg.norm(a - b) < min_color_distance:
                    continue

                va, vb = flow[y, x], flow[yy, xx]
                if np.linalg.norm(va) < 0.5 or np.linalg.norm(vb) < 0.5:
                    continue
                if np.linalg.norm(va - vb) > 0.1:
                    continue

                examples.append(pair_feature(a, b))

    return examples


def calibrate_memory_radius(
    examples: list[np.ndarray],
    quantile: float = 0.99,
) -> float:
    x = np.stack(examples)
    nearest = []
    for i in range(len(x)):
        distances = np.sum((x - x[i]) ** 2, axis=1)
        distances[i] = np.inf
        nearest.append(float(np.min(distances)))
    return float(np.quantile(nearest, quantile))


def build_affinity(
    image: np.ndarray,
    memory: np.ndarray | None = None,
    memory_radius: float | None = None,
    appearance_sigma: float = 0.18,
    epsilon: float = 1e-4,
    min_color_distance: float = 0.25,
) -> np.ndarray:
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
                color_distance = float(np.linalg.norm(a - b))
                weight = epsilon + math.exp(
                    -(color_distance**2)
                    / (2.0 * appearance_sigma * appearance_sigma)
                )

                if (
                    memory is not None
                    and memory_radius is not None
                    and color_distance >= min_color_distance
                ):
                    feature = pair_feature(a, b)
                    distances = np.sum((memory - feature) ** 2, axis=1)
                    if float(np.min(distances)) <= memory_radius:
                        weight = 1.0

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
    predicted: np.ndarray,
) -> float:
    truth = truth.ravel()
    predicted = predicted.ravel()
    object_ids = [x for x in np.unique(truth) if x != 0]
    return float(
        np.mean(
            [
                len(np.unique(predicted[truth == object_id]))
                for object_id in object_ids
            ]
        )
    )


def evaluate(
    image: np.ndarray,
    truth: np.ndarray,
    memory: np.ndarray | None,
    radius: float | None,
) -> dict:
    predicted = threshold_components(
        build_affinity(image, memory, radius)
    )
    return {
        "ari": adjusted_rand_index(truth, predicted),
        "component_count": int(np.unique(predicted).size),
        "object_fragmentation": object_fragmentation(truth, predicted),
    }


def summarize(rows: list[dict], key: str) -> dict:
    return {
        "median_ari": float(np.median([r[key]["ari"] for r in rows])),
        "min_ari": float(np.min([r[key]["ari"] for r in rows])),
        "median_component_count": float(
            np.median([r[key]["component_count"] for r in rows])
        ),
        "median_object_fragmentation": float(
            np.median([r[key]["object_fragmentation"] for r in rows])
        ),
        "max_object_fragmentation": float(
            np.max([r[key]["object_fragmentation"] for r in rows])
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-scenes", type=int, default=12)
    parser.add_argument("--json", type=str, default=None)
    args = parser.parse_args()

    estimated_examples: list[np.ndarray] = []
    oracle_examples: list[np.ndarray] = []
    training_frames = []
    flow_rows = []

    for episode, (pos1, pos2, shift1, shift2, texture_seed) in enumerate(
        TRAINING_EPISODES
    ):
        frame0, labels0, _ = render_scene(
            pos1,
            pos2,
            texture_seed=texture_seed,
            sensor_seed=100 + episode,
        )
        next_pos1 = (
            pos1[0] + shift1[0],
            pos1[1] + shift1[1],
        )
        next_pos2 = (
            pos2[0] + shift2[0],
            pos2[1] + shift2[1],
        )
        frame1, _, _ = render_scene(
            next_pos1,
            next_pos2,
            texture_seed=texture_seed,
            sensor_seed=200 + episode,
        )

        estimated_flow, confidence, fb_error = estimate_bidirectional_flow(
            frame0, frame1
        )
        oracle_flow = true_flow(labels0, (shift1, shift2))

        valid = (confidence > 0.3) & (fb_error < 0.1)
        foreground = (labels0 > 0) & valid
        background = (labels0 == 0) & valid

        flow_rows.append(
            {
                "episode": episode,
                "valid_foreground_pixels": int(foreground.sum()),
                "foreground_exact_displacement_fraction": float(
                    np.mean(
                        np.all(
                            estimated_flow[foreground]
                            == oracle_flow[foreground],
                            axis=1,
                        )
                    )
                ),
                "foreground_mean_endpoint_error": float(
                    np.mean(
                        np.linalg.norm(
                            estimated_flow[foreground]
                            - oracle_flow[foreground],
                            axis=1,
                        )
                    )
                ),
                "valid_background_pixels": int(background.sum()),
                "background_zero_displacement_fraction": float(
                    np.mean(
                        np.all(
                            estimated_flow[background] == 0.0,
                            axis=1,
                        )
                    )
                ),
            }
        )

        estimated_examples.extend(
            collect_binding_examples(
                frame0,
                estimated_flow,
                confidence,
                fb_error,
            )
        )
        oracle_examples.extend(
            collect_binding_examples(frame0, oracle_flow)
        )
        training_frames.append((frame0, frame1))

    estimated_radius = calibrate_memory_radius(estimated_examples)
    oracle_radius = calibrate_memory_radius(oracle_examples)
    estimated_memory = np.stack(estimated_examples)
    oracle_memory = np.stack(oracle_examples)

    # Pair each source frame with another episode's destination frame.
    shuffled_examples: list[np.ndarray] = []
    for i in range(len(training_frames)):
        frame0 = training_frames[i][0]
        frame1 = training_frames[(i + 1) % len(training_frames)][1]
        flow, confidence, fb_error = estimate_bidirectional_flow(
            frame0, frame1
        )
        shuffled_examples.extend(
            collect_binding_examples(
                frame0,
                flow,
                confidence,
                fb_error,
            )
        )

    rows = []
    for sample in range(args.test_scenes):
        pos1, pos2 = TEST_POSITIONS[sample % len(TEST_POSITIONS)]
        image, truth, _ = render_scene(
            pos1,
            pos2,
            texture_seed=100 + sample,
            sensor_seed=500 + sample,
            sensor_noise=0.006,
        )

        if shuffled_examples:
            shuffled_memory = np.stack(shuffled_examples)
            shuffled_radius = calibrate_memory_radius(shuffled_examples)
        else:
            shuffled_memory = None
            shuffled_radius = None

        rows.append(
            {
                "sample": sample,
                "pos1": list(pos1),
                "pos2": list(pos2),
                "static": evaluate(image, truth, None, None),
                "oracle_flow_memory": evaluate(
                    image,
                    truth,
                    oracle_memory,
                    oracle_radius,
                ),
                "estimated_flow_memory": evaluate(
                    image,
                    truth,
                    estimated_memory,
                    estimated_radius,
                ),
                "time_shuffled": evaluate(
                    image,
                    truth,
                    shuffled_memory,
                    shuffled_radius,
                ),
            }
        )

    report = {
        "training_episodes": len(TRAINING_EPISODES),
        "test_scenes": args.test_scenes,
        "estimated_binding_examples": len(estimated_examples),
        "oracle_binding_examples": len(oracle_examples),
        "time_shuffled_binding_examples": len(shuffled_examples),
        "estimated_memory_radius": estimated_radius,
        "oracle_memory_radius": oracle_radius,
        "flow_quality": flow_rows,
        "summary": {
            "static": summarize(rows, "static"),
            "oracle_flow_memory": summarize(rows, "oracle_flow_memory"),
            "estimated_flow_memory": summarize(
                rows, "estimated_flow_memory"
            ),
            "time_shuffled": summarize(rows, "time_shuffled"),
        },
        "interpretation": (
            "Local RGB correspondence is sufficient in this controlled world "
            "to recover common-fate evidence without oracle motion. Estimated "
            "flow reaches the same later static binding result as oracle flow, "
            "while time-shuffling destroys the binding evidence. High static "
            "ARI is again not enough: the static/shuffled conditions leave each "
            "true object split into two components, whereas both coherent-flow "
            "memories reduce fragmentation to one."
        ),
        "rows": rows,
    }

    print(json.dumps(report, indent=2))

    assert len(estimated_examples) > 0
    assert len(oracle_examples) > 0
    assert len(shuffled_examples) == 0
    assert report["summary"]["estimated_flow_memory"][
        "median_object_fragmentation"
    ] == 1.0
    assert report["summary"]["static"][
        "median_object_fragmentation"
    ] > 1.0

    if args.json:
        Path(args.json).write_text(
            json.dumps(report, indent=2) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
