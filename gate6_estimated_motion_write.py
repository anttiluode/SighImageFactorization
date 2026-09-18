#!/usr/bin/env python3
"""Gate 6: remove oracle motion using only consecutive RGB frames.

The remaining scaffold is explicit:

1. Split frame t into local *appearance-coherent connected regions*.
2. Estimate one small translation per region by RGB template matching into
   frame t+1.
3. If two adjacent appearance regions have the same confident non-zero
   translation, store their continuous RGB boundary pair as a future affinity.
4. Stop the objects at new positions and test whether those learned relations
   bind the later static image.

No object labels, part IDs, or true motion enter the estimated-motion learner.
Ground-truth ownership/motion are used only for evaluation and for an oracle
upper-bound control.

Controls:
  * static appearance only,
  * oracle-motion memory,
  * estimated-motion memory,
  * time-shuffled frame pairs.

The readout remains local connected components so object fragmentation cannot
hide behind a global clustering assignment.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from gate5_continuous_common_fate import (
    adjusted_rand_index,
    build_affinity,
    calibrate_radius,
    mean_object_fragmentation,
    prepare_memory,
    threshold_components,
)


PALETTE = np.array(
    [
        [0.35, 0.20, 0.20],  # background
        [0.75, 0.20, 0.20],  # top part shared by both objects
        [0.20, 0.25, 0.80],  # object-1 lower part
        [0.20, 0.75, 0.25],  # object-2 lower part
    ],
    dtype=np.float64,
)


def render_scene(
    pos1: tuple[int, int],
    pos2: tuple[int, int],
    seed: int,
    n: int = 24,
    size: int = 8,
    sensor_noise: float = 0.004,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Render two rigid, textured, two-part objects.

    The returned part map exists only for diagnostics/oracle construction; the
    estimated learner never consumes it.
    """
    rng = np.random.default_rng(seed)
    part = np.zeros((n, n), dtype=np.int64)
    labels = np.zeros((n, n), dtype=np.int64)

    yy, xx = np.mgrid[0:n, 0:n]
    background_texture = (
        0.025 * np.sin(0.9 * xx + 0.7 * yy)
        + 0.015 * np.cos(1.3 * xx - 0.4 * yy)
    )
    image = np.empty((n, n, 3), dtype=np.float64)
    image[:] = PALETTE[0]
    image += background_texture[..., None] * np.array([0.7, -0.3, 0.2])

    for obj, (y, x) in enumerate((pos1, pos2), start=1):
        labels[y : y + size, x : x + size] = obj

        ly, lx = np.mgrid[0:size, 0:size]
        texture = (
            0.035 * np.sin(1.4 * lx + 0.8 * ly + obj * 0.9)
            + 0.020 * np.cos(0.7 * lx - 1.2 * ly + obj)
        )

        part[y : y + size // 2, x : x + size] = 1
        part[y + size // 2 : y + size, x : x + size] = obj + 1

        block = np.empty((size, size, 3), dtype=np.float64)
        block[: size // 2] = PALETTE[1]
        block[size // 2 :] = PALETTE[obj + 1]
        block += texture[..., None] * np.array([0.5, 0.3, -0.4])
        image[y : y + size, x : x + size] = block

    image += rng.normal(0.0, sensor_noise, image.shape)
    return np.clip(image, 0.0, 1.0), labels, part


def appearance_components(image: np.ndarray) -> np.ndarray:
    """Connected regions under the same static RGB affinity used at test time."""
    w = build_affinity(image)
    flat = threshold_components(w, threshold=0.5)
    return flat.reshape(image.shape[:2])


def estimate_component_translations(
    frame0: np.ndarray,
    frame1: np.ndarray,
    components: np.ndarray,
    search_radius: int = 2,
    displacement_penalty: float = 1e-5,
) -> tuple[dict[int, tuple[int, int]], dict[int, float]]:
    """Estimate one translation per appearance component from RGB only."""
    n = frame0.shape[0]
    motions: dict[int, tuple[int, int]] = {}
    confidences: dict[int, float] = {}

    for comp in np.unique(components):
        ys, xs = np.where(components == comp)
        best = (float("inf"), 0, 0)
        second = float("inf")

        for dy in range(-search_radius, search_radius + 1):
            for dx in range(-search_radius, search_radius + 1):
                yy = ys + dy
                xx = xs + dx
                valid = (yy >= 0) & (yy < n) & (xx >= 0) & (xx < n)
                if float(np.mean(valid)) < 0.90:
                    continue

                diff = frame0[ys[valid], xs[valid]] - frame1[yy[valid], xx[valid]]
                score = float(
                    np.mean(diff * diff)
                    + displacement_penalty * (dy * dy + dx * dx)
                )
                if score < best[0]:
                    second = best[0]
                    best = (score, dy, dx)
                elif score < second:
                    second = score

        motions[int(comp)] = (int(best[1]), int(best[2]))
        confidences[int(comp)] = float(
            (second - best[0]) / (second + 1e-30)
        )

    return motions, confidences


def true_flow(
    labels: np.ndarray,
    motion1: tuple[int, int],
    motion2: tuple[int, int],
) -> np.ndarray:
    flow = np.zeros((*labels.shape, 2), dtype=np.float64)
    flow[labels == 1] = motion1
    flow[labels == 2] = motion2
    return flow


def collect_boundary_examples_from_component_motion(
    image: np.ndarray,
    components: np.ndarray,
    motions: dict[int, tuple[int, int]],
    confidences: dict[int, float] | None = None,
    min_confidence: float = 0.50,
    min_rgb_contrast: float = 0.15,
) -> list[tuple[np.ndarray, np.ndarray, float]]:
    """Store only boundaries between confident moving appearance regions.

    Requiring both sides to be confident intentionally means the huge stationary
    background component need not provide negative evidence.  Static RGB
    appearance already separates it in this gate; the learned memory is being
    asked specifically to bridge unlike parts that demonstrably move together.
    """
    n = image.shape[0]
    examples: list[tuple[np.ndarray, np.ndarray, float]] = []

    for y in range(n):
        for x in range(n):
            for dy, dx in ((1, 0), (0, 1)):
                yy, xx = y + dy, x + dx
                if yy >= n or xx >= n:
                    continue

                if np.linalg.norm(image[y, x] - image[yy, xx]) < min_rgb_contrast:
                    continue

                ca = int(components[y, x])
                cb = int(components[yy, xx])
                if ca == cb:
                    continue

                if confidences is not None and (
                    confidences[ca] < min_confidence
                    or confidences[cb] < min_confidence
                ):
                    continue

                va = np.asarray(motions[ca], dtype=np.float64)
                vb = np.asarray(motions[cb], dtype=np.float64)
                na = float(np.linalg.norm(va))
                nb = float(np.linalg.norm(vb))
                if na < 1e-12 and nb < 1e-12:
                    continue

                same_motion = (
                    na > 1e-12
                    and nb > 1e-12
                    and np.linalg.norm(va - vb) < 0.1
                )
                examples.append(
                    (
                        image[y, x].copy(),
                        image[yy, xx].copy(),
                        1.0 if same_motion else 0.0,
                    )
                )

    return examples


def component_oracle_motion(
    components: np.ndarray,
    labels: np.ndarray,
    motion1: tuple[int, int],
    motion2: tuple[int, int],
) -> dict[int, tuple[int, int]]:
    """Evaluation/oracle control only: majority ownership determines true motion."""
    motions: dict[int, tuple[int, int]] = {}
    for comp in np.unique(components):
        owners = labels[components == comp]
        values, counts = np.unique(owners, return_counts=True)
        owner = int(values[np.argmax(counts)])
        if owner == 1:
            motions[int(comp)] = motion1
        elif owner == 2:
            motions[int(comp)] = motion2
        else:
            motions[int(comp)] = (0, 0)
    return motions


def evaluate_static(
    image: np.ndarray,
    truth: np.ndarray,
    examples: list[tuple[np.ndarray, np.ndarray, float]],
) -> dict:
    if examples:
        radius = calibrate_radius(examples)
        memory = prepare_memory(examples)
        w = build_affinity(image, memory, radius)
    else:
        radius = None
        w = build_affinity(image)

    pred = threshold_components(w, threshold=0.5)
    return {
        "ari": adjusted_rand_index(truth, pred),
        "component_count": int(np.unique(pred).size),
        "mean_object_fragmentation": mean_object_fragmentation(truth, pred),
        "memory_radius": radius,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-scenes", type=int, default=12)
    parser.add_argument("--json", type=str, default=None)
    args = parser.parse_args()

    # The two rigid objects move in opposite horizontal directions.
    training = (
        ((3, 2), (13, 14), (3, 3), (13, 13)),
        ((3, 3), (13, 13), (3, 4), (13, 12)),
        ((3, 4), (13, 12), (3, 5), (13, 11)),
    )
    motion1 = (0, 1)
    motion2 = (0, -1)

    estimated_examples: list[tuple[np.ndarray, np.ndarray, float]] = []
    oracle_examples: list[tuple[np.ndarray, np.ndarray, float]] = []
    motion_rows = []

    for episode, (p1, p2, q1, q2) in enumerate(training):
        frame0, labels0, _ = render_scene(p1, p2, seed=2 * episode)
        frame1, _, _ = render_scene(q1, q2, seed=2 * episode + 1)

        components = appearance_components(frame0)
        motions, confidences = estimate_component_translations(
            frame0, frame1, components
        )
        estimated_examples.extend(
            collect_boundary_examples_from_component_motion(
                frame0,
                components,
                motions,
                confidences,
            )
        )

        oracle_motions = component_oracle_motion(
            components, labels0, motion1, motion2
        )
        oracle_examples.extend(
            collect_boundary_examples_from_component_motion(
                frame0,
                components,
                oracle_motions,
                confidences=None,
                min_confidence=0.0,
            )
        )

        moving_correct = []
        moving_confidence = []
        background_confidence = []
        for comp in np.unique(components):
            comp = int(comp)
            owners = labels0[components == comp]
            values, counts = np.unique(owners, return_counts=True)
            owner = int(values[np.argmax(counts)])
            expected = (
                motion1 if owner == 1
                else motion2 if owner == 2
                else (0, 0)
            )
            if owner == 0:
                background_confidence.append(confidences[comp])
            else:
                moving_correct.append(motions[comp] == expected)
                moving_confidence.append(confidences[comp])

        motion_rows.append(
            {
                "episode": episode,
                "appearance_component_count": int(np.unique(components).size),
                "moving_component_motion_accuracy": float(np.mean(moving_correct)),
                "moving_component_median_confidence": float(
                    np.median(moving_confidence)
                ),
                "background_component_median_confidence": float(
                    np.median(background_confidence)
                ),
            }
        )

    # Time-shuffled control: each frame0 is paired with an unrelated scene well
    # outside the search radius.  A valid estimator should report low confidence
    # rather than confidently inventing common fate.
    shuffled_targets = (
        ((12, 2), (3, 14)),
        ((11, 3), (2, 13)),
        ((12, 4), (4, 12)),
    )
    shuffled_examples: list[tuple[np.ndarray, np.ndarray, float]] = []
    shuffled_confidences = []

    for episode, (p1, p2, _, _) in enumerate(training):
        frame0, _, _ = render_scene(p1, p2, seed=2 * episode)
        q1, q2 = shuffled_targets[episode]
        wrong_next, _, _ = render_scene(q1, q2, seed=50 + episode)

        components = appearance_components(frame0)
        motions, confidences = estimate_component_translations(
            frame0, wrong_next, components
        )
        shuffled_examples.extend(
            collect_boundary_examples_from_component_motion(
                frame0,
                components,
                motions,
                confidences,
            )
        )
        shuffled_confidences.extend(confidences.values())

    test_positions = (
        ((12, 2), (3, 14)),
        ((11, 3), (2, 13)),
        ((12, 4), (4, 12)),
    )

    rows = []
    for sample in range(args.test_scenes):
        p1, p2 = test_positions[sample % len(test_positions)]
        image, truth, _ = render_scene(p1, p2, seed=100 + sample)

        rows.append(
            {
                "sample": sample,
                "pos1": list(p1),
                "pos2": list(p2),
                "static": evaluate_static(image, truth, []),
                "oracle_motion": evaluate_static(
                    image, truth, oracle_examples
                ),
                "estimated_motion": evaluate_static(
                    image, truth, estimated_examples
                ),
                "time_shuffled": evaluate_static(
                    image, truth, shuffled_examples
                ),
            }
        )

    def summarize(name: str) -> dict:
        ari = [r[name]["ari"] for r in rows]
        comps = [r[name]["component_count"] for r in rows]
        frag = [r[name]["mean_object_fragmentation"] for r in rows]
        return {
            "median_ari": float(np.median(ari)),
            "min_ari": float(np.min(ari)),
            "median_component_count": float(np.median(comps)),
            "median_object_fragmentation": float(np.median(frag)),
            "max_object_fragmentation": float(np.max(frag)),
        }

    report = {
        "training_episodes": len(training),
        "test_scenes": args.test_scenes,
        "estimated_training_examples": len(estimated_examples),
        "oracle_training_examples": len(oracle_examples),
        "time_shuffled_training_examples": len(shuffled_examples),
        "moving_component_motion_accuracy": float(
            np.mean(
                [
                    row["moving_component_motion_accuracy"]
                    for row in motion_rows
                ]
            )
        ),
        "moving_component_median_confidence": float(
            np.median(
                [
                    row["moving_component_median_confidence"]
                    for row in motion_rows
                ]
            )
        ),
        "background_component_median_confidence": float(
            np.median(
                [
                    row["background_component_median_confidence"]
                    for row in motion_rows
                ]
            )
        ),
        "time_shuffled_component_median_confidence": float(
            np.median(shuffled_confidences)
        ),
        "summary": {
            "static": summarize("static"),
            "oracle_motion": summarize("oracle_motion"),
            "estimated_motion": summarize("estimated_motion"),
            "time_shuffled": summarize("time_shuffled"),
        },
        "motion_rows": motion_rows,
        "rows": rows,
        "interpretation": (
            "Oracle motion is no longer required. Appearance-coherent regions "
            "estimated from frame t can recover their small translations from "
            "frame t+1 using RGB template matching alone. Common motion then "
            "writes boundary-pair affinities that bind the later static objects "
            "at new positions. Time-shuffled frames produce no confident write. "
            "The remaining scaffold is the appearance-region decomposition; dense "
            "correspondence, occlusion and natural images are still untested."
        ),
    }

    print(json.dumps(report, indent=2))

    assert len(estimated_examples) > 0
    assert report["moving_component_motion_accuracy"] > 0.95
    assert len(shuffled_examples) == 0
    assert (
        report["summary"]["estimated_motion"]["median_object_fragmentation"]
        <= 1.0
    )

    if args.json:
        Path(args.json).write_text(
            json.dumps(report, indent=2) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
