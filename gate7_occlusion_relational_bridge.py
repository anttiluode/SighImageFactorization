#!/usr/bin/env python3
"""Gate 7: occlusion exposes the locality limit, region relations bridge it.

Gates 4--6 learned that unlike appearance regions can belong together because
they move together.  But the learned operator was still local: it only changed
edges between neighbouring pixels.

This gate removes that contact at test time.  A one-pixel background occluder
cuts each two-part object into two visible islands.  A novel distractor is
placed at the same short spatial gap from one object.

Two memory scales are kept deliberately separate:

* local edge memory: noisy RGB boundary-pixel pairs, as in Gate 6;
* object relation memory: mean RGB descriptors of whole appearance regions
  that moved together.

The second distinction matters.  An earlier version of this gate allowed one
lucky pixel match to authorize a nonlocal edge; one test scene produced a false
bridge.  Nonlocal identity now requires region-level evidence.

We compare:
  * local learned affinity only;
  * relation-specific short-range bridge edges;
  * proximity-only short-range bridge edges.

The kill condition is per-scene, not median: relational bridging must repair
every object without a single cross-object collision.
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
from gate6_estimated_motion_write import (
    PALETTE,
    appearance_components,
    collect_boundary_examples_from_component_motion,
    estimate_component_translations,
    render_scene,
)


CLUTTER_RGB = np.array([0.82, 0.78, 0.12], dtype=np.float64)


def direct_adjacencies(component_map: np.ndarray) -> set[tuple[int, int]]:
    n = component_map.shape[0]
    out: set[tuple[int, int]] = set()
    for y in range(n):
        for x in range(n):
            a = int(component_map[y, x])
            for dy, dx in ((1, 0), (0, 1)):
                yy, xx = y + dy, x + dx
                if yy >= n or xx >= n:
                    continue
                b = int(component_map[yy, xx])
                if a != b:
                    out.add(tuple(sorted((a, b))))
    return out


def component_means(
    image: np.ndarray,
    component_map: np.ndarray,
) -> dict[int, np.ndarray]:
    return {
        int(comp): image[component_map == comp].mean(axis=0)
        for comp in np.unique(component_map)
    }


def train_estimated_common_fate_memories():
    """Train both local pixel-edge and region-relation memories from RGB video."""
    training = (
        ((3, 2), (13, 14), (3, 3), (13, 13)),
        ((3, 3), (13, 13), (3, 4), (13, 12)),
        ((3, 4), (13, 12), (3, 5), (13, 11)),
    )

    local_examples = []
    relation_examples = []

    for episode, (p1, p2, q1, q2) in enumerate(training):
        frame0, _, _ = render_scene(p1, p2, seed=2 * episode)
        frame1, _, _ = render_scene(q1, q2, seed=2 * episode + 1)

        components = appearance_components(frame0)
        motions, confidences = estimate_component_translations(
            frame0, frame1, components
        )

        local_examples.extend(
            collect_boundary_examples_from_component_motion(
                frame0,
                components,
                motions,
                confidences,
            )
        )

        means = component_means(frame0, components)
        for ca, cb in direct_adjacencies(components):
            if confidences[ca] < 0.50 or confidences[cb] < 0.50:
                continue
            va = np.asarray(motions[ca], dtype=np.float64)
            vb = np.asarray(motions[cb], dtype=np.float64)
            if np.linalg.norm(va) < 1e-12 or np.linalg.norm(vb) < 1e-12:
                continue
            if np.linalg.norm(va - vb) < 0.1:
                relation_examples.append(
                    (
                        means[ca].copy(),
                        means[cb].copy(),
                        1.0,
                    )
                )

    if not local_examples:
        raise RuntimeError("expected local common-fate examples")
    if len(relation_examples) < 2:
        raise RuntimeError("expected repeated region relation examples")

    return {
        "local_examples": local_examples,
        "local_memory": prepare_memory(local_examples),
        "local_radius": calibrate_radius(local_examples),
        "relation_examples": relation_examples,
        "relation_memory": prepare_memory(relation_examples),
        "relation_radius": calibrate_radius(relation_examples),
    }


def make_occluded_clutter_scene(
    pos1: tuple[int, int],
    pos2: tuple[int, int],
    seed: int,
    size: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    """Static scene with each learned object split by a one-pixel occluder."""
    image, labels, _ = render_scene(pos1, pos2, seed=seed)
    rng = np.random.default_rng(10_000 + seed)

    # Remove one horizontal row at the original internal part boundary.  Fill it
    # with background-like pixels and mark it background in the evaluation mask.
    for y, x in (pos1, pos2):
        row = y + size // 2
        image[row, x : x + size] = (
            PALETTE[0]
            + rng.normal(0.0, 0.003, size=(size, 3))
        )
        labels[row, x : x + size] = 0

    # Novel distractor: same geometric gap as the occluded object relation, but
    # an appearance relation never observed during common motion.
    y1, x1 = pos1
    cy = y1
    cx = x1 + size + 1
    image[cy : cy + 3, cx : cx + 3] = (
        CLUTTER_RGB
        + rng.normal(0.0, 0.004, size=(3, 3, 3))
    )
    labels[cy : cy + 3, cx : cx + 3] = 3

    return np.clip(image, 0.0, 1.0), labels


def positive_memory_match(
    rgb_a: np.ndarray,
    rgb_b: np.ndarray,
    memory: tuple[np.ndarray, np.ndarray, np.ndarray],
    radius: float,
) -> bool:
    p, q, target = memory
    direct = np.sum((p - rgb_a) ** 2, axis=1) + np.sum(
        (q - rgb_b) ** 2, axis=1
    )
    swapped = np.sum((p - rgb_b) ** 2, axis=1) + np.sum(
        (q - rgb_a) ** 2, axis=1
    )
    distance = np.minimum(direct, swapped)
    match = int(np.argmin(distance))
    return bool(
        float(distance[match]) <= radius
        and float(target[match]) > 0.5
    )


def candidate_gap_pairs(
    component_map: np.ndarray,
) -> dict[tuple[int, int], tuple[int, int]]:
    """Return one pixel-index bridge for every non-adjacent component pair at L1=2."""
    n = component_map.shape[0]
    direct = direct_adjacencies(component_map)
    offsets = (
        (2, 0), (-2, 0), (0, 2), (0, -2),
        (1, 1), (1, -1), (-1, 1), (-1, -1),
    )
    candidates: dict[tuple[int, int], tuple[int, int]] = {}

    for y in range(n):
        for x in range(n):
            i = y * n + x
            ca = int(component_map[y, x])
            for dy, dx in offsets:
                yy, xx = y + dy, x + dx
                if yy < 0 or yy >= n or xx < 0 or xx >= n:
                    continue
                j = yy * n + xx
                if j <= i:
                    continue
                cb = int(component_map[yy, xx])
                if ca == cb:
                    continue

                pair = tuple(sorted((ca, cb)))
                if pair in direct:
                    # Ordinary appearance boundaries are not occlusion gaps.
                    continue
                candidates.setdefault(pair, (i, j))

    return candidates


def add_gap_bridges(
    w: np.ndarray,
    image: np.ndarray,
    component_map: np.ndarray,
    mode: str,
    relation_memory: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    relation_radius: float | None = None,
) -> tuple[np.ndarray, int]:
    if mode not in {"relational", "proximity"}:
        raise ValueError("mode must be relational or proximity")
    if mode == "relational" and (
        relation_memory is None or relation_radius is None
    ):
        raise ValueError("relational mode requires region relation memory")

    out = w.copy()
    candidates = candidate_gap_pairs(component_map)
    means = component_means(image, component_map)
    added = 0

    for (ca, cb), (i, j) in candidates.items():
        allow = mode == "proximity"
        if mode == "relational":
            allow = positive_memory_match(
                means[ca],
                means[cb],
                relation_memory,
                relation_radius,
            )

        if allow:
            out[i, j] = out[j, i] = 1.0
            added += 1

    return out, added


def collision_metrics(
    truth: np.ndarray,
    predicted: np.ndarray,
) -> dict[str, int]:
    true_flat = np.asarray(truth).ravel()
    pred_flat = np.asarray(predicted).ravel()
    mixed = 0
    object_collisions = 0

    for cluster in np.unique(pred_flat):
        labels = np.unique(true_flat[pred_flat == cluster])
        if len(labels) > 1:
            mixed += 1
        nonzero = labels[labels != 0]
        if len(nonzero) > 1:
            object_collisions += 1

    return {
        "mixed_truth_component_count": mixed,
        "cross_object_collision_count": object_collisions,
    }


def evaluate(
    image: np.ndarray,
    truth: np.ndarray,
    local_memory,
    local_radius,
    bridge_mode: str | None,
    relation_memory=None,
    relation_radius=None,
) -> dict:
    local_w = build_affinity(image, local_memory, local_radius)
    local_components = threshold_components(
        local_w, threshold=0.5
    ).reshape(image.shape[:2])

    bridge_count = 0
    w = local_w
    if bridge_mode is not None:
        w, bridge_count = add_gap_bridges(
            local_w,
            image,
            local_components,
            bridge_mode,
            relation_memory,
            relation_radius,
        )

    predicted = threshold_components(w, threshold=0.5)
    result = {
        "ari": adjusted_rand_index(truth, predicted),
        "component_count": int(np.unique(predicted).size),
        "mean_object_fragmentation": mean_object_fragmentation(
            truth, predicted
        ),
        "bridge_component_pairs": bridge_count,
    }
    result.update(collision_metrics(truth, predicted))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenes", type=int, default=12)
    parser.add_argument("--json", type=str, default=None)
    args = parser.parse_args()

    memories = train_estimated_common_fate_memories()

    positions = (
        ((12, 2), (3, 14)),
        ((11, 3), (2, 13)),
        ((12, 4), (4, 12)),
    )

    rows = []
    for sample in range(args.scenes):
        p1, p2 = positions[sample % len(positions)]
        image, truth = make_occluded_clutter_scene(
            p1, p2, seed=200 + sample
        )

        rows.append(
            {
                "sample": sample,
                "local_memory": evaluate(
                    image,
                    truth,
                    memories["local_memory"],
                    memories["local_radius"],
                    None,
                ),
                "relational_gap_bridge": evaluate(
                    image,
                    truth,
                    memories["local_memory"],
                    memories["local_radius"],
                    "relational",
                    memories["relation_memory"],
                    memories["relation_radius"],
                ),
                "proximity_gap_bridge": evaluate(
                    image,
                    truth,
                    memories["local_memory"],
                    memories["local_radius"],
                    "proximity",
                ),
            }
        )

    def summarize(name: str) -> dict:
        keys = (
            "ari",
            "component_count",
            "mean_object_fragmentation",
            "bridge_component_pairs",
            "mixed_truth_component_count",
            "cross_object_collision_count",
        )
        return {
            key: float(np.median([row[name][key] for row in rows]))
            for key in keys
        }

    report = {
        "local_training_examples": len(memories["local_examples"]),
        "region_relation_training_examples": len(
            memories["relation_examples"]
        ),
        "local_memory_radius": memories["local_radius"],
        "relation_memory_radius": memories["relation_radius"],
        "scenes": args.scenes,
        "summary": {
            "local_memory": summarize("local_memory"),
            "relational_gap_bridge": summarize("relational_gap_bridge"),
            "proximity_gap_bridge": summarize("proximity_gap_bridge"),
        },
        "interpretation": (
            "Occlusion breaks the contact required by a purely local learned "
            "operator. Proximity can reconnect the visible islands but also "
            "merges nearby clutter. Region-level relations learned from common "
            "motion authorize exactly the nonlocal bridges supported by prior "
            "history. The earlier single-pixel bridge criterion was unsafe; "
            "nonlocal identity requires evidence at the abstraction level of "
            "the edge being created."
        ),
        "rows": rows,
    }

    print(json.dumps(report, indent=2))

    for row in rows:
        rel = row["relational_gap_bridge"]
        local = row["local_memory"]
        prox = row["proximity_gap_bridge"]

        assert rel["mean_object_fragmentation"] <= 1.0
        assert rel["mixed_truth_component_count"] == 0
        assert rel["cross_object_collision_count"] == 0
        assert local["mean_object_fragmentation"] > 1.0
        assert prox["cross_object_collision_count"] > 0

    if args.json:
        Path(args.json).write_text(
            json.dumps(report, indent=2) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
