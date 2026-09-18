#!/usr/bin/env python3
"""Gate 10: local-only vector synchrony creates instance addresses without repulsion.

Gate 9 used a signed oscillator graph:
  local same-motion neighbours attract,
  every other moving pair weakly repels.

That global repulsion guaranteed address separation but also injected knowledge
that distant moving components should be different.

This gate deletes every nonlocal edge.

Each moving appearance component starts as a random D-dimensional unit vector.
Only adjacent components with the same non-zero motion interact.  A projected
unit-sphere update synchronizes each disconnected object locally.  Different
objects never communicate, so their final addresses are independent symmetry-
broken orientations inherited from random initial state.

Two questions are separated:

1. binding: does local vector synchrony create a shared address for the parts
   of one object, allowing later occluded binding?
2. capacity: how often do two independent instance addresses accidentally lie
   within the same-address cosine threshold as dimensionality grows?

The full scene uses D=8.  A Monte Carlo address-capacity sweep covers
D={2,3,4,8,16}.  No object IDs and no cross-instance repulsion enter the
dynamics.

This is a mechanism-isolation experiment, not an AKOrN reproduction.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from gate5_continuous_common_fate import build_affinity, threshold_components
from gate6_estimated_motion_write import appearance_components, render_scene
from gate7_occlusion_relational_bridge import (
    direct_adjacencies,
    make_occluded_clutter_scene,
    train_estimated_common_fate_memories,
)
from gate8_persistent_phase_address import (
    add_relation_phase_bridges,
    estimate_component_translations_robust,
    evaluate_graph,
    keep_phase_where_observation_persists,
)
from gate9_emergent_phase_address import moving_components


ADDRESS_THRESHOLD = 0.99
CAPACITY_DIMS = (2, 3, 4, 8, 16)


def normalize_rows(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-30)


def local_attractive_edges(
    components: np.ndarray,
    motions: dict[int, tuple[int, int]],
) -> tuple[list[int], list[tuple[int, int]]]:
    """Only spatially adjacent same-motion moving components may interact."""
    nodes = moving_components(components, motions)
    index = {comp: i for i, comp in enumerate(nodes)}
    edges: list[tuple[int, int]] = []

    for ca, cb in direct_adjacencies(components):
        if ca not in index or cb not in index:
            continue
        va = np.asarray(motions[ca], dtype=np.float64)
        vb = np.asarray(motions[cb], dtype=np.float64)
        if np.linalg.norm(va - vb) < 0.1:
            edges.append((index[ca], index[cb]))

    edges.sort()
    return nodes, edges


def projected_vector_relax(
    initial: np.ndarray,
    edges: list[tuple[int, int]],
    steps: int = 240,
    gamma: float = 0.20,
) -> np.ndarray:
    """Attractive synchronization of unit vectors using tangent projection."""
    x = normalize_rows(np.asarray(initial, dtype=np.float64).copy())

    for _ in range(steps):
        drive = np.zeros_like(x)
        for i, j in edges:
            drive[i] += x[j]
            drive[j] += x[i]

        tangent = drive - np.sum(drive * x, axis=1, keepdims=True) * x
        x = normalize_rows(x + gamma * tangent)

    return x


def synchronization_receipt(
    x: np.ndarray,
    edges: list[tuple[int, int]],
) -> dict[str, float]:
    cosines = [float(np.dot(x[i], x[j])) for i, j in edges]
    return {
        "minimum_local_edge_cosine": float(np.min(cosines)),
        "mean_local_edge_cosine": float(np.mean(cosines)),
    }


def vector_field_from_components(
    components: np.ndarray,
    nodes: list[int],
    addresses: np.ndarray,
) -> np.ndarray:
    dim = addresses.shape[1]
    field = np.zeros((*components.shape, dim), dtype=np.float64)
    for comp, address in zip(nodes, addresses):
        field[components == comp] = address
    return field


def advect_vector_field(
    field0: np.ndarray,
    components: np.ndarray,
    motions: dict[int, tuple[int, int]],
) -> np.ndarray:
    n = components.shape[0]
    out = np.zeros_like(field0)

    for comp in np.unique(components):
        comp = int(comp)
        dy, dx = motions[comp]
        ys, xs = np.where(components == comp)
        yy = ys + dy
        xx = xs + dx
        valid = (yy >= 0) & (yy < n) & (xx >= 0) & (xx < n)
        out[yy[valid], xx[valid]] = field0[ys[valid], xs[valid]]

    return out


def component_vector_addresses(
    field: np.ndarray,
    component_map: np.ndarray,
    support_threshold: float = 0.25,
) -> dict[int, np.ndarray]:
    dim = field.shape[-1]
    out: dict[int, np.ndarray] = {}

    for comp in np.unique(component_map):
        comp = int(comp)
        value = field[component_map == comp].mean(axis=0)
        norm = float(np.linalg.norm(value))
        out[comp] = (
            value / norm
            if norm > support_threshold
            else np.zeros(dim, dtype=np.float64)
        )

    return out


def capacity_sweep(
    trials: int,
    threshold: float,
    seed: int = 991,
) -> dict[str, dict[str, float]]:
    """Collision probability for independent random unit-vector addresses."""
    rng = np.random.default_rng(seed)
    out = {}

    for dim in CAPACITY_DIMS:
        a = normalize_rows(rng.normal(size=(trials, dim)))
        b = normalize_rows(rng.normal(size=(trials, dim)))
        cosine = np.sum(a * b, axis=1)
        out[str(dim)] = {
            "trials": trials,
            "collision_fraction": float(np.mean(cosine >= threshold)),
            "cosine_p95": float(np.quantile(cosine, 0.95)),
            "cosine_p99": float(np.quantile(cosine, 0.99)),
            "maximum_cosine": float(np.max(cosine)),
        }

    return out


def bridge_with_addresses(
    final_image: np.ndarray,
    truth: np.ndarray,
    local_w: np.ndarray,
    final_components: np.ndarray,
    memories,
    addresses: dict[int, np.ndarray] | None,
    require_address: bool,
) -> dict:
    w, count = add_relation_phase_bridges(
        local_w,
        final_image,
        final_components,
        memories["relation_memory"],
        memories["relation_radius"],
        phases=addresses,
        require_phase=require_address,
    )
    return evaluate_graph(w, truth, count)


def run_scene(seed: int, memories, dim: int) -> dict:
    # Same velocity, as Gates 8--9: velocity cannot identify the two instances.
    p1_0, p2_0 = (12, 3), (4, 11)
    p1_1, p2_1 = (12, 4), (4, 12)

    frame0, _, _ = render_scene(p1_0, p2_0, seed=5000 + 2 * seed)
    frame1, _, _ = render_scene(p1_1, p2_1, seed=5001 + 2 * seed)

    components0 = appearance_components(frame0)
    motions, _ = estimate_component_translations_robust(
        frame0, frame1, components0
    )
    nodes, edges = local_attractive_edges(components0, motions)

    rng = np.random.default_rng(6000 + seed)
    initial = normalize_rows(rng.normal(size=(len(nodes), dim)))
    relaxed = projected_vector_relax(initial, edges)

    field0 = vector_field_from_components(components0, nodes, relaxed)
    frozen0 = vector_field_from_components(components0, nodes, initial)

    field1 = advect_vector_field(field0, components0, motions)
    frozen1 = advect_vector_field(frozen0, components0, motions)

    final_image, truth = make_occluded_clutter_scene(
        p1_1, p2_1, seed=7000 + seed
    )
    visible = keep_phase_where_observation_persists(
        field1, frame1, final_image
    )
    frozen_visible = keep_phase_where_observation_persists(
        frozen1, frame1, final_image
    )

    local_w = build_affinity(
        final_image,
        memories["local_memory"],
        memories["local_radius"],
    )
    final_components = threshold_components(
        local_w, threshold=0.5
    ).reshape(final_image.shape[:2])

    addresses = component_vector_addresses(visible, final_components)
    frozen_addresses = component_vector_addresses(
        frozen_visible, final_components
    )
    reset_addresses = {
        comp: np.zeros(dim, dtype=np.float64)
        for comp in addresses
    }

    # Diagnostic only: because there are two local attractive edges, their
    # synchronized endpoint orientations are the two instance addresses.
    edge_addresses = [
        normalize_rows((relaxed[[i, j]].mean(axis=0))[None, :])[0]
        for i, j in edges
    ]
    cross_cosine = float(np.dot(edge_addresses[0], edge_addresses[1]))

    return {
        "seed": seed,
        "moving_component_count": len(nodes),
        "local_attractive_edges": [list(edge) for edge in edges],
        "local_edge_count": len(edges),
        "synchronization": synchronization_receipt(relaxed, edges),
        "cross_instance_address_cosine": cross_cosine,
        "cross_instance_address_collision": bool(
            cross_cosine >= ADDRESS_THRESHOLD
        ),
        "local_memory": evaluate_graph(local_w, truth, 0),
        "relation_only": bridge_with_addresses(
            final_image,
            truth,
            local_w,
            final_components,
            memories,
            None,
            False,
        ),
        "frozen_random_vector": bridge_with_addresses(
            final_image,
            truth,
            local_w,
            final_components,
            memories,
            frozen_addresses,
            True,
        ),
        "local_vector_sync": bridge_with_addresses(
            final_image,
            truth,
            local_w,
            final_components,
            memories,
            addresses,
            True,
        ),
        "reset_vector": bridge_with_addresses(
            final_image,
            truth,
            local_w,
            final_components,
            memories,
            reset_addresses,
            True,
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenes", type=int, default=24)
    parser.add_argument("--dim", type=int, default=8)
    parser.add_argument("--capacity-trials", type=int, default=20000)
    parser.add_argument("--json", type=str, default=None)
    args = parser.parse_args()

    memories = train_estimated_common_fate_memories()
    rows = [
        run_scene(seed, memories, args.dim)
        for seed in range(args.scenes)
    ]

    conditions = (
        "local_memory",
        "relation_only",
        "frozen_random_vector",
        "local_vector_sync",
        "reset_vector",
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

    capacity = capacity_sweep(
        args.capacity_trials,
        ADDRESS_THRESHOLD,
    )

    report = {
        "scenes": args.scenes,
        "state_dimension": args.dim,
        "same_address_cosine_threshold": ADDRESS_THRESHOLD,
        "median_moving_component_count": float(
            np.median([row["moving_component_count"] for row in rows])
        ),
        "median_local_edge_count": float(
            np.median([row["local_edge_count"] for row in rows])
        ),
        "minimum_local_edge_cosine": float(
            np.min(
                [
                    row["synchronization"]["minimum_local_edge_cosine"]
                    for row in rows
                ]
            )
        ),
        "maximum_cross_instance_cosine": float(
            np.max(
                [row["cross_instance_address_cosine"] for row in rows]
            )
        ),
        "address_collision_scenes": int(
            sum(row["cross_instance_address_collision"] for row in rows)
        ),
        "capacity_sweep": capacity,
        "summary": {
            name: summarize(name)
            for name in conditions
        },
        "collision_scenes": {
            name: int(
                sum(
                    row[name]["cross_object_collision_count"] > 0
                    for row in rows
                )
            )
            for name in conditions
        },
        "fragmented_scenes": {
            name: int(
                sum(
                    row[name]["mean_object_fragmentation"] > 1.0
                    for row in rows
                )
            )
            for name in conditions
        },
        "interpretation": (
            "Explicit cross-instance repulsion is unnecessary. Purely local "
            "attractive synchronization gives each disconnected common-fate "
            "island a shared orientation inherited from random symmetry breaking. "
            "The remaining failure mode is accidental address collision. The "
            "Monte Carlo sweep isolates dimensionality as address capacity: "
            "higher-dimensional unit vectors make unrelated instance orientations "
            "far less likely to fall inside the same-address threshold."
        ),
        "rows": rows,
    }

    print(json.dumps(report, indent=2))

    assert report["median_moving_component_count"] == 4.0
    assert report["median_local_edge_count"] == 2.0
    assert report["minimum_local_edge_cosine"] > 0.99
    assert capacity["8"]["collision_fraction"] < capacity["2"]["collision_fraction"]

    for row in rows:
        sync = row["local_vector_sync"]
        relation = row["relation_only"]
        frozen = row["frozen_random_vector"]
        reset = row["reset_vector"]

        assert sync["mean_object_fragmentation"] <= 1.0
        assert sync["cross_object_collision_count"] == 0
        assert relation["cross_object_collision_count"] > 0
        assert frozen["mean_object_fragmentation"] > 1.0
        assert reset["mean_object_fragmentation"] > 1.0

    if args.json:
        Path(args.json).write_text(
            json.dumps(report, indent=2) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
