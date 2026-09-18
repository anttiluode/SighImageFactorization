#!/usr/bin/env python3
"""Gate 9: local oscillator dynamics generate the instance address.

Gate 8 proved that an instance-specific persistent phase-like variable can
resolve the ambiguity left by generic part relations. But Gate 8 *assigned*
one address after explicitly discovering each common-fate group.

This gate removes that enumerator.

Each moving appearance component starts with a random scalar phase. Coupling is
constructed without object IDs:

  * adjacent components with the same non-zero motion attract;
  * all other moving component pairs weakly repel.

The two objects deliberately have the SAME velocity, so velocity cannot serve
as identity. Kuramoto dynamics must synchronize the parts of each object while
separating the two disconnected instances in phase.

The resulting phase is then carried exactly as in Gate 8 into an occluded
static scene. Generic learned part relations may bridge only when the visible
regions also carry the same emergent address.

Attackers:
  * relation only -- no instance address;
  * frozen random phase -- persistence without within-object synchronization;
  * collapsed initial phase -- exact symmetry, so Kuramoto cannot invent a
    distinction from nothing;
  * reset phase -- erase the address when motion stops.

This is still a mechanism-isolation gate, not an AKOrN reproduction.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from gate5_continuous_common_fate import (
    build_affinity,
    threshold_components,
)
from gate6_estimated_motion_write import (
    appearance_components,
    render_scene,
)
from gate7_occlusion_relational_bridge import (
    direct_adjacencies,
    make_occluded_clutter_scene,
    train_estimated_common_fate_memories,
)
from gate8_persistent_phase_address import (
    add_relation_phase_bridges,
    advect_phase_field,
    component_phase_addresses,
    estimate_component_translations_robust,
    evaluate_graph,
    keep_phase_where_observation_persists,
    phase_field_from_components,
)


def moving_components(
    components: np.ndarray,
    motions: dict[int, tuple[int, int]],
) -> list[int]:
    """All non-substrate components with non-zero robust displacement."""
    labels, counts = np.unique(components, return_counts=True)
    substrate = int(labels[int(np.argmax(counts))])
    out = []
    for comp in labels:
        comp = int(comp)
        if comp == substrate:
            continue
        if np.linalg.norm(np.asarray(motions[comp], dtype=np.float64)) > 1e-12:
            out.append(comp)
    return sorted(out)


def common_fate_coupling(
    components: np.ndarray,
    motions: dict[int, tuple[int, int]],
    attraction: float = 1.0,
    repulsion: float = 0.20,
) -> tuple[list[int], np.ndarray]:
    """Build a signed oscillator coupling without enumerating object groups."""
    nodes = moving_components(components, motions)
    index = {comp: i for i, comp in enumerate(nodes)}
    coupling = np.full((len(nodes), len(nodes)), -repulsion, dtype=np.float64)
    np.fill_diagonal(coupling, 0.0)

    for ca, cb in direct_adjacencies(components):
        if ca not in index or cb not in index:
            continue
        va = np.asarray(motions[ca], dtype=np.float64)
        vb = np.asarray(motions[cb], dtype=np.float64)
        if np.linalg.norm(va - vb) < 0.1:
            ia, ib = index[ca], index[cb]
            coupling[ia, ib] = attraction
            coupling[ib, ia] = attraction

    return nodes, coupling


def kuramoto_relax(
    coupling: np.ndarray,
    initial_phase: np.ndarray,
    steps: int = 300,
    dt: float = 0.08,
) -> np.ndarray:
    """Plain scalar Kuramoto relaxation on the signed component graph."""
    theta = np.asarray(initial_phase, dtype=np.float64).copy()

    for _ in range(steps):
        # diff[i,j] = theta_j - theta_i
        diff = theta[None, :] - theta[:, None]
        velocity = np.sum(coupling * np.sin(diff), axis=1)
        theta += dt * velocity
        theta = (theta + math.pi) % (2.0 * math.pi) - math.pi

    return theta


def addresses_from_theta(
    nodes: list[int],
    theta: np.ndarray,
) -> dict[int, np.ndarray]:
    return {
        comp: np.array(
            [math.cos(float(angle)), math.sin(float(angle))],
            dtype=np.float64,
        )
        for comp, angle in zip(nodes, theta)
    }


def coupling_phase_receipt(
    coupling: np.ndarray,
    theta: np.ndarray,
) -> dict[str, float]:
    positive = []
    negative = []
    for i in range(len(theta)):
        for j in range(i + 1, len(theta)):
            cosine = math.cos(float(theta[i] - theta[j]))
            if coupling[i, j] > 0:
                positive.append(cosine)
            elif coupling[i, j] < 0:
                negative.append(cosine)

    return {
        "minimum_attractive_edge_cosine": float(np.min(positive)),
        "maximum_repulsive_edge_cosine": float(np.max(negative)),
    }


def bridge_with_addresses(
    final_image: np.ndarray,
    truth: np.ndarray,
    local_w: np.ndarray,
    final_components: np.ndarray,
    memories,
    addresses: dict[int, np.ndarray] | None,
    require_phase: bool,
) -> dict:
    w, count = add_relation_phase_bridges(
        local_w,
        final_image,
        final_components,
        memories["relation_memory"],
        memories["relation_radius"],
        phases=addresses,
        require_phase=require_phase,
    )
    return evaluate_graph(w, truth, count)


def run_scene(seed: int, memories) -> dict:
    # Same velocity for both objects: motion cannot label the instances.
    p1_0, p2_0 = (12, 3), (4, 11)
    p1_1, p2_1 = (12, 4), (4, 12)

    frame0, _, _ = render_scene(
        p1_0, p2_0, seed=2000 + 2 * seed
    )
    frame1, _, _ = render_scene(
        p1_1, p2_1, seed=2001 + 2 * seed
    )

    components0 = appearance_components(frame0)
    motions, _ = estimate_component_translations_robust(
        frame0, frame1, components0
    )
    nodes, coupling = common_fate_coupling(components0, motions)

    rng = np.random.default_rng(3000 + seed)
    theta0 = rng.uniform(-math.pi, math.pi, size=len(nodes))
    theta = kuramoto_relax(coupling, theta0)
    oscillator_addresses = addresses_from_theta(nodes, theta)
    frozen_addresses = addresses_from_theta(nodes, theta0)

    collapsed_theta0 = np.zeros(len(nodes), dtype=np.float64)
    collapsed_theta = kuramoto_relax(coupling, collapsed_theta0)
    collapsed_addresses = addresses_from_theta(nodes, collapsed_theta)

    phase0 = phase_field_from_components(
        components0, oscillator_addresses
    )
    frozen0 = phase_field_from_components(
        components0, frozen_addresses
    )
    collapsed0 = phase_field_from_components(
        components0, collapsed_addresses
    )

    phase1 = advect_phase_field(phase0, components0, motions)
    frozen1 = advect_phase_field(frozen0, components0, motions)
    collapsed1 = advect_phase_field(collapsed0, components0, motions)

    final_image, truth = make_occluded_clutter_scene(
        p1_1, p2_1, seed=4000 + seed
    )

    phase_visible = keep_phase_where_observation_persists(
        phase1, frame1, final_image
    )
    frozen_visible = keep_phase_where_observation_persists(
        frozen1, frame1, final_image
    )
    collapsed_visible = keep_phase_where_observation_persists(
        collapsed1, frame1, final_image
    )

    local_w = build_affinity(
        final_image,
        memories["local_memory"],
        memories["local_radius"],
    )
    final_components = threshold_components(
        local_w, threshold=0.5
    ).reshape(final_image.shape[:2])

    phase_by_region = component_phase_addresses(
        phase_visible, final_components
    )
    frozen_by_region = component_phase_addresses(
        frozen_visible, final_components
    )
    collapsed_by_region = component_phase_addresses(
        collapsed_visible, final_components
    )
    reset_by_region = {
        comp: np.zeros(2, dtype=np.float64)
        for comp in phase_by_region
    }

    receipt = coupling_phase_receipt(coupling, theta)

    return {
        "seed": seed,
        "moving_component_count": len(nodes),
        "coupling": coupling.tolist(),
        "phase_receipt": receipt,
        "local_memory": evaluate_graph(local_w, truth, 0),
        "relation_only": bridge_with_addresses(
            final_image,
            truth,
            local_w,
            final_components,
            memories,
            addresses=None,
            require_phase=False,
        ),
        "frozen_random_phase": bridge_with_addresses(
            final_image,
            truth,
            local_w,
            final_components,
            memories,
            frozen_by_region,
            require_phase=True,
        ),
        "emergent_phase": bridge_with_addresses(
            final_image,
            truth,
            local_w,
            final_components,
            memories,
            phase_by_region,
            require_phase=True,
        ),
        "collapsed_initial_phase": bridge_with_addresses(
            final_image,
            truth,
            local_w,
            final_components,
            memories,
            collapsed_by_region,
            require_phase=True,
        ),
        "reset_phase": bridge_with_addresses(
            final_image,
            truth,
            local_w,
            final_components,
            memories,
            reset_by_region,
            require_phase=True,
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenes", type=int, default=12)
    parser.add_argument("--json", type=str, default=None)
    args = parser.parse_args()

    memories = train_estimated_common_fate_memories()
    rows = [run_scene(seed, memories) for seed in range(args.scenes)]

    conditions = (
        "local_memory",
        "relation_only",
        "frozen_random_phase",
        "emergent_phase",
        "collapsed_initial_phase",
        "reset_phase",
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
        "scenes": args.scenes,
        "median_moving_component_count": float(
            np.median([row["moving_component_count"] for row in rows])
        ),
        "minimum_attractive_edge_cosine": float(
            np.min(
                [
                    row["phase_receipt"]["minimum_attractive_edge_cosine"]
                    for row in rows
                ]
            )
        ),
        "maximum_repulsive_edge_cosine": float(
            np.max(
                [
                    row["phase_receipt"]["maximum_repulsive_edge_cosine"]
                    for row in rows
                ]
            )
        ),
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
            "Gate 8's external instance enumerator is no longer required. "
            "Random component phases, attractive coupling across adjacent "
            "same-motion parts, and weak repulsion elsewhere self-organize into "
            "shared within-instance and separated between-instance addresses. "
            "The emergent phase resolves later occluded binding. Frozen random "
            "phase tests persistence without synchronization; collapsed initial "
            "phase tests exact symmetry; reset phase tests persistence itself."
        ),
        "rows": rows,
    }

    print(json.dumps(report, indent=2))

    assert report["minimum_attractive_edge_cosine"] > 0.99
    assert report["maximum_repulsive_edge_cosine"] < -0.99

    for row in rows:
        emergent = row["emergent_phase"]
        relation = row["relation_only"]
        collapsed = row["collapsed_initial_phase"]
        reset = row["reset_phase"]

        assert row["moving_component_count"] == 4
        assert emergent["mean_object_fragmentation"] <= 1.0
        assert emergent["cross_object_collision_count"] == 0
        assert relation["cross_object_collision_count"] > 0
        assert collapsed["cross_object_collision_count"] > 0
        assert reset["mean_object_fragmentation"] > 1.0

    if args.json:
        Path(args.json).write_text(
            json.dumps(report, indent=2) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
