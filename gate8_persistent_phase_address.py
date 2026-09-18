#!/usr/bin/env python3
"""Gate 8: a persistent phase-like address resolves instance binding.

Gate 7 established that a correct generic part relation is not enough when
multiple compatible object instances come close.  This gate adds the missing
variable explicitly: an instance-specific dynamical address.

Crucially, the two objects move with the SAME velocity.  Velocity itself cannot
serve as the ID.  We discover two instances as separate connected groups of
appearance regions whose neighbouring parts share motion.  Each discovered
group receives an arbitrary 2-D unit-vector phase address.

That phase is advected with the object into the next frame.  The objects then
stop, an occluder removes direct contact between their parts, and a distractor
appears.  Generic learned part relations may create a nonlocal bridge only when
the two visible regions also carry the same persistent phase.

Attackers:
  * relation only -- Gate-7 type relation, no instance address;
  * collapsed phase -- all objects receive the same phase;
  * reset phase -- phase disappears when motion stops.

This is a mechanism isolation test, not a reproduction of AKOrN.  The phase is
assigned from discovered common-fate groups rather than learned end-to-end.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from gate5_continuous_common_fate import (
    adjusted_rand_index,
    build_affinity,
    mean_object_fragmentation,
    threshold_components,
)
from gate6_estimated_motion_write import (
    appearance_components,
    estimate_component_translations,
    render_scene,
)
from gate7_occlusion_relational_bridge import (
    candidate_gap_pairs,
    collision_metrics,
    component_means,
    direct_adjacencies,
    make_occluded_clutter_scene,
    positive_memory_match,
    train_estimated_common_fate_memories,
)


def discover_common_fate_groups(
    components: np.ndarray,
    motions: dict[int, tuple[int, int]],
    confidences: dict[int, float],
    min_confidence: float = 0.50,
) -> list[list[int]]:
    """Connected moving component groups, not global velocity bins."""
    moving = set()
    for comp in np.unique(components):
        comp = int(comp)
        motion = np.asarray(motions[comp], dtype=np.float64)
        if confidences[comp] >= min_confidence and np.linalg.norm(motion) > 1e-12:
            moving.add(comp)

    neighbours: dict[int, set[int]] = {comp: set() for comp in moving}
    for ca, cb in direct_adjacencies(components):
        if ca not in moving or cb not in moving:
            continue
        va = np.asarray(motions[ca], dtype=np.float64)
        vb = np.asarray(motions[cb], dtype=np.float64)
        if np.linalg.norm(va - vb) < 0.1:
            neighbours[ca].add(cb)
            neighbours[cb].add(ca)

    groups = []
    unseen = set(moving)
    while unseen:
        root = min(unseen)
        stack = [root]
        unseen.remove(root)
        group = []
        while stack:
            node = stack.pop()
            group.append(node)
            for nxt in neighbours[node]:
                if nxt in unseen:
                    unseen.remove(nxt)
                    stack.append(nxt)
        groups.append(sorted(group))

    groups.sort(key=lambda g: g[0])
    return groups


def assign_phase_addresses(
    groups: list[list[int]],
) -> dict[int, np.ndarray]:
    """Arbitrary continuous addresses on the unit circle."""
    out: dict[int, np.ndarray] = {}
    count = len(groups)
    if count == 0:
        return out

    for index, group in enumerate(groups):
        angle = 2.0 * math.pi * index / count
        phase = np.array([math.cos(angle), math.sin(angle)], dtype=np.float64)
        for comp in group:
            out[comp] = phase.copy()
    return out


def phase_field_from_components(
    components: np.ndarray,
    addresses: dict[int, np.ndarray],
) -> np.ndarray:
    field = np.zeros((*components.shape, 2), dtype=np.float64)
    for comp, phase in addresses.items():
        field[components == comp] = phase
    return field


def advect_phase_field(
    phase0: np.ndarray,
    components: np.ndarray,
    motions: dict[int, tuple[int, int]],
) -> np.ndarray:
    """Carry the address with each tracked appearance component."""
    n = components.shape[0]
    out = np.zeros_like(phase0)

    for comp in np.unique(components):
        comp = int(comp)
        dy, dx = motions[comp]
        ys, xs = np.where(components == comp)
        yy = ys + dy
        xx = xs + dx
        valid = (yy >= 0) & (yy < n) & (xx >= 0) & (xx < n)
        out[yy[valid], xx[valid]] = phase0[ys[valid], xs[valid]]

    return out


def keep_phase_where_observation_persists(
    phase: np.ndarray,
    previous_image: np.ndarray,
    current_image: np.ndarray,
    rgb_change_threshold: float = 0.15,
) -> np.ndarray:
    """Clear carried state where an occluder/new clutter replaced the observation."""
    out = phase.copy()
    changed = np.linalg.norm(previous_image - current_image, axis=-1) > rgb_change_threshold
    out[changed] = 0.0
    return out


def component_phase_addresses(
    phase_field: np.ndarray,
    component_map: np.ndarray,
) -> dict[int, np.ndarray]:
    out: dict[int, np.ndarray] = {}
    for comp in np.unique(component_map):
        comp = int(comp)
        value = phase_field[component_map == comp].mean(axis=0)
        norm = float(np.linalg.norm(value))
        if norm > 0.25:
            out[comp] = value / norm
        else:
            out[comp] = np.zeros(2, dtype=np.float64)
    return out


def same_phase(
    a: np.ndarray,
    b: np.ndarray,
    cosine_threshold: float = 0.99,
) -> bool:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 0.5 or nb < 0.5:
        return False
    return float(np.dot(a, b) / (na * nb)) >= cosine_threshold


def add_relation_phase_bridges(
    w: np.ndarray,
    image: np.ndarray,
    component_map: np.ndarray,
    relation_memory,
    relation_radius: float,
    phases: dict[int, np.ndarray] | None,
    require_phase: bool,
) -> tuple[np.ndarray, int]:
    out = w.copy()
    candidates = candidate_gap_pairs(component_map)
    means = component_means(image, component_map)
    added = 0

    for (ca, cb), (i, j) in candidates.items():
        if not positive_memory_match(
            means[ca],
            means[cb],
            relation_memory,
            relation_radius,
        ):
            continue

        if require_phase:
            assert phases is not None
            if not same_phase(phases[ca], phases[cb]):
                continue

        out[i, j] = out[j, i] = 1.0
        added += 1

    return out, added


def evaluate_graph(
    w: np.ndarray,
    truth: np.ndarray,
    bridge_count: int,
) -> dict:
    predicted = threshold_components(w, threshold=0.5)
    result = {
        "ari": adjusted_rand_index(truth, predicted),
        "component_count": int(np.unique(predicted).size),
        "mean_object_fragmentation": mean_object_fragmentation(truth, predicted),
        "bridge_component_pairs": bridge_count,
    }
    result.update(collision_metrics(truth, predicted))
    return result


def run_scene(seed: int, memories) -> dict:
    # Same-velocity movement: velocity cannot distinguish the two instances.
    p1_0, p2_0 = (12, 3), (4, 11)
    p1_1, p2_1 = (12, 4), (4, 12)

    frame0, _, _ = render_scene(p1_0, p2_0, seed=1000 + 2 * seed)
    frame1, _, _ = render_scene(p1_1, p2_1, seed=1001 + 2 * seed)

    components0 = appearance_components(frame0)
    motions, confidences = estimate_component_translations(
        frame0, frame1, components0
    )

    groups = discover_common_fate_groups(
        components0,
        motions,
        confidences,
    )
    addresses = assign_phase_addresses(groups)
    phase0 = phase_field_from_components(components0, addresses)
    phase1 = advect_phase_field(phase0, components0, motions)

    final_image, truth = make_occluded_clutter_scene(
        p1_1, p2_1, seed=1200 + seed
    )
    visible_phase = keep_phase_where_observation_persists(
        phase1,
        frame1,
        final_image,
    )

    local_w = build_affinity(
        final_image,
        memories["local_memory"],
        memories["local_radius"],
    )
    final_components = threshold_components(
        local_w, threshold=0.5
    ).reshape(final_image.shape[:2])
    phases = component_phase_addresses(
        visible_phase,
        final_components,
    )

    relation_w, relation_count = add_relation_phase_bridges(
        local_w,
        final_image,
        final_components,
        memories["relation_memory"],
        memories["relation_radius"],
        phases=None,
        require_phase=False,
    )

    phase_w, phase_count = add_relation_phase_bridges(
        local_w,
        final_image,
        final_components,
        memories["relation_memory"],
        memories["relation_radius"],
        phases=phases,
        require_phase=True,
    )

    collapsed = {
        comp: (
            np.array([1.0, 0.0], dtype=np.float64)
            if np.linalg.norm(value) > 0.5
            else np.zeros(2, dtype=np.float64)
        )
        for comp, value in phases.items()
    }
    collapsed_w, collapsed_count = add_relation_phase_bridges(
        local_w,
        final_image,
        final_components,
        memories["relation_memory"],
        memories["relation_radius"],
        phases=collapsed,
        require_phase=True,
    )

    reset = {
        comp: np.zeros(2, dtype=np.float64)
        for comp in phases
    }
    reset_w, reset_count = add_relation_phase_bridges(
        local_w,
        final_image,
        final_components,
        memories["relation_memory"],
        memories["relation_radius"],
        phases=reset,
        require_phase=True,
    )

    moving_components = []
    for comp in np.unique(components0):
        comp = int(comp)
        if np.linalg.norm(motions[comp]) > 1e-12 and confidences[comp] >= 0.50:
            moving_components.append(
                {
                    "component": comp,
                    "motion": list(motions[comp]),
                    "confidence": confidences[comp],
                }
            )

    return {
        "seed": seed,
        "discovered_instance_groups": [list(g) for g in groups],
        "group_count": len(groups),
        "moving_components": moving_components,
        "local_memory": evaluate_graph(local_w, truth, 0),
        "relation_only": evaluate_graph(
            relation_w, truth, relation_count
        ),
        "persistent_phase": evaluate_graph(
            phase_w, truth, phase_count
        ),
        "collapsed_phase": evaluate_graph(
            collapsed_w, truth, collapsed_count
        ),
        "reset_phase": evaluate_graph(
            reset_w, truth, reset_count
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenes", type=int, default=8)
    parser.add_argument("--json", type=str, default=None)
    args = parser.parse_args()

    memories = train_estimated_common_fate_memories()
    rows = [run_scene(seed, memories) for seed in range(args.scenes)]

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
        "median_discovered_group_count": float(
            np.median([row["group_count"] for row in rows])
        ),
        "summary": {
            "local_memory": summarize("local_memory"),
            "relation_only": summarize("relation_only"),
            "persistent_phase": summarize("persistent_phase"),
            "collapsed_phase": summarize("collapsed_phase"),
            "reset_phase": summarize("reset_phase"),
        },
        "collision_scenes": {
            name: int(
                sum(row[name]["cross_object_collision_count"] > 0 for row in rows)
            )
            for name in (
                "relation_only",
                "persistent_phase",
                "collapsed_phase",
                "reset_phase",
            )
        },
        "fragmented_scenes": {
            name: int(
                sum(row[name]["mean_object_fragmentation"] > 1.0 for row in rows)
            )
            for name in (
                "relation_only",
                "persistent_phase",
                "collapsed_phase",
                "reset_phase",
            )
        },
        "interpretation": (
            "Two separate objects can share the same velocity and the same generic "
            "part relations. Local connected common-fate groups nevertheless give "
            "them distinct instance addresses. Carrying a phase-like unit vector "
            "with each group through time disambiguates the later occluded static "
            "scene. Collapsing all phases recreates the Gate-7 cross-binding error; "
            "resetting phase recreates fragmentation. The useful role of phase here "
            "is therefore instance address, not a better diffusion solver."
        ),
        "rows": rows,
    }

    print(json.dumps(report, indent=2))

    for row in rows:
        phase = row["persistent_phase"]
        relation = row["relation_only"]
        collapsed = row["collapsed_phase"]
        reset = row["reset_phase"]

        assert row["group_count"] == 2
        assert phase["mean_object_fragmentation"] <= 1.0
        assert phase["cross_object_collision_count"] == 0
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
