#!/usr/bin/env python3
"""Gate 11: remove appearance-region presegmentation from instance-address formation.

Gates 8--10 formed oscillator states on presegmented appearance components.
This gate moves the dynamical state down to pixels.

Input to the address-forming mechanism:
  * two consecutive noisy RGB frames;
  * local +/-2 pixel correspondence;
  * four-neighbour locality.

No appearance components are used to construct the moving oscillator graph.

Pipeline:
  1. estimate dense local RGB displacement with center-anchored patch matching;
  2. keep forward/backward-consistent moving pixels;
  3. connect neighbouring moving pixels only when their estimated displacement
     agrees;
  4. initialize one random D-dimensional unit vector per active pixel;
  5. projected local attraction synchronizes each moving island;
  6. advect the vector field with the estimated displacement;
  7. after motion stops and occlusion appears, use the persistent vector address
     to gate the already-learned generic nonlocal part relation.

The generic relation memory and the final occlusion readout still use the older
region machinery.  This gate removes presegmentation only from *instance address
formation*.  A later gate can attack the remaining region-level readout.

Controls:
  * frozen random pixel vectors;
  * component-level Gate-10 address as an upper-bound reference;
  * local relation only, with no address.

This is a controlled synthetic mechanism test, not dense optical flow for
natural video and not an AKOrN reproduction.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from gate5_continuous_common_fate import build_affinity, threshold_components
from gate6_estimated_motion_write import appearance_components, render_scene
from gate7_occlusion_relational_bridge import (
    make_occluded_clutter_scene,
    train_estimated_common_fate_memories,
)
from gate8_persistent_phase_address import (
    add_relation_phase_bridges,
    estimate_component_translations_robust,
    evaluate_graph,
    keep_phase_where_observation_persists,
)
from gate10_local_vector_address import (
    component_vector_addresses,
    local_attractive_edges,
    normalize_rows,
    projected_vector_relax,
    vector_field_from_components,
    advect_vector_field,
)


def estimate_center_anchored_patch_flow(
    frame0: np.ndarray,
    frame1: np.ndarray,
    patch_radius: int = 1,
    search_radius: int = 2,
    patch_weight: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray]:
    """Dense local correspondence; center RGB dominates, patch breaks ties."""
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
            candidates: list[tuple[float, int, int]] = []

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
                    candidates.append(
                        (
                            center_cost + patch_weight * patch_cost,
                            dy,
                            dx,
                        )
                    )

            candidates.sort()
            best, second = candidates[0], candidates[1]
            flow[y, x] = best[1:]
            confidence[y, x] = (
                second[0] - best[0]
            ) / (second[0] + 1e-30)

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


def active_moving_mask(
    flow: np.ndarray,
    confidence: np.ndarray,
    fb_error: np.ndarray,
    confidence_threshold: float = 0.15,
    fb_threshold: float = 0.1,
) -> np.ndarray:
    return (
        (np.linalg.norm(flow, axis=-1) > 0.5)
        & (confidence > confidence_threshold)
        & (fb_error < fb_threshold)
    )


def pixel_motion_graph(
    flow: np.ndarray,
    active: np.ndarray,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Return active flat pixel indices and local equal-displacement edges."""
    n = active.shape[0]
    nodes = np.flatnonzero(active.ravel())
    local_index = {int(flat): i for i, flat in enumerate(nodes)}
    edges: list[tuple[int, int]] = []

    for y in range(n):
        for x in range(n):
            if not active[y, x]:
                continue
            i_flat = y * n + x
            for dy, dx in ((1, 0), (0, 1)):
                yy, xx = y + dy, x + dx
                if yy >= n or xx >= n or not active[yy, xx]:
                    continue
                if np.linalg.norm(flow[y, x] - flow[yy, xx]) > 0.1:
                    continue
                j_flat = yy * n + xx
                edges.append(
                    (local_index[i_flat], local_index[j_flat])
                )

    return nodes, edges


def pixel_vector_field(
    shape: tuple[int, int],
    nodes: np.ndarray,
    vectors: np.ndarray,
) -> np.ndarray:
    n, m = shape
    dim = vectors.shape[1]
    field = np.zeros((n, m, dim), dtype=np.float64)
    flat = field.reshape(n * m, dim)
    flat[nodes] = vectors
    return field


def advect_pixel_vector_field(
    field0: np.ndarray,
    flow: np.ndarray,
    active: np.ndarray,
) -> np.ndarray:
    n = active.shape[0]
    out = np.zeros_like(field0)

    ys, xs = np.where(active)
    for y, x in zip(ys, xs):
        dy, dx = np.rint(flow[y, x]).astype(int)
        yy, xx = y + dy, x + dx
        if 0 <= yy < n and 0 <= xx < n:
            out[yy, xx] = field0[y, x]

    return out


def field_component_addresses(
    field: np.ndarray,
    component_map: np.ndarray,
    support_threshold: float = 0.12,
) -> dict[int, np.ndarray]:
    """Mean persistent pixel vector inside each final visible region."""
    dim = field.shape[-1]
    out: dict[int, np.ndarray] = {}

    for comp in np.unique(component_map):
        comp = int(comp)
        vectors = field[component_map == comp]
        supported = np.linalg.norm(vectors, axis=1) > 0.5
        support_fraction = float(np.mean(supported))

        if support_fraction < support_threshold:
            out[comp] = np.zeros(dim, dtype=np.float64)
            continue

        value = vectors[supported].mean(axis=0)
        norm = float(np.linalg.norm(value))
        out[comp] = (
            value / norm
            if norm > 1e-12
            else np.zeros(dim, dtype=np.float64)
        )

    return out


def true_flow(
    labels: np.ndarray,
    shift: tuple[int, int],
) -> np.ndarray:
    flow = np.zeros((*labels.shape, 2), dtype=np.float64)
    flow[labels > 0] = shift
    return flow


def flow_receipt(
    estimated: np.ndarray,
    active: np.ndarray,
    labels: np.ndarray,
    shift: tuple[int, int],
) -> dict[str, float]:
    oracle = true_flow(labels, shift)
    fg = (labels > 0) & active
    bg = (labels == 0) & active

    return {
        "active_foreground_pixels": int(fg.sum()),
        "active_background_pixels": int(bg.sum()),
        "foreground_active_fraction": float(
            fg.sum() / max(1, int((labels > 0).sum()))
        ),
        "foreground_exact_displacement_fraction": float(
            np.mean(
                np.all(
                    estimated[fg] == oracle[fg],
                    axis=1,
                )
            )
        )
        if np.any(fg)
        else 0.0,
        "background_false_moving_fraction": float(
            bg.sum() / max(1, int((labels == 0).sum()))
        ),
    }


def within_object_vector_coherence(
    field: np.ndarray,
    labels: np.ndarray,
) -> dict[str, float]:
    values = []
    means = []

    for object_id in (1, 2):
        vectors = field[labels == object_id]
        vectors = vectors[np.linalg.norm(vectors, axis=1) > 0.5]
        mean = normalize_rows(vectors.mean(axis=0, keepdims=True))[0]
        means.append(mean)
        values.extend(vectors @ mean)

    return {
        "minimum_foreground_cosine_to_object_mean": float(np.min(values)),
        "mean_foreground_cosine_to_object_mean": float(np.mean(values)),
        "cross_object_mean_cosine": float(np.dot(means[0], means[1])),
    }


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


def component_upper_bound(
    frame0: np.ndarray,
    frame1: np.ndarray,
    final_image: np.ndarray,
    final_components: np.ndarray,
    local_w: np.ndarray,
    truth: np.ndarray,
    memories,
    seed: int,
    dim: int,
) -> dict:
    """Gate-10 component address reference on the same input frames."""
    components0 = appearance_components(frame0)
    motions, _ = estimate_component_translations_robust(
        frame0, frame1, components0
    )
    nodes, edges = local_attractive_edges(components0, motions)
    rng = np.random.default_rng(seed)
    initial = normalize_rows(rng.normal(size=(len(nodes), dim)))
    relaxed = projected_vector_relax(initial, edges)
    field0 = vector_field_from_components(
        components0, nodes, relaxed
    )
    field1 = advect_vector_field(field0, components0, motions)
    visible = keep_phase_where_observation_persists(
        field1, frame1, final_image
    )
    addresses = component_vector_addresses(
        visible, final_components
    )
    return bridge_with_addresses(
        final_image,
        truth,
        local_w,
        final_components,
        memories,
        addresses,
        True,
    )


def run_scene(seed: int, memories, dim: int) -> dict:
    # Both objects have the same one-pixel horizontal displacement.
    p1_0, p2_0 = (12, 3), (4, 11)
    p1_1, p2_1 = (12, 4), (4, 12)
    shift = (0, 1)

    frame0, labels0, _ = render_scene(
        p1_0, p2_0, seed=8000 + 2 * seed
    )
    frame1, _, _ = render_scene(
        p1_1, p2_1, seed=8001 + 2 * seed
    )

    flow, confidence, fb_error = estimate_bidirectional_flow(
        frame0, frame1
    )
    active = active_moving_mask(flow, confidence, fb_error)
    nodes, edges = pixel_motion_graph(flow, active)

    rng = np.random.default_rng(9000 + seed)
    initial = normalize_rows(rng.normal(size=(len(nodes), dim)))
    relaxed = projected_vector_relax(
        initial,
        edges,
        steps=320,
        gamma=0.18,
    )

    field0 = pixel_vector_field(
        frame0.shape[:2],
        nodes,
        relaxed,
    )
    frozen0 = pixel_vector_field(
        frame0.shape[:2],
        nodes,
        initial,
    )
    field1 = advect_pixel_vector_field(field0, flow, active)
    frozen1 = advect_pixel_vector_field(frozen0, flow, active)

    final_image, truth = make_occluded_clutter_scene(
        p1_1, p2_1, seed=10000 + seed
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

    addresses = field_component_addresses(
        visible, final_components
    )
    frozen_addresses = field_component_addresses(
        frozen_visible, final_components
    )
    reset_addresses = {
        comp: np.zeros(dim, dtype=np.float64)
        for comp in addresses
    }

    return {
        "seed": seed,
        "active_pixel_count": int(active.sum()),
        "pixel_edge_count": len(edges),
        "flow_receipt": flow_receipt(
            flow, active, labels0, shift
        ),
        "vector_coherence": within_object_vector_coherence(
            field0, labels0
        ),
        "relation_only": bridge_with_addresses(
            final_image,
            truth,
            local_w,
            final_components,
            memories,
            None,
            False,
        ),
        "frozen_random_pixel_vectors": bridge_with_addresses(
            final_image,
            truth,
            local_w,
            final_components,
            memories,
            frozen_addresses,
            True,
        ),
        "pixel_vector_sync": bridge_with_addresses(
            final_image,
            truth,
            local_w,
            final_components,
            memories,
            addresses,
            True,
        ),
        "reset_pixel_vectors": bridge_with_addresses(
            final_image,
            truth,
            local_w,
            final_components,
            memories,
            reset_addresses,
            True,
        ),
        "component_vector_reference": component_upper_bound(
            frame0,
            frame1,
            final_image,
            final_components,
            local_w,
            truth,
            memories,
            seed=11000 + seed,
            dim=dim,
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenes", type=int, default=12)
    parser.add_argument("--dim", type=int, default=8)
    parser.add_argument("--json", type=str, default=None)
    args = parser.parse_args()

    memories = train_estimated_common_fate_memories()
    rows = [
        run_scene(seed, memories, args.dim)
        for seed in range(args.scenes)
    ]

    conditions = (
        "relation_only",
        "frozen_random_pixel_vectors",
        "pixel_vector_sync",
        "reset_pixel_vectors",
        "component_vector_reference",
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
        "state_dimension": args.dim,
        "median_active_pixel_count": float(
            np.median([r["active_pixel_count"] for r in rows])
        ),
        "median_pixel_edge_count": float(
            np.median([r["pixel_edge_count"] for r in rows])
        ),
        "median_foreground_active_fraction": float(
            np.median(
                [
                    r["flow_receipt"]["foreground_active_fraction"]
                    for r in rows
                ]
            )
        ),
        "minimum_foreground_exact_displacement_fraction": float(
            np.min(
                [
                    r["flow_receipt"][
                        "foreground_exact_displacement_fraction"
                    ]
                    for r in rows
                ]
            )
        ),
        "maximum_background_false_moving_fraction": float(
            np.max(
                [
                    r["flow_receipt"]["background_false_moving_fraction"]
                    for r in rows
                ]
            )
        ),
        "minimum_foreground_cosine_to_object_mean": float(
            np.min(
                [
                    r["vector_coherence"][
                        "minimum_foreground_cosine_to_object_mean"
                    ]
                    for r in rows
                ]
            )
        ),
        "maximum_cross_object_mean_cosine": float(
            np.max(
                [
                    r["vector_coherence"]["cross_object_mean_cosine"]
                    for r in rows
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
            "Instance-address formation no longer requires appearance-region "
            "presegmentation. Dense local RGB correspondence defines a moving "
            "pixel graph, and local vector synchronization produces coherent "
            "instance addresses directly on pixels. The final nonlocal relation "
            "readout remains region-based and is the next scaffold to attack."
        ),
        "rows": rows,
    }

    print(json.dumps(report, indent=2))

    assert report["median_active_pixel_count"] > 100
    assert report[
        "minimum_foreground_exact_displacement_fraction"
    ] > 0.85
    assert report[
        "maximum_background_false_moving_fraction"
    ] < 0.08

    for row in rows:
        sync = row["pixel_vector_sync"]
        frozen = row["frozen_random_pixel_vectors"]
        relation = row["relation_only"]

        assert sync["mean_object_fragmentation"] <= 1.0
        assert sync["cross_object_collision_count"] == 0
        assert frozen["mean_object_fragmentation"] > 1.0
        assert relation["cross_object_collision_count"] > 0

    if args.json:
        Path(args.json).write_text(
            json.dumps(report, indent=2) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
