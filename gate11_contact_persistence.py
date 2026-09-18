#!/usr/bin/env python3
"""Gate 11: temporary contact attacks local instance addresses.

Gate 10 established that D-dimensional local vector synchrony can give two
*disconnected* objects independent persistent addresses without any global
repulsion.

This gate attacks the word "disconnected".

Each trial starts with four local components:
    object A = nodes 0--1
    object B = nodes 2--3

The two internal edges synchronize first, creating two independent D=8
addresses exactly as in Gate 10.  Then a temporary same-motion contact appears
between nodes 1 and 2.

Two update rules are compared:

1. instantaneous:
   a newly observed local edge immediately receives coupling weight 1.

2. persistence-gated:
   a new edge starts weak and accumulates authority only while the contact
   persists,

       w[t+1] = w[t] + (1 - w[t]) / tau.

The contact is removed after 1,2,4,...,128 updates.  Address collision means the
two object-address cosines cross Gate 10's same-address threshold (0.99).

The point is not that tau=64 is special.  The duration sweep asks whether a
separate relation timescale can reject transient contact while still allowing
long-lived evidence to merge states.

This is a mechanism-isolation experiment.  No image labels or object IDs enter
the dynamics after the two local islands have formed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


ADDRESS_THRESHOLD = 0.99
CONTACT_DURATIONS = (1, 2, 4, 8, 16, 32, 64, 128)


def normalize(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-30)


def local_step(
    x: np.ndarray,
    cross_weight: float,
    gamma: float,
) -> np.ndarray:
    """One batched projected unit-sphere synchronization update."""
    drive = np.zeros_like(x)

    # Persistent within-object edges.
    drive[:, 0] += x[:, 1]
    drive[:, 1] += x[:, 0]
    drive[:, 2] += x[:, 3]
    drive[:, 3] += x[:, 2]

    # Temporary contact between the two already-established objects.
    if cross_weight > 0.0:
        drive[:, 1] += cross_weight * x[:, 2]
        drive[:, 2] += cross_weight * x[:, 1]

    tangent = drive - np.sum(drive * x, axis=-1, keepdims=True) * x
    return normalize(x + gamma * tangent)


def instance_cosine(x: np.ndarray) -> np.ndarray:
    """Cosine between the two object-level addresses."""
    a = normalize(x[:, 0] + x[:, 1])
    b = normalize(x[:, 2] + x[:, 3])
    return np.sum(a * b, axis=-1)


def internal_edge_cosines(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.sum(x[:, 0] * x[:, 1], axis=-1),
        np.sum(x[:, 2] * x[:, 3], axis=-1),
    )


def prepare_addresses(
    trials: int,
    dim: int,
    seed: int,
    gamma: float,
    pre_steps: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = normalize(rng.normal(size=(trials, 4, dim)))
    for _ in range(pre_steps):
        x = local_step(x, cross_weight=0.0, gamma=gamma)
    return x


def summarize_cosine(cosine: np.ndarray) -> dict:
    return {
        "median_cosine": float(np.median(cosine)),
        "p95_cosine": float(np.quantile(cosine, 0.95)),
        "p99_cosine": float(np.quantile(cosine, 0.99)),
        "maximum_cosine": float(np.max(cosine)),
        "collision_fraction": float(
            np.mean(cosine >= ADDRESS_THRESHOLD)
        ),
    }


def run_sweep(
    trials: int,
    dim: int,
    tau: float,
    gamma: float,
    pre_steps: int,
    post_separation_steps: int,
    seed: int,
) -> dict:
    base = prepare_addresses(
        trials=trials,
        dim=dim,
        seed=seed,
        gamma=gamma,
        pre_steps=pre_steps,
    )

    edge_a, edge_b = internal_edge_cosines(base)
    pre_cosine = instance_cosine(base)

    instantaneous = base.copy()
    gated = base.copy()
    gated_weight = 0.0

    checkpoint_states = {}
    sweep = {}

    max_duration = max(CONTACT_DURATIONS)
    for step in range(1, max_duration + 1):
        instantaneous = local_step(
            instantaneous,
            cross_weight=1.0,
            gamma=gamma,
        )

        gated_weight += (1.0 - gated_weight) / tau
        gated = local_step(
            gated,
            cross_weight=gated_weight,
            gamma=gamma,
        )

        if step in CONTACT_DURATIONS:
            sweep[str(step)] = {
                "contact_steps": step,
                "contact_fraction_of_tau": float(step / tau),
                "persistence_gated_edge_weight": float(gated_weight),
                "instantaneous": summarize_cosine(
                    instance_cosine(instantaneous)
                ),
                "persistence_gated": summarize_cosine(
                    instance_cosine(gated)
                ),
            }

        if step == 32:
            checkpoint_states["instantaneous_32"] = instantaneous.copy()
            checkpoint_states["gated_32"] = gated.copy()

    # Remove contact entirely and ask whether the damage / protection persists.
    post_instantaneous = checkpoint_states["instantaneous_32"]
    post_gated = checkpoint_states["gated_32"]
    for _ in range(post_separation_steps):
        post_instantaneous = local_step(
            post_instantaneous,
            cross_weight=0.0,
            gamma=gamma,
        )
        post_gated = local_step(
            post_gated,
            cross_weight=0.0,
            gamma=gamma,
        )

    return {
        "trials": trials,
        "dimension": dim,
        "address_collision_cosine_threshold": ADDRESS_THRESHOLD,
        "gamma": gamma,
        "persistence_tau": tau,
        "pre_sync_steps": pre_steps,
        "post_separation_steps": post_separation_steps,
        "pre_contact": {
            "minimum_internal_edge_cosine": float(
                min(np.min(edge_a), np.min(edge_b))
            ),
            **summarize_cosine(pre_cosine),
        },
        "duration_sweep": sweep,
        "after_32_step_contact_then_separation": {
            "instantaneous": summarize_cosine(
                instance_cosine(post_instantaneous)
            ),
            "persistence_gated": summarize_cosine(
                instance_cosine(post_gated)
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=5000)
    parser.add_argument("--dim", type=int, default=8)
    parser.add_argument("--tau", type=float, default=64.0)
    parser.add_argument("--gamma", type=float, default=0.20)
    parser.add_argument("--pre-steps", type=int, default=240)
    parser.add_argument("--post-separation-steps", type=int, default=120)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--json", type=str, default=None)
    args = parser.parse_args()

    report = run_sweep(
        trials=args.trials,
        dim=args.dim,
        tau=args.tau,
        gamma=args.gamma,
        pre_steps=args.pre_steps,
        post_separation_steps=args.post_separation_steps,
        seed=args.seed,
    )

    report["interpretation"] = (
        "Gate 10's address is vulnerable when two previously independent "
        "instances become local same-motion neighbours. Instantaneous coupling "
        "collapses those addresses after a moderate contact. A slowly accrued "
        "new-edge eligibility suppresses transient identity loss, but deliberately "
        "does not make merging impossible: sufficiently persistent contact "
        "eventually wins. Identity therefore needs a relation timescale, not only "
        "address dimensionality."
    )

    print(json.dumps(report, indent=2))

    pre = report["pre_contact"]
    d16 = report["duration_sweep"]["16"]
    d32 = report["duration_sweep"]["32"]
    d64 = report["duration_sweep"]["64"]
    post = report["after_32_step_contact_then_separation"]

    assert pre["minimum_internal_edge_cosine"] > 0.999
    assert pre["collision_fraction"] < 0.01

    # Brief/moderate contact attacks the instantaneous rule.
    assert (
        d16["instantaneous"]["collision_fraction"]
        > d16["persistence_gated"]["collision_fraction"]
    )
    assert d32["instantaneous"]["collision_fraction"] > 0.95
    assert d32["persistence_gated"]["collision_fraction"] < 0.01

    # Long contact is allowed to become strong evidence.
    assert d64["persistence_gated"]["collision_fraction"] > 0.95

    # With no repulsive mechanism, a collapsed address does not spontaneously
    # separate again merely because contact ended.
    assert post["instantaneous"]["collision_fraction"] > 0.95
    assert post["persistence_gated"]["collision_fraction"] < 0.01

    if args.json:
        Path(args.json).write_text(
            json.dumps(report, indent=2) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
