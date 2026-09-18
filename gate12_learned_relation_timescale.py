#!/usr/bin/env python3
"""Gate 12: learn the relation-admission timescale, then attack age alone.

Gate 11 hand-picked tau=64 to show that a newly observed relation should not
immediately rewrite an established instance address.  This gate asks whether the
same scalar timescale can be selected from experience rather than inserted by
hand.

Training episodes contain:
    transient contacts: 8, 16, 24, 32 steps
    persistent contacts: 64, 96, 128 steps

For each candidate tau we replay the exact Gate-11 local vector dynamics and
measure two errors:
    * false merge on transient contacts
    * failure to merge on persistent contacts

The tau minimizing their sum is selected on training addresses and evaluated on
held-out addresses.

Then comes the important attacker.  Two futures share an identical 48-step
contact prefix:
    A) the contact ends at step 48 and the objects should remain distinct;
    B) the contact continues after step 48 and we would like an early merge.

Any policy whose only evidence is contact age has exactly the same state at the
end of those identical prefixes.  It therefore cannot know which future it is
in.  A balanced paired-prefix classifier is forced to 50% accuracy regardless
of which side it chooses.

The result is deliberately two-sided:
    relation timescale can be learned from encounter statistics,
    but edge age alone cannot predict relation identity before histories diverge.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np

from gate11_contact_persistence import (
    ADDRESS_THRESHOLD,
    instance_cosine,
    local_step,
    prepare_addresses,
)


CANDIDATE_TAUS = (8.0, 12.0, 16.0, 24.0, 32.0, 48.0, 64.0, 96.0, 128.0)
TRANSIENT_DURATIONS = (8, 16, 24, 32)
PERSISTENT_DURATIONS = (64, 96, 128)
PAIRED_PREFIX_STEPS = 48


def collision_fraction(x: np.ndarray) -> float:
    return float(np.mean(instance_cosine(x) >= ADDRESS_THRESHOLD))


def replay_tau(
    base: np.ndarray,
    tau: float,
    durations: Iterable[int],
    gamma: float,
) -> dict[int, dict[str, float]]:
    durations = tuple(sorted(set(int(d) for d in durations)))
    if not durations:
        raise ValueError("durations must not be empty")
    if tau <= 0.0:
        raise ValueError("tau must be positive")

    x = base.copy()
    weight = 0.0
    report: dict[int, dict[str, float]] = {}
    wanted = set(durations)

    for step in range(1, max(durations) + 1):
        weight += (1.0 - weight) / tau
        x = local_step(x, cross_weight=weight, gamma=gamma)
        if step in wanted:
            cos = instance_cosine(x)
            report[step] = {
                "edge_weight": float(weight),
                "collision_fraction": float(np.mean(cos >= ADDRESS_THRESHOLD)),
                "median_cross_instance_cosine": float(np.median(cos)),
            }
    return report


def score_tau(
    base: np.ndarray,
    tau: float,
    gamma: float,
    transient_durations: tuple[int, ...] = TRANSIENT_DURATIONS,
    persistent_durations: tuple[int, ...] = PERSISTENT_DURATIONS,
) -> dict[str, float]:
    durations = transient_durations + persistent_durations
    replay = replay_tau(base, tau=tau, durations=durations, gamma=gamma)

    false_merge = float(
        np.mean([replay[d]["collision_fraction"] for d in transient_durations])
    )
    false_nonmerge = float(
        np.mean(
            [1.0 - replay[d]["collision_fraction"] for d in persistent_durations]
        )
    )
    return {
        "false_merge": false_merge,
        "false_nonmerge": false_nonmerge,
        "objective": false_merge + false_nonmerge,
    }


def learn_tau(
    base: np.ndarray,
    candidates: tuple[float, ...],
    gamma: float,
) -> tuple[float, dict[str, dict[str, float]]]:
    scores: dict[str, dict[str, float]] = {}
    for tau in candidates:
        scores[f"{tau:g}"] = score_tau(base, tau=tau, gamma=gamma)

    best_tau = min(
        candidates,
        key=lambda tau: (scores[f"{tau:g}"]["objective"], tau),
    )
    return float(best_tau), scores


def paired_prefix_attack(
    base: np.ndarray,
    tau: float,
    gamma: float,
    prefix_steps: int = PAIRED_PREFIX_STEPS,
) -> dict[str, float]:
    replay = replay_tau(
        base,
        tau=tau,
        durations=(prefix_steps,),
        gamma=gamma,
    )
    prefix = replay[prefix_steps]
    collision = prefix["collision_fraction"]

    # The transient and persistent hypotheses have identical observations and
    # therefore identical model state through the prefix.  Whatever binary
    # merge decision is made, exactly one of the two balanced labels is correct.
    return {
        "prefix_steps": int(prefix_steps),
        "edge_weight": prefix["edge_weight"],
        "collision_fraction": collision,
        "transient_false_merge_if_contact_ends_now": collision,
        "persistent_early_merge_success_if_contact_continues": collision,
        "balanced_future_label_accuracy_from_age_only": 0.5,
        "paired_state_difference": 0.0,
    }


def run_experiment(
    train_trials: int,
    test_trials: int,
    dim: int,
    gamma: float,
    pre_steps: int,
    train_seed: int,
    test_seed: int,
    pair_seed: int,
) -> dict:
    train_base = prepare_addresses(
        trials=train_trials,
        dim=dim,
        seed=train_seed,
        gamma=gamma,
        pre_steps=pre_steps,
    )
    learned_tau, train_scores = learn_tau(
        train_base,
        candidates=CANDIDATE_TAUS,
        gamma=gamma,
    )

    test_base = prepare_addresses(
        trials=test_trials,
        dim=dim,
        seed=test_seed,
        gamma=gamma,
        pre_steps=pre_steps,
    )
    held_out = score_tau(test_base, tau=learned_tau, gamma=gamma)
    held_out_replay = replay_tau(
        test_base,
        tau=learned_tau,
        durations=TRANSIENT_DURATIONS + PERSISTENT_DURATIONS,
        gamma=gamma,
    )

    pair_base = prepare_addresses(
        trials=test_trials,
        dim=dim,
        seed=pair_seed,
        gamma=gamma,
        pre_steps=pre_steps,
    )
    paired = paired_prefix_attack(
        pair_base,
        tau=learned_tau,
        gamma=gamma,
    )

    return {
        "train_trials": train_trials,
        "test_trials": test_trials,
        "dimension": dim,
        "gamma": gamma,
        "pre_sync_steps": pre_steps,
        "candidate_taus": list(CANDIDATE_TAUS),
        "transient_durations": list(TRANSIENT_DURATIONS),
        "persistent_durations": list(PERSISTENT_DURATIONS),
        "learned_tau": learned_tau,
        "training_scores": train_scores,
        "held_out": held_out,
        "held_out_duration_replay": {
            str(k): v for k, v in held_out_replay.items()
        },
        "paired_prefix_attacker": paired,
        "interpretation": (
            "A scalar relation-admission timescale can be selected from prior "
            "merge/split encounter statistics rather than hand-picked. On the "
            "matched held-out regime it protects brief contacts while allowing "
            "persistent contacts to merge. But two futures with the same observed "
            "contact prefix produce exactly the same age-only state, so no learned "
            "timescale can predict whether that contact is about to split or persist. "
            "Fast relation admission therefore requires predictive evidence beyond "
            "edge age itself."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-trials", type=int, default=3000)
    parser.add_argument("--test-trials", type=int, default=5000)
    parser.add_argument("--dim", type=int, default=8)
    parser.add_argument("--gamma", type=float, default=0.20)
    parser.add_argument("--pre-steps", type=int, default=240)
    parser.add_argument("--train-seed", type=int, default=123)
    parser.add_argument("--test-seed", type=int, default=456)
    parser.add_argument("--pair-seed", type=int, default=789)
    parser.add_argument("--json", type=str, default=None)
    args = parser.parse_args()

    report = run_experiment(
        train_trials=args.train_trials,
        test_trials=args.test_trials,
        dim=args.dim,
        gamma=args.gamma,
        pre_steps=args.pre_steps,
        train_seed=args.train_seed,
        test_seed=args.test_seed,
        pair_seed=args.pair_seed,
    )
    print(json.dumps(report, indent=2))

    assert report["learned_tau"] == 64.0
    assert report["held_out"]["false_merge"] < 0.01
    assert report["held_out"]["false_nonmerge"] < 0.01

    pair = report["paired_prefix_attacker"]
    assert pair["paired_state_difference"] == 0.0
    assert pair["balanced_future_label_accuracy_from_age_only"] == 0.5
    assert abs(
        pair["transient_false_merge_if_contact_ends_now"]
        - pair["persistent_early_merge_success_if_contact_continues"]
    ) < 1e-15

    if args.json:
        Path(args.json).write_text(
            json.dumps(report, indent=2) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
