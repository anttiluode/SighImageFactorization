#!/usr/bin/env python3
"""Gate 17: false-clean identical-prefix attacker.

Gate 16 established that local relation authority can retreat during ambiguous
tracking and later recover. Validation weakly preferred separate attack/recovery
timescales, but the held-out test could not distinguish symmetric from
asymmetric eligibility.

This gate attacks the remaining question at the causal boundary.

After established clean evidence and an ambiguous interval, a relation receives
a short clean-looking streak. We pair two worlds with the *exact same observed
RGB-derived evidence prefix*:

    stable recovery:
        the clean streak continues

    false-clean glitch:
        ambiguity returns on the next observation

At the split point the entire evidence history is identical. Therefore any
deterministic causal policy whose state is a function only of that history must
also be identical in the paired worlds. No extra EMA timescale, hysteresis
parameter, or nonlinear state update can classify the two futures before new
information arrives.

The experiment does three things:

1. build paired noisy-RGB histories with randomized ambiguity and clean-streak
   lengths, sharing the prefix exactly;
2. verify raw, cumulative, symmetric, and asymmetric state families all have
   chance balanced accuracy for "stable recovery vs relapse" at the split;
3. show that the first post-split RGB observation restores information and can
   classify the futures well.

This is an observability gate, not a claim that uncertainty memory is useless.
It identifies what temporal memory alone cannot buy.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from gate16_recoverable_uncertainty import (
    clean_observation,
    asymmetric_trace,
    cumulative_trace,
    symmetric_trace,
)
from gate15_track_confidence_admission import challenged_observation


PRE_CLEAR = 8
ATTACK_LENGTHS = (2, 3, 4, 5)
FALSE_CLEAN_LENGTHS = (1, 2, 3, 4)
FUTURE_OBSERVATIONS = 4

SYMMETRIC_TAU = 1.5
SYMMETRIC_THRESHOLD = 0.90
ASYMMETRIC_ATTACK_TAU = 1.5
ASYMMETRIC_RECOVERY_TAU = 6.0
ASYMMETRIC_THRESHOLD = 0.75


def observation_target(
    kind: str,
    rng: np.random.Generator,
    seed: int,
) -> tuple[float, float, float]:
    """Return authority target, confidence, and exact motion receipt."""
    if kind == "clear":
        residual, confidence, exact = clean_observation(
            1,
            rng,
            seed,
        )
    elif kind == "ambiguous":
        residual, _oracle, confidence, exact = challenged_observation(
            1,
            rng,
            seed,
        )
    else:
        raise ValueError(f"unknown observation kind: {kind}")

    target = confidence if residual <= 0.0 else 0.0
    return float(target), float(confidence), float(exact)


def make_pair(seed: int) -> dict:
    """Create one exact-prefix stable/relapse pair from RGB observations."""
    rng = np.random.default_rng(seed)
    attack_len = int(rng.choice(ATTACK_LENGTHS))
    false_clean_len = int(rng.choice(FALSE_CLEAN_LENGTHS))

    shared = []
    shared_confidence = []
    shared_exact = []

    step = 0
    for _ in range(PRE_CLEAR):
        target, confidence, exact = observation_target(
            "clear",
            rng,
            seed + 10_000 * (step + 1),
        )
        shared.append(target)
        shared_confidence.append(confidence)
        shared_exact.append(exact)
        step += 1

    for _ in range(attack_len):
        target, confidence, exact = observation_target(
            "ambiguous",
            rng,
            seed + 10_000 * (step + 1),
        )
        shared.append(target)
        shared_confidence.append(confidence)
        shared_exact.append(exact)
        step += 1

    for _ in range(false_clean_len):
        target, confidence, exact = observation_target(
            "clear",
            rng,
            seed + 10_000 * (step + 1),
        )
        shared.append(target)
        shared_confidence.append(confidence)
        shared_exact.append(exact)
        step += 1

    stable_rng = np.random.default_rng(seed + 700_001)
    relapse_rng = np.random.default_rng(seed + 900_001)

    stable_future = []
    relapse_future = []
    stable_confidence = []
    relapse_confidence = []

    for j in range(FUTURE_OBSERVATIONS):
        target, confidence, _ = observation_target(
            "clear",
            stable_rng,
            seed + 300_000 + 10_000 * j,
        )
        stable_future.append(target)
        stable_confidence.append(confidence)

        target, confidence, _ = observation_target(
            "ambiguous",
            relapse_rng,
            seed + 500_000 + 10_000 * j,
        )
        relapse_future.append(target)
        relapse_confidence.append(confidence)

    shared = np.asarray(shared, dtype=np.float64)
    stable = np.concatenate(
        [shared, np.asarray(stable_future, dtype=np.float64)]
    )
    relapse = np.concatenate(
        [shared, np.asarray(relapse_future, dtype=np.float64)]
    )

    return {
        "attack_length": attack_len,
        "false_clean_length": false_clean_len,
        "split_index": len(shared) - 1,
        "shared_target": shared,
        "stable_target": stable,
        "relapse_target": relapse,
        "shared_confidence": np.asarray(
            shared_confidence,
            dtype=np.float64,
        ),
        "shared_motion_exact": np.asarray(
            shared_exact,
            dtype=np.float64,
        ),
        "stable_future_confidence": np.asarray(
            stable_confidence,
            dtype=np.float64,
        ),
        "relapse_future_confidence": np.asarray(
            relapse_confidence,
            dtype=np.float64,
        ),
    }


def state_at_split(
    target: np.ndarray,
    family: str,
) -> float:
    """Evaluate representative Gate-16 temporal states at the split."""
    row = np.asarray(target, dtype=np.float64)[None, :]
    if family == "raw":
        return float(row[0, -1])
    if family == "cumulative_mean":
        return float(cumulative_trace(row)[0, -1])
    if family == "symmetric_ema":
        return float(
            symmetric_trace(row, SYMMETRIC_TAU)[0, -1]
        )
    if family == "asymmetric_eligibility":
        return float(
            asymmetric_trace(
                row,
                ASYMMETRIC_ATTACK_TAU,
                ASYMMETRIC_RECOVERY_TAU,
            )[0, -1]
        )
    raise ValueError(f"unknown family: {family}")


def paired_split_dataset(
    pairs: list[dict],
    family: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Duplicate each identical state with opposite future labels."""
    stable_state = np.array(
        [
            state_at_split(
                pair["stable_target"][: pair["split_index"] + 1],
                family,
            )
            for pair in pairs
        ],
        dtype=np.float64,
    )
    relapse_state = np.array(
        [
            state_at_split(
                pair["relapse_target"][: pair["split_index"] + 1],
                family,
            )
            for pair in pairs
        ],
        dtype=np.float64,
    )
    states = np.concatenate([stable_state, relapse_state])
    labels = np.concatenate(
        [
            np.ones(len(pairs), dtype=np.int64),
            np.zeros(len(pairs), dtype=np.int64),
        ]
    )
    return states, labels, stable_state - relapse_state


def balanced_accuracy(
    labels: np.ndarray,
    prediction: np.ndarray,
) -> float:
    positive = labels == 1
    negative = labels == 0
    tpr = float(np.mean(prediction[positive]))
    tnr = float(np.mean(~prediction[negative]))
    return 0.5 * (tpr + tnr)


def best_high_threshold_classifier(
    values: np.ndarray,
    labels: np.ndarray,
) -> dict[str, float]:
    """Fit stable iff value >= threshold by balanced accuracy."""
    ordered = np.sort(np.unique(values))
    candidates = np.concatenate(
        [
            np.array([-np.inf]),
            ordered,
            np.array([np.inf]),
        ]
    )

    best = None
    for threshold in candidates:
        prediction = values >= threshold
        accuracy = balanced_accuracy(labels, prediction)
        key = (
            -accuracy,
            -float(threshold)
            if np.isfinite(threshold)
            else np.inf,
        )
        if best is None or key < best[0]:
            best = (key, float(threshold), accuracy)
    assert best is not None
    return {
        "threshold": best[1],
        "balanced_accuracy": best[2],
    }


def one_step_future_dataset(
    pairs: list[dict],
) -> tuple[np.ndarray, np.ndarray]:
    """First observation that differs between the two paired futures."""
    stable = np.array(
        [
            pair["stable_target"][pair["split_index"] + 1]
            for pair in pairs
        ],
        dtype=np.float64,
    )
    relapse = np.array(
        [
            pair["relapse_target"][pair["split_index"] + 1]
            for pair in pairs
        ],
        dtype=np.float64,
    )
    values = np.concatenate([stable, relapse])
    labels = np.concatenate(
        [
            np.ones(len(pairs), dtype=np.int64),
            np.zeros(len(pairs), dtype=np.int64),
        ]
    )
    return values, labels


def family_split_report(
    pairs: list[dict],
    family: str,
) -> dict:
    values, labels, differences = paired_split_dataset(
        pairs,
        family,
    )
    classifier = best_high_threshold_classifier(
        values,
        labels,
    )

    return {
        "max_paired_state_difference": float(
            np.max(np.abs(differences))
        ),
        "median_split_state": float(np.median(values)),
        "best_threshold": classifier["threshold"],
        "best_balanced_accuracy": classifier["balanced_accuracy"],
    }


def fast_fraction_by_clean_streak(
    pairs: list[dict],
) -> dict:
    """How often Gate-16 policies have already re-authorized at the split."""
    out = {}
    for family, threshold in (
        ("symmetric_ema", SYMMETRIC_THRESHOLD),
        ("asymmetric_eligibility", ASYMMETRIC_THRESHOLD),
    ):
        by_length = {}
        for length in FALSE_CLEAN_LENGTHS:
            selected = [
                pair
                for pair in pairs
                if pair["false_clean_length"] == length
            ]
            states = np.array(
                [
                    state_at_split(
                        pair["shared_target"],
                        family,
                    )
                    for pair in selected
                ],
                dtype=np.float64,
            )
            by_length[str(length)] = {
                "pairs": len(selected),
                "fast_fraction": float(
                    np.mean(states >= threshold)
                )
                if len(states)
                else None,
                "median_authority": float(np.median(states))
                if len(states)
                else None,
            }
        out[family] = by_length
    return out


def make_pairs(count: int, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    seeds = rng.integers(
        0,
        2**31 - 1,
        size=count,
        dtype=np.int64,
    )
    return [make_pair(int(item)) for item in seeds]


def run_reference(
    train_pairs: int,
    test_pairs: int,
    train_seed: int,
    test_seed: int,
) -> dict:
    train = make_pairs(train_pairs, train_seed)
    test = make_pairs(test_pairs, test_seed)

    families = (
        "raw",
        "cumulative_mean",
        "symmetric_ema",
        "asymmetric_eligibility",
    )

    split = {
        family: family_split_report(test, family)
        for family in families
    }

    train_future_values, train_future_labels = one_step_future_dataset(
        train
    )
    learned_future_classifier = best_high_threshold_classifier(
        train_future_values,
        train_future_labels,
    )
    test_future_values, test_future_labels = one_step_future_dataset(
        test
    )
    test_future_prediction = (
        test_future_values
        >= learned_future_classifier["threshold"]
    )
    post_split_accuracy = balanced_accuracy(
        test_future_labels,
        test_future_prediction,
    )

    prefix_max_difference = max(
        float(
            np.max(
                np.abs(
                    pair["stable_target"][: pair["split_index"] + 1]
                    - pair["relapse_target"][: pair["split_index"] + 1]
                )
            )
        )
        for pair in test
    )

    return {
        "train_pairs": train_pairs,
        "test_pairs": test_pairs,
        "pre_clear_observations": PRE_CLEAR,
        "attack_lengths": list(ATTACK_LENGTHS),
        "false_clean_lengths": list(FALSE_CLEAN_LENGTHS),
        "future_observations": FUTURE_OBSERVATIONS,
        "prefix_identity_receipt": {
            "max_target_prefix_difference": prefix_max_difference,
            "paired_future_labels_per_prefix": [
                "stable_recovery",
                "relapse_to_ambiguity",
            ],
        },
        "split_point_family_results": split,
        "gate16_policy_reauthorization_at_split": (
            fast_fraction_by_clean_streak(test)
        ),
        "first_post_split_observation": {
            "learned_threshold": learned_future_classifier["threshold"],
            "training_balanced_accuracy": learned_future_classifier[
                "balanced_accuracy"
            ],
            "held_out_balanced_accuracy": post_split_accuracy,
            "median_stable_target": float(
                np.median(
                    test_future_values[
                        test_future_labels == 1
                    ]
                )
            ),
            "median_relapse_target": float(
                np.median(
                    test_future_values[
                        test_future_labels == 0
                    ]
                )
            ),
        },
        "interpretation": (
            "Stable recovery and false-clean relapse worlds are paired with an "
            "exactly identical RGB-derived evidence prefix. Every deterministic "
            "causal state computed from that prefix is therefore identical in "
            "the pair, so future-type classification at the split is bounded at "
            "chance. Symmetric or asymmetric eligibility can choose a different "
            "speed/safety trade-off, but extra timescales cannot create missing "
            "information. The first post-split RGB observation breaks the "
            "identity and restores predictive information."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-pairs", type=int, default=80)
    parser.add_argument("--test-pairs", type=int, default=160)
    parser.add_argument("--train-seed", type=int, default=1701)
    parser.add_argument("--test-seed", type=int, default=1702)
    parser.add_argument("--json", type=str, default=None)
    args = parser.parse_args()

    report = run_reference(
        train_pairs=args.train_pairs,
        test_pairs=args.test_pairs,
        train_seed=args.train_seed,
        test_seed=args.test_seed,
    )
    print(json.dumps(report, indent=2))

    assert (
        report["prefix_identity_receipt"][
            "max_target_prefix_difference"
        ]
        == 0.0
    )
    for family, result in report[
        "split_point_family_results"
    ].items():
        assert result["max_paired_state_difference"] == 0.0, family
        assert abs(result["best_balanced_accuracy"] - 0.5) < 1e-12, family

    assert (
        report["first_post_split_observation"][
            "held_out_balanced_accuracy"
        ]
        > 0.85
    )

    if args.json:
        Path(args.json).write_text(
            json.dumps(report, indent=2) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
