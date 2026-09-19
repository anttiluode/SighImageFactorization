#!/usr/bin/env python3
"""Gate 18: motion-model surprise supplies information Gate 17 discarded.

Gate 17 proved that no temporal state can predict two futures from an exactly
identical evidence prefix. To escape that boundary we must add information that
is available before ordinary relation confidence diverges.

The admission evidence used by Gates 14-17 asks essentially:

    * did both regions move together?
    * was that correspondence confident?

It discards the *absolute shared motion vector* once common fate is established.

Gate 18 keeps the ordinary admission bit deliberately matched while exposing one
additional local measurement: short-horizon motion-model surprise.

Every pre-relapse RGB observation contains two foreground regions moving with the
same high-confidence translation. Thus ordinary common-fate admission is TRUE for
every accepted trial in both future classes.

The classes differ only statistically in the shared trajectory:

    stable future:
        shared direction usually persists

    future relapse:
        shared direction becomes unstable before confidence/common-fate fails

All directions have equal speed, so the cue is not a speed shortcut.

The predictor receives only RGB-estimated translations and computes

    surprise = mean_t ||vhat_t - vhat_(t-1)||^2

where vhat_t is the mean of the two candidate translation estimates.

Training learns one surprise threshold. Test controls include:
    * ordinary matched admission bit
    * oracle-motion surprise upper bound
    * RGB-estimated surprise
    * shuffled surprise
    * always-fast and always-cautious clocks

This is a controlled predictive-cue gate, not a natural-video claim. The question
is whether an observable discarded by the previous relation representation can
escape Gate 17's information boundary before the ordinary confidence trace
changes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from gate11_contact_persistence import prepare_addresses
from gate13_predictive_relation_admission import (
    CAUTIOUS_TAU,
    FAST_TAU,
    balanced_error,
    evaluate_policy,
    learn_residual_threshold,
)
from gate14_rgb_predictive_relation_admission import (
    render_frame,
    make_static_textures,
    estimate_pair_motion,
)


HISTORY_STEPS = 5
CONTACT_STEPS = 32
ORDINARY_MIN_CONFIDENCE = 0.95

# Same-speed orthogonal translations. Direction changes cannot be detected from
# speed magnitude alone.
DIRECTIONS = np.array(
    [
        [0, 1],
        [1, 0],
    ],
    dtype=np.int64,
)

STABLE_SWITCH_PROB = 0.04
RELAPSE_SWITCH_PROB = 0.65
RELAPSE_UNSTABLE_START = 2
MAX_REJECTION_ATTEMPTS = 200


def choose_motion_history(
    label: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate equal-speed shared motion with class-dependent directional stability."""
    motion = np.empty((HISTORY_STEPS, 2), dtype=np.int64)
    direction = int(rng.integers(0, len(DIRECTIONS)))
    motion[0] = DIRECTIONS[direction]

    for step in range(1, HISTORY_STEPS):
        if label == 1:
            switch_probability = STABLE_SWITCH_PROB
        else:
            switch_probability = (
                STABLE_SWITCH_PROB
                if step < RELAPSE_UNSTABLE_START
                else RELAPSE_SWITCH_PROB
            )

        if rng.random() < switch_probability:
            direction = 1 - direction
        motion[step] = DIRECTIONS[direction]

    return motion


def estimate_motion_probe(
    motion: np.ndarray,
    textures: tuple[np.ndarray, np.ndarray, np.ndarray],
    seed: int,
) -> tuple[np.ndarray, float, float, bool]:
    """Render one RGB frame pair and recover its shared candidate translation."""
    pos_a = np.array([1, 1], dtype=np.int64)
    pos_b = np.array([7, 7], dtype=np.int64)

    frame0 = render_frame(
        tuple(pos_a),
        tuple(pos_b),
        textures,
        noise_seed=seed + 100_000,
    )
    frame1 = render_frame(
        tuple(pos_a + motion),
        tuple(pos_b + motion),
        textures,
        noise_seed=seed + 200_000,
    )

    va, vb, confidence = estimate_pair_motion(frame0, frame1)
    residual = float(np.sum((va - vb) ** 2))
    ordinary = (
        residual <= 0.0
        and confidence >= ORDINARY_MIN_CONFIDENCE
    )
    shared = 0.5 * (va + vb)
    exact = bool(
        np.array_equal(va, motion)
        and np.array_equal(vb, motion)
    )
    return shared, residual, float(confidence), exact


def trajectory_surprise(motion: np.ndarray) -> float:
    delta = np.diff(
        np.asarray(motion, dtype=np.float64),
        axis=0,
    )
    return float(np.mean(np.sum(delta * delta, axis=1)))


def make_trial(
    label: int,
    seed: int,
) -> dict | None:
    """Return one RGB prefix only if ordinary admission is matched throughout."""
    rng = np.random.default_rng(seed)
    true_motion = choose_motion_history(label, rng)
    textures = make_static_textures(seed + 300_000)

    estimated = np.empty_like(true_motion, dtype=np.float64)
    confidences = np.empty(HISTORY_STEPS, dtype=np.float64)
    residuals = np.empty(HISTORY_STEPS, dtype=np.float64)
    exact = np.empty(HISTORY_STEPS, dtype=np.float64)
    ordinary = np.empty(HISTORY_STEPS, dtype=bool)

    for step in range(HISTORY_STEPS):
        shared, residual, confidence, is_exact = estimate_motion_probe(
            true_motion[step],
            textures,
            seed + 10_000 * (step + 1),
        )
        estimated[step] = shared
        residuals[step] = residual
        confidences[step] = confidence
        exact[step] = float(is_exact)
        ordinary[step] = (
            residual <= 0.0
            and confidence >= ORDINARY_MIN_CONFIDENCE
        )

    # The matched-evidence attacker conditions on the exact decision-level input
    # consumed by the old admission rule: common fate + high confidence at every
    # observation. Trials that do not meet that condition are not part of this
    # boundary test.
    if not bool(np.all(ordinary)):
        return None

    return {
        "label": int(label),
        "rgb_surprise": trajectory_surprise(estimated),
        "oracle_surprise": trajectory_surprise(true_motion),
        "minimum_confidence": float(np.min(confidences)),
        "mean_confidence": float(np.mean(confidences)),
        "mean_motion_accuracy": float(np.mean(exact)),
        "max_common_fate_residual": float(np.max(residuals)),
        "ordinary_bit": True,
        "true_motion": true_motion,
        "estimated_motion": estimated,
    }


def generate_dataset(
    trials: int,
    seed: int,
) -> dict[str, np.ndarray]:
    """Balanced accepted trials with an exactly matched ordinary admission bit."""
    if trials % 2:
        raise ValueError("trials must be even")

    rng = np.random.default_rng(seed)
    per_class = trials // 2
    rows = []

    for label in (0, 1):
        accepted = 0
        attempts = 0
        while accepted < per_class:
            attempts += 1
            if attempts > per_class * MAX_REJECTION_ATTEMPTS:
                raise RuntimeError(
                    "could not generate enough matched-admission RGB trials"
                )
            trial_seed = int(
                rng.integers(0, 2**31 - 1)
            )
            row = make_trial(label, trial_seed)
            if row is None:
                continue
            rows.append(row)
            accepted += 1

    rng.shuffle(rows)

    return {
        "labels": np.asarray(
            [row["label"] for row in rows],
            dtype=np.int64,
        ),
        "rgb_surprise": np.asarray(
            [row["rgb_surprise"] for row in rows],
            dtype=np.float64,
        ),
        "oracle_surprise": np.asarray(
            [row["oracle_surprise"] for row in rows],
            dtype=np.float64,
        ),
        "minimum_confidence": np.asarray(
            [row["minimum_confidence"] for row in rows],
            dtype=np.float64,
        ),
        "mean_confidence": np.asarray(
            [row["mean_confidence"] for row in rows],
            dtype=np.float64,
        ),
        "mean_motion_accuracy": np.asarray(
            [row["mean_motion_accuracy"] for row in rows],
            dtype=np.float64,
        ),
        "max_common_fate_residual": np.asarray(
            [row["max_common_fate_residual"] for row in rows],
            dtype=np.float64,
        ),
        "ordinary_bit": np.asarray(
            [row["ordinary_bit"] for row in rows],
            dtype=bool,
        ),
    }


def cue_metrics(
    labels: np.ndarray,
    prediction: np.ndarray,
) -> dict[str, float]:
    error, fnr, fpr = balanced_error(
        labels,
        prediction,
    )
    return {
        "accuracy": float(
            np.mean(prediction == (labels == 1))
        ),
        "balanced_error": error,
        "false_negative_rate": fnr,
        "false_positive_rate": fpr,
        "stable_prediction_rate": float(
            np.mean(prediction[labels == 1])
        ),
        "relapse_prediction_rate": float(
            np.mean(prediction[labels == 0])
        ),
    }


def run_reference(
    train_trials: int,
    test_trials: int,
    dim: int,
    gamma: float,
    pre_steps: int,
    train_seed: int,
    test_seed: int,
    shuffle_seed: int,
    address_seed: int,
) -> dict:
    train = generate_dataset(
        train_trials,
        train_seed,
    )
    rgb_threshold, rgb_training = learn_residual_threshold(
        train["rgb_surprise"],
        train["labels"],
    )
    oracle_threshold, oracle_training = learn_residual_threshold(
        train["oracle_surprise"],
        train["labels"],
    )

    test = generate_dataset(
        test_trials,
        test_seed,
    )
    labels = test["labels"]

    ordinary_prediction = test["ordinary_bit"].copy()
    rgb_prediction = (
        test["rgb_surprise"] <= rgb_threshold
    )
    oracle_prediction = (
        test["oracle_surprise"] <= oracle_threshold
    )

    rng = np.random.default_rng(shuffle_seed)
    shuffled_surprise = test["rgb_surprise"][
        rng.permutation(test_trials)
    ]
    shuffled_prediction = (
        shuffled_surprise <= rgb_threshold
    )

    base = prepare_addresses(
        trials=test_trials,
        dim=dim,
        seed=address_seed,
        gamma=gamma,
        pre_steps=pre_steps,
    )

    ordinary_policy = evaluate_policy(
        base,
        labels,
        np.where(
            ordinary_prediction,
            FAST_TAU,
            CAUTIOUS_TAU,
        ),
        gamma=gamma,
        contact_steps=CONTACT_STEPS,
    )
    predictive_policy = evaluate_policy(
        base,
        labels,
        np.where(
            rgb_prediction,
            FAST_TAU,
            CAUTIOUS_TAU,
        ),
        gamma=gamma,
        contact_steps=CONTACT_STEPS,
    )
    shuffled_policy = evaluate_policy(
        base,
        labels,
        np.where(
            shuffled_prediction,
            FAST_TAU,
            CAUTIOUS_TAU,
        ),
        gamma=gamma,
        contact_steps=CONTACT_STEPS,
    )
    cautious = evaluate_policy(
        base,
        labels,
        np.full(test_trials, CAUTIOUS_TAU),
        gamma=gamma,
        contact_steps=CONTACT_STEPS,
    )

    return {
        "train_trials": train_trials,
        "test_trials": test_trials,
        "dimension": dim,
        "history_steps": HISTORY_STEPS,
        "stable_switch_probability": STABLE_SWITCH_PROB,
        "relapse_switch_probability": RELAPSE_SWITCH_PROB,
        "relapse_unstable_start": RELAPSE_UNSTABLE_START,
        "ordinary_min_confidence": ORDINARY_MIN_CONFIDENCE,
        "learned_rgb_surprise_threshold": rgb_threshold,
        "learned_oracle_surprise_threshold": oracle_threshold,
        "training": {
            "rgb_surprise": rgb_training,
            "oracle_surprise": oracle_training,
        },
        "matched_old_evidence_receipt": {
            "ordinary_bit_true_fraction": float(
                np.mean(test["ordinary_bit"])
            ),
            "ordinary_cue": cue_metrics(
                labels,
                ordinary_prediction,
            ),
            "minimum_confidence": float(
                np.min(test["minimum_confidence"])
            ),
            "median_minimum_confidence": float(
                np.median(test["minimum_confidence"])
            ),
            "max_common_fate_residual": float(
                np.max(test["max_common_fate_residual"])
            ),
        },
        "tracker_receipt": {
            "mean_motion_accuracy": float(
                np.mean(test["mean_motion_accuracy"])
            ),
            "median_rgb_surprise_stable": float(
                np.median(
                    test["rgb_surprise"][labels == 1]
                )
            ),
            "median_rgb_surprise_relapse": float(
                np.median(
                    test["rgb_surprise"][labels == 0]
                )
            ),
            "median_oracle_surprise_stable": float(
                np.median(
                    test["oracle_surprise"][labels == 1]
                )
            ),
            "median_oracle_surprise_relapse": float(
                np.median(
                    test["oracle_surprise"][labels == 0]
                )
            ),
        },
        "held_out_cues": {
            "ordinary_matched_admission": cue_metrics(
                labels,
                ordinary_prediction,
            ),
            "rgb_motion_surprise": cue_metrics(
                labels,
                rgb_prediction,
            ),
            "oracle_motion_surprise": cue_metrics(
                labels,
                oracle_prediction,
            ),
            "shuffled_rgb_surprise": cue_metrics(
                labels,
                shuffled_prediction,
            ),
        },
        "policies": {
            "ordinary_matched_fast": ordinary_policy,
            "rgb_surprise_fast_or_cautious": predictive_policy,
            "shuffled_surprise": shuffled_policy,
            "global_cautious": cautious,
        },
        "interpretation": (
            "Conditioning on high-confidence common fate makes the previous "
            "admission evidence identical across stable and future-relapse "
            "classes, so it predicts stable for everyone and has balanced "
            "accuracy 0.5. The absolute shared-motion vector was still present "
            "in the RGB correspondence but had been discarded. Its short-horizon "
            "directional surprise predicts future relapse before ordinary "
            "confidence changes. Shuffling that surprise destroys the gain. "
            "This escapes Gate 17 by adding observability, not by adding more "
            "memory to the same evidence."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-trials", type=int, default=400)
    parser.add_argument("--test-trials", type=int, default=800)
    parser.add_argument("--dim", type=int, default=8)
    parser.add_argument("--gamma", type=float, default=0.20)
    parser.add_argument("--pre-steps", type=int, default=240)
    parser.add_argument("--train-seed", type=int, default=1801)
    parser.add_argument("--test-seed", type=int, default=1802)
    parser.add_argument("--shuffle-seed", type=int, default=1803)
    parser.add_argument("--address-seed", type=int, default=1804)
    parser.add_argument("--json", type=str, default=None)
    args = parser.parse_args()

    report = run_reference(
        train_trials=args.train_trials,
        test_trials=args.test_trials,
        dim=args.dim,
        gamma=args.gamma,
        pre_steps=args.pre_steps,
        train_seed=args.train_seed,
        test_seed=args.test_seed,
        shuffle_seed=args.shuffle_seed,
        address_seed=args.address_seed,
    )
    print(json.dumps(report, indent=2))

    ordinary = report["held_out_cues"][
        "ordinary_matched_admission"
    ]
    predictive = report["held_out_cues"][
        "rgb_motion_surprise"
    ]
    oracle = report["held_out_cues"][
        "oracle_motion_surprise"
    ]
    shuffled = report["held_out_cues"][
        "shuffled_rgb_surprise"
    ]

    assert abs(ordinary["balanced_error"] - 0.5) < 1e-12
    assert predictive["accuracy"] > 0.80
    assert predictive["balanced_error"] < 0.20
    assert oracle["accuracy"] >= predictive["accuracy"] - 0.05
    assert shuffled["accuracy"] < 0.60

    policy = report["policies"][
        "rgb_surprise_fast_or_cautious"
    ]
    ordinary_policy = report["policies"][
        "ordinary_matched_fast"
    ]
    assert (
        ordinary_policy["accidental_false_merge_fraction"]
        > 0.65
    )
    assert policy["genuine_merge_fraction"] > 0.55
    assert policy["accidental_false_merge_fraction"] < 0.12
    assert policy["balanced_relation_accuracy"] > 0.72

    if args.json:
        Path(args.json).write_text(
            json.dumps(report, indent=2) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
