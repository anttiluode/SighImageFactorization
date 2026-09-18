#!/usr/bin/env python3
"""Gate 13: predictive relation admission from pre-contact common fate.

Gate 12 proved a causal boundary: contact age cannot distinguish two futures
with identical observed prefixes. This gate adds one observable that exists
before contact and asks whether it can buy faster safe binding.

Each encounter has the same 32-step same-motion contact. The only class
information available before that contact is a short noisy motion history:

    genuine relation:
        two local regions share one latent velocity process + sensor noise

    accidental relation:
        the two regions have independent latent velocity processes + sensor noise

The learner sees only a scalar local statistic,

    mean_t ||v_a(t) - v_b(t)||^2,

and, during training only, the eventual relation outcome. A threshold is chosen
to minimize balanced classification error.

At test time the future outcome is hidden. The learned threshold chooses which
clock a new edge is allowed to use:

    predicted genuine    -> fast admission, tau=16
    predicted accidental -> cautious admission, tau=96

All contacts last exactly 32 steps. Therefore duration itself has zero class
information in this gate.

Controls:
  * Gate-12 age-only tau=64
  * always-fast tau=16
  * always-cautious tau=96
  * shuffled predictive evidence
  * higher sensor-noise distribution shift

This is still a mechanism-isolation experiment. The motion histories are
synthetic and the fast/slow clocks are fixed controls. The claim being tested
is narrower: a predictive pre-contact observable can route relation admission
better than age alone when contact duration is matched.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from gate11_contact_persistence import (
    ADDRESS_THRESHOLD,
    instance_cosine,
    prepare_addresses,
)


CONTACT_STEPS = 32
AGE_ONLY_TAU = 64.0
FAST_TAU = 16.0
CAUTIOUS_TAU = 96.0
HISTORY_STEPS = 12
TRAIN_SENSOR_NOISE = 0.15
SHIFT_SENSOR_NOISE = 0.20
LATENT_DRIFT = 0.18


def normalize(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-30)


def local_step_per_trial(
    x: np.ndarray,
    cross_weight: np.ndarray,
    gamma: float,
) -> np.ndarray:
    """Gate-11 local vector step with one temporary-edge weight per trial."""
    weight = np.asarray(cross_weight, dtype=np.float64).reshape(-1, 1)
    if len(weight) != len(x):
        raise ValueError("cross_weight must contain one value per trial")

    drive = np.zeros_like(x)
    drive[:, 0] += x[:, 1]
    drive[:, 1] += x[:, 0]
    drive[:, 2] += x[:, 3]
    drive[:, 3] += x[:, 2]
    drive[:, 1] += weight * x[:, 2]
    drive[:, 2] += weight * x[:, 1]

    tangent = drive - np.sum(drive * x, axis=-1, keepdims=True) * x
    return normalize(x + gamma * tangent)


def generate_motion_histories(
    trials: int,
    seed: int,
    history_steps: int = HISTORY_STEPS,
    sensor_noise: float = TRAIN_SENSOR_NOISE,
    latent_drift: float = LATENT_DRIFT,
) -> tuple[np.ndarray, np.ndarray]:
    """Return eventual relation labels and a pre-contact local residual statistic.

    label 1: the pair shares one latent AR-like velocity process.
    label 0: each side has an independent latent velocity process.

    Both classes later receive the exact same 32-step same-motion contact.
    """
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, 2, size=trials, dtype=np.int64)
    va = np.zeros((trials, history_steps, 2), dtype=np.float64)
    vb = np.zeros_like(va)

    for i in range(trials):
        if labels[i] == 1:
            latent = rng.normal(0.0, 0.5, size=2)
            for t in range(history_steps):
                latent = 0.85 * latent + rng.normal(
                    0.0, latent_drift, size=2
                )
                va[i, t] = latent + rng.normal(0.0, sensor_noise, size=2)
                vb[i, t] = latent + rng.normal(0.0, sensor_noise, size=2)
        else:
            latent_a = rng.normal(0.0, 0.5, size=2)
            latent_b = rng.normal(0.0, 0.5, size=2)
            for t in range(history_steps):
                latent_a = 0.85 * latent_a + rng.normal(
                    0.0, latent_drift, size=2
                )
                latent_b = 0.85 * latent_b + rng.normal(
                    0.0, latent_drift, size=2
                )
                va[i, t] = latent_a + rng.normal(
                    0.0, sensor_noise, size=2
                )
                vb[i, t] = latent_b + rng.normal(
                    0.0, sensor_noise, size=2
                )

    residual = np.mean(np.sum((va - vb) ** 2, axis=-1), axis=1)
    return labels, residual


def balanced_error(
    labels: np.ndarray,
    predictions: np.ndarray,
) -> tuple[float, float, float]:
    labels = np.asarray(labels, dtype=np.int64)
    predictions = np.asarray(predictions, dtype=bool)
    genuine = labels == 1
    accidental = labels == 0
    false_negative = float(np.mean(~predictions[genuine]))
    false_positive = float(np.mean(predictions[accidental]))
    return (
        0.5 * (false_negative + false_positive),
        false_negative,
        false_positive,
    )


def learn_residual_threshold(
    residual: np.ndarray,
    labels: np.ndarray,
) -> tuple[float, dict[str, float]]:
    """Learn genuine iff residual <= threshold from past encounter outcomes."""
    order = np.argsort(residual)
    ordered = residual[order]
    candidates = np.concatenate(
        (
            np.array([-np.inf]),
            0.5 * (ordered[:-1] + ordered[1:]),
            np.array([np.inf]),
        )
    )

    best = None
    for threshold in candidates:
        prediction = residual <= threshold
        error, fnr, fpr = balanced_error(labels, prediction)
        key = (
            error,
            abs(float(threshold)) if np.isfinite(threshold) else np.inf,
        )
        if best is None or key < best[0]:
            best = (
                key,
                float(threshold),
                {
                    "balanced_error": error,
                    "false_negative_rate": fnr,
                    "false_positive_rate": fpr,
                },
            )

    assert best is not None
    return best[1], best[2]


def replay_relation_policy(
    base: np.ndarray,
    tau_per_trial: np.ndarray,
    contact_steps: int,
    gamma: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Replay matched-duration contact with a per-trial admission timescale."""
    tau = np.asarray(tau_per_trial, dtype=np.float64)
    if tau.shape != (len(base),):
        raise ValueError("tau_per_trial must have shape (trials,)")
    if np.any(tau <= 0.0):
        raise ValueError("all admission timescales must be positive")

    x = base.copy()
    weight = np.zeros(len(base), dtype=np.float64)
    for _ in range(contact_steps):
        weight += (1.0 - weight) / tau
        x = local_step_per_trial(x, weight, gamma=gamma)
    return instance_cosine(x), weight


def relation_metrics(
    cosine: np.ndarray,
    labels: np.ndarray,
) -> dict[str, float]:
    merged = np.asarray(cosine) >= ADDRESS_THRESHOLD
    genuine = labels == 1
    accidental = labels == 0
    genuine_merge = float(np.mean(merged[genuine]))
    accidental_merge = float(np.mean(merged[accidental]))
    return {
        "genuine_merge_fraction": genuine_merge,
        "accidental_false_merge_fraction": accidental_merge,
        "balanced_relation_accuracy": 0.5
        * (genuine_merge + (1.0 - accidental_merge)),
        "overall_relation_accuracy": float(
            np.mean(merged == (labels == 1))
        ),
    }


def evaluate_policy(
    base: np.ndarray,
    labels: np.ndarray,
    tau_per_trial: np.ndarray,
    gamma: float,
    contact_steps: int = CONTACT_STEPS,
) -> dict:
    cosine, final_weight = replay_relation_policy(
        base,
        tau_per_trial=tau_per_trial,
        contact_steps=contact_steps,
        gamma=gamma,
    )
    report = relation_metrics(cosine, labels)
    report.update(
        {
            "median_final_edge_weight": float(np.median(final_weight)),
            "median_cross_instance_cosine": float(np.median(cosine)),
        }
    )
    return report


def run_reference(
    train_trials: int,
    test_trials: int,
    dim: int,
    gamma: float,
    pre_steps: int,
    train_seed: int,
    test_seed: int,
    address_seed: int,
    shuffle_seed: int,
    shift_seed: int,
    shift_address_seed: int,
) -> dict:
    train_labels, train_residual = generate_motion_histories(
        train_trials,
        seed=train_seed,
    )
    threshold, training = learn_residual_threshold(
        train_residual,
        train_labels,
    )

    test_labels, test_residual = generate_motion_histories(
        test_trials,
        seed=test_seed,
    )
    prediction = test_residual <= threshold
    cue_error, cue_fnr, cue_fpr = balanced_error(
        test_labels,
        prediction,
    )

    base = prepare_addresses(
        trials=test_trials,
        dim=dim,
        seed=address_seed,
        gamma=gamma,
        pre_steps=pre_steps,
    )

    age_only = evaluate_policy(
        base,
        test_labels,
        np.full(test_trials, AGE_ONLY_TAU),
        gamma,
    )
    always_fast = evaluate_policy(
        base,
        test_labels,
        np.full(test_trials, FAST_TAU),
        gamma,
    )
    always_cautious = evaluate_policy(
        base,
        test_labels,
        np.full(test_trials, CAUTIOUS_TAU),
        gamma,
    )
    predictive_tau = np.where(
        prediction,
        FAST_TAU,
        CAUTIOUS_TAU,
    )
    predictive = evaluate_policy(
        base,
        test_labels,
        predictive_tau,
        gamma,
    )

    rng = np.random.default_rng(shuffle_seed)
    shuffled_prediction = prediction[rng.permutation(test_trials)]
    shuffled_tau = np.where(
        shuffled_prediction,
        FAST_TAU,
        CAUTIOUS_TAU,
    )
    shuffled = evaluate_policy(
        base,
        test_labels,
        shuffled_tau,
        gamma,
    )

    shift_labels, shift_residual = generate_motion_histories(
        test_trials,
        seed=shift_seed,
        sensor_noise=SHIFT_SENSOR_NOISE,
    )
    shift_prediction = shift_residual <= threshold
    shift_error, shift_fnr, shift_fpr = balanced_error(
        shift_labels,
        shift_prediction,
    )
    shift_base = prepare_addresses(
        trials=test_trials,
        dim=dim,
        seed=shift_address_seed,
        gamma=gamma,
        pre_steps=pre_steps,
    )
    shifted = evaluate_policy(
        shift_base,
        shift_labels,
        np.where(
            shift_prediction,
            FAST_TAU,
            CAUTIOUS_TAU,
        ),
        gamma,
    )

    return {
        "train_trials": train_trials,
        "test_trials": test_trials,
        "dimension": dim,
        "contact_steps": CONTACT_STEPS,
        "history_steps": HISTORY_STEPS,
        "training_sensor_noise": TRAIN_SENSOR_NOISE,
        "shift_sensor_noise": SHIFT_SENSOR_NOISE,
        "age_only_tau": AGE_ONLY_TAU,
        "fast_tau": FAST_TAU,
        "cautious_tau": CAUTIOUS_TAU,
        "learned_residual_threshold": threshold,
        "training_cue": training,
        "held_out_cue": {
            "balanced_error": cue_error,
            "false_negative_rate": cue_fnr,
            "false_positive_rate": cue_fpr,
            "accuracy": float(
                np.mean(prediction == (test_labels == 1))
            ),
            "genuine_prediction_rate": float(
                np.mean(prediction[test_labels == 1])
            ),
            "accidental_prediction_rate": float(
                np.mean(prediction[test_labels == 0])
            ),
            "median_genuine_residual": float(
                np.median(test_residual[test_labels == 1])
            ),
            "median_accidental_residual": float(
                np.median(test_residual[test_labels == 0])
            ),
        },
        "policies": {
            "age_only_tau64": age_only,
            "always_fast_tau16": always_fast,
            "always_cautious_tau96": always_cautious,
            "predictive_fast_or_cautious": predictive,
            "shuffled_predictive_evidence": shuffled,
        },
        "noise_shift": {
            "cue_balanced_error": shift_error,
            "cue_false_negative_rate": shift_fnr,
            "cue_false_positive_rate": shift_fpr,
            "cue_accuracy": float(
                np.mean(shift_prediction == (shift_labels == 1))
            ),
            "policy": shifted,
        },
        "interpretation": (
            "With contact duration held fixed, age carries no class information. "
            "A threshold learned from noisy pre-contact common-fate history can "
            "route likely genuine relations onto a fast admission clock while "
            "keeping likely accidental contacts on a cautious clock. Shuffling "
            "the predictive evidence destroys the advantage, and higher sensor "
            "noise degrades it, showing that the gain comes from informative "
            "history rather than the two-timescale policy by itself."
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
    parser.add_argument("--address-seed", type=int, default=789)
    parser.add_argument("--shuffle-seed", type=int, default=999)
    parser.add_argument("--shift-seed", type=int, default=654)
    parser.add_argument("--shift-address-seed", type=int, default=987)
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
        address_seed=args.address_seed,
        shuffle_seed=args.shuffle_seed,
        shift_seed=args.shift_seed,
        shift_address_seed=args.shift_address_seed,
    )
    print(json.dumps(report, indent=2))

    cue = report["held_out_cue"]
    age = report["policies"]["age_only_tau64"]
    fast = report["policies"]["always_fast_tau16"]
    predictive = report["policies"]["predictive_fast_or_cautious"]
    shuffled = report["policies"]["shuffled_predictive_evidence"]
    shifted = report["noise_shift"]["policy"]

    assert cue["accuracy"] > 0.98
    assert age["genuine_merge_fraction"] < 0.01
    assert fast["accidental_false_merge_fraction"] > 0.75
    assert predictive["genuine_merge_fraction"] > 0.75
    assert predictive["accidental_false_merge_fraction"] < 0.03
    assert predictive["balanced_relation_accuracy"] > 0.85
    assert shuffled["balanced_relation_accuracy"] < 0.55
    assert shifted["genuine_merge_fraction"] > 0.45
    assert shifted["accidental_false_merge_fraction"] < 0.02

    if args.json:
        Path(args.json).write_text(
            json.dumps(report, indent=2) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
