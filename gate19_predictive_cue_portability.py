#!/usr/bin/env python3
"""Gate 19: predictive-cue portability follows the world contingency.

Gate 18 found a genuinely new pre-relapse observable: short-horizon motion
surprise. In that synthetic world, stable recoveries changed shared motion
direction with probability 0.10 while relapse-prone recoveries changed with
probability 0.70.

That result is useful only if we distinguish two things:

    observable:
        RGB-derived short-horizon shared-motion surprise

    learned contingency:
        how that observable maps to future relapse risk in the current world

Gate 19 freezes the Gate-18 predictor and changes the world:

    preserved shift:
        new motion vocabulary + extra sensor noise, same 0.10 / 0.70 relation

    weakened:
        0.25 / 0.55

    broken:
        0.40 / 0.40

    reversed:
        0.70 / 0.10

The desired result is not invariance. It is calibrated failure:

    * transfer when the contingency survives,
    * degrade when it weakens,
    * fall to chance when it disappears,
    * become systematically wrong when it reverses.

Then a small in-world calibration set is allowed to relearn only the univariate
surprise->future rule. Recalibration should recover the reversed world by
flipping the rule polarity, but it should not manufacture information in the
broken world.

All surprise values remain derived from noisy RGB correspondence; oracle motion
is recorded only as a diagnostic upper bound.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from gate14_rgb_predictive_relation_admission import (
    estimate_pair_motion,
    make_static_textures,
    render_frame,
)
from gate18_pre_relapse_predictive_cue import (
    RECOVERY_MOTIONS,
    balanced_accuracy,
    learn_univariate_rule,
    make_pairs as make_gate18_pairs,
    paired_dataset as gate18_paired_dataset,
    apply_rule,
)


RECOVERY_OBSERVATIONS = 4

ALTERNATE_MOTIONS = np.array(
    [
        [0, -1],
        [-1, 0],
    ],
    dtype=np.int64,
)


SCENARIOS = {
    "baseline": {
        "stable_p": 0.10,
        "relapse_p": 0.70,
        "motion_vocabulary": "original",
        "extra_sensor_noise": 0.0,
    },
    "preserved_covariate_shift": {
        "stable_p": 0.10,
        "relapse_p": 0.70,
        "motion_vocabulary": "alternate",
        "extra_sensor_noise": 0.004,
    },
    "weakened_contingency": {
        "stable_p": 0.25,
        "relapse_p": 0.55,
        "motion_vocabulary": "original",
        "extra_sensor_noise": 0.0,
    },
    "broken_contingency": {
        "stable_p": 0.40,
        "relapse_p": 0.40,
        "motion_vocabulary": "original",
        "extra_sensor_noise": 0.0,
    },
    "reversed_contingency": {
        "stable_p": 0.70,
        "relapse_p": 0.10,
        "motion_vocabulary": "original",
        "extra_sensor_noise": 0.0,
    },
}


def _add_noise(
    frame: np.ndarray,
    sigma: float,
    seed: int,
) -> np.ndarray:
    if sigma <= 0.0:
        return frame
    rng = np.random.default_rng(seed)
    return np.clip(
        frame + rng.normal(0.0, sigma, frame.shape),
        0.0,
        1.0,
    )


def rgb_motion_observation(
    motion: np.ndarray,
    seed: int,
    extra_sensor_noise: float,
) -> dict[str, float | np.ndarray]:
    """One clean common-fate observation, measured from RGB only."""
    textures = make_static_textures(seed + 100_000)
    # Interior starts support both positive and negative one-pixel motions.
    pos_a = np.array([2, 2], dtype=np.int64)
    pos_b = np.array([8, 8], dtype=np.int64)

    frame0 = render_frame(
        tuple(pos_a),
        tuple(pos_b),
        textures,
        noise_seed=seed + 200_000,
    )
    frame1 = render_frame(
        tuple(pos_a + motion),
        tuple(pos_b + motion),
        textures,
        noise_seed=seed + 200_001,
    )
    frame0 = _add_noise(
        frame0,
        extra_sensor_noise,
        seed + 300_000,
    )
    frame1 = _add_noise(
        frame1,
        extra_sensor_noise,
        seed + 300_001,
    )

    va, vb, confidence = estimate_pair_motion(frame0, frame1)
    residual = float(np.sum((va - vb) ** 2))
    shared_motion = 0.5 * (
        np.asarray(va, dtype=np.float64)
        + np.asarray(vb, dtype=np.float64)
    )
    exact = float(
        np.array_equal(va, motion)
        and np.array_equal(vb, motion)
    )
    return {
        "shared_motion": shared_motion,
        "confidence": float(confidence),
        "residual": residual,
        "exact": exact,
    }


def coupled_motion_sequences(
    rng: np.random.Generator,
    stable_p: float,
    relapse_p: float,
    vocabulary: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Counterfactual stable/relapse paths sharing the same random uniforms."""
    start = int(rng.integers(0, 2))
    stable = [start]
    relapse = [start]

    for _ in range(1, RECOVERY_OBSERVATIONS):
        u = float(rng.random())
        stable_next = stable[-1]
        relapse_next = relapse[-1]
        if u < stable_p:
            stable_next = 1 - stable_next
        if u < relapse_p:
            relapse_next = 1 - relapse_next
        stable.append(stable_next)
        relapse.append(relapse_next)

    return (
        vocabulary[np.asarray(stable, dtype=np.int64)],
        vocabulary[np.asarray(relapse, dtype=np.int64)],
    )


def motion_surprise(motion: np.ndarray) -> float:
    if len(motion) < 2:
        return 0.0
    return float(
        np.mean(
            np.sum(
                np.diff(motion, axis=0) ** 2,
                axis=1,
            )
        )
    )


def sequence_features(
    true_motion: np.ndarray,
    seed: int,
    extra_sensor_noise: float,
) -> dict[str, float]:
    estimated = []
    confidence = []
    residual = []
    exact = []

    for step, motion in enumerate(true_motion):
        observation = rgb_motion_observation(
            motion,
            seed + 10_000 * (step + 1),
            extra_sensor_noise,
        )
        estimated.append(
            np.asarray(
                observation["shared_motion"],
                dtype=np.float64,
            )
        )
        confidence.append(float(observation["confidence"]))
        residual.append(float(observation["residual"]))
        exact.append(float(observation["exact"]))

    estimated = np.stack(estimated, axis=0)
    return {
        "rgb_motion_surprise": motion_surprise(estimated),
        "oracle_motion_surprise": motion_surprise(
            np.asarray(true_motion, dtype=np.float64)
        ),
        "mean_confidence": float(np.mean(confidence)),
        "common_fate_fraction": float(
            np.mean(np.asarray(residual) <= 0.0)
        ),
        "exact_motion_fraction": float(np.mean(exact)),
    }


def make_scenario_pairs(
    count: int,
    seed: int,
    stable_p: float,
    relapse_p: float,
    motion_vocabulary: str,
    extra_sensor_noise: float,
) -> list[dict]:
    rng = np.random.default_rng(seed)
    if motion_vocabulary == "original":
        vocabulary = RECOVERY_MOTIONS
    elif motion_vocabulary == "alternate":
        vocabulary = ALTERNATE_MOTIONS
    else:
        raise ValueError(
            f"unknown motion vocabulary: {motion_vocabulary}"
        )

    seeds = rng.integers(
        0,
        2**31 - 1,
        size=count,
        dtype=np.int64,
    )
    pairs = []
    for item in seeds:
        pair_seed = int(item)
        pair_rng = np.random.default_rng(pair_seed)
        stable_motion, relapse_motion = coupled_motion_sequences(
            pair_rng,
            stable_p=stable_p,
            relapse_p=relapse_p,
            vocabulary=vocabulary,
        )
        stable = sequence_features(
            stable_motion,
            pair_seed + 100_000,
            extra_sensor_noise,
        )
        relapse = sequence_features(
            relapse_motion,
            pair_seed + 100_000,
            extra_sensor_noise,
        )
        pairs.append(
            {
                "stable": stable,
                "relapse": relapse,
            }
        )
    return pairs


def scenario_dataset(
    pairs: list[dict],
    feature: str,
) -> tuple[np.ndarray, np.ndarray]:
    stable = np.asarray(
        [pair["stable"][feature] for pair in pairs],
        dtype=np.float64,
    )
    relapse = np.asarray(
        [pair["relapse"][feature] for pair in pairs],
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


def evaluate_scenario(
    frozen_rule: dict[str, float | str],
    calibration_pairs: list[dict],
    test_pairs: list[dict],
) -> dict:
    calibration_values, calibration_labels = scenario_dataset(
        calibration_pairs,
        "rgb_motion_surprise",
    )
    test_values, test_labels = scenario_dataset(
        test_pairs,
        "rgb_motion_surprise",
    )
    oracle_values, _ = scenario_dataset(
        test_pairs,
        "oracle_motion_surprise",
    )

    frozen_prediction = apply_rule(
        test_values,
        frozen_rule,
    )
    frozen_accuracy = balanced_accuracy(
        test_labels,
        frozen_prediction,
    )

    recalibrated_rule = learn_univariate_rule(
        calibration_values,
        calibration_labels,
    )
    recalibrated_prediction = apply_rule(
        test_values,
        recalibrated_rule,
    )
    recalibrated_accuracy = balanced_accuracy(
        test_labels,
        recalibrated_prediction,
    )

    # Apply the frozen Gate-18 rule to oracle motion as an estimator ceiling.
    oracle_accuracy = balanced_accuracy(
        test_labels,
        apply_rule(oracle_values, frozen_rule),
    )

    stable = test_labels == 1
    relapse = test_labels == 0

    confidence_values, _ = scenario_dataset(
        test_pairs,
        "mean_confidence",
    )
    exact_values, _ = scenario_dataset(
        test_pairs,
        "exact_motion_fraction",
    )
    common_fate_values, _ = scenario_dataset(
        test_pairs,
        "common_fate_fraction",
    )

    return {
        "frozen_balanced_accuracy": frozen_accuracy,
        "recalibrated_balanced_accuracy": recalibrated_accuracy,
        "oracle_frozen_balanced_accuracy": oracle_accuracy,
        "recalibrated_rule": recalibrated_rule,
        "frozen_authority": {
            "stable_fast_fraction": float(
                np.mean(frozen_prediction[stable])
            ),
            "relapse_false_fast_fraction": float(
                np.mean(frozen_prediction[relapse])
            ),
        },
        "median_surprise": {
            "stable": float(np.median(test_values[stable])),
            "relapse": float(np.median(test_values[relapse])),
        },
        "rgb_receipt": {
            "median_confidence": float(
                np.median(confidence_values)
            ),
            "mean_exact_motion_fraction": float(
                np.mean(exact_values)
            ),
            "mean_common_fate_fraction": float(
                np.mean(common_fate_values)
            ),
        },
    }


def run_reference(
    gate18_train_pairs: int,
    calibration_pairs: int,
    test_pairs: int,
    gate18_train_seed: int,
    scenario_seed: int,
) -> dict:
    # Freeze the mapping learned in the original Gate-18 world.
    original_train = make_gate18_pairs(
        gate18_train_pairs,
        gate18_train_seed,
    )
    original_features, original_labels = gate18_paired_dataset(
        original_train
    )
    frozen_rule = learn_univariate_rule(
        original_features["motion_surprise"],
        original_labels,
    )

    scenarios = {}
    for index, (name, spec) in enumerate(SCENARIOS.items()):
        calibration = make_scenario_pairs(
            calibration_pairs,
            seed=scenario_seed + 10_000 * index,
            stable_p=float(spec["stable_p"]),
            relapse_p=float(spec["relapse_p"]),
            motion_vocabulary=str(spec["motion_vocabulary"]),
            extra_sensor_noise=float(spec["extra_sensor_noise"]),
        )
        test = make_scenario_pairs(
            test_pairs,
            seed=scenario_seed + 10_000 * index + 1,
            stable_p=float(spec["stable_p"]),
            relapse_p=float(spec["relapse_p"]),
            motion_vocabulary=str(spec["motion_vocabulary"]),
            extra_sensor_noise=float(spec["extra_sensor_noise"]),
        )
        scenarios[name] = {
            "world": spec,
            **evaluate_scenario(
                frozen_rule,
                calibration,
                test,
            ),
        }

    return {
        "gate18_train_pairs": gate18_train_pairs,
        "calibration_pairs_per_scenario": calibration_pairs,
        "test_pairs_per_scenario": test_pairs,
        "frozen_gate18_rule": frozen_rule,
        "scenarios": scenarios,
        "interpretation": (
            "The motion-surprise observable is not treated as a universal law. "
            "A frozen Gate-18 mapping transfers when the same instability/relapse "
            "contingency survives covariate shift, degrades when the contingency "
            "weakens, becomes chance when instability is statistically unrelated "
            "to relapse, and becomes anti-predictive when the relationship "
            "reverses. Small in-world recalibration can flip the mapping in the "
            "reversed world, but cannot recover information in the broken world. "
            "This separates the reusable RGB observable from the learned world-"
            "specific contingency."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate18-train-pairs", type=int, default=40)
    parser.add_argument("--calibration-pairs", type=int, default=30)
    parser.add_argument("--test-pairs", type=int, default=80)
    parser.add_argument("--gate18-train-seed", type=int, default=1801)
    parser.add_argument("--scenario-seed", type=int, default=1901)
    parser.add_argument("--json", type=str, default=None)
    args = parser.parse_args()

    report = run_reference(
        gate18_train_pairs=args.gate18_train_pairs,
        calibration_pairs=args.calibration_pairs,
        test_pairs=args.test_pairs,
        gate18_train_seed=args.gate18_train_seed,
        scenario_seed=args.scenario_seed,
    )
    print(json.dumps(report, indent=2))

    scenarios = report["scenarios"]
    assert (
        scenarios["baseline"]["frozen_balanced_accuracy"]
        > 0.75
    )
    assert (
        scenarios["preserved_covariate_shift"][
            "frozen_balanced_accuracy"
        ]
        > 0.70
    )
    assert (
        scenarios["weakened_contingency"][
            "frozen_balanced_accuracy"
        ]
        > 0.60
    )
    assert abs(
        scenarios["broken_contingency"][
            "frozen_balanced_accuracy"
        ]
        - 0.5
    ) < 0.12
    assert (
        scenarios["reversed_contingency"][
            "frozen_balanced_accuracy"
        ]
        < 0.35
    )
    assert (
        scenarios["reversed_contingency"][
            "recalibrated_balanced_accuracy"
        ]
        > 0.75
    )
    assert (
        scenarios["broken_contingency"][
            "recalibrated_balanced_accuracy"
        ]
        < 0.65
    )

    if args.json:
        Path(args.json).write_text(
            json.dumps(report, indent=2) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
