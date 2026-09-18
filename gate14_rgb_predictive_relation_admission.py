#!/usr/bin/env python3
"""Gate 14: derive predictive relation evidence from noisy RGB frame history.

Gate 13 showed that a pre-contact common-fate cue can route relation admission
onto a fast or cautious clock, but the cue was generated directly from velocity
histories. This gate removes that velocity input.

The learner now receives short sequences of noisy RGB frames. For every
consecutive frame pair it reuses Gate 6's machinery:

    RGB frame
      -> appearance-connected regions
      -> local template-match translation estimate
      -> confidence

Two foreground regions are candidate future contact partners. Their estimated
pre-contact motion disagreement is accumulated as

    e_rgb = mean_t ||vhat_a(t) - vhat_b(t)||^2.

During training only, eventual relation fate is used to learn one threshold on
that image-derived scalar. At test time no true velocity or ownership label is
given to the predictor.

All later contacts still last exactly 32 steps, preserving Gate 13's causal
control: contact duration itself contains zero class information.

A genuine pair shares all pre-contact motions. An accidental pair shares each
motion only with probability 0.5, so some accidental histories are genuinely
ambiguous even with perfect motion estimation.

Controls:
  * oracle-motion residual upper bound
  * age-only tau=64
  * always-fast tau=16
  * RGB-predictive fast/cautious clocks
  * shuffled RGB cue

This remains a synthetic image-world gate. Stable coloured/textured patches make
region discovery intentionally easy so the experiment isolates one question:
can image-derived temporal evidence control relation admission?
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from gate6_estimated_motion_write import (
    appearance_components,
    estimate_component_translations,
)
from gate11_contact_persistence import prepare_addresses
from gate13_predictive_relation_admission import (
    AGE_ONLY_TAU,
    CAUTIOUS_TAU,
    FAST_TAU,
    balanced_error,
    evaluate_policy,
    learn_residual_threshold,
)


GRID = 14
PATCH = 3
HISTORY_STEPS = 4
CONTACT_STEPS = 32
SENSOR_NOISE = 0.006
ACCIDENTAL_SAME_MOTION_PROB = 0.50
MIN_CONFIDENCE = 0.45
INVALID_RESIDUAL = 10.0
MOTION_CHOICES = np.array(
    [
        [0, 1],
        [1, 0],
        [1, 1],
    ],
    dtype=np.int64,
)

BACKGROUND = np.array([0.32, 0.27, 0.22], dtype=np.float64)
PATCH_A = np.array([0.78, 0.22, 0.20], dtype=np.float64)
PATCH_B = np.array([0.18, 0.72, 0.30], dtype=np.float64)


def make_static_textures(seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create one persistent textured world; only sensor noise changes by frame."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:GRID, 0:GRID]
    background = (
        0.018 * np.sin(0.8 * xx + 0.5 * yy + rng.uniform(-np.pi, np.pi))
        + 0.012 * np.cos(1.1 * xx - 0.6 * yy + rng.uniform(-np.pi, np.pi))
    )

    ly, lx = np.mgrid[0:PATCH, 0:PATCH]
    tex_a = (
        0.045 * np.sin(1.4 * lx + 0.9 * ly + rng.uniform(-np.pi, np.pi))
        + 0.020 * np.cos(0.7 * lx - 1.2 * ly)
    )
    tex_b = (
        0.045 * np.cos(1.2 * lx - 0.8 * ly + rng.uniform(-np.pi, np.pi))
        + 0.020 * np.sin(0.9 * lx + 1.1 * ly)
    )
    return background, tex_a, tex_b


def render_frame(
    pos_a: tuple[int, int],
    pos_b: tuple[int, int],
    textures: tuple[np.ndarray, np.ndarray, np.ndarray],
    noise_seed: int,
) -> np.ndarray:
    background, tex_a, tex_b = textures
    image = np.empty((GRID, GRID, 3), dtype=np.float64)
    image[:] = BACKGROUND
    image += background[..., None] * np.array([0.5, -0.25, 0.20])

    for pos, base, texture, direction in (
        (pos_a, PATCH_A, tex_a, np.array([0.45, 0.20, -0.30])),
        (pos_b, PATCH_B, tex_b, np.array([-0.25, 0.35, 0.25])),
    ):
        y, x = pos
        block = np.empty((PATCH, PATCH, 3), dtype=np.float64)
        block[:] = base
        block += texture[..., None] * direction
        image[y : y + PATCH, x : x + PATCH] = block

    rng = np.random.default_rng(noise_seed)
    image += rng.normal(0.0, SENSOR_NOISE, image.shape)
    return np.clip(image, 0.0, 1.0)


def choose_foreground_components(
    image: np.ndarray,
    components: np.ndarray,
) -> tuple[int, int] | None:
    """Pick the two largest non-background appearance components, label-free."""
    ids, counts = np.unique(components, return_counts=True)
    if len(ids) < 3:
        return None

    background_id = int(ids[np.argmax(counts)])
    foreground = [
        (int(count), int(comp))
        for comp, count in zip(ids, counts)
        if int(comp) != background_id
    ]
    foreground.sort(reverse=True)
    if len(foreground) < 2:
        return None

    first = foreground[0][1]
    second = foreground[1][1]

    # Deterministic image-derived ordering by mean red channel. The final motion
    # disagreement is symmetric, so this is only for stable bookkeeping.
    mean_first = float(np.mean(image[components == first, 0]))
    mean_second = float(np.mean(image[components == second, 0]))
    if mean_first < mean_second:
        first, second = second, first
    return first, second


def estimate_pair_motion(
    frame0: np.ndarray,
    frame1: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Estimate the two foreground translations from RGB only."""
    components = appearance_components(frame0)
    selected = choose_foreground_components(frame0, components)
    if selected is None:
        return np.zeros(2), np.zeros(2), 0.0

    motions, confidences = estimate_component_translations(
        frame0,
        frame1,
        components,
        search_radius=1,
    )
    a, b = selected
    va = np.asarray(motions[a], dtype=np.float64)
    vb = np.asarray(motions[b], dtype=np.float64)
    confidence = float(min(confidences[a], confidences[b]))
    return va, vb, confidence


def make_motion_history(
    label: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Create bounded positive-step histories for two candidate regions."""
    a = np.empty((HISTORY_STEPS, 2), dtype=np.int64)
    b = np.empty_like(a)

    for t in range(HISTORY_STEPS):
        choice_a = int(rng.integers(len(MOTION_CHOICES)))
        a[t] = MOTION_CHOICES[choice_a]

        if label == 1 or rng.random() < ACCIDENTAL_SAME_MOTION_PROB:
            b[t] = a[t]
        else:
            alternatives = [
                i for i in range(len(MOTION_CHOICES))
                if i != choice_a
            ]
            b[t] = MOTION_CHOICES[
                alternatives[int(rng.integers(len(alternatives)))]
            ]
    return a, b


def history_residuals(
    label: int,
    seed: int,
) -> tuple[float, float, float, float]:
    """Return RGB residual, oracle residual, mean confidence, valid fraction."""
    rng = np.random.default_rng(seed)
    motion_a, motion_b = make_motion_history(label, rng)
    textures = make_static_textures(seed + 100_000)

    pos_a = np.array([1, 1], dtype=np.int64)
    pos_b = np.array([7, 7], dtype=np.int64)
    frames = [
        render_frame(
            tuple(pos_a),
            tuple(pos_b),
            textures,
            noise_seed=seed + 200_000,
        )
    ]

    positions_a = pos_a.copy()
    positions_b = pos_b.copy()
    for t in range(HISTORY_STEPS):
        positions_a = positions_a + motion_a[t]
        positions_b = positions_b + motion_b[t]
        frames.append(
            render_frame(
                tuple(positions_a),
                tuple(positions_b),
                textures,
                noise_seed=seed + 200_001 + t,
            )
        )

    estimated_disagreement = []
    confidences = []
    valid = []
    for t in range(HISTORY_STEPS):
        va, vb, confidence = estimate_pair_motion(frames[t], frames[t + 1])
        confidences.append(confidence)
        is_valid = confidence >= MIN_CONFIDENCE
        valid.append(is_valid)
        if is_valid:
            estimated_disagreement.append(float(np.sum((va - vb) ** 2)))
        else:
            estimated_disagreement.append(INVALID_RESIDUAL)

    oracle = float(
        np.mean(np.sum((motion_a - motion_b) ** 2, axis=1))
    )
    rgb = float(np.mean(estimated_disagreement))
    return (
        rgb,
        oracle,
        float(np.mean(confidences)),
        float(np.mean(valid)),
    )


def generate_rgb_dataset(
    trials: int,
    seed: int,
) -> dict[str, np.ndarray]:
    """Balanced eventual outcomes with RGB-only and oracle prehistory receipts."""
    rng = np.random.default_rng(seed)
    labels = np.arange(trials, dtype=np.int64) % 2
    rng.shuffle(labels)

    rgb = np.empty(trials, dtype=np.float64)
    oracle = np.empty(trials, dtype=np.float64)
    confidence = np.empty(trials, dtype=np.float64)
    valid_fraction = np.empty(trials, dtype=np.float64)

    seeds = rng.integers(0, 2**31 - 1, size=trials, dtype=np.int64)
    for i in range(trials):
        rgb[i], oracle[i], confidence[i], valid_fraction[i] = history_residuals(
            int(labels[i]),
            int(seeds[i]),
        )

    return {
        "labels": labels,
        "rgb_residual": rgb,
        "oracle_residual": oracle,
        "mean_confidence": confidence,
        "valid_fraction": valid_fraction,
    }


def cue_report(
    labels: np.ndarray,
    residual: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    prediction = residual <= threshold
    error, fnr, fpr = balanced_error(labels, prediction)
    return {
        "accuracy": float(np.mean(prediction == (labels == 1))),
        "balanced_error": error,
        "false_negative_rate": fnr,
        "false_positive_rate": fpr,
        "genuine_prediction_rate": float(np.mean(prediction[labels == 1])),
        "accidental_prediction_rate": float(np.mean(prediction[labels == 0])),
        "median_genuine_residual": float(np.median(residual[labels == 1])),
        "median_accidental_residual": float(np.median(residual[labels == 0])),
    }


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
) -> dict:
    train = generate_rgb_dataset(train_trials, train_seed)
    rgb_threshold, rgb_training = learn_residual_threshold(
        train["rgb_residual"],
        train["labels"],
    )
    oracle_threshold, _ = learn_residual_threshold(
        train["oracle_residual"],
        train["labels"],
    )

    test = generate_rgb_dataset(test_trials, test_seed)
    labels = test["labels"]
    rgb_prediction = test["rgb_residual"] <= rgb_threshold

    base = prepare_addresses(
        trials=test_trials,
        dim=dim,
        seed=address_seed,
        gamma=gamma,
        pre_steps=pre_steps,
    )

    age = evaluate_policy(
        base,
        labels,
        np.full(test_trials, AGE_ONLY_TAU),
        gamma=gamma,
        contact_steps=CONTACT_STEPS,
    )
    fast = evaluate_policy(
        base,
        labels,
        np.full(test_trials, FAST_TAU),
        gamma=gamma,
        contact_steps=CONTACT_STEPS,
    )
    predictive = evaluate_policy(
        base,
        labels,
        np.where(rgb_prediction, FAST_TAU, CAUTIOUS_TAU),
        gamma=gamma,
        contact_steps=CONTACT_STEPS,
    )

    rng = np.random.default_rng(shuffle_seed)
    shuffled_prediction = rgb_prediction[rng.permutation(test_trials)]
    shuffled = evaluate_policy(
        base,
        labels,
        np.where(shuffled_prediction, FAST_TAU, CAUTIOUS_TAU),
        gamma=gamma,
        contact_steps=CONTACT_STEPS,
    )

    return {
        "train_trials": train_trials,
        "test_trials": test_trials,
        "dimension": dim,
        "history_steps": HISTORY_STEPS,
        "contact_steps": CONTACT_STEPS,
        "accidental_same_motion_probability": ACCIDENTAL_SAME_MOTION_PROB,
        "sensor_noise": SENSOR_NOISE,
        "minimum_motion_confidence": MIN_CONFIDENCE,
        "learned_rgb_threshold": rgb_threshold,
        "training_rgb_cue": rgb_training,
        "held_out_rgb_cue": cue_report(
            labels,
            test["rgb_residual"],
            rgb_threshold,
        ),
        "held_out_oracle_cue": cue_report(
            labels,
            test["oracle_residual"],
            oracle_threshold,
        ),
        "estimator_receipt": {
            "mean_confidence": float(np.mean(test["mean_confidence"])),
            "median_confidence": float(np.median(test["mean_confidence"])),
            "mean_valid_fraction": float(np.mean(test["valid_fraction"])),
        },
        "policies": {
            "age_only_tau64": age,
            "always_fast_tau16": fast,
            "rgb_predictive_fast_or_cautious": predictive,
            "shuffled_rgb_cue": shuffled,
        },
        "interpretation": (
            "With matched 32-step contact duration, the predictor receives no "
            "true motion. Gate-6 RGB region correspondence reconstructs a short "
            "pre-contact common-fate history whose learned residual can select "
            "fast versus cautious relation admission. The oracle cue is retained "
            "only as an upper bound, and shuffling the RGB cue tests whether the "
            "gain belongs to the particular visual history rather than to the "
            "two-clock policy itself."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-trials", type=int, default=240)
    parser.add_argument("--test-trials", type=int, default=400)
    parser.add_argument("--dim", type=int, default=8)
    parser.add_argument("--gamma", type=float, default=0.20)
    parser.add_argument("--pre-steps", type=int, default=240)
    parser.add_argument("--train-seed", type=int, default=1401)
    parser.add_argument("--test-seed", type=int, default=1402)
    parser.add_argument("--address-seed", type=int, default=1403)
    parser.add_argument("--shuffle-seed", type=int, default=1404)
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
    )
    print(json.dumps(report, indent=2))

    rgb = report["held_out_rgb_cue"]
    oracle = report["held_out_oracle_cue"]
    age = report["policies"]["age_only_tau64"]
    fast = report["policies"]["always_fast_tau16"]
    predictive = report["policies"]["rgb_predictive_fast_or_cautious"]
    shuffled = report["policies"]["shuffled_rgb_cue"]
    receipt = report["estimator_receipt"]

    assert receipt["mean_valid_fraction"] > 0.90
    assert rgb["accuracy"] > 0.80
    assert oracle["accuracy"] >= rgb["accuracy"] - 0.05
    assert age["genuine_merge_fraction"] < 0.05
    assert fast["accidental_false_merge_fraction"] > 0.65
    assert predictive["genuine_merge_fraction"] > 0.65
    assert predictive["accidental_false_merge_fraction"] < 0.20
    assert predictive["balanced_relation_accuracy"] > 0.72
    assert shuffled["balanced_relation_accuracy"] < 0.60

    if args.json:
        Path(args.json).write_text(
            json.dumps(report, indent=2) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
