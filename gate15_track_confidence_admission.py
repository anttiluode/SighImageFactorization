#!/usr/bin/env python3
"""Gate 15: track confidence vetoes spurious common-fate evidence.

Gate 14 established that RGB-derived motion history can control the speed at
which a new relation rewrites an established instance address. Its tracker was
still easy: stable appearance, no occlusion, no plausible local distractor.

This gate attacks the correspondence itself.

Each pre-contact observation is a noisy RGB frame pair containing two candidate
foreground regions. The second candidate undergoes three simultaneous insults in
the target frame:

    * an appearance jump,
    * one-column partial occlusion,
    * a look-alike continuation near the displacement predicted by candidate A.

If the two candidates truly share motion, that look-alike continuation overlaps
the true continuation and tracking remains coherent. If their true motions
differ, the B tracker is tempted by a visually plausible false continuation that
moves like A. This is the dangerous error: estimated motion can falsely say
"common fate" even though the true motions differ.

Gate 6's template matcher already emits a confidence score. We therefore split
the evidence into two local quantities:

    common-fate residual:
        e = mean_t ||vhat_a(t) - vhat_b(t)||^2

    track confidence:
        q = min_t confidence_t

Training outcomes select:
    * a residual threshold, then
    * a confidence threshold for trusting a "genuine" residual.

At test time:
    residual-only prediction:
        e <= theta_e

    confidence-gated prediction:
        e <= theta_e and q >= theta_q

Both predictions route the same relation clocks used in Gates 13-14:
    predicted genuine    -> tau=16
    otherwise            -> tau=96

All final contacts remain exactly 32 steps.

Controls:
    * oracle-motion residual upper bound
    * residual-only RGB cue (ignores confidence)
    * confidence-gated RGB cue
    * shuffled confidence
    * globally cautious tau=96
    * always-fast tau=16

This is still a synthetic tracking attacker. The point is not natural-video
tracking performance; it is whether locally available uncertainty can prevent a
bad correspondence from acquiring fast causal authority.
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
    CAUTIOUS_TAU,
    FAST_TAU,
    balanced_error,
    evaluate_policy,
    learn_residual_threshold,
)
from gate14_rgb_predictive_relation_admission import (
    BACKGROUND,
    CONTACT_STEPS,
    GRID,
    HISTORY_STEPS,
    MOTION_CHOICES,
    PATCH,
    PATCH_A,
    PATCH_B,
    SENSOR_NOISE,
    choose_foreground_components,
    make_static_textures,
)


TRUE_TARGET_DRIFT = 0.15
TRUE_TARGET_OCCLUSION_COLUMNS = 1
LOOKALIKE_BLEND = 0.50
ACCIDENTAL_SAME_MOTION_PROB = 0.50


def render_tracking_frame(
    pos_a: tuple[int, int],
    pos_b: tuple[int, int],
    textures: tuple[np.ndarray, np.ndarray, np.ndarray],
    noise_seed: int,
    *,
    drift_b: float = 0.0,
    occlude_b: bool = False,
    lookalike_pos: tuple[int, int] | None = None,
) -> np.ndarray:
    """Render one challenged RGB frame without exposing labels to the tracker."""
    background_texture, tex_a, tex_b = textures

    image = np.empty((GRID, GRID, 3), dtype=np.float64)
    image[:] = BACKGROUND
    image += background_texture[..., None] * np.array(
        [0.5, -0.25, 0.20],
        dtype=np.float64,
    )

    ay, ax = pos_a
    block_a = np.empty((PATCH, PATCH, 3), dtype=np.float64)
    block_a[:] = PATCH_A
    block_a += tex_a[..., None] * np.array(
        [0.45, 0.20, -0.30],
        dtype=np.float64,
    )
    image[ay : ay + PATCH, ax : ax + PATCH] = block_a

    by, bx = pos_b
    block_b = np.empty((PATCH, PATCH, 3), dtype=np.float64)
    block_b[:] = PATCH_B + drift_b
    block_b += tex_b[..., None] * np.array(
        [-0.25, 0.35, 0.25],
        dtype=np.float64,
    )
    image[by : by + PATCH, bx : bx + PATCH] = block_b

    if occlude_b:
        image[
            by : by + PATCH,
            bx : bx + TRUE_TARGET_OCCLUSION_COLUMNS,
        ] = BACKGROUND

    if lookalike_pos is not None:
        gy, gx = lookalike_pos
        lookalike = np.empty((PATCH, PATCH, 3), dtype=np.float64)
        lookalike[:] = PATCH_B
        lookalike += tex_b[..., None] * np.array(
            [-0.25, 0.35, 0.25],
            dtype=np.float64,
        )
        existing = image[gy : gy + PATCH, gx : gx + PATCH]
        image[gy : gy + PATCH, gx : gx + PATCH] = (
            LOOKALIKE_BLEND * lookalike
            + (1.0 - LOOKALIKE_BLEND) * existing
        )

    rng = np.random.default_rng(noise_seed)
    image += rng.normal(0.0, SENSOR_NOISE, image.shape)
    return np.clip(image, 0.0, 1.0)


def estimate_pair_motion(
    frame0: np.ndarray,
    frame1: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Estimate candidate motions plus the tracker's own weakest confidence."""
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
    return (
        np.asarray(motions[a], dtype=np.float64),
        np.asarray(motions[b], dtype=np.float64),
        float(min(confidences[a], confidences[b])),
    )


def choose_motion_pair(
    label: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    idx_a = int(rng.integers(len(MOTION_CHOICES)))
    motion_a = MOTION_CHOICES[idx_a].copy()

    if label == 1 or rng.random() < ACCIDENTAL_SAME_MOTION_PROB:
        return motion_a, motion_a.copy()

    alternatives = [
        idx
        for idx in range(len(MOTION_CHOICES))
        if idx != idx_a
    ]
    idx_b = alternatives[int(rng.integers(len(alternatives)))]
    return motion_a, MOTION_CHOICES[idx_b].copy()


def challenged_observation(
    label: int,
    rng: np.random.Generator,
    seed: int,
) -> tuple[float, float, float, float]:
    """One two-frame RGB tracking probe.

    Returns estimated motion disagreement, oracle disagreement, confidence and
    whether both estimated motions equal their hidden true translations.
    """
    motion_a, motion_b = choose_motion_pair(label, rng)
    textures = make_static_textures(seed + 100_000)

    pos_a = np.array([1, 1], dtype=np.int64)
    pos_b = np.array([7, 7], dtype=np.int64)

    frame0 = render_tracking_frame(
        tuple(pos_a),
        tuple(pos_b),
        textures,
        noise_seed=seed + 200_000,
    )

    frame1 = render_tracking_frame(
        tuple(pos_a + motion_a),
        tuple(pos_b + motion_b),
        textures,
        noise_seed=seed + 200_001,
        drift_b=TRUE_TARGET_DRIFT,
        occlude_b=True,
        # A look-alike B continuation is placed where A's motion predicts.
        lookalike_pos=tuple(pos_b + motion_a),
    )

    estimate_a, estimate_b, confidence = estimate_pair_motion(
        frame0,
        frame1,
    )

    estimated_disagreement = float(
        np.sum((estimate_a - estimate_b) ** 2)
    )
    oracle_disagreement = float(
        np.sum((motion_a - motion_b) ** 2)
    )
    correct = float(
        np.array_equal(estimate_a, motion_a)
        and np.array_equal(estimate_b, motion_b)
    )
    return (
        estimated_disagreement,
        oracle_disagreement,
        confidence,
        correct,
    )


def track_history(
    label: int,
    seed: int,
) -> tuple[float, float, float, float, float]:
    """Aggregate four independent local pre-contact tracking probes."""
    rng = np.random.default_rng(seed)

    estimated = []
    oracle = []
    confidence = []
    correctness = []

    for step in range(HISTORY_STEPS):
        e, o, q, correct = challenged_observation(
            label,
            rng,
            seed + 1_000 * step,
        )
        estimated.append(e)
        oracle.append(o)
        confidence.append(q)
        correctness.append(correct)

    return (
        float(np.mean(estimated)),
        float(np.mean(oracle)),
        float(np.min(confidence)),
        float(np.mean(confidence)),
        float(np.mean(correctness)),
    )


def generate_tracking_dataset(
    trials: int,
    seed: int,
) -> dict[str, np.ndarray]:
    """Balanced relation outcomes with RGB tracking uncertainty receipts."""
    rng = np.random.default_rng(seed)
    labels = np.arange(trials, dtype=np.int64) % 2
    rng.shuffle(labels)
    seeds = rng.integers(
        0,
        2**31 - 1,
        size=trials,
        dtype=np.int64,
    )

    rgb_residual = np.empty(trials, dtype=np.float64)
    oracle_residual = np.empty(trials, dtype=np.float64)
    min_confidence = np.empty(trials, dtype=np.float64)
    mean_confidence = np.empty(trials, dtype=np.float64)
    motion_accuracy = np.empty(trials, dtype=np.float64)

    for i, (label, trial_seed) in enumerate(zip(labels, seeds)):
        (
            rgb_residual[i],
            oracle_residual[i],
            min_confidence[i],
            mean_confidence[i],
            motion_accuracy[i],
        ) = track_history(
            int(label),
            int(trial_seed),
        )

    return {
        "labels": labels,
        "rgb_residual": rgb_residual,
        "oracle_residual": oracle_residual,
        "min_confidence": min_confidence,
        "mean_confidence": mean_confidence,
        "motion_accuracy": motion_accuracy,
    }


def learn_confidence_threshold(
    residual: np.ndarray,
    confidence: np.ndarray,
    labels: np.ndarray,
    residual_threshold: float,
) -> tuple[float, dict[str, float]]:
    """Learn when a low residual is trustworthy enough to earn the fast clock."""
    ordered = np.sort(np.asarray(confidence, dtype=np.float64))
    candidates = np.concatenate(
        (
            np.array([-np.inf]),
            0.5 * (ordered[:-1] + ordered[1:]),
            np.array([np.inf]),
        )
    )

    best = None
    for threshold in candidates:
        prediction = (
            (residual <= residual_threshold)
            & (confidence >= threshold)
        )
        error, fnr, fpr = balanced_error(labels, prediction)
        key = (
            error,
            -float(threshold)
            if np.isfinite(threshold)
            else np.inf,
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


def cue_metrics(
    labels: np.ndarray,
    prediction: np.ndarray,
) -> dict[str, float]:
    error, fnr, fpr = balanced_error(labels, prediction)
    return {
        "accuracy": float(
            np.mean(prediction == (labels == 1))
        ),
        "balanced_error": error,
        "false_negative_rate": fnr,
        "false_positive_rate": fpr,
        "genuine_prediction_rate": float(
            np.mean(prediction[labels == 1])
        ),
        "accidental_prediction_rate": float(
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
    address_seed: int,
    shuffle_seed: int,
) -> dict:
    train = generate_tracking_dataset(
        train_trials,
        train_seed,
    )

    residual_threshold, residual_training = learn_residual_threshold(
        train["rgb_residual"],
        train["labels"],
    )
    confidence_threshold, confidence_training = learn_confidence_threshold(
        train["rgb_residual"],
        train["min_confidence"],
        train["labels"],
        residual_threshold,
    )
    oracle_threshold, _ = learn_residual_threshold(
        train["oracle_residual"],
        train["labels"],
    )

    test = generate_tracking_dataset(
        test_trials,
        test_seed,
    )
    labels = test["labels"]

    residual_only = (
        test["rgb_residual"] <= residual_threshold
    )
    confidence_gated = (
        residual_only
        & (test["min_confidence"] >= confidence_threshold)
    )
    oracle_prediction = (
        test["oracle_residual"] <= oracle_threshold
    )

    rng = np.random.default_rng(shuffle_seed)
    shuffled_confidence = test["min_confidence"][
        rng.permutation(test_trials)
    ]
    shuffled_gate = (
        residual_only
        & (shuffled_confidence >= confidence_threshold)
    )

    base = prepare_addresses(
        trials=test_trials,
        dim=dim,
        seed=address_seed,
        gamma=gamma,
        pre_steps=pre_steps,
    )

    global_cautious = evaluate_policy(
        base,
        labels,
        np.full(test_trials, CAUTIOUS_TAU),
        gamma=gamma,
        contact_steps=CONTACT_STEPS,
    )
    always_fast = evaluate_policy(
        base,
        labels,
        np.full(test_trials, FAST_TAU),
        gamma=gamma,
        contact_steps=CONTACT_STEPS,
    )
    residual_policy = evaluate_policy(
        base,
        labels,
        np.where(
            residual_only,
            FAST_TAU,
            CAUTIOUS_TAU,
        ),
        gamma=gamma,
        contact_steps=CONTACT_STEPS,
    )
    confidence_policy = evaluate_policy(
        base,
        labels,
        np.where(
            confidence_gated,
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
            shuffled_gate,
            FAST_TAU,
            CAUTIOUS_TAU,
        ),
        gamma=gamma,
        contact_steps=CONTACT_STEPS,
    )

    return {
        "train_trials": train_trials,
        "test_trials": test_trials,
        "dimension": dim,
        "history_observations": HISTORY_STEPS,
        "contact_steps": CONTACT_STEPS,
        "appearance_jump": TRUE_TARGET_DRIFT,
        "occlusion_columns": TRUE_TARGET_OCCLUSION_COLUMNS,
        "lookalike_blend": LOOKALIKE_BLEND,
        "accidental_same_motion_probability": ACCIDENTAL_SAME_MOTION_PROB,
        "learned_residual_threshold": residual_threshold,
        "learned_confidence_threshold": confidence_threshold,
        "training": {
            "residual_only": residual_training,
            "confidence_gated": confidence_training,
        },
        "held_out_cues": {
            "residual_only_rgb": cue_metrics(
                labels,
                residual_only,
            ),
            "confidence_gated_rgb": cue_metrics(
                labels,
                confidence_gated,
            ),
            "oracle_motion": cue_metrics(
                labels,
                oracle_prediction,
            ),
            "shuffled_confidence": cue_metrics(
                labels,
                shuffled_gate,
            ),
        },
        "tracker_receipt": {
            "median_min_confidence_genuine": float(
                np.median(
                    test["min_confidence"][labels == 1]
                )
            ),
            "median_min_confidence_accidental": float(
                np.median(
                    test["min_confidence"][labels == 0]
                )
            ),
            "mean_motion_accuracy_genuine": float(
                np.mean(
                    test["motion_accuracy"][labels == 1]
                )
            ),
            "mean_motion_accuracy_accidental": float(
                np.mean(
                    test["motion_accuracy"][labels == 0]
                )
            ),
            "median_rgb_residual_genuine": float(
                np.median(
                    test["rgb_residual"][labels == 1]
                )
            ),
            "median_rgb_residual_accidental": float(
                np.median(
                    test["rgb_residual"][labels == 0]
                )
            ),
        },
        "policies": {
            "global_cautious_tau96": global_cautious,
            "always_fast_tau16": always_fast,
            "residual_only_fast_or_cautious": residual_policy,
            "confidence_gated_fast_or_cautious": confidence_policy,
            "shuffled_confidence": shuffled_policy,
        },
        "interpretation": (
            "The look-alike tracking attacker makes RGB motion disagreement "
            "alone unsafe: a bad correspondence can imitate common fate. The "
            "template matcher nevertheless exposes local uncertainty. Requiring "
            "both low disagreement and sufficient track confidence restores "
            "selective fast relation admission. Shuffling confidence destroys "
            "that rescue, while a globally cautious clock remains safe but "
            "cannot bind clear genuine relations quickly."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-trials", type=int, default=240)
    parser.add_argument("--test-trials", type=int, default=400)
    parser.add_argument("--dim", type=int, default=8)
    parser.add_argument("--gamma", type=float, default=0.20)
    parser.add_argument("--pre-steps", type=int, default=240)
    parser.add_argument("--train-seed", type=int, default=1501)
    parser.add_argument("--test-seed", type=int, default=1502)
    parser.add_argument("--address-seed", type=int, default=1503)
    parser.add_argument("--shuffle-seed", type=int, default=1504)
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

    residual = report["held_out_cues"]["residual_only_rgb"]
    gated = report["held_out_cues"]["confidence_gated_rgb"]
    oracle = report["held_out_cues"]["oracle_motion"]
    tracker = report["tracker_receipt"]
    global_cautious = report["policies"]["global_cautious_tau96"]
    residual_policy = report["policies"]["residual_only_fast_or_cautious"]
    gated_policy = report["policies"]["confidence_gated_fast_or_cautious"]
    shuffled = report["policies"]["shuffled_confidence"]

    assert residual["balanced_error"] > 0.35
    assert gated["accuracy"] > 0.85
    assert oracle["accuracy"] > 0.90
    assert (
        tracker["median_min_confidence_genuine"]
        > tracker["median_min_confidence_accidental"] + 0.15
    )
    assert (
        residual_policy["accidental_false_merge_fraction"]
        > 0.60
    )
    assert (
        gated_policy["genuine_merge_fraction"]
        > 0.65
    )
    assert (
        gated_policy["accidental_false_merge_fraction"]
        < 0.15
    )
    assert (
        gated_policy["balanced_relation_accuracy"]
        > 0.78
    )
    assert global_cautious["genuine_merge_fraction"] < 0.05
    assert shuffled["balanced_relation_accuracy"] < 0.62

    if args.json:
        Path(args.json).write_text(
            json.dumps(report, indent=2) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
