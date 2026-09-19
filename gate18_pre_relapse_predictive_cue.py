#!/usr/bin/env python3
"""Gate 18: search for a pre-relapse predictive observable.

Gate 17 proved that no deterministic state computed from an exactly identical
ordinary evidence prefix can predict which future follows. Gate 18 therefore
does not add another memory timescale. It adds candidate observables from the
RGB correspondence process itself and asks whether any of them differs *before*
the ordinary confidence/common-fate trace diverges.

Each counterfactual pair shares:
    * an established clear prefix,
    * the same ambiguous attack,
    * the same first clean recovery observation.

The recovery regime then differs only in motion-model stability:

    stable-recovery world:
        shared A/B motion switches direction with probability 0.10 per step

    relapse-prone world:
        shared A/B motion switches direction with probability 0.70 per step

Both A and B still move together in every recovery observation. Therefore the
ordinary common-fate residual remains zero and template confidence stays high in
both classes. After the split, the stable world remains clean while the
relapse-prone world returns to the Gate-15 ambiguous tracker.

Four pre-relapse RGB diagnostics compete:
    * absolute template reconstruction error,
    * absolute best-vs-second-best margin,
    * forward/backward cycle inconsistency,
    * short-horizon shared-motion surprise.

Thresholds are fit on training pairs. Candidate choice is frozen on a separate
validation split. The final test also includes:
    * Gate-17 temporal states built only from the ordinary authority target,
    * mean/min confidence controls,
    * a shuffled selected-cue control.

The intended claim is narrow: a new observable may beat Gate 17 only if it
contains information that the old evidence history did not.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from gate6_estimated_motion_write import appearance_components
from gate14_rgb_predictive_relation_admission import (
    MOTION_CHOICES,
    choose_foreground_components,
    make_static_textures,
    render_frame,
)
from gate15_track_confidence_admission import challenged_observation
from gate17_false_clean_identical_prefix import (
    asymmetric_trace,
    cumulative_trace,
    symmetric_trace,
)


PRE_CLEAR = 4
ATTACK_OBSERVATIONS = 3
RECOVERY_OBSERVATIONS = 4

STABLE_SWITCH_PROBABILITY = 0.10
RELAPSE_SWITCH_PROBABILITY = 0.70

SYMMETRIC_TAU = 1.5
ASYMMETRIC_ATTACK_TAU = 1.5
ASYMMETRIC_RECOVERY_TAU = 6.0

# Equal-norm translations minimize trivial confidence differences between the
# two recovery regimes.
RECOVERY_MOTIONS = np.array(
    [
        [0, 1],
        [1, 0],
    ],
    dtype=np.int64,
)


def _component_match(
    frame0: np.ndarray,
    frame1: np.ndarray,
    components: np.ndarray,
    comp: int,
    search_radius: int = 1,
    displacement_penalty: float = 1e-5,
) -> dict[str, float | np.ndarray]:
    """Gate-6 template matcher with its hidden score diagnostics exposed."""
    h, w = frame0.shape[:2]
    ys, xs = np.where(components == comp)
    candidates: list[tuple[float, int, int]] = []

    for dy in range(-search_radius, search_radius + 1):
        for dx in range(-search_radius, search_radius + 1):
            yy = ys + dy
            xx = xs + dx
            valid = (
                (yy >= 0)
                & (yy < h)
                & (xx >= 0)
                & (xx < w)
            )
            if float(np.mean(valid)) < 0.90:
                continue

            diff = (
                frame0[ys[valid], xs[valid]]
                - frame1[yy[valid], xx[valid]]
            )
            score = float(
                np.mean(diff * diff)
                + displacement_penalty * (dy * dy + dx * dx)
            )
            candidates.append((score, dy, dx))

    candidates.sort(key=lambda item: item[0])
    if not candidates:
        return {
            "motion": np.zeros(2, dtype=np.float64),
            "best_score": float("inf"),
            "second_score": float("inf"),
            "absolute_margin": 0.0,
            "confidence": 0.0,
        }

    best_score, dy, dx = candidates[0]
    second_score = (
        candidates[1][0]
        if len(candidates) > 1
        else best_score
    )
    absolute_margin = float(second_score - best_score)
    confidence = float(
        absolute_margin / (second_score + 1e-30)
    )
    return {
        "motion": np.array([dy, dx], dtype=np.float64),
        "best_score": float(best_score),
        "second_score": float(second_score),
        "absolute_margin": absolute_margin,
        "confidence": confidence,
    }


def pair_diagnostics(
    frame0: np.ndarray,
    frame1: np.ndarray,
) -> dict[str, float | np.ndarray]:
    """RGB-only relation diagnostics for the two foreground candidates."""
    components0 = appearance_components(frame0)
    selected0 = choose_foreground_components(frame0, components0)
    if selected0 is None:
        return {
            "residual": 10.0,
            "confidence": 0.0,
            "match_error": 1.0,
            "absolute_margin": 0.0,
            "cycle_error": 10.0,
            "shared_motion": np.zeros(2, dtype=np.float64),
        }

    a0, b0 = selected0
    forward_a = _component_match(
        frame0, frame1, components0, a0
    )
    forward_b = _component_match(
        frame0, frame1, components0, b0
    )

    va = np.asarray(forward_a["motion"], dtype=np.float64)
    vb = np.asarray(forward_b["motion"], dtype=np.float64)
    residual = float(np.sum((va - vb) ** 2))
    confidence = float(
        min(
            float(forward_a["confidence"]),
            float(forward_b["confidence"]),
        )
    )
    match_error = float(
        0.5
        * (
            float(forward_a["best_score"])
            + float(forward_b["best_score"])
        )
    )
    absolute_margin = float(
        min(
            float(forward_a["absolute_margin"]),
            float(forward_b["absolute_margin"]),
        )
    )

    components1 = appearance_components(frame1)
    selected1 = choose_foreground_components(frame1, components1)
    if selected1 is None:
        cycle_error = 10.0
    else:
        a1, b1 = selected1
        backward_a = _component_match(
            frame1, frame0, components1, a1
        )
        backward_b = _component_match(
            frame1, frame0, components1, b1
        )
        ba = np.asarray(
            backward_a["motion"],
            dtype=np.float64,
        )
        bb = np.asarray(
            backward_b["motion"],
            dtype=np.float64,
        )
        cycle_error = float(
            0.5
            * (
                np.sum((va + ba) ** 2)
                + np.sum((vb + bb) ** 2)
            )
        )

    return {
        "residual": residual,
        "confidence": confidence,
        "match_error": match_error,
        "absolute_margin": absolute_margin,
        "cycle_error": cycle_error,
        "shared_motion": 0.5 * (va + vb),
    }


def clean_common_fate_observation(
    motion: np.ndarray,
    seed: int,
) -> dict[str, float | np.ndarray]:
    """Render one clean RGB common-fate observation with a chosen motion."""
    textures = make_static_textures(seed + 100_000)
    pos_a = np.array([1, 1], dtype=np.int64)
    pos_b = np.array([7, 7], dtype=np.int64)

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
    return pair_diagnostics(frame0, frame1)


def authority_target(
    diagnostic: dict[str, float | np.ndarray],
) -> float:
    return (
        float(diagnostic["confidence"])
        if float(diagnostic["residual"]) <= 0.0
        else 0.0
    )


def recovery_motion_sequences(
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Coupled stable/relapse motion paths with different switch hazards."""
    start = int(rng.integers(0, 2))
    stable = [start]
    relapse = [start]

    # Coupled uniforms mean every stable switch is also a relapse-prone switch.
    for _ in range(1, RECOVERY_OBSERVATIONS):
        u = float(rng.random())
        stable_next = stable[-1]
        relapse_next = relapse[-1]
        if u < STABLE_SWITCH_PROBABILITY:
            stable_next = 1 - stable_next
        if u < RELAPSE_SWITCH_PROBABILITY:
            relapse_next = 1 - relapse_next
        stable.append(stable_next)
        relapse.append(relapse_next)

    return (
        RECOVERY_MOTIONS[np.asarray(stable, dtype=np.int64)],
        RECOVERY_MOTIONS[np.asarray(relapse, dtype=np.int64)],
    )


def make_pair(seed: int) -> dict:
    """One stable-recovery / relapse-prone counterfactual pair."""
    rng = np.random.default_rng(seed)

    # Shared ordinary evidence before the recovery streak.
    shared_target = []

    base_motion = RECOVERY_MOTIONS[int(rng.integers(0, 2))]
    for step in range(PRE_CLEAR):
        diagnostic = clean_common_fate_observation(
            base_motion,
            seed + 10_000 * (step + 1),
        )
        shared_target.append(authority_target(diagnostic))

    for step in range(ATTACK_OBSERVATIONS):
        residual, _oracle, confidence, _exact = challenged_observation(
            1,
            rng,
            seed + 100_000 + 10_000 * step,
        )
        shared_target.append(
            float(confidence) if residual <= 0.0 else 0.0
        )

    stable_motion, relapse_motion = recovery_motion_sequences(rng)

    stable_diagnostics = []
    relapse_diagnostics = []
    stable_target = list(shared_target)
    relapse_target = list(shared_target)

    for step in range(RECOVERY_OBSERVATIONS):
        observation_seed = seed + 300_000 + 10_000 * step

        stable_diag = clean_common_fate_observation(
            stable_motion[step],
            observation_seed,
        )
        relapse_diag = clean_common_fate_observation(
            relapse_motion[step],
            observation_seed,
        )
        stable_diagnostics.append(stable_diag)
        relapse_diagnostics.append(relapse_diag)
        stable_target.append(authority_target(stable_diag))
        relapse_target.append(authority_target(relapse_diag))

    # The first genuinely future observation remains hidden from the pre-relapse
    # classifier. It is retained only as a receipt that the two futures really do
    # diverge after the split.
    future_seed = seed + 900_000
    stable_future = clean_common_fate_observation(
        stable_motion[-1],
        future_seed,
    )
    relapse_residual, _oracle, relapse_confidence, _exact = (
        challenged_observation(
            1,
            np.random.default_rng(seed + 901_000),
            future_seed,
        )
    )
    relapse_future_target = (
        float(relapse_confidence)
        if relapse_residual <= 0.0
        else 0.0
    )

    return {
        "stable_target": np.asarray(stable_target, dtype=np.float64),
        "relapse_target": np.asarray(relapse_target, dtype=np.float64),
        "stable_recovery": stable_diagnostics,
        "relapse_recovery": relapse_diagnostics,
        "stable_motion": stable_motion.astype(np.float64),
        "relapse_motion": relapse_motion.astype(np.float64),
        "stable_future_target": authority_target(stable_future),
        "relapse_future_target": relapse_future_target,
    }


def temporal_state(target: np.ndarray, family: str) -> float:
    row = np.asarray(target, dtype=np.float64)[None, :]
    if family == "raw":
        return float(row[0, -1])
    if family == "cumulative_mean":
        return float(cumulative_trace(row)[0, -1])
    if family == "symmetric_ema":
        return float(symmetric_trace(row, SYMMETRIC_TAU)[0, -1])
    if family == "asymmetric_eligibility":
        return float(
            asymmetric_trace(
                row,
                ASYMMETRIC_ATTACK_TAU,
                ASYMMETRIC_RECOVERY_TAU,
            )[0, -1]
        )
    raise ValueError(f"unknown family: {family}")


def recovery_features(
    diagnostics: list[dict[str, float | np.ndarray]],
) -> dict[str, float]:
    motion = np.stack(
        [
            np.asarray(item["shared_motion"], dtype=np.float64)
            for item in diagnostics
        ],
        axis=0,
    )
    if len(motion) > 1:
        motion_surprise = float(
            np.mean(
                np.sum(
                    np.diff(motion, axis=0) ** 2,
                    axis=1,
                )
            )
        )
    else:
        motion_surprise = 0.0

    confidence = np.array(
        [float(item["confidence"]) for item in diagnostics],
        dtype=np.float64,
    )
    return {
        "mean_confidence": float(np.mean(confidence)),
        "min_confidence": float(np.min(confidence)),
        "match_error": float(
            np.mean(
                [
                    float(item["match_error"])
                    for item in diagnostics
                ]
            )
        ),
        "absolute_margin": float(
            np.mean(
                [
                    float(item["absolute_margin"])
                    for item in diagnostics
                ]
            )
        ),
        "cycle_error": float(
            np.mean(
                [
                    float(item["cycle_error"])
                    for item in diagnostics
                ]
            )
        ),
        "motion_surprise": motion_surprise,
    }


def paired_dataset(pairs: list[dict]) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Return stable=1 / relapse=0 features from paired worlds."""
    labels = np.concatenate(
        [
            np.ones(len(pairs), dtype=np.int64),
            np.zeros(len(pairs), dtype=np.int64),
        ]
    )

    feature_names = (
        "raw_state",
        "cumulative_state",
        "symmetric_state",
        "asymmetric_state",
        "mean_confidence",
        "min_confidence",
        "match_error",
        "absolute_margin",
        "cycle_error",
        "motion_surprise",
    )
    values = {name: [] for name in feature_names}

    for future_key, recovery_key in (
        ("stable_target", "stable_recovery"),
        ("relapse_target", "relapse_recovery"),
    ):
        for pair in pairs:
            target = pair[future_key]
            features = recovery_features(pair[recovery_key])
            values["raw_state"].append(temporal_state(target, "raw"))
            values["cumulative_state"].append(
                temporal_state(target, "cumulative_mean")
            )
            values["symmetric_state"].append(
                temporal_state(target, "symmetric_ema")
            )
            values["asymmetric_state"].append(
                temporal_state(target, "asymmetric_eligibility")
            )
            for name in (
                "mean_confidence",
                "min_confidence",
                "match_error",
                "absolute_margin",
                "cycle_error",
                "motion_surprise",
            ):
                values[name].append(features[name])

    return (
        {
            name: np.asarray(items, dtype=np.float64)
            for name, items in values.items()
        },
        labels,
    )


def balanced_accuracy(labels: np.ndarray, prediction: np.ndarray) -> float:
    positive = labels == 1
    negative = labels == 0
    tpr = float(np.mean(prediction[positive]))
    tnr = float(np.mean(~prediction[negative]))
    return 0.5 * (tpr + tnr)


def learn_univariate_rule(
    values: np.ndarray,
    labels: np.ndarray,
) -> dict[str, float | str]:
    """Choose threshold and polarity on training data only."""
    unique = np.sort(np.unique(values))
    if len(unique) > 1:
        mids = 0.5 * (unique[:-1] + unique[1:])
    else:
        mids = unique.copy()
    candidates = np.concatenate(
        [np.array([-np.inf]), mids, np.array([np.inf])]
    )

    best = None
    for polarity in ("high", "low"):
        for threshold in candidates:
            prediction = (
                values >= threshold
                if polarity == "high"
                else values <= threshold
            )
            accuracy = balanced_accuracy(labels, prediction)
            key = (
                -accuracy,
                0 if polarity == "low" else 1,
                abs(float(threshold))
                if np.isfinite(threshold)
                else np.inf,
            )
            if best is None or key < best[0]:
                best = (
                    key,
                    float(threshold),
                    polarity,
                    accuracy,
                )

    assert best is not None
    return {
        "threshold": best[1],
        "polarity": best[2],
        "training_balanced_accuracy": best[3],
    }


def apply_rule(
    values: np.ndarray,
    rule: dict[str, float | str],
) -> np.ndarray:
    threshold = float(rule["threshold"])
    if rule["polarity"] == "high":
        return values >= threshold
    return values <= threshold


def evaluate_rule(
    values: np.ndarray,
    labels: np.ndarray,
    rule: dict[str, float | str],
) -> float:
    return balanced_accuracy(labels, apply_rule(values, rule))


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
    validation_pairs: int,
    test_pairs: int,
    train_seed: int,
    validation_seed: int,
    test_seed: int,
    shuffle_seed: int,
) -> dict:
    train_pairs_data = make_pairs(train_pairs, train_seed)
    validation_pairs_data = make_pairs(
        validation_pairs,
        validation_seed,
    )
    test_pairs_data = make_pairs(test_pairs, test_seed)

    train, train_labels = paired_dataset(train_pairs_data)
    validation, validation_labels = paired_dataset(validation_pairs_data)
    test, test_labels = paired_dataset(test_pairs_data)

    rules = {
        name: learn_univariate_rule(values, train_labels)
        for name, values in train.items()
    }
    validation_accuracy = {
        name: evaluate_rule(
            validation[name],
            validation_labels,
            rule,
        )
        for name, rule in rules.items()
    }

    candidate_cues = (
        "match_error",
        "absolute_margin",
        "cycle_error",
        "motion_surprise",
    )
    selected_cue = max(
        candidate_cues,
        key=lambda name: (
            validation_accuracy[name],
            -candidate_cues.index(name),
        ),
    )

    test_accuracy = {
        name: evaluate_rule(
            test[name],
            test_labels,
            rule,
        )
        for name, rule in rules.items()
    }

    rng = np.random.default_rng(shuffle_seed)
    shuffled = test[selected_cue][rng.permutation(len(test_labels))]
    shuffled_accuracy = evaluate_rule(
        shuffled,
        test_labels,
        rules[selected_cue],
    )

    stable_future = np.array(
        [
            pair["stable_future_target"]
            for pair in test_pairs_data
        ],
        dtype=np.float64,
    )
    relapse_future = np.array(
        [
            pair["relapse_future_target"]
            for pair in test_pairs_data
        ],
        dtype=np.float64,
    )

    paired_motion_difference = []
    for pair in test_pairs_data:
        stable = pair["stable_motion"]
        relapse = pair["relapse_motion"]
        paired_motion_difference.append(
            float(
                np.mean(
                    np.sum(
                        (stable - relapse) ** 2,
                        axis=1,
                    )
                )
            )
        )

    symmetric_fast = test["symmetric_state"] >= 0.90
    asymmetric_fast = test["asymmetric_state"] >= 0.75
    selected_prediction = apply_rule(
        test[selected_cue],
        rules[selected_cue],
    )
    stable = test_labels == 1
    relapse = test_labels == 0

    return {
        "train_pairs": train_pairs,
        "validation_pairs": validation_pairs,
        "test_pairs": test_pairs,
        "pre_clear_observations": PRE_CLEAR,
        "attack_observations": ATTACK_OBSERVATIONS,
        "recovery_observations": RECOVERY_OBSERVATIONS,
        "stable_switch_probability": STABLE_SWITCH_PROBABILITY,
        "relapse_switch_probability": RELAPSE_SWITCH_PROBABILITY,
        "learned_rules": rules,
        "validation_balanced_accuracy": validation_accuracy,
        "selected_predictive_cue": selected_cue,
        "held_out_balanced_accuracy": test_accuracy,
        "selected_cue_shuffled_balanced_accuracy": shuffled_accuracy,
        "reauthorization_at_split": {
            "gate16_symmetric": {
                "stable_fast_fraction": float(
                    np.mean(symmetric_fast[stable])
                ),
                "relapse_false_fast_fraction": float(
                    np.mean(symmetric_fast[relapse])
                ),
            },
            "gate16_asymmetric": {
                "stable_fast_fraction": float(
                    np.mean(asymmetric_fast[stable])
                ),
                "relapse_false_fast_fraction": float(
                    np.mean(asymmetric_fast[relapse])
                ),
            },
            "predictive_selected_cue": {
                "stable_fast_fraction": float(
                    np.mean(selected_prediction[stable])
                ),
                "relapse_false_fast_fraction": float(
                    np.mean(selected_prediction[relapse])
                ),
            },
        },
        "paired_recovery_receipt": {
            "mean_hidden_motion_difference": float(
                np.mean(paired_motion_difference)
            ),
            "median_stable_motion_surprise": float(
                np.median(
                    test["motion_surprise"][:test_pairs]
                )
            ),
            "median_relapse_motion_surprise": float(
                np.median(
                    test["motion_surprise"][test_pairs:]
                )
            ),
            "median_stable_mean_confidence": float(
                np.median(
                    test["mean_confidence"][:test_pairs]
                )
            ),
            "median_relapse_mean_confidence": float(
                np.median(
                    test["mean_confidence"][test_pairs:]
                )
            ),
        },
        "first_post_split_receipt": {
            "median_stable_target": float(np.median(stable_future)),
            "median_relapse_target": float(np.median(relapse_future)),
        },
        "interpretation": (
            "Gate 17's temporal states remain limited to the ordinary "
            "confidence/common-fate target. Gate 18 adds diagnostics from the "
            "correspondence process itself. If motion-model surprise survives "
            "validation, held-out testing and cue shuffling while the old target "
            "states remain near chance, then the system has found genuinely new "
            "pre-relapse information rather than another way of smoothing the "
            "same evidence. The reauthorization receipt then interprets that "
            "prediction causally: old confidence-memory states may already be "
            "ready to restore fast authority, while the new cue can keep only "
            "the instability-marked relation cautious."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-pairs", type=int, default=40)
    parser.add_argument("--validation-pairs", type=int, default=30)
    parser.add_argument("--test-pairs", type=int, default=80)
    parser.add_argument("--train-seed", type=int, default=1801)
    parser.add_argument("--validation-seed", type=int, default=1802)
    parser.add_argument("--test-seed", type=int, default=1803)
    parser.add_argument("--shuffle-seed", type=int, default=1804)
    parser.add_argument("--json", type=str, default=None)
    args = parser.parse_args()

    report = run_reference(
        train_pairs=args.train_pairs,
        validation_pairs=args.validation_pairs,
        test_pairs=args.test_pairs,
        train_seed=args.train_seed,
        validation_seed=args.validation_seed,
        test_seed=args.test_seed,
        shuffle_seed=args.shuffle_seed,
    )
    print(json.dumps(report, indent=2))

    selected = report["selected_predictive_cue"]
    selected_test = report["held_out_balanced_accuracy"][selected]

    old_states = (
        "raw_state",
        "cumulative_state",
        "symmetric_state",
        "asymmetric_state",
    )
    assert max(
        report["held_out_balanced_accuracy"][name]
        for name in old_states
    ) < 0.65
    assert selected_test > 0.75
    assert (
        report["selected_cue_shuffled_balanced_accuracy"]
        < 0.62
    )

    if args.json:
        Path(args.json).write_text(
            json.dumps(report, indent=2) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
