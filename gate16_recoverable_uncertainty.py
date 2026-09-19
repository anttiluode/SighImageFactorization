#!/usr/bin/env python3
"""Gate 16: recoverable local uncertainty with asymmetric eligibility dynamics.

Gate 15 established that track confidence can veto a dangerous false-common-fate
match. Its veto was one-shot: one history was compressed to one confidence
number, then the relation was either trusted or slowed.

This gate asks what happens over time when the *same candidate relation* moves
through:

    clear evidence -> ambiguous/occluded evidence -> clear evidence again

The desired behavior is stricter than a binary veto:

    1. clear history may earn fast relation authority,
    2. ambiguity should remove that authority quickly,
    3. one clean sample after ambiguity should NOT restore it immediately,
    4. repeated clean evidence should eventually restore it.

That fourth condition rules out a permanent veto. The third rules out a
fast symmetric smoother that forgets danger immediately.

Every observation still comes from noisy RGB and the Gate-6 template matcher.
For a clean observation the two candidates are rendered without occlusion or a
look-alike. For an ambiguous observation we reuse Gate 15's appearance jump,
partial occlusion and look-alike attacker.

A local observation is compressed to a continuous authority target:

    target_t = confidence_t       if estimated common-fate residual == 0
               0                  otherwise

The experiment compares four temporal policies:

    permanent veto:
        once target drops below the policy threshold, never recover

    cumulative mean:
        average all past target values equally

    symmetric eligibility:
        one EMA timescale for attack and recovery

    asymmetric eligibility:
        fast attack timescale, slower recovery timescale

Thresholds and timescales are selected on training sequences by balanced error
over four phase checkpoints. The held-out test then evaluates both the authority
state itself and the downstream D=8 relation-admission dynamics.

This is still a synthetic mechanism-isolation gate. The claim is about local
uncertainty dynamics, not natural-video tracking.
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
    evaluate_policy,
    learn_residual_threshold,
)
from gate14_rgb_predictive_relation_admission import (
    MOTION_CHOICES,
    estimate_pair_motion as estimate_clean_pair_motion,
    make_static_textures,
    render_frame,
)
from gate15_track_confidence_admission import (
    challenged_observation,
    generate_tracking_dataset,
)


CLEAR_PRE = 8
AMBIGUOUS = 4
CLEAR_POST = 8
EARLY_RECOVERY_INDEX = CLEAR_PRE + AMBIGUOUS
CHECKPOINTS = (
    CLEAR_PRE - 1,
    CLEAR_PRE + AMBIGUOUS - 1,
    EARLY_RECOVERY_INDEX,
    CLEAR_PRE + AMBIGUOUS + CLEAR_POST - 1,
)
CHECKPOINT_NAMES = (
    "pre_attack",
    "under_ambiguity",
    "one_clean_after_attack",
    "full_recovery",
)

THRESHOLD_GRID = np.array(
    [0.70, 0.75, 0.80, 0.85, 0.90],
    dtype=np.float64,
)
SYMMETRIC_TAU_GRID = np.array(
    [1.5, 2.0, 3.0, 4.0, 6.0, 8.0],
    dtype=np.float64,
)
ATTACK_TAU_GRID = np.array(
    [1.5, 2.0, 3.0, 4.0],
    dtype=np.float64,
)
RECOVERY_TAU_GRID = np.array(
    [2.0, 3.0, 4.0, 6.0, 8.0],
    dtype=np.float64,
)


def choose_clear_motion_pair(
    label: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Clear evidence is genuinely diagnostic: accidental motions differ."""
    idx_a = int(rng.integers(len(MOTION_CHOICES)))
    motion_a = MOTION_CHOICES[idx_a].copy()
    if label == 1:
        return motion_a, motion_a.copy()

    alternatives = [
        idx for idx in range(len(MOTION_CHOICES))
        if idx != idx_a
    ]
    idx_b = alternatives[int(rng.integers(len(alternatives)))]
    return motion_a, MOTION_CHOICES[idx_b].copy()


def clean_observation(
    label: int,
    rng: np.random.Generator,
    seed: int,
) -> tuple[float, float, float]:
    """One clean RGB tracking observation.

    Returns estimated disagreement, confidence, and exact hidden-motion recovery.
    """
    motion_a, motion_b = choose_clear_motion_pair(label, rng)
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
        tuple(pos_a + motion_a),
        tuple(pos_b + motion_b),
        textures,
        noise_seed=seed + 200_001,
    )

    estimate_a, estimate_b, confidence = estimate_clean_pair_motion(
        frame0,
        frame1,
    )
    residual = float(
        np.sum((estimate_a - estimate_b) ** 2)
    )
    exact = float(
        np.array_equal(estimate_a, motion_a)
        and np.array_equal(estimate_b, motion_b)
    )
    return residual, confidence, exact


def make_sequence(
    label: int,
    seed: int,
    residual_threshold: float,
) -> dict[str, np.ndarray]:
    """Generate one clear -> ambiguous -> clear RGB evidence trajectory."""
    rng = np.random.default_rng(seed)
    total = CLEAR_PRE + AMBIGUOUS + CLEAR_POST

    residual = np.empty(total, dtype=np.float64)
    confidence = np.empty(total, dtype=np.float64)
    exact = np.empty(total, dtype=np.float64)
    phase = np.empty(total, dtype=np.int64)

    for step in range(total):
        obs_seed = seed + 10_000 * (step + 1)
        if step < CLEAR_PRE or step >= CLEAR_PRE + AMBIGUOUS:
            e, q, correct = clean_observation(
                label,
                rng,
                obs_seed,
            )
            phase[step] = 0 if step < CLEAR_PRE else 2
        else:
            e, _oracle, q, correct = challenged_observation(
                label,
                rng,
                obs_seed,
            )
            phase[step] = 1

        residual[step] = e
        confidence[step] = q
        exact[step] = correct

    target = np.where(
        residual <= residual_threshold,
        confidence,
        0.0,
    )
    return {
        "residual": residual,
        "confidence": confidence,
        "target": target,
        "exact": exact,
        "phase": phase,
    }


def generate_sequence_dataset(
    trials: int,
    seed: int,
    residual_threshold: float,
) -> dict[str, np.ndarray]:
    """Balanced held-out relation trajectories."""
    rng = np.random.default_rng(seed)
    labels = np.arange(trials, dtype=np.int64) % 2
    rng.shuffle(labels)
    seeds = rng.integers(
        0,
        2**31 - 1,
        size=trials,
        dtype=np.int64,
    )

    total = CLEAR_PRE + AMBIGUOUS + CLEAR_POST
    target = np.empty((trials, total), dtype=np.float64)
    confidence = np.empty_like(target)
    residual = np.empty_like(target)
    exact = np.empty_like(target)

    for i, (label, trial_seed) in enumerate(zip(labels, seeds)):
        seq = make_sequence(
            int(label),
            int(trial_seed),
            residual_threshold,
        )
        target[i] = seq["target"]
        confidence[i] = seq["confidence"]
        residual[i] = seq["residual"]
        exact[i] = seq["exact"]

    return {
        "labels": labels,
        "target": target,
        "confidence": confidence,
        "residual": residual,
        "exact": exact,
    }


def desired_fast(labels: np.ndarray) -> np.ndarray:
    """Desired fast/cautious authority at the four checkpoints."""
    desired = np.zeros(
        (len(labels), len(CHECKPOINTS)),
        dtype=bool,
    )
    genuine = labels == 1
    desired[genuine, 0] = True
    desired[genuine, 3] = True
    return desired


def checkpoint_values(trace: np.ndarray) -> np.ndarray:
    return trace[:, CHECKPOINTS]


def balanced_checkpoint_error(
    predictions: np.ndarray,
    desired: np.ndarray,
) -> float:
    positive = desired
    negative = ~desired
    fnr = float(np.mean(~predictions[positive]))
    fpr = float(np.mean(predictions[negative]))
    return 0.5 * (fnr + fpr)


def cumulative_trace(target: np.ndarray) -> np.ndarray:
    denom = np.arange(1, target.shape[1] + 1, dtype=np.float64)
    return np.cumsum(target, axis=1) / denom[None, :]


def permanent_veto_trace(
    target: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """Current target is usable until the first sub-threshold observation."""
    trace = np.zeros_like(target)
    alive = np.ones(len(target), dtype=bool)

    for step in range(target.shape[1]):
        current = target[:, step]
        alive &= current >= threshold
        trace[:, step] = np.where(alive, current, 0.0)
    return trace


def symmetric_trace(
    target: np.ndarray,
    tau: float,
) -> np.ndarray:
    state = np.zeros(len(target), dtype=np.float64)
    trace = np.empty_like(target)
    for step in range(target.shape[1]):
        state += (target[:, step] - state) / tau
        trace[:, step] = state
    return trace


def asymmetric_trace(
    target: np.ndarray,
    attack_tau: float,
    recovery_tau: float,
) -> np.ndarray:
    """Fast downward updates, slower upward recovery."""
    state = np.zeros(len(target), dtype=np.float64)
    trace = np.empty_like(target)

    for step in range(target.shape[1]):
        observation = target[:, step]
        falling = observation < state
        tau = np.where(
            falling,
            attack_tau,
            recovery_tau,
        )
        state += (observation - state) / tau
        trace[:, step] = state
    return trace


def learn_cumulative_policy(
    target: np.ndarray,
    labels: np.ndarray,
) -> dict[str, float]:
    desired = desired_fast(labels)
    checkpoints = checkpoint_values(cumulative_trace(target))
    best = None
    for threshold in THRESHOLD_GRID:
        pred = checkpoints >= threshold
        error = balanced_checkpoint_error(pred, desired)
        key = (error, -float(threshold))
        if best is None or key < best[0]:
            best = (key, float(threshold))
    assert best is not None
    return {
        "threshold": best[1],
        "balanced_checkpoint_error": best[0][0],
    }


def learn_permanent_policy(
    target: np.ndarray,
    labels: np.ndarray,
) -> dict[str, float]:
    desired = desired_fast(labels)
    best = None
    for threshold in THRESHOLD_GRID:
        checkpoints = checkpoint_values(
            permanent_veto_trace(target, threshold)
        )
        pred = checkpoints >= threshold
        error = balanced_checkpoint_error(pred, desired)
        key = (error, -float(threshold))
        if best is None or key < best[0]:
            best = (key, float(threshold))
    assert best is not None
    return {
        "threshold": best[1],
        "balanced_checkpoint_error": best[0][0],
    }


def learn_symmetric_policy(
    target: np.ndarray,
    labels: np.ndarray,
) -> dict[str, float]:
    desired = desired_fast(labels)
    best = None
    for tau in SYMMETRIC_TAU_GRID:
        checkpoints = checkpoint_values(
            symmetric_trace(target, float(tau))
        )
        for threshold in THRESHOLD_GRID:
            pred = checkpoints >= threshold
            error = balanced_checkpoint_error(pred, desired)
            key = (
                error,
                float(tau),
                -float(threshold),
            )
            if best is None or key < best[0]:
                best = (
                    key,
                    float(tau),
                    float(threshold),
                )
    assert best is not None
    return {
        "tau": best[1],
        "threshold": best[2],
        "balanced_checkpoint_error": best[0][0],
    }


def learn_asymmetric_policy(
    target: np.ndarray,
    labels: np.ndarray,
) -> dict[str, float]:
    desired = desired_fast(labels)
    best = None

    for attack_tau in ATTACK_TAU_GRID:
        for recovery_tau in RECOVERY_TAU_GRID:
            checkpoints = checkpoint_values(
                asymmetric_trace(
                    target,
                    float(attack_tau),
                    float(recovery_tau),
                )
            )
            for threshold in THRESHOLD_GRID:
                pred = checkpoints >= threshold
                error = balanced_checkpoint_error(pred, desired)
                key = (
                    error,
                    float(attack_tau),
                    -float(recovery_tau),
                    -float(threshold),
                )
                if best is None or key < best[0]:
                    best = (
                        key,
                        float(attack_tau),
                        float(recovery_tau),
                        float(threshold),
                    )
    assert best is not None
    return {
        "attack_tau": best[1],
        "recovery_tau": best[2],
        "threshold": best[3],
        "balanced_checkpoint_error": best[0][0],
    }


def policy_predictions(
    target: np.ndarray,
    learned: dict[str, dict[str, float]],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    threshold = learned["permanent_veto"]["threshold"]
    trace = permanent_veto_trace(target, threshold)
    out["permanent_veto"] = (
        checkpoint_values(trace),
        checkpoint_values(trace) >= threshold,
    )

    threshold = learned["cumulative_mean"]["threshold"]
    trace = cumulative_trace(target)
    out["cumulative_mean"] = (
        checkpoint_values(trace),
        checkpoint_values(trace) >= threshold,
    )

    symmetric = learned["symmetric_ema"]
    trace = symmetric_trace(
        target,
        symmetric["tau"],
    )
    out["symmetric_ema"] = (
        checkpoint_values(trace),
        checkpoint_values(trace) >= symmetric["threshold"],
    )

    asymmetric = learned["asymmetric_eligibility"]
    trace = asymmetric_trace(
        target,
        asymmetric["attack_tau"],
        asymmetric["recovery_tau"],
    )
    out["asymmetric_eligibility"] = (
        checkpoint_values(trace),
        checkpoint_values(trace) >= asymmetric["threshold"],
    )

    return out


def checkpoint_report(
    labels: np.ndarray,
    authority: np.ndarray,
    fast: np.ndarray,
) -> dict:
    desired = desired_fast(labels)
    report = {
        "balanced_checkpoint_error": balanced_checkpoint_error(
            fast,
            desired,
        ),
        "checkpoints": {},
    }

    for j, name in enumerate(CHECKPOINT_NAMES):
        report["checkpoints"][name] = {
            "median_authority_genuine": float(
                np.median(authority[labels == 1, j])
            ),
            "median_authority_accidental": float(
                np.median(authority[labels == 0, j])
            ),
            "genuine_fast_fraction": float(
                np.mean(fast[labels == 1, j])
            ),
            "accidental_fast_fraction": float(
                np.mean(fast[labels == 0, j])
            ),
        }
    return report


def downstream_report(
    base: np.ndarray,
    labels: np.ndarray,
    fast: np.ndarray,
    gamma: float,
) -> dict:
    report = {}
    for j, name in enumerate(CHECKPOINT_NAMES):
        tau = np.where(
            fast[:, j],
            FAST_TAU,
            CAUTIOUS_TAU,
        )
        report[name] = evaluate_policy(
            base,
            labels,
            tau,
            gamma=gamma,
            contact_steps=32,
        )
    return report


def tracker_phase_receipt(
    data: dict[str, np.ndarray],
) -> dict:
    labels = data["labels"]
    target = data["target"]
    confidence = data["confidence"]
    exact = data["exact"]

    phases = {
        "clear_pre": slice(0, CLEAR_PRE),
        "ambiguous": slice(CLEAR_PRE, CLEAR_PRE + AMBIGUOUS),
        "clear_post": slice(CLEAR_PRE + AMBIGUOUS, None),
    }
    out = {}
    for name, slc in phases.items():
        out[name] = {
            "mean_target_genuine": float(
                np.mean(target[labels == 1, slc])
            ),
            "mean_target_accidental": float(
                np.mean(target[labels == 0, slc])
            ),
            "mean_confidence_genuine": float(
                np.mean(confidence[labels == 1, slc])
            ),
            "mean_confidence_accidental": float(
                np.mean(confidence[labels == 0, slc])
            ),
            "motion_accuracy_genuine": float(
                np.mean(exact[labels == 1, slc])
            ),
            "motion_accuracy_accidental": float(
                np.mean(exact[labels == 0, slc])
            ),
        }
    return out


def run_reference(
    train_trials: int,
    test_trials: int,
    dim: int,
    gamma: float,
    pre_steps: int,
    threshold_train_trials: int,
    threshold_seed: int,
    train_seed: int,
    test_seed: int,
    address_seed: int,
) -> dict:
    # Relearn Gate 15's residual threshold from attacked RGB histories rather
    # than hard-coding the previous CI receipt.
    threshold_train = generate_tracking_dataset(
        threshold_train_trials,
        threshold_seed,
    )
    residual_threshold, _ = learn_residual_threshold(
        threshold_train["rgb_residual"],
        threshold_train["labels"],
    )

    train = generate_sequence_dataset(
        train_trials,
        train_seed,
        residual_threshold,
    )
    learned = {
        "permanent_veto": learn_permanent_policy(
            train["target"],
            train["labels"],
        ),
        "cumulative_mean": learn_cumulative_policy(
            train["target"],
            train["labels"],
        ),
        "symmetric_ema": learn_symmetric_policy(
            train["target"],
            train["labels"],
        ),
        "asymmetric_eligibility": learn_asymmetric_policy(
            train["target"],
            train["labels"],
        ),
    }

    test = generate_sequence_dataset(
        test_trials,
        test_seed,
        residual_threshold,
    )
    predictions = policy_predictions(
        test["target"],
        learned,
    )

    base = prepare_addresses(
        trials=test_trials,
        dim=dim,
        seed=address_seed,
        gamma=gamma,
        pre_steps=pre_steps,
    )

    policies = {}
    downstream = {}
    for name, (authority, fast) in predictions.items():
        policies[name] = checkpoint_report(
            test["labels"],
            authority,
            fast,
        )
        downstream[name] = downstream_report(
            base,
            test["labels"],
            fast,
            gamma,
        )

    return {
        "train_trials": train_trials,
        "test_trials": test_trials,
        "dimension": dim,
        "clear_pre_observations": CLEAR_PRE,
        "ambiguous_observations": AMBIGUOUS,
        "clear_post_observations": CLEAR_POST,
        "checkpoint_names": list(CHECKPOINT_NAMES),
        "residual_threshold": residual_threshold,
        "learned_policies": learned,
        "tracker_phase_receipt": tracker_phase_receipt(test),
        "authority_policies": policies,
        "downstream_relation_dynamics": downstream,
        "interpretation": (
            "A permanent veto is safe under ambiguity but cannot recover. A "
            "cumulative history is reluctant to forget earlier clean evidence "
            "and therefore stays overconfident during the attack. A fast "
            "symmetric state can attack quickly but tends to re-authorize after "
            "too little clean evidence. The learned asymmetric eligibility "
            "state separates those timescales: authority falls quickly under "
            "local uncertainty, remains cautious after one clean sample, then "
            "returns after repeated clean evidence. The state remains local to "
            "the candidate relation, so other relations need not be slowed."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-trials", type=int, default=100)
    parser.add_argument("--test-trials", type=int, default=160)
    parser.add_argument("--threshold-train-trials", type=int, default=80)
    parser.add_argument("--dim", type=int, default=8)
    parser.add_argument("--gamma", type=float, default=0.20)
    parser.add_argument("--pre-steps", type=int, default=240)
    parser.add_argument("--threshold-seed", type=int, default=1600)
    parser.add_argument("--train-seed", type=int, default=1601)
    parser.add_argument("--test-seed", type=int, default=1602)
    parser.add_argument("--address-seed", type=int, default=1603)
    parser.add_argument("--json", type=str, default=None)
    args = parser.parse_args()

    report = run_reference(
        train_trials=args.train_trials,
        test_trials=args.test_trials,
        dim=args.dim,
        gamma=args.gamma,
        pre_steps=args.pre_steps,
        threshold_train_trials=args.threshold_train_trials,
        threshold_seed=args.threshold_seed,
        train_seed=args.train_seed,
        test_seed=args.test_seed,
        address_seed=args.address_seed,
    )
    print(json.dumps(report, indent=2))

    asym = report["authority_policies"]["asymmetric_eligibility"]
    permanent = report["authority_policies"]["permanent_veto"]
    cumulative = report["authority_policies"]["cumulative_mean"]
    symmetric = report["authority_policies"]["symmetric_ema"]

    pre = asym["checkpoints"]["pre_attack"]
    attack = asym["checkpoints"]["under_ambiguity"]
    early = asym["checkpoints"]["one_clean_after_attack"]
    final = asym["checkpoints"]["full_recovery"]

    assert pre["genuine_fast_fraction"] > 0.80
    assert attack["genuine_fast_fraction"] < 0.20
    assert early["genuine_fast_fraction"] < 0.35
    assert final["genuine_fast_fraction"] > 0.75
    assert final["accidental_fast_fraction"] < 0.15

    assert (
        asym["balanced_checkpoint_error"]
        < permanent["balanced_checkpoint_error"]
    )
    assert (
        asym["balanced_checkpoint_error"]
        < cumulative["balanced_checkpoint_error"]
    )
    assert (
        asym["balanced_checkpoint_error"]
        <= symmetric["balanced_checkpoint_error"]
    )

    asym_down = report["downstream_relation_dynamics"][
        "asymmetric_eligibility"
    ]
    assert (
        asym_down["under_ambiguity"][
            "accidental_false_merge_fraction"
        ]
        < 0.15
    )
    assert (
        asym_down["full_recovery"][
            "genuine_merge_fraction"
        ]
        > 0.65
    )

    if args.json:
        Path(args.json).write_text(
            json.dumps(report, indent=2) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
