#!/usr/bin/env python3
"""Gate 20: online contingency adaptation with delayed outcomes.

Gate 19 allowed a magic labeled calibration batch after the world changed.
Gate 20 removes that pause. Predictions remain live while the meaning of the
Gate-18 motion-surprise observable changes over time, and the eventual
stable/relapse outcome is revealed only after a delay.

The stream contains five consecutive regimes:

    baseline:
        original 0.10 / 0.70 instability->relapse contingency

    preserved covariate shift:
        alternate motion vocabulary + extra RGB noise, same contingency

    reversed:
        0.70 / 0.10, so the old predictor is actively wrong

    broken:
        0.40 / 0.40, so motion surprise contains no future information

    restored:
        original 0.10 / 0.70 contingency returns

Four online policies compete:

    frozen:
        never update the Gate-18 mapping

    all_history:
        refit one scalar threshold/polarity from every revealed outcome

    recent_window:
        refit from a finite recent window

    reliability_gated_recent:
        fit on the older portion of the recent window, validate on its newer
        portion, and grant predictive authority only when held-out recent
        accuracy exceeds a trust threshold. Otherwise the relation stays
        cautious instead of fitting noise.

No policy sees the current episode's future label when deciding authority.
Labels arrive OUTCOME_DELAY episodes later.

The intended mechanism is not "always adapt." It is:
    * ignore harmless covariate shift when predictive meaning survives,
    * withdraw authority while a reversal invalidates the old mapping,
    * relearn the reversed mapping from delayed outcomes,
    * become cautious when no mapping is predictive,
    * and recover when the old contingency returns.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from gate18_pre_relapse_predictive_cue import (
    apply_rule,
    balanced_accuracy,
    learn_univariate_rule,
    make_pairs as make_gate18_pairs,
    paired_dataset as gate18_paired_dataset,
)
from gate19_predictive_cue_portability import (
    SCENARIOS,
    make_scenario_pairs,
)


OUTCOME_DELAY = 8
RECENT_WINDOW = 48
MIN_REFIT = 24
GUARD_FIT_COUNT = 28
GUARD_VALIDATION_COUNT = 20
GUARD_TRUST_ACCURACY = 0.70
TRUST_STREAK = 6

STREAM_SEGMENTS = (
    ("baseline", "baseline"),
    ("preserved_covariate_shift", "preserved_covariate_shift"),
    ("reversed_contingency", "reversed_contingency"),
    ("broken_contingency", "broken_contingency"),
    ("restored_contingency", "baseline"),
)


def flatten_pairs(
    pairs: list[dict],
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Turn paired stable/relapse worlds into a randomized balanced stream."""
    stable = np.asarray(
        [
            pair["stable"]["rgb_motion_surprise"]
            for pair in pairs
        ],
        dtype=np.float64,
    )
    relapse = np.asarray(
        [
            pair["relapse"]["rgb_motion_surprise"]
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
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(values))
    return values[order], labels[order]


def build_stream(
    pairs_per_segment: int,
    seed: int,
) -> dict:
    """Generate a nonstationary RGB-derived surprise/outcome stream."""
    all_values = []
    all_labels = []
    segment_ranges = {}
    cursor = 0

    for index, (segment_name, scenario_name) in enumerate(
        STREAM_SEGMENTS
    ):
        spec = SCENARIOS[scenario_name]
        pairs = make_scenario_pairs(
            pairs_per_segment,
            seed=seed + 10_000 * index,
            stable_p=float(spec["stable_p"]),
            relapse_p=float(spec["relapse_p"]),
            motion_vocabulary=str(spec["motion_vocabulary"]),
            extra_sensor_noise=float(spec["extra_sensor_noise"]),
        )
        values, labels = flatten_pairs(
            pairs,
            seed=seed + 10_000 * index + 1,
        )
        start = cursor
        stop = cursor + len(values)
        segment_ranges[segment_name] = {
            "start": start,
            "stop": stop,
            "scenario": scenario_name,
        }
        cursor = stop
        all_values.append(values)
        all_labels.append(labels)

    return {
        "values": np.concatenate(all_values),
        "labels": np.concatenate(all_labels),
        "segments": segment_ranges,
    }


def _has_both_classes(labels: np.ndarray) -> bool:
    return bool(
        np.any(labels == 0)
        and np.any(labels == 1)
    )


def fit_rule_or_prior(
    values: np.ndarray,
    labels: np.ndarray,
    prior_rule: dict[str, float | str],
) -> dict[str, float | str]:
    if len(values) < MIN_REFIT or not _has_both_classes(labels):
        return prior_rule
    return learn_univariate_rule(values, labels)


def guarded_recent_state(
    values: np.ndarray,
    labels: np.ndarray,
    prior_rule: dict[str, float | str],
) -> tuple[dict[str, float | str], bool, float]:
    """Cross-time validate a recent candidate before granting authority."""
    needed = GUARD_FIT_COUNT + GUARD_VALIDATION_COUNT
    if len(values) < needed:
        return prior_rule, True, 1.0

    recent_values = values[-needed:]
    recent_labels = labels[-needed:]
    fit_values = recent_values[:GUARD_FIT_COUNT]
    fit_labels = recent_labels[:GUARD_FIT_COUNT]
    validation_values = recent_values[GUARD_FIT_COUNT:]
    validation_labels = recent_labels[GUARD_FIT_COUNT:]

    if (
        not _has_both_classes(fit_labels)
        or not _has_both_classes(validation_labels)
    ):
        return prior_rule, False, 0.5

    candidate = learn_univariate_rule(
        fit_values,
        fit_labels,
    )
    validation_prediction = apply_rule(
        validation_values,
        candidate,
    )
    validation_accuracy = balanced_accuracy(
        validation_labels,
        validation_prediction,
    )
    active = validation_accuracy >= GUARD_TRUST_ACCURACY
    return candidate, active, validation_accuracy


def online_run(
    values: np.ndarray,
    labels: np.ndarray,
    frozen_rule: dict[str, float | str],
    outcome_delay: int = OUTCOME_DELAY,
) -> dict:
    """Run all policies causally over one shared stream."""
    n = len(values)
    policy_names = (
        "frozen",
        "all_history",
        "recent_window",
        "reliability_gated_recent",
    )
    predictions = {
        name: np.zeros(n, dtype=bool)
        for name in policy_names
    }
    active = {
        name: np.ones(n, dtype=bool)
        for name in policy_names
    }
    polarity = {
        name: np.empty(n, dtype=object)
        for name in policy_names
    }
    validation_accuracy = np.full(n, np.nan, dtype=np.float64)

    revealed_values: list[float] = []
    revealed_labels: list[int] = []

    for t in range(n):
        reveal = t - outcome_delay
        if reveal >= 0:
            revealed_values.append(float(values[reveal]))
            revealed_labels.append(int(labels[reveal]))

        history_values = np.asarray(
            revealed_values,
            dtype=np.float64,
        )
        history_labels = np.asarray(
            revealed_labels,
            dtype=np.int64,
        )

        rules: dict[str, dict[str, float | str]] = {
            "frozen": frozen_rule,
            "all_history": fit_rule_or_prior(
                history_values,
                history_labels,
                frozen_rule,
            ),
            "recent_window": fit_rule_or_prior(
                history_values[-RECENT_WINDOW:],
                history_labels[-RECENT_WINDOW:],
                frozen_rule,
            ),
        }

        guarded_rule, guarded_active, guarded_validation = (
            guarded_recent_state(
                history_values,
                history_labels,
                frozen_rule,
            )
        )
        rules["reliability_gated_recent"] = guarded_rule
        active["reliability_gated_recent"][t] = guarded_active
        validation_accuracy[t] = guarded_validation

        for name in policy_names:
            polarity[name][t] = rules[name]["polarity"]
            if not active[name][t]:
                # Cautious means no fast relation authority.
                predictions[name][t] = False
            else:
                predictions[name][t] = bool(
                    apply_rule(
                        np.asarray([values[t]], dtype=np.float64),
                        rules[name],
                    )[0]
                )

    return {
        "predictions": predictions,
        "active": active,
        "polarity": polarity,
        "validation_accuracy": validation_accuracy,
    }


def interval_metrics(
    labels: np.ndarray,
    prediction: np.ndarray,
    active: np.ndarray,
    polarity: np.ndarray,
) -> dict:
    stable = labels == 1
    relapse = labels == 0
    stable_fast = float(np.mean(prediction[stable]))
    relapse_fast = float(np.mean(prediction[relapse]))

    active_accuracy = None
    selected = active
    if np.any(selected):
        selected_labels = labels[selected]
        selected_prediction = prediction[selected]
        if _has_both_classes(selected_labels):
            active_accuracy = balanced_accuracy(
                selected_labels,
                selected_prediction,
            )

    low_fraction = float(np.mean(polarity == "low"))
    high_fraction = float(np.mean(polarity == "high"))

    return {
        "balanced_relation_accuracy": 0.5
        * (stable_fast + (1.0 - relapse_fast)),
        "stable_fast_fraction": stable_fast,
        "relapse_false_fast_fraction": relapse_fast,
        "active_fraction": float(np.mean(active)),
        "active_balanced_accuracy": (
            float(active_accuracy)
            if active_accuracy is not None
            else None
        ),
        "low_polarity_fraction": low_fraction,
        "high_polarity_fraction": high_fraction,
    }


def first_trusted_polarity_streak(
    active: np.ndarray,
    polarity: np.ndarray,
    expected_polarity: str,
    streak: int = TRUST_STREAK,
) -> int | None:
    good = active & (polarity == expected_polarity)
    for start in range(0, len(good) - streak + 1):
        if bool(np.all(good[start : start + streak])):
            return start
    return None


def segment_report(
    stream: dict,
    online: dict,
) -> dict:
    labels = stream["labels"]
    reports = {}

    for segment_name, meta in stream["segments"].items():
        start = int(meta["start"])
        stop = int(meta["stop"])
        midpoint = start + (stop - start) // 2
        segment_labels = labels[start:stop]
        tail_labels = labels[midpoint:stop]

        expected_polarity = (
            "high"
            if segment_name == "reversed_contingency"
            else (
                None
                if segment_name == "broken_contingency"
                else "low"
            )
        )

        policy_reports = {}
        for name in online["predictions"]:
            prediction = online["predictions"][name]
            active = online["active"][name]
            polarity = online["polarity"][name]

            whole = interval_metrics(
                segment_labels,
                prediction[start:stop],
                active[start:stop],
                polarity[start:stop],
            )
            tail = interval_metrics(
                tail_labels,
                prediction[midpoint:stop],
                active[midpoint:stop],
                polarity[midpoint:stop],
            )

            latency = None
            if expected_polarity is not None:
                latency = first_trusted_polarity_streak(
                    active[start:stop],
                    polarity[start:stop],
                    expected_polarity,
                )

            policy_reports[name] = {
                "whole_segment": whole,
                "tail_half": tail,
                "trusted_expected_polarity_latency": latency,
            }

        reports[segment_name] = {
            "start": start,
            "stop": stop,
            "episodes": stop - start,
            "expected_polarity": expected_polarity,
            "policies": policy_reports,
        }

    return reports


def run_reference(
    gate18_train_pairs: int,
    pairs_per_segment: int,
    gate18_train_seed: int,
    stream_seed: int,
    outcome_delay: int,
) -> dict:
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

    stream = build_stream(
        pairs_per_segment=pairs_per_segment,
        seed=stream_seed,
    )
    online = online_run(
        stream["values"],
        stream["labels"],
        frozen_rule,
        outcome_delay=outcome_delay,
    )
    segments = segment_report(stream, online)

    return {
        "gate18_train_pairs": gate18_train_pairs,
        "pairs_per_segment": pairs_per_segment,
        "episodes_per_segment": 2 * pairs_per_segment,
        "outcome_delay": outcome_delay,
        "recent_window": RECENT_WINDOW,
        "guard_fit_count": GUARD_FIT_COUNT,
        "guard_validation_count": GUARD_VALIDATION_COUNT,
        "guard_trust_accuracy": GUARD_TRUST_ACCURACY,
        "frozen_gate18_rule": frozen_rule,
        "segments": segments,
        "interpretation": (
            "A frozen predictor survives pure covariate shift but becomes "
            "anti-predictive after a contingency reversal. All-history "
            "recalibration is slow because obsolete outcomes retain equal weight. "
            "A recent window can relearn reversal but remains willing to fit "
            "noise when the cue becomes uninformative. Reliability-gated recent "
            "calibration uses delayed outcomes to withdraw predictive authority "
            "when a recent fitted mapping fails chronological validation, then "
            "restore authority when a stable predictive contingency is learned "
            "again."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate18-train-pairs", type=int, default=40)
    parser.add_argument("--pairs-per-segment", type=int, default=60)
    parser.add_argument("--gate18-train-seed", type=int, default=1801)
    parser.add_argument("--stream-seed", type=int, default=2001)
    parser.add_argument("--outcome-delay", type=int, default=OUTCOME_DELAY)
    parser.add_argument("--json", type=str, default=None)
    args = parser.parse_args()

    report = run_reference(
        gate18_train_pairs=args.gate18_train_pairs,
        pairs_per_segment=args.pairs_per_segment,
        gate18_train_seed=args.gate18_train_seed,
        stream_seed=args.stream_seed,
        outcome_delay=args.outcome_delay,
    )
    print(json.dumps(report, indent=2))

    segments = report["segments"]
    guarded = "reliability_gated_recent"
    recent = "recent_window"

    assert (
        segments["baseline"]["policies"][guarded]["tail_half"][
            "balanced_relation_accuracy"
        ]
        > 0.72
    )
    assert (
        segments["preserved_covariate_shift"]["policies"][guarded][
            "tail_half"
        ]["balanced_relation_accuracy"]
        > 0.70
    )
    assert (
        segments["reversed_contingency"]["policies"]["frozen"][
            "tail_half"
        ]["balanced_relation_accuracy"]
        < 0.35
    )
    assert (
        segments["reversed_contingency"]["policies"][recent][
            "tail_half"
        ]["balanced_relation_accuracy"]
        > 0.68
    )
    assert (
        segments["reversed_contingency"]["policies"][guarded][
            "tail_half"
        ]["balanced_relation_accuracy"]
        > 0.65
    )
    assert (
        segments["broken_contingency"]["policies"][guarded][
            "tail_half"
        ]["active_fraction"]
        < 0.50
    )
    assert (
        segments["restored_contingency"]["policies"][guarded][
            "tail_half"
        ]["balanced_relation_accuracy"]
        > 0.68
    )

    if args.json:
        Path(args.json).write_text(
            json.dumps(report, indent=2) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
