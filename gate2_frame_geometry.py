#!/usr/bin/env python3
"""Gate 2: exact reconstruction is not an isometry."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np

from residue import build_sigh_filter, encoder_matrix
from gate1_source_separation import make_dataset, make_sources


def invsqrt_spd(g: np.ndarray) -> np.ndarray:
    vals, vecs = np.linalg.eigh(g)
    if vals.min() <= 1e-14:
        raise ValueError("frame operator is not numerically positive definite")
    return (vecs * (1.0 / np.sqrt(vals))) @ vecs.T


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=16)
    p.add_argument("--depth", type=int, default=8)
    p.add_argument("--samples", type=int, default=256)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--json", type=str, default=None)
    args = p.parse_args()

    filt = build_sigh_filter(args.n)
    t = encoder_matrix(args.n, args.depth, filt)
    gram = t @ t.T
    vals = np.linalg.eigvalsh(gram)
    tight = invsqrt_spd(gram) @ t
    tight_error = float(
        np.linalg.norm(tight @ tight.T - np.eye(t.shape[0]), ord=2)
    )

    sources = make_sources(args.n).reshape(3, -1)
    source_gains = (
        np.linalg.norm(sources @ t, axis=1) / np.linalg.norm(sources, axis=1)
    )
    source_tight_gains = (
        np.linalg.norm(sources @ tight, axis=1) / np.linalg.norm(sources, axis=1)
    )

    images, _, _ = make_dataset(args.n, args.samples, args.seed, attacked=True)
    raw = images.reshape(args.samples, -1)
    tight_features = raw @ tight
    s_raw = np.linalg.svd(raw - raw.mean(axis=0), compute_uv=False)
    s_tight = np.linalg.svd(
        tight_features - tight_features.mean(axis=0), compute_uv=False
    )
    sv_rel = float(
        np.linalg.norm(s_raw - s_tight) / (np.linalg.norm(s_raw) + 1e-30)
    )

    report = {
        "grid": [args.n, args.n],
        "depth": args.depth,
        "encoder_shape": list(t.shape),
        "frame_eigenvalue_min": float(vals.min()),
        "frame_eigenvalue_max": float(vals.max()),
        "frame_condition_number": float(vals.max() / vals.min()),
        "encoder_singular_min": float(np.sqrt(vals.min())),
        "encoder_singular_max": float(np.sqrt(vals.max())),
        "source_norm_gain_in_raw_residue_space": source_gains.tolist(),
        "source_norm_gain_after_tightening": source_tight_gains.tolist(),
        "tight_frame_identity_operator_error": tight_error,
        "raw_vs_tight_nonzero_singular_spectrum_relative_error": sv_rel,
        "interpretation": (
            "The residue transform is exactly reconstructible but not isometric. "
            "Tightening removes its metric weighting and makes the embedding an "
            "isometry, so global PCA sees the same nonzero singular spectrum as "
            "raw pixels. Linear residue coordinates alone do not create source "
            "separation."
        ),
    }
    print(json.dumps(report, indent=2))

    assert vals.min() > 0.0
    assert tight_error < 1e-10
    assert np.max(np.abs(source_tight_gains - 1.0)) < 1e-10
    assert sv_rel < 1e-10

    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
