#!/usr/bin/env python3
"""Gate 1: does Sigh residue geometry help ICA under a tight PCA bottleneck?

The clean rank-3 receipt is intentionally boring: linear coordinate changes
should not create source information. The attacked condition adds structured
nuisance and pixel noise, then forces every method through the same 3-D PCA
bottleneck before ICA.

Controls:
  raw pixels
  orthogonal tall embedding
  random embedding with the same singular values as the Sigh encoder
  Sigh residue stack
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np

from factorization import best_assignment_score, pca_fastica
from residue import build_sigh_filter, encode_residues, encoder_matrix


def normalize(v: np.ndarray) -> np.ndarray:
    v = v - v.mean()
    return v / (np.linalg.norm(v) + 1e-30)


def make_sources(n: int) -> np.ndarray:
    y, x = np.mgrid[0:n, 0:n]
    xx = (x - (n - 1) / 2) / n
    yy = (y - (n - 1) / 2) / n

    g1 = np.exp(-((xx + 0.20) ** 2 + (yy + 0.08) ** 2) / 0.020)
    s1 = g1

    g2 = np.exp(-((xx - 0.16) ** 2 + (yy - 0.12) ** 2) / 0.030)
    s2 = g2 * np.cos(2 * np.pi * (4.0 * xx + 1.0 * yy))

    g3 = np.exp(-((xx + 0.03) ** 2 + (yy - 0.20) ** 2) / 0.025)
    s3 = g3 * np.cos(2 * np.pi * (7.0 * xx - 6.0 * yy))

    return np.stack([normalize(s1), normalize(s2), normalize(s3)])


def laplace(rng: np.random.Generator, shape: tuple[int, ...]) -> np.ndarray:
    return rng.laplace(0.0, 1.0 / np.sqrt(2.0), size=shape)


def smooth_noise(rng: np.random.Generator, count: int, n: int) -> np.ndarray:
    z = rng.standard_normal((count, n, n))
    fy = np.fft.fftfreq(n)
    yy, xx = np.meshgrid(fy, fy, indexing="ij")
    h = np.exp(-((xx * xx + yy * yy) / (0.10**2)))
    out = np.fft.ifft2(np.fft.fft2(z, axes=(-2, -1)) * h, axes=(-2, -1)).real
    out -= out.mean(axis=(-2, -1), keepdims=True)
    out /= np.linalg.norm(out, axis=(-2, -1), keepdims=True) + 1e-30
    return out


def make_dataset(
    n: int, samples: int, seed: int, attacked: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    sources = make_sources(n)
    coeff = laplace(rng, (samples, 3))
    images = np.einsum("sc,cnm->snm", coeff, sources)

    if attacked:
        nuisance_maps = smooth_noise(rng, 10, n)
        nuisance_coeff = rng.normal(0.0, 0.45, size=(samples, len(nuisance_maps)))
        images += np.einsum("sj,jnm->snm", nuisance_coeff, nuisance_maps)
        images += rng.normal(0.0, 0.035, size=images.shape)

    return images, coeff, sources


def matched_random_encoder(t: np.ndarray, seed: int) -> np.ndarray:
    """Random orientation with exactly T's singular values; same input/output sizes."""
    rng = np.random.default_rng(seed)
    in_dim, out_dim = t.shape
    s = np.linalg.svd(t, compute_uv=False)
    q_in, _ = np.linalg.qr(rng.standard_normal((in_dim, in_dim)))
    q_out, _ = np.linalg.qr(rng.standard_normal((out_dim, in_dim)))
    return q_in @ np.diag(s) @ q_out.T


def orthogonal_tall_encoder(in_dim: int, out_dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.standard_normal((out_dim, in_dim)))
    return q.T


def score_space(features: np.ndarray, coeff: np.ndarray, seed: int) -> dict:
    fit = pca_fastica(features, n_components=3, seed=seed)
    score, perm, corr = best_assignment_score(coeff, fit.activations)
    return {
        "mean_abs_latent_correlation": score,
        "assignment": perm,
        "correlation_matrix": corr.tolist(),
        "ica_converged": fit.converged,
        "ica_iterations": fit.iterations,
    }


def run_condition(
    n: int, depth: int, samples: int, seed: int, attacked: bool
) -> dict:
    images, coeff, _ = make_dataset(n, samples, seed, attacked)
    raw = images.reshape(samples, -1)
    filt = build_sigh_filter(n)
    sigh = encode_residues(images, filt, depth).flatten()

    t = encoder_matrix(n, depth, filt)
    ortho_t = orthogonal_tall_encoder(t.shape[0], t.shape[1], seed + 101)
    random_t = matched_random_encoder(t, seed + 202)

    spaces = {
        "raw_pixels": raw,
        "orthogonal_redundant": raw @ ortho_t,
        "matched_spectrum_random": raw @ random_t,
        "sigh_residue": sigh,
    }
    return {
        name: score_space(feat, coeff, seed + i * 17)
        for i, (name, feat) in enumerate(spaces.items())
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=16)
    p.add_argument("--depth", type=int, default=8)
    p.add_argument("--samples", type=int, default=512)
    p.add_argument("--seeds", type=int, default=8)
    p.add_argument("--json", type=str, default=None)
    args = p.parse_args()

    all_rows = []
    for seed in range(args.seeds):
        clean = run_condition(args.n, args.depth, args.samples, seed, attacked=False)
        attacked = run_condition(args.n, args.depth, args.samples, seed, attacked=True)
        all_rows.append({"seed": seed, "clean": clean, "attacked": attacked})

    methods = [
        "raw_pixels",
        "orthogonal_redundant",
        "matched_spectrum_random",
        "sigh_residue",
    ]
    summary = {}
    for condition in ("clean", "attacked"):
        summary[condition] = {}
        for method in methods:
            vals = [
                r[condition][method]["mean_abs_latent_correlation"]
                for r in all_rows
            ]
            summary[condition][method] = {
                "median": float(np.median(vals)),
                "mean": float(np.mean(vals)),
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
            }

    sigh = summary["attacked"]["sigh_residue"]["median"]
    matched = summary["attacked"]["matched_spectrum_random"]["median"]
    raw = summary["attacked"]["raw_pixels"]["median"]
    report = {
        "grid": [args.n, args.n],
        "depth": args.depth,
        "samples_per_seed": args.samples,
        "seeds": args.seeds,
        "metric": (
            "mean absolute correlation between recovered ICA activations and "
            "true independent source amplitudes after optimal permutation"
        ),
        "summary": summary,
        "attacked_sigh_minus_raw_median": sigh - raw,
        "attacked_sigh_minus_matched_random_median": sigh - matched,
        "rows": all_rows,
        "interpretation_rule": (
            "The clean condition is an equivalence receipt, not a win condition. "
            "A Sigh-specific effect requires attacked Sigh to beat both raw pixels "
            "and the matched-singular-spectrum random encoder."
        ),
    }
    print(json.dumps(report, indent=2))

    for row in all_rows:
        for condition in ("clean", "attacked"):
            for method in methods:
                assert row[condition][method]["ica_converged"]
                assert 0.0 <= row[condition][method]["mean_abs_latent_correlation"] <= 1.0000001

    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
