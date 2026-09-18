#!/usr/bin/env python3
"""Gate 0: the purification trajectory becomes an exact finite-depth representation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np

from residue import build_sigh_filter, encode_residues


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=32)
    p.add_argument("--depth", type=int, default=12)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--json", type=str, default=None)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    x0 = rng.standard_normal((args.n, args.n))
    filt = build_sigh_filter(args.n)
    stack = encode_residues(x0, filt, args.depth)
    recon = stack.reconstruct()

    err = float(np.linalg.norm(recon - x0) / (np.linalg.norm(x0) + 1e-30))
    energies = np.sum(stack.residuals**2, axis=(-2, -1))
    report = {
        "grid": [args.n, args.n],
        "depth": args.depth,
        "relative_reconstruction_error": err,
        "residual_energy_by_depth": energies.tolist(),
        "terminal_energy": float(np.sum(stack.terminal**2)),
        "identity": "x0 = sum_k (x_k - x_{k+1}) + x_N",
        "interpretation": "Sigh purification need not discard information if every departure is retained.",
    }
    print(json.dumps(report, indent=2))
    assert err < 1e-12
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
