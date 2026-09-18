# SighImageFactorization

**Perfect-reconstruction residue trajectories for PCA, ICA, independent subspaces, and source separation.**

This repo takes one small step sideways from [SighImageSuper](https://github.com/anttiluode/SighImageSuper).

SighImageSuper repeatedly applies a known linear spectral operator,

[
x_{k+1}=A x_k,
]

and watches the image purify toward the operator's maximum-gain mode. The obvious movie keeps only the current state. This repo keeps **what disappears**:

[
r_k=x_k-x_{k+1}.
]

Then the whole forward trajectory is an exact finite-depth representation:

[
oxed{x_0=sum_{k=0}^{N-1}r_k+x_N}.
]

Nothing needs to be numerically inverted. The purification loop carries its own bookkeeping.

## Why this might matter

For a fixed operator,

[
r_k=(I-A)A^k x_0.
]

So one iterative operator automatically generates a filter bank indexed by **operator lifetime**: what dies immediately, what survives several passes, and what reaches the terminal state.

The scientific question is not whether this representation is lossless; Gate 0 makes that algebraically trivial. The question is:

> **Does this residue geometry make independently generated structure easier to factor under constrained observation than equally large, equally conditioned generic linear coordinates?**

That wording is deliberate. A linear transform cannot create source information. PCA/ICA are microscopes here, not magic. If the residue coordinates help, the gain must come from an inductive bias in *which directions are made important under a bottleneck*.

## Gate 0 — exact residue receipt

`gate0_exact_residue.py` uses the same radial high-pass preset as SighImageSuper and verifies

```text
x0 -> x1 -> x2 -> ... -> xN
 |     |     |
 r0    r1    r2          + terminal xN
```

with machine-precision reconstruction.

## Gate 1 — source separation with attackers

`gate1_source_separation.py` constructs three independently varying localized source images with different carrier/persistence structure. Every representation is forced through the **same three-dimensional PCA bottleneck** before the same NumPy FastICA implementation.

Four representations compete:

1. **raw pixels** — ordinary baseline;
2. **orthogonal redundant embedding** — same expanded dimensionality, no metric distortion;
3. **matched-spectrum random encoder** — same sizes and exactly the same singular values as the Sigh residue encoder, but random singular directions;
4. **Sigh residue stack** — actual temporal-residue geometry.

The first attacked run is a **negative result**:

| representation | attacked median latent recovery (3 seeds) |
|---|---:|
| raw pixels | 0.938 |
| orthogonal redundant | 0.938 |
| matched-spectrum random | 0.942 |
| **Sigh residue** | **0.672** |

The clean rank-3 condition is about 0.99 for every representation, as expected.

## Gate 2 — frame geometry explains the loss

Exact reconstruction is not the same as an isometric representation. The stacked residue encoder is a redundant **frame** whose Euclidean norm weights different spectral directions differently.

For the three source maps in Gate 1, the induced norm gains are approximately:

```text
source 1   0.994
source 2   0.986
source 3   0.682
```

Gate 2 tightens the frame,

[
T_{tight}=(TT^	op)^{-1/2}T,
]

and measures:

```text
frame eigenvalue range             0.138 .. 1.000
frame condition number             7.246
tight-frame identity error         8.55e-15
raw vs tight singular spectrum     1.26e-15 relative error
```

Once tightened, the expanded residue representation preserves every input inner product. Global PCA therefore sees the same nonzero singular spectrum as raw pixels.

> **A full-rank linear residue transform alone does not create source separation. Any useful gain must come from its metric bias or from a downstream constraint/nonlinearity that can exploit the residue axis.**

That changes the next experiment. The promising directions are local/grouped residue features (ISA), sparse/limited access to depth channels, nonlinear residue energy, and real temporal motion where object parts can bind by common trajectory.

## What this repo is *not* claiming

It is not claiming that ICA components are semantic objects. The stronger future hypothesis is that an object could become a **coherent subspace of residue trajectories**, especially once real time/motion is added.

A plausible progression is:

```text
Sigh residue transform
        -> PCA whitening/compression
        -> ICA (independent factors)
        -> ISA / grouped factors
        -> moving-source tests
        -> learn the operator questions themselves
```

The interesting endpoint would be: **do not merely learn objects; learn a sequence of questions under which objects separate themselves.**

## Lineage

```text
SighImageSuper
  purification trajectory
        |
        v
save every departure r_k
        |
        v
SighImageFactorization
  exact residue space
        |
        +--> PCA / ICA attackers
        +--> frame-geometry boundary
        +--> independent subspaces / motion (next)
```

The high-pass filter implementation is intentionally copied mathematically from SighImageSuper's Gate 0 so this is a real branch of that experiment rather than a new toy with the same name.

## Requirements

- Python 3.11+
- NumPy

Run:

```bash
python -m unittest discover -s tests -v
python gate0_exact_residue.py --json results/gate0.json
python gate1_source_separation.py --samples 256 --seeds 3 --json results/gate1.json
python gate2_frame_geometry.py --json results/gate2.json
```

No SciPy or scikit-learn is required; the small symmetric FastICA implementation is included so the gate stays inspectable.
