# Latest result — v0 linear factorization boundary

## Gate 0: exact residue representation

For the SighImageSuper high-pass operator, storing each departure

[
r_k=x_k-x_{k+1}
]

makes the finite trajectory exactly reconstructible:

[
x_0=sum_k r_k+x_N.
]

Measured relative reconstruction error at 32x32, depth 12: **1.64e-16**.

## Gate 1: PCA -> ICA source separation

Three independently varying localized sources were mixed, then every representation was forced through the same 3-D PCA bottleneck and the same NumPy FastICA implementation.

Controls were raw pixels, an orthogonal tall embedding, and a random tall encoder with **exactly the same singular values** as the Sigh residue encoder.

Initial 3-seed attacked medians:

| representation | score |
|---|---:|
| raw pixels | 0.93795 |
| orthogonal redundant | 0.93795 |
| matched-spectrum random | 0.94231 |
| **Sigh residue** | **0.67195** |

Sigh therefore **fails** the first source-separation claim. We do not tune the scene until it wins.

## Gate 2: why the failure happens

The residue encoder is exact but not isometric. Its frame operator has eigenvalues from **0.137998** to **1.0** (condition number **7.246**). The three synthetic source maps receive norm gains **0.994, 0.986, 0.682** respectively in residue space.

After Parseval/tight-frame correction,

[
T_{tight}=(TT^	op)^{-1/2}T,
]

we measure:

- `||T_tight T_tight^T - I||_2 = 8.55e-15`
- raw-vs-tight nonzero singular-spectrum relative error = `1.26e-15`
- all three source norm gains = 1 to numerical precision.

## Boundary established

A global full-rank **linear** residue coordinate transform cannot conjure independent objects. With an isometric frame it is just a redundant change of coordinates; with the raw frame it adds a particular metric bias, and Gate 1 shows that this bias can be harmful.

The next scientifically meaningful mechanisms must use something the equivalence argument does not remove: locality, grouping/ISA, sparse access to residue depth, nonlinear residue energy, learned noncommuting operator questions, or real world-time/motion.
