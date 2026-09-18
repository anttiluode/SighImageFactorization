# Latest result — affinity binding boundary

## Gates 0--2: residue space boundary

The finite Sigh trajectory remains exactly reconstructible:

[
x_0=sum_k(x_k-x_{k+1})+x_N.
]

Gate 1 nevertheless found that global PCA -> ICA on the raw Sigh residue stack
does **not** improve additive source separation under nuisance.  Initial
three-seed attacked medians were:

| representation | recovery |
|---|---:|
| raw pixels | 0.93795 |
| orthogonal redundant | 0.93795 |
| matched-spectrum random | 0.94231 |
| Sigh residue | 0.67195 |

Gate 2 explains why.  The residue encoder is exact but non-isometric
(frame condition number **7.246**).  Tightening makes it an isometry and
reproduces the raw nonzero singular spectrum to **1.26e-15** relative error.

Boundary: **a global full-rank linear residue transform does not create object
separation.**

## Gate 3: same graph, two dynamics

A local RGB affinity graph was built from a 16x16 synthetic scene.  Two
spatially separate objects share the same colour, so raw colour+position
clustering cannot simply label them separately.

The graph was held fixed while the same random 8-D unit-vector field evolved
under either linear diffusion or a stripped-down vector Kuramoto projected
update.

Six-seed medians:

| measurement | linear | Kuramoto |
|---|---:|---:|
| ARI at step 128 | 0.421 | 0.444 |
| ARI at step 256 | 0.734 | 0.723 |
| ARI at step 512 | 1.000 | 1.000 |
| first checkpoint ARI >= .95 | 384 | 384 |

Controls:

- raw colour+xy median ARI: **0.439**
- graph-only spectral clustering median ARI: **1.000**

Both dynamic trajectories retain the exact telescoping residue receipt to
about 1e-15.

Interpretation: in this controlled static case the **input-derived graph**
contains the useful object partition.  Nonlinear synchronization is an
alternative local solver/binder, but does not add measurable grouping
information beyond matched linear diffusion.

## Gate 4: common fate writes a persistent operator

Each object now consists of two unlike appearance regions.  Static appearance
affinity tends to split those regions.

During a synthetic motion episode we observe local feature pairs plus oracle
motion vectors.  Unlike neighbouring features that share the same non-zero
motion write a persistent compatibility into the next affinity operator.

The objects are then moved to new positions and stopped.

Across three new placements:

| operator | median ARI |
|---|---:|
| static appearance | 0.75487 |
| **common-fate feature memory** | **1.00000** |
| part-shuffled motion | 0.75487 |
| coordinate-edge memory | 0.75487 |

The coherent-motion episode learns feature compatibilities 1<->2 and 1<->3
at **1.0**.  When the two parts are assigned opposing motion, the same
compatibilities are only about **1.27e-14**.

The coordinate-memory attacker stores the old absolute boundary edges.  It
does not help once the objects stop elsewhere.

### What this establishes

This is still a synthetic mechanism gate: motion is oracle-provided and
appearance is discretized.

But it isolates a substantially stronger primitive than the original
PCA/ICA idea:

[
oxed{
	ext{shared history}
ightarrow
	ext{operator rewrite}
ightarrow
	ext{future static grouping}
}
]

The operator creates the grouping.  The residue stack is now best viewed as a
record of the grouping trajectory rather than the source of grouping itself.

## Next attacker

Remove the scaffolding in order:

1. continuous RGB/local feature embeddings instead of discrete appearance IDs;
2. estimated correspondence/flow instead of oracle motion;
3. occlusion and clutter;
4. a learned persistent state that writes the affinity without an explicit
   hand-coded common-fate rule.

The strongest kill condition remains location transfer: a system that only
remembers where the moving blob was must fail when the same object stops
somewhere else.
