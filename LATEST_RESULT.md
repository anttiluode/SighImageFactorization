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

## Gate 5: continuous RGB-pair memory

The discrete appearance IDs from Gate 4 were removed.

The learner now receives only:
- noisy local RGB pairs;
- oracle motion vectors.

Three motion frames produce continuous pair examples. Same non-zero motion is a
positive affinity target; differential motion is negative evidence; two
stationary pixels are ignored. A nearest-pair radius is calibrated solely from
leave-one-out distances among the training examples.

At test time the objects stop at new positions with fresh RGB noise. Affinities
are thresholded locally and connected components are measured.

Across 12 unseen static scenes:

| operator | median ARI | median components | mean object fragmentation |
|---|---:|---:|---:|
| static RGB affinity | 0.48112 | 3 | 2.0 |
| **coherent common-fate memory** | **0.99660** | **3** | **1.0** |
| part-split motion | 0.94871 | 5 | 2.0 |

The coherent condition's minimum ARI over the 12 scenes is **0.89702**; every
true object remains a single connected component in every scene.

The attacker is revealing: part-split motion still reaches high ARI because it
learns foreground/background boundaries, but each true object remains split
into two pieces. Therefore ARI alone would have let us overclaim.

New boundary:

[
\boxed{\text{foreground separation} \neq \text{object binding}}
]

For this lineage, object binding now means at least both:
1. high agreement with ownership labels; and
2. low fragmentation of each true object.

Gate 5 is still synthetic and still uses oracle motion, but the persistent
operator write now works in continuous noisy appearance space rather than a
hand-coded feature-ID table.

## Gate 6: oracle motion removed

Gate 6 receives only frame (t) and frame (t+1).

Each frame is first split into local appearance-coherent connected regions.
For every region, RGB template matching searches a small displacement window in
the next frame. Only confident non-zero translations are allowed to write
persistent cross-appearance affinity.

The estimator was evaluated against ground truth **without exposing that ground
truth to the learner**.

GitHub Actions measured:

```text
moving-component translation accuracy     1.000
moving-component median confidence        0.995734
background median confidence              0.037952
time-shuffled median confidence           0.001830
estimated boundary-pair writes            48
time-shuffled writes                       0
```

Six later static scenes at unseen positions:

| condition | median ARI | components | object fragmentation |
|---|---:|---:|---:|
| static appearance | 0.97368 | 5 | 2.0 |
| oracle motion | **1.00000** | **3** | **1.0** |
| **RGB-estimated motion** | **1.00000** | **3** | **1.0** |
| time-shuffled frames | 0.97368 | 5 | 2.0 |

Thus the Gate-5 mechanism no longer requires an externally supplied velocity
field in this controlled setting. The moving image pair itself supplies enough
evidence to rewrite the later static grouping operator.

The remaining scaffold is now narrower and clearer:

[
\boxed{
\text{appearance regions}
\rightarrow
\text{estimated common motion}
\rightarrow
\text{persistent relation}
}
]

The next question is whether the appearance-region presegmentation can be
weakened without reintroducing boundary-motion hallucinations.

## Gate 7: occlusion reaches the instance-binding wall

The learned common-fate operator from Gate 6 was still local. Gate 7 inserts a
one-pixel background occluder through each two-part object and adds a novel
distractor at the same short gap.

Three mechanisms are compared on six scenes:

| mechanism | median ARI | components | fragmentation | collision scenes |
|---|---:|---:|---:|---:|
| local common-fate memory | 0.97992 | 6 | 1.667 | 0 / 6 |
| region-relation gap bridge | **1.00000** | **4** | **1.000** | **2 / 6** |
| proximity-only gap bridge | 0.94467 | 2 | 1.000 | 6 / 6 |

The local operator cannot reconnect visible islands once direct contact is
removed. A nonlocal relation bridge fixes fragmentation, and it rejects the
novel distractor better than proximity alone.

However, the stronger per-scene attacker exposes a deeper failure: in two
layouts the two *real* objects come close diagonally. Their visible parts
satisfy a genuinely learned common-fate relation, so the generic region rule
can connect the wrong instances.

An earlier implementation allowed one pixel match to authorize the nonlocal
edge. Replacing that with whole-region mean descriptors removed the accidental
pixel-level failure but **did not remove these two instance collisions**. That
is evidence that the remaining problem is not a noisy descriptor threshold.

Boundary:

[
\boxed{\text{correct feature relation} \neq \text{correct instance binding}}
]

A generic operator can know that red+green or red+blue parts compose objects
and still not know *which red belongs to which green/blue right now*.

This sharply motivates the next variable: an instance-specific persistent
address—phase, oscillator orientation, slot identity, track state, or an
equivalent dynamical code.

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
