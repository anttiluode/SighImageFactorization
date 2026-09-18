# SighImageFactorization

**From purification residues to self-written grouping operators.**

This repo is a direct branch of [SighImageSuper](https://github.com/anttiluode/SighImageSuper).

The starting observation is still tiny.  If an iterative process is

[
x_{k+1}=A_k x_k,
]

save every departure

[
r_k=x_k-x_{k+1}.
]

Then, regardless of whether the operators change,

[
oxed{x_0=sum_{k=0}^{N-1}r_k+x_N}.
]

The purification trajectory can therefore be kept instead of discarded.

The repo began by asking whether this **temporal residue space** itself makes
independent sources or objects easier to discover.  Gates 0--2 established an
important negative boundary.  Gates 3--4 now move the interesting mechanism
from the coordinate transform into the **operator that performs the grouping**.

## Gate 0 — exact residue receipt

`gate0_exact_residue.py` uses the same radial high-pass preset as
SighImageSuper and verifies machine-precision reconstruction from all saved
departures plus the terminal state.

Measured relative reconstruction error: about **1.6e-16**.

## Gate 1 — PCA / ICA source-separation attack

`gate1_source_separation.py` forces four representations through the same
three-dimensional PCA bottleneck and the same NumPy FastICA implementation:

1. raw pixels;
2. orthogonal redundant coordinates;
3. a random encoder with the same singular values as the Sigh encoder;
4. the Sigh residue stack.

Initial attacked medians over three seeds:

| representation | latent recovery |
|---|---:|
| raw pixels | 0.938 |
| orthogonal redundant | 0.938 |
| matched-spectrum random | 0.942 |
| **Sigh residue** | **0.672** |

So **residue time did not create source separation**.

## Gate 2 — frame geometry explains the failure

The residue transform is exactly reconstructible but not isometric.  Its frame
operator has eigenvalues roughly **0.138 .. 1.000**, condition number **7.246**,
and one of the Gate-1 source maps receives only about **0.682** norm gain.

After tightening the frame,

[
T_{tight}=(TT^	op)^{-1/2}T,
]

we measure:

```text
||T_tight T_tight^T - I||_2       8.55e-15
raw vs tight singular spectrum     1.26e-15 relative error
```

Thus a global full-rank linear residue transform cannot conjure objects merely
by renaming coordinates.

## The AKOrN clue

Miyato, Lowe, Geiger & Welling's
[Artificial Kuramoto Oscillatory Neurons](https://arxiv.org/abs/2410.13821)
(ICLR 2025) is useful here because its object binding does **not** live in a
fixed linear image transform.  It uses repeated projected oscillator dynamics,
input-conditioned stimuli and learned convolutional/attention connectivity.

That suggests a stricter question for this repo:

> **If the input writes the interaction operator, what is due to the operator
> itself and what is due to nonlinear synchronization?**

## Gate 3 — same graph: diffusion versus Kuramoto

`gate3_same_graph_binding.py` makes a local RGB affinity graph from a synthetic
image.  Two spatially separate objects deliberately have the same colour, so a
global colour/position clustering cannot simply assign them unique identities.

The graph is then held fixed.

The same random **8-D unit-vector field** is evolved in two ways:

```text
linear:
    X <- (1-gamma) X + gamma P X

vector Kuramoto-style:
    Y <- P X
    X <- normalize(X + gamma Proj_X(Y))
```

Both get the same graph, initialization, step size, checkpoints and k-means
readout.  A static spectral clustering result is included as the graph-only
reference.

Six-seed initial result:

```text
raw colour+xy median ARI                 0.439
spectral graph median ARI                1.000

step 128:
    linear diffusion median ARI          0.421
    vector Kuramoto median ARI           0.444

step 256:
    linear diffusion median ARI          0.734
    vector Kuramoto median ARI           0.723

step 512:
    linear diffusion median ARI          1.000
    vector Kuramoto median ARI           1.000

median first checkpoint with ARI >= .95:
    linear                              384
    Kuramoto                            384
```

So in this stripped-down setting **synchronization is not yet the extra
ingredient**.  The input-derived graph contains the useful partition and both
local dynamics eventually reveal it on essentially the same timescale.

This is useful because it prevents us from crediting an oscillator merely for
solving a clustering problem already encoded in its connectivity.

The residue identity still holds for either trajectory: saving every state
departure reconstructs its initial distributed state to floating-point
precision.

## Gate 4 — common fate writes the operator

`gate4_common_fate_write.py` tests the more interesting hypothesis.

Each synthetic object contains two appearance regions.  Static appearance
affinity therefore wants to split the object.  During a controlled motion
episode, the learner observes local feature pairs and their motion vectors.

If two unlike neighbouring features repeatedly have the same non-zero motion,

[
	ext{common fate}
longrightarrow
	ext{persistent feature-pair compatibility}.
]

The objects are then placed at **new coordinates and stopped**.  The same
spectral readout is applied to four operators:

```text
static appearance
common-fate feature memory
part-shuffled motion
absolute-coordinate edge memory
```

Across three new placements:

| operator | median ARI |
|---|---:|
| static appearance | 0.755 |
| **common-fate feature memory** | **1.000** |
| part-shuffled motion | 0.755 |
| coordinate memory | 0.755 |

The learned relation is specifically:

```text
shared feature 1 <-> object-1 feature 2    ~1.0
shared feature 1 <-> object-2 feature 3    ~1.0
```

When the two parts are given opposing motion, those compatibilities collapse
to about `1e-14`.

This gate is deliberately modest.  It uses **oracle motion** and discrete
appearance types.  It does not claim natural-image object discovery.

What it does establish is the mechanism we wanted to isolate:

> **history can rewrite a grouping operator in feature space, and that operator
> can bind a later static scene at new locations.**

That is the first place in this repo where history changes what a future image
means.

## Gate 5 — continuous RGB-pair common fate

`gate5_continuous_common_fate.py` removes the discrete feature-ID table from
Gate 4. The learner receives only noisy continuous RGB pairs plus oracle motion.

Three motion frames generate local pair examples:

```text
same non-zero motion                    target affinity 1
different motion / moving vs stationary target affinity 0
stationary vs stationary                 no evidence
```

At test time the objects are stopped at new coordinates with fresh RGB noise.
A pair can override ordinary appearance affinity only if it lies within a
nearest-pair radius calibrated entirely from the training examples.

The readout is deliberately local: threshold affinity at 0.5 and take connected
components. This exposes a subtle failure that ARI alone hides.

Across 12 unseen noisy static scenes:

| operator | median ARI | median components | object fragmentation |
|---|---:|---:|---:|
| static RGB affinity | 0.481 | 3 | 2.0 |
| **coherent common-fate memory** | **0.997** | **3** | **1.0** |
| part-split motion | 0.949 | 5 | 2.0 |

The part-split attacker is the interesting one. It learns that moving foreground
differs from stationary background, so ARI becomes deceptively high, but every
true object remains split into its two appearance parts. Only coherent common
fate binds those parts into one connected object.

So Gate 5 adds another measurement rule:

> **foreground separation is not object binding; always measure fragmentation.**

The continuous memory is still intentionally primitive: nearest RGB-pair
examples, oracle motion, and synthetic colours. But the discrete type lookup is
gone.

## Gate 6 — RGB-only motion write

`gate6_estimated_motion_write.py` removes the oracle motion vectors from Gate 5.

The learner receives only consecutive RGB frames. It first forms local
appearance-coherent connected regions, then estimates one small translation per
region by template matching into the next frame. Two adjacent appearance
regions write a persistent affinity only when both translations are confident,
non-zero, and equal.

This keeps the remaining scaffold explicit: **appearance-region decomposition
still comes before motion estimation**. We have removed oracle motion, not yet
solved dense optical flow.

GitHub Actions result:

```text
moving component motion accuracy          1.000
moving component median confidence        0.9957
background component median confidence    0.0380
time-shuffled median confidence           0.00183
estimated-motion training examples        48
time-shuffled training examples           0
```

On later static scenes at new positions:

| operator | median ARI | median components | object fragmentation |
|---|---:|---:|---:|
| static appearance | 0.974 | 5 | 2.0 |
| oracle motion | **1.000** | **3** | **1.0** |
| **RGB-estimated motion** | **1.000** | **3** | **1.0** |
| time-shuffled frames | 0.974 | 5 | 2.0 |

So in this controlled world, the true motion vector is no longer needed. The
system can infer which appearance regions move together from the frames
themselves, write that relation into the operator, and reuse it after the
motion evidence is gone.

The time-shuffled attacker is important: unrelated frame pairs do not merely
produce a worse write; the confidence gate refuses to write anything at all.

## Gate 7 — occlusion reveals the instance-binding boundary

Gate 6 can learn that two unlike appearance regions belong together from RGB
motion alone. But that learned relation is still **generic**: it says which
types of parts have belonged together, not which particular instance owns a
part now.

`gate7_occlusion_relational_bridge.py` attacks that distinction.

At test time a one-pixel background occluder removes the direct contact between
the two visible parts of each learned object. A novel clutter patch is placed at
the same short geometric gap.

The purely local learned operator cannot cross the missing row:

```text
local learned memory
    median ARI                 0.980
    components                 6
    mean object fragmentation  1.667
    cross-object collisions    0
```

A relation-specific nonlocal bridge uses **region-level** appearance relations
learned during common motion. It repairs fragmentation:

```text
region relation bridge
    median ARI                 1.000
    components                 4
    mean object fragmentation  1.000
```

But the per-scene attacker matters more than the median. In **2 of 6** layouts,
the two real objects themselves come close enough that a legitimately learned
part relation also matches the wrong instance, producing a cross-object bind.

A proximity-only bridge is worse: it cross-binds in **6 of 6** scenes and has
median ARI **0.945**.

So Gate 7 establishes another boundary:

[
\boxed{\text{part relation} \neq \text{instance identity}}
]

The important failure is not noise. The relation can be correct and still be
insufficient. Once multiple compatible objects coexist, a system needs an
instance-specific persistent variable—a phase, slot, track, address, or
equivalent state—to say *which copy of the relation is currently this object*.

This is the point where the AKOrN/synchrony clue becomes more than analogy:
oscillatory state is potentially useful not because synchronization is a better
diffusion solver (Gate 3 says it was not), but because **relative phase/state can
serve as an instance-binding address while generic feature relations remain
shared**.

## Current picture

```text
SighImageSuper
    |
    | save what disappears
    v
exact residue trajectory
    |
    +-- global PCA / ICA ----------------------> boundary: no magic objects
    |
    v
input-derived affinity operator
    |
    +-- linear diffusion
    +-- vector synchrony ----------------------> same graph, same answer
    |
    v
common fate writes persistent affinity
    |
    v
later static grouping at a new location
```

The new working hypothesis is no longer

> residue time itself is the object space.

It is:

> **the operator creates the grouping; the residue records the grouping
> trajectory; history can rewrite the operator.**

The next gate should remove one piece of scaffolding at a time: replace
discrete appearance types with continuous local features, replace oracle motion
with estimated correspondence/flow, then ask whether the learned operator
survives clutter, occlusion and novel arrangements.

## Requirements

- Python 3.11+
- NumPy

Run:

```bash
python -m unittest discover -s tests -v
python gate0_exact_residue.py
python gate1_source_separation.py --samples 256 --seeds 3
python gate2_frame_geometry.py
python gate3_same_graph_binding.py
python gate4_common_fate_write.py
```

No SciPy or scikit-learn is required.
