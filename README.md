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

## Gate 8 — persistent phase is useful as an instance address

Gate 3 already showed that vector synchrony is **not** a better solver for a
fixed grouping graph. Gate 7 showed a different missing variable: a generic
learned part relation can be correct while still binding the wrong object
instance.

`gate8_persistent_phase_address.py` isolates that role.

Two separate objects are deliberately given the **same velocity** and share
the same generic learned part relations. Velocity therefore cannot be the
identity. Robust RGB region correspondence finds the moving appearance
components, and local common-fate connectivity separates them into two
disconnected instance groups.

Each group receives an arbitrary 2-D unit-vector address on the circle. That
address is carried with the object into the next frame. The objects then stop,
occlusion removes direct part contact, and a learned nonlocal relation is
allowed to bridge only when both visible regions also carry the same address.

A first attempt failed because mean region matching gave the huge background
region the same apparent translation as the moving objects. Replacing that with
interior-pixel, median photometric matching removed boundary-ownership
contamination. A second attempt exposed another conflation: using an arbitrary
confidence threshold to decide object membership omitted correctly moving
parts. The final gate treats the largest appearance region explicitly as the
current substrate/reference and lets non-zero robust displacement, not a
confidence cutoff, define moving candidates.

Six-scene GitHub Actions result:

| mechanism | median ARI | components | fragmentation | collision scenes |
|---|---:|---:|---:|---:|
| local memory | 0.980 | 6 | 1.667 | 0 / 6 |
| relation only | 0.958 | 3 | 1.000 | **6 / 6** |
| **persistent phase** | **1.000** | **4** | **1.000** | **0 / 6** |
| collapsed phase | 0.958 | 3 | 1.000 | **6 / 6** |
| reset phase | 0.980 | 6 | 1.667 | 0 / 6 |

The two discovered instance groups are present in every tested scene.

This is the first gate where phase has a specifically demonstrated function:

[
\boxed{\text{phase-like persistent state} = \text{instance address}}
]

Collapse the addresses and the old cross-binding error returns. Remove the
addresses when motion stops and the object fragments again.

This is **not** yet an AKOrN reproduction or evidence that biological phase is
the unique solution. A slot, track ID, persistent vector, or another dynamical
address could play the same abstract role. What the gate establishes is the
need for an instance-specific state variable beyond generic feature relations.

## Gate 9 — oscillator dynamics generate the address

Gate 8 still assigned one arbitrary phase after explicitly enumerating each
common-fate group. `gate9_emergent_phase_address.py` removes that helper.

Each moving appearance component starts with a random scalar phase. The signed
coupling graph is written only from local relations:

```text
adjacent + same non-zero motion    attractive coupling +1.0
other moving-component pairs       weak repulsion      -0.2
```

Both object instances deliberately move with the **same velocity**, so velocity
cannot act as the instance label.

Plain Kuramoto relaxation produces the address. Across the six CI scenes:

```text
minimum attractive-edge phase cosine     1.000000
maximum repulsive-edge phase cosine     -1.000000
moving appearance components             4
```

The resulting phase is then carried into exactly the same occluded static
binding problem used by Gate 8.

| mechanism | median ARI | components | fragmentation | collision scenes |
|---|---:|---:|---:|---:|
| local memory | 0.97992 | 6 | 1.667 | 0 / 6 |
| relation only | 0.95829 | 3 | 1.000 | **6 / 6** |
| frozen random phase | 0.97992 | 6 | 1.667 | 0 / 6 |
| **emergent synchronized phase** | **1.00000** | **4** | **1.000** | **0 / 6** |
| collapsed initial phase | 0.95829 | 3 | 1.000 | **6 / 6** |
| reset phase | 0.97992 | 6 | 1.667 | 0 / 6 |

The frozen-random attacker matters: **persistent arbitrary vectors are not
enough under this readout**. The two parts of an object must first synchronize
into a shared address.

The collapsed-phase attacker is the complementary intervention. With exact
initial symmetry every sine difference is zero, so the oscillator dynamics
cannot invent an instance distinction. The generic relation then cross-binds
exactly as in Gate 7.

So the demonstrated mechanism is now:

```text
local common-fate relation
        -> signed oscillator coupling
        -> within-instance synchrony
        -> between-instance phase separation
        -> persistent instance address
        -> correct later occluded binding
```

This is still not an AKOrN reproduction, and phase is still not claimed as the
unique representation. What changed is that the address is no longer assigned
by an external instance enumerator: **the local oscillator dynamics generate
it.**

## Gate 10 — local-only vector address capacity

Gate 9 still used weak **global repulsion** between moving components that were
not local same-motion neighbours. Gate 10 deletes those edges completely.

Each moving appearance component begins as a random (D)-dimensional unit
vector. The only interaction is local attraction between adjacent components
that share the same non-zero motion. Disconnected objects never communicate.

With (D=8), twelve occluded static test scenes give:

```text
moving components                  4
local attractive edges             2
minimum synchronized-edge cosine   1.000000
maximum cross-instance cosine      0.514826
address collisions                 0 / 12
```

| mechanism | median ARI | components | fragmentation | collision scenes |
|---|---:|---:|---:|---:|
| local memory | 0.97992 | 6 | 1.667 | 0 / 12 |
| relation only | 0.95829 | 3 | 1.000 | **12 / 12** |
| frozen random vector | 0.97992 | 6 | 1.667 | 0 / 12 |
| **local vector synchrony** | **1.00000** | **4** | **1.000** | **0 / 12** |
| reset vector | 0.97992 | 6 | 1.667 | 0 / 12 |

So explicit cross-instance repulsion is unnecessary. Each disconnected island
can synchronize locally and inherit an independent orientation through random
symmetry breaking.

The remaining failure mode is **address collision**. A 20,000-pair Monte Carlo
sweep at the same cosine >= 0.99 identity threshold measures:

| vector dimension | accidental collision fraction |
|---:|---:|
| 2 | 0.04435 |
| 3 | 0.00450 |
| 4 | 0.00055 |
| 8 | 0 / 20,000 |
| 16 | 0 / 20,000 |

The 99th-percentile cross-instance cosine falls from **0.9996 in 2-D** to
**0.7598 in 8-D** and **0.5480 in 16-D**.

That gives high-dimensional oscillator orientation a very concrete function in
this lineage:

> **dimension is instance-address capacity.**

It is not evidence that 8-D is optimal, nor that biological binding uses this
code. It shows why a multidimensional unit-vector state can be useful even when
all interactions are strictly local and attractive.

## Gate 11 — temporary contact needs a relation timescale

Gate 10 succeeds because different instances are disconnected. Gate 11 attacks
that assumption.

After each object's two local components have synchronized into a D=8 address,
the two objects temporarily become **local same-motion neighbours**. Under the
Gate-10 rule this is a perfectly legal new attractive edge.

Two edge policies compete:

```text
instantaneous:
    new local edge weight = 1 immediately

persistence-gated:
    w[t+1] = w[t] + (1 - w[t]) / tau
    tau = 64
```

The second rule is not claimed to be optimal. It asks one narrow question:
should new relational evidence have to persist before it is allowed to rewrite
an already-established instance identity?

Across **5,000 random D=8 address trials**:

| same-motion contact | instantaneous collision | persistence-gated collision |
|---:|---:|---:|
| 8 steps | 0.04% | 0% |
| 16 steps | 13.02% | 0% |
| 32 steps | **100%** | **0.08%** |
| 64 steps | 100% | 99.98% |
| 128 steps | 100% | 100% |

Collision means the two object addresses cross the same cosine >= 0.99 identity
threshold used by Gate 10.

At 32 contact steps the median cross-instance cosine is:

```text
instantaneous coupling        0.999612
persistence-gated coupling    0.866135
```

Then the contact is removed and both objects evolve for another 120 steps using
only their original internal edges. The collision fractions remain **100%**
versus **0.08%**. With no separating force, an identity collapse does not
repair itself merely because the objects separate again.

The important result is therefore a new boundary:

> **address capacity is not enough; instance identity also needs a timescale
> for admitting new relations.**

The persistence gate is deliberately finite rather than absolute. Long enough
shared contact eventually wins. It acts as a temporal causal filter: transient
contact should not instantly redefine identity, while sustained evidence may.


## Gate 12 — learn the relation timescale, then hit the causal boundary

Gate 11 used a hand-picked `tau=64`. Gate 12 removes that choice.

The training world contains only encounter histories:

```text
transient contacts:   8, 16, 24, 32 steps
persistent contacts: 64, 96, 128 steps
```

For each candidate admission timescale, the exact Gate-11 vector dynamics are
replayed. The selected timescale minimizes

```text
false merge on transient contacts
+
failure to merge on persistent contacts.
```

Across 3,000 training address pairs the minimum occurs at **tau = 64**. On a
separate 5,000-pair held-out set, the mean transient false-merge rate is
**0.00015** and the persistent non-merge rate is **0.0**.

So the slow relation timescale does not have to be inserted by hand: it can be
selected from prior merge/split statistics.

Then the attacker gives the learner two possible futures with an identical
48-step contact prefix:

```text
future A: contact ends now      -> keep two identities
future B: contact continues     -> merge as early as possible
```

At the end of the shared prefix, an age-only policy has exactly the same state
in both futures. With the learned tau, about **40%** of address pairs have
already crossed the merge threshold at this point. That number is
simultaneously the early-merge success rate for future B and the false-merge
rate for future A.

For balanced paired futures, no age-only classifier can exceed **50%** accuracy
before the histories diverge.

The new boundary is therefore:

> **experience can learn how cautious relation admission should be, but contact
> age alone cannot predict whether a currently identical relation is about to
> split or persist. Fast safe binding needs another predictive cue.**

That cue is the next thing to earn: motion consistency, appearance continuity,
prediction error, consequence, or some learned local statistic must supply
information that is absent from age itself.


## Gate 13 — predictive relation admission

Gate 12 proved that contact age alone cannot resolve two futures with an
identical observed prefix. Gate 13 therefore adds information that exists
**before contact**: short noisy common-fate history.

Every test encounter has exactly the same **32-step same-motion contact**.
Duration has no class information.

Before contact, each candidate relation supplies twelve noisy local velocity
samples. Genuine relations share one latent velocity process; accidental pairs
have independent latent histories. The learner receives only

```text
mean_t ||v_a(t) - v_b(t)||^2
```

plus, during training only, the eventual merge/split outcome. A single threshold
is learned from 3,000 previous encounters.

Reference result:

```text
learned residual threshold             0.186985
held-out cue accuracy                  0.9922
genuine predicted genuine              0.99961
accidental predicted genuine           0.01550
```

That prediction does not directly merge anything. It chooses which relation
admission clock gets authority:

```text
predicted genuine       tau = 16
predicted accidental    tau = 96
Gate-12 age-only        tau = 64
```

On 5,000 held-out D=8 address pairs, all with the same 32-step contact:

| policy | genuine merge | accidental false merge | balanced relation accuracy |
|---|---:|---:|---:|
| age-only tau=64 | 0.00078 | 0.00041 | 0.50019 |
| always-fast tau=16 | 0.80306 | 0.80783 | 0.49761 |
| always-cautious tau=96 | 0.00000 | 0.00041 | 0.49980 |
| **predictive fast/cautious** | **0.80267** | **0.01306** | **0.89481** |
| shuffled predictive cue | 0.40094 | 0.42391 | 0.48852 |

This is the first gate in the lineage where the system can bind a likely genuine
new relation **quickly** without paying the same merge rate on an equal-duration
accidental contact.

The shuffled attacker is decisive: merely owning a fast and a slow clock does
not help. The advantage disappears when the learned pre-contact evidence is
detached from the relation it describes.

A sensor-noise shift from 0.15 to 0.20 reduces cue accuracy to **0.8624** and
genuine early merging to **0.5853**, while accidental false merging stays low at
**0.00160**. So the mechanism degrades rather than remaining magically robust.

The new working boundary is:

> **relation admission can be accelerated by predictive history, but its quality
> is limited by the quality and transfer of that predictive observable.**

This gate is still synthetic. It does not yet infer the cue from RGB video.
That is now the obvious next attacker.


## Gate 14 — predictive evidence comes from RGB history

Gate 13 still received a generated velocity-history residual. Gate 14 removes
that direct motion input and reuses the RGB correspondence machinery introduced
in Gate 6.

Each candidate relation is observed for four pre-contact frame transitions.
Every frame contains two small textured foreground regions plus a textured
background. The learner receives only noisy RGB.

For each consecutive frame pair:

```text
RGB
 -> appearance-connected regions
 -> local template-match translation estimate
 -> confidence
```

The two candidate foreground regions then produce

```text
e_rgb = mean_t ||vhat_a(t) - vhat_b(t)||^2
```

and one threshold is learned from past outcomes. True velocities are used only
for an oracle upper-bound measurement.

The accidental class is deliberately ambiguous: at each pre-contact step it
copies the partner's motion with probability 0.5. Some accidental relations
therefore have exactly the same observable four-step common-fate history as a
genuine relation.

CI reference, **120 training / 200 held-out histories**:

```text
learned RGB threshold              0.0
RGB cue accuracy                   0.975
oracle-motion cue accuracy         0.975
RGB false-negative rate            0.000
RGB false-positive rate            0.050
mean translation confidence        0.99598
valid estimated transitions        1.000
```

The equality with the oracle is important. In this controlled image world the
remaining 5% cue error is not caused by motion estimation; it is ambiguity in
the observed history itself.

Every later contact is still exactly **32 steps**.

| policy | genuine merge | accidental false merge | balanced relation accuracy |
|---|---:|---:|---:|
| age-only tau=64 | 0.000 | 0.000 | 0.500 |
| always-fast tau=16 | 0.770 | 0.780 | 0.495 |
| **RGB-predictive fast/cautious** | **0.770** | **0.030** | **0.870** |
| shuffled RGB cue | 0.420 | 0.410 | 0.505 |

So the predictive mechanism has now moved back into the image/world loop:

> **visual history can decide how quickly a new relation is allowed to rewrite
> an established instance address.**

Shuffling the image-derived cue removes the benefit, while the always-fast
control shows why prediction is needed at all.

This still uses easy foreground-region discovery and stable local texture. The
next useful attacker is no longer “can RGB supply motion?” It is whether the
predictive relation survives **appearance ambiguity, partial occlusion, and
track uncertainty** without relying on clean pre-segmented regions.

## Current picture

```text
SighImageSuper
    |
    | save what disappears
    v
exact residue trajectory
    |
    +-- global PCA / ICA ----------------------> boundary: linear coordinates
    |
    v
input-derived affinity operator
    |
    +-- diffusion vs synchrony ----------------> same graph, same grouping
    |
    v
common fate writes persistent affinity
    |
    +-- RGB-only correspondence ---------------> oracle motion removed
    +-- occlusion / relational bridge ---------> nonlocal identity boundary
    |
    v
persistent oscillator / vector address
    |
    +-- local-only synchronization ------------> dimension = address capacity
    +-- temporary contact ---------------------> relation admission needs time
    |
    v
learned global admission timescale
    |
    +-- identical-prefix attacker -------------> age cannot predict relation fate
    |
    v
predictive relation admission
    |
    +-- pre-contact common fate ---------------> fast vs cautious edge clock
    +-- shuffled cue ---------------------------> advantage disappears
    |
    v
RGB predictive relation admission
    |
    +-- estimated visual motion ---------------> matches oracle cue here
    +-- matched-duration contact --------------> 77% genuine / 3% accidental
```

The working hypothesis is now:

> **the operator creates the grouping; local dynamics create an instance address;
> visual history can rewrite the operator and can also control how fast a new
> relation is allowed to rewrite that address.**

Residues remain useful as an exact trajectory microscope, but they are no longer
being asked to manufacture objects by themselves.

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
python gate5_continuous_common_fate.py --samples 6
python gate6_estimated_motion_write.py --test-scenes 6
python gate7_occlusion_relational_bridge.py --scenes 6
python gate8_persistent_phase_address.py --scenes 6
python gate9_emergent_phase_address.py --scenes 6
python gate10_local_vector_address.py --scenes 12 --capacity-trials 20000
python gate11_contact_persistence.py --trials 5000
python gate12_learned_relation_timescale.py --train-trials 3000 --test-trials 5000
python gate13_predictive_relation_admission.py --train-trials 3000 --test-trials 5000
python gate14_rgb_predictive_relation_admission.py --train-trials 120 --test-trials 200
```

No SciPy or scikit-learn is required.
