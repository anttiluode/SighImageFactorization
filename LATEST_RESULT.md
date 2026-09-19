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

## Gate 8: phase becomes an instance address

Gate 7's failure was not that the learned part relation was wrong. It was that
the same correct relation could apply to multiple nearby object instances.

Gate 8 therefore adds one extra state variable per discovered moving instance:
an arbitrary 2-D unit vector (phase-like address).

The attacker is deliberately strong: **both objects move with the same
velocity**, so velocity itself cannot label them.

The first implementation failed because boundary ownership changes caused the
large background component to inherit the objects' apparent translation. The
motion estimator was changed to score eroded region interiors with a median
photometric loss. The second implementation failed because a confidence
threshold omitted correctly moving parts; confidence is now diagnostic only,
while the single largest appearance region is explicitly treated as the
current substrate/reference.

With those estimator confounds removed, local common-fate connectivity
discovers two disconnected moving groups in every tested scene. Each gets a
different unit-vector address, which is carried forward after motion stops.

Six-scene CI result:

| condition | median ARI | components | fragmentation | collision scenes |
|---|---:|---:|---:|---:|
| local memory | 0.97992 | 6 | 1.667 | 0 / 6 |
| relation only | 0.95829 | 3 | 1.000 | **6 / 6** |
| **persistent phase** | **1.00000** | **4** | **1.000** | **0 / 6** |
| collapsed phase | 0.95829 | 3 | 1.000 | **6 / 6** |
| reset phase | 0.97992 | 6 | 1.667 | 0 / 6 |

The intervention is unusually clean:

- **collapse all phases** -> generic relation cross-binding returns;
- **reset phase at the stop** -> objects fragment;
- **preserve distinct phase** -> correct binding survives occlusion.

So the role of phase is now sharply different from Gate 3:

[
\boxed{
\text{phase is not the grouping solver; phase carries instance identity}
}
]

The experiment does not establish phase as the unique representation. A slot,
track, persistent vector, or another address could substitute. It establishes
that generic learned relations need an instance-specific persistent variable
when several compatible instances coexist.

## Gate 9: phase address emerges from local oscillator dynamics

Gate 8 still contained an external enumerator: common-fate groups were found
first, then each group was simply assigned a different phase.

Gate 9 removes that assignment. Every moving appearance component starts with a
random phase. The only phase dynamics are:

[
\dot\theta_i = \sum_j J_{ij}\sin(\theta_j-\theta_i)
]

with (J_{ij}=+1) for adjacent components sharing the same non-zero motion and
(J_{ij}=-0.2) for other moving-component pairs.

The two objects have the **same velocity**, so velocity cannot carry identity.

Six-scene CI receipt:

```text
minimum attractive-edge cosine       1.000000
maximum repulsive-edge cosine       -1.000000
moving component count               4
```

| condition | median ARI | components | fragmentation | collision scenes |
|---|---:|---:|---:|---:|
| local memory | 0.97992 | 6 | 1.667 | 0 / 6 |
| relation only | 0.95829 | 3 | 1.000 | **6 / 6** |
| frozen random phase | 0.97992 | 6 | 1.667 | 0 / 6 |
| **emergent phase** | **1.00000** | **4** | **1.000** | **0 / 6** |
| collapsed initial phase | 0.95829 | 3 | 1.000 | **6 / 6** |
| reset phase | 0.97992 | 6 | 1.667 | 0 / 6 |

This closes the main cheat in Gate 8.

- Persisting **random unsynchronized** phase does not bind the two parts.
- Letting local attractive coupling synchronize the parts produces a shared
  within-object address.
- Weak repulsion pushes the two disconnected instances to opposite phase.
- Erasing that state after motion restores fragmentation.
- Starting in exact collapsed symmetry leaves the dynamics at that symmetry
  fixed point and restores cross-binding.

The useful claim is therefore narrower and stronger:

[
\boxed{
\text{local dynamics can generate a persistent instance address}
}
]

rather than merely “a phase variable can store an externally assigned ID.”

## Gate 10: global repulsion removed

Gate 9 generated the phase address dynamically but still used weak all-to-all
repulsion between moving components that were not local same-motion neighbours.

Gate 10 removes that global information entirely.

Each moving appearance component starts as a random (D)-dimensional unit
vector. Only local adjacent same-motion components attract. The two object
instances are disconnected and never communicate.

Twelve-scene CI result at (D=8):

| condition | median ARI | components | fragmentation | collision scenes |
|---|---:|---:|---:|---:|
| local memory | 0.97992 | 6 | 1.667 | 0 / 12 |
| relation only | 0.95829 | 3 | 1.000 | **12 / 12** |
| frozen random vector | 0.97992 | 6 | 1.667 | 0 / 12 |
| **local vector synchrony** | **1.00000** | **4** | **1.000** | **0 / 12** |
| reset vector | 0.97992 | 6 | 1.667 | 0 / 12 |

The two local attractive edges synchronize to cosine **1.0**. Across the 12
actual scenes the maximum cosine between the two independent instance addresses
is only **0.514826**, far below the 0.99 same-address threshold.

Address-capacity attacker, 20,000 independent address pairs:

```text
D=2    collision fraction  0.04435
D=3                        0.00450
D=4                        0.00055
D=8                        0.00000
D=16                       0.00000
```

At D=8 the maximum cosine in all 20,000 random address pairs is **0.95625**;
at D=16 it is **0.80901**.

New boundary:

[
\boxed{
\text{local synchrony creates the address; dimension supplies address capacity}
}
]

No cross-instance repulsion or object enumerator is required in this controlled
case. Frozen random vectors still fragment, showing that capacity without
within-object synchronization is not enough.

## Gate 11: temporary contact attacks the address

Gate 10's local-only address has one hidden assumption: separate instances never
become local same-motion neighbours.

Gate 11 initializes exactly the Gate-10 object:

```text
object A: two locally synchronized D=8 components
object B: two locally synchronized D=8 components
cross-instance coupling: none
```

Across 5,000 random trials the internal cosine is 1.0 to numerical precision,
while the maximum pre-contact cross-instance cosine is **0.92560** and there are
zero collisions at the 0.99 identity threshold.

Then one temporary same-motion edge is inserted between the two established
objects.

### Instantaneous edge admission fails

If a new local edge receives full coupling weight immediately:

```text
contact 8 steps      collision fraction   0.0004
contact 16 steps                          0.1302
contact 32 steps                          1.0000
```

At 32 steps the median cross-instance cosine is **0.999612**. The two instance
addresses have effectively become one.

Removing contact and running 120 additional steps with only the original
within-object edges does not repair the loss: collision remains **1.0000**.

### Persistence-gated edge admission

Give the new edge a slow eligibility state instead:

[
w_{t+1}=w_t+\frac{1-w_t}{64}.
]

Results:

```text
contact 8 steps      collision fraction   0.0000
contact 16 steps                          0.0000
contact 32 steps                          0.0008
contact 64 steps                          0.9998
contact 128 steps                         1.0000
```

At 32 steps the new edge has reached weight **0.39586**, but the median
cross-instance cosine is still only **0.86614**. After contact is removed and
120 separation steps pass, collision remains only **0.0008**.

The gate therefore does not make identity permanent. Sustained evidence
eventually wins. It creates a **timescale separation**:

[
\boxed{
\text{brief contact} \not\Rightarrow \text{identity rewrite}
\qquad
\text{persistent contact} \Rightarrow \text{possible rewrite}
}
]

### Boundary

Gate 10 established:

[
\text{local synchrony creates address; dimension supplies capacity}.
]

Gate 11 adds:

[
\boxed{\text{identity also needs a timescale for admitting new relations}.}
]

This is the first point in the lineage where the *age of an edge* matters
independently of its instantaneous evidence.

## Gate 12: learn the relation timescale

Gate 11's `tau=64` was a hand-written protection timescale. Gate 12 turns that
number into an empirical model-selection problem.

Training uses 3,000 independent D=8 address pairs and seven encounter lengths:

```text
transient:   8, 16, 24, 32
persistent: 64, 96, 128
```

Nine candidate timescales compete by minimizing the sum of transient false
merges and persistent failures to merge. The selected value is:

```text
learned tau = 64
```

On 5,000 held-out address pairs:

```text
mean transient false-merge rate    0.00015
mean persistent non-merge rate     0.00000
```

So the timescale itself can be selected from prior encounter statistics rather
than fixed by the experimenter.

### Identical-prefix attacker

Now construct two futures that are causally identical for 48 contact steps.

```text
A: the relation ends at step 48
B: the relation continues after step 48
```

At step 48, any policy that receives only relation age has the same input
history and therefore the same internal state in A and B. In the reference run,
the learned rule has already merged **40.12%** of address pairs.

That single number has opposite meanings:

```text
A: 40.12% false merge
B: 40.12% early-merge success
```

On balanced A/B pairs, an age-only decision is therefore exactly **50%**
accurate about the future relation label. Tuning tau cannot remove this
ambiguity; it only moves the point on the caution/latency tradeoff.

New boundary:

[
\boxed{
\text{learned caution is possible}
\quad\text{but}\quad
\text{age alone cannot predict relation fate}
}
]

## Gate 13: predictive history controls the relation clock

Gate 12 showed that age cannot distinguish two identical contact prefixes. Gate
13 gives the system one pre-contact observable that can contain information the
age variable lacks.

Every held-out encounter lasts exactly **32 contact steps**. Before contact, a
pair has twelve noisy velocity observations.

```text
genuine relation:
    both sides share one latent velocity process + sensor noise

accidental relation:
    each side has an independent latent velocity process + sensor noise
```

The learner compresses that local history to

[
e = \frac{1}{T}\sum_t ||v_a(t)-v_b(t)||^2
]

and learns one threshold from 3,000 past outcomes.

Reference threshold and held-out cue result:

```text
threshold                          0.1869849
training balanced error            0.006342
held-out accuracy                  0.9922
held-out genuine hit rate          0.999608
held-out accidental false hit      0.015504
```

The cue then chooses a relation-admission timescale:

```text
predicted genuine       tau=16
predicted accidental    tau=96
age-only control        tau=64
```

All policies receive the same 5,000 held-out D=8 address pairs and the same
32-step contact duration.

| policy | genuine merge | accidental merge | balanced accuracy |
|---|---:|---:|---:|
| age-only tau=64 | 0.000785 | 0.000408 | 0.500188 |
| always-fast tau=16 | 0.803060 | 0.807834 | 0.497613 |
| always-cautious tau=96 | 0.000000 | 0.000408 | 0.499796 |
| **predictive clock** | **0.802668** | **0.013056** | **0.894806** |
| shuffled cue | 0.400942 | 0.423909 | 0.488516 |

That separation is the mechanism receipt.

The age-only system is safe because it is almost inert at 32 steps. The
always-fast system is responsive but destroys identity indiscriminately. The
predictive system gets almost the always-fast genuine merge rate while keeping
accidental merges near one percent.

Shuffling the cue across relations destroys the result. Therefore the benefit is
not the mere existence of two clocks; it depends on the cue carrying information
about the particular relation.

### Distribution shift

Increase velocity sensor noise from 0.15 to 0.20 without relearning the
threshold:

```text
cue accuracy                      0.8624
genuine merge fraction            0.58530
accidental false merge            0.00160
balanced relation accuracy        0.79185
```

Performance degrades substantially, as it should. The learned predictive
observable is useful, not invariant.

New boundary:

[
\boxed{
\text{predictive history can buy fast safe binding}
\quad\text{only to the extent that the predictor transfers}
}
]

## Gate 14: RGB history supplies the predictive cue

Gate 13's predictor still consumed generated velocity histories. Gate 14 removes
that input.

The learner sees only four consecutive noisy RGB transitions before contact.
Gate 6's appearance-region/template-matching estimator supplies one translation
per visible foreground region. The local predictive statistic is

[
e_{rgb}=\frac{1}{T}\sum_t ||\hat v_a(t)-\hat v_b(t)||^2.
]

Training outcome labels are used only to choose one threshold on this scalar.

The accidental class copies the partner's motion independently with probability
0.5 at each pre-contact step, so an accidental relation can genuinely look
coherent for all four observations.

CI reference with 120 training and 200 held-out histories:

```text
learned RGB threshold              0.0
training balanced cue error        0.00833

held-out RGB cue accuracy          0.975
held-out oracle cue accuracy       0.975
RGB false-negative rate            0.000
RGB false-positive rate            0.050

mean estimator confidence          0.995981
mean valid-transition fraction     1.000
```

In this controlled world RGB correspondence reaches the oracle-motion ceiling.
The residual 5% error is therefore observational ambiguity, not motion-estimator
error.

All later contacts remain exactly 32 steps:

| policy | genuine merge | accidental merge | balanced accuracy |
|---|---:|---:|---:|
| age-only tau=64 | 0.000 | 0.000 | 0.500 |
| always-fast tau=16 | 0.770 | 0.780 | 0.495 |
| **RGB predictive clock** | **0.770** | **0.030** | **0.870** |
| shuffled RGB cue | 0.420 | 0.410 | 0.505 |

The mechanism therefore survives removal of direct velocity input.

[
\boxed{
\text{RGB history}
\rightarrow
\text{estimated common fate}
\rightarrow
\text{relation clock}
\rightarrow
\text{faster selective binding}
}
]

This also sharpens the next limitation. Gate 14 intentionally makes appearance
regions easy to recover and keeps their texture stable.

## Gate 15: confidence belongs to the relation

Gate 14's RGB motion estimator was essentially oracle-perfect. Gate 15 makes its
correspondence actively ambiguous.

For each of four pre-contact local tracking probes, candidate B undergoes:

```text
appearance jump       0.15
partial occlusion     1 column
look-alike blend      0.50
```

A visually plausible B continuation is placed near the displacement predicted by
candidate A. If true B motion differs, template matching is often lured onto the
look-alike.

That creates a particularly dangerous error: **false common fate**.

CI reference, 80 training / 120 held-out histories:

```text
learned residual threshold              0.000000
learned confidence threshold            0.596747

residual-only cue:
    accuracy                             0.5167
    false positive                       0.9667

confidence-gated cue:
    accuracy                             0.9250
    false negative                       0.0500
    false positive                       0.1000

oracle-motion cue accuracy               0.9750
```

The motion residual has been successfully attacked:

```text
median RGB residual, genuine             0.0
median RGB residual, accidental          0.0
```

But the tracker's own confidence has not:

```text
median minimum confidence, genuine       0.64925
median minimum confidence, accidental    0.32217
```

The downstream relation dynamics make the consequence concrete:

| policy | genuine merge | accidental merge | balanced accuracy |
|---|---:|---:|---:|
| global cautious tau=96 | 0.000 | 0.000 | 0.500 |
| always fast tau=16 | 0.850 | 0.767 | 0.542 |
| residual-only clock | 0.850 | 0.733 | 0.558 |
| **confidence-gated clock** | **0.800** | **0.050** | **0.875** |
| shuffled confidence | 0.533 | 0.317 | 0.608 |

This establishes a new piece of the mechanism:

[
\boxed{
\text{evidence says what relation to propose}
\quad + \quad
\text{confidence says how much causal authority it earns}
}
]

Global caution also prevents false merges, but it cannot bind any genuine
relation quickly. The confidence gate localizes the caution to the doubtful
track.

## Gate 16: relation authority becomes a recoverable state

The same candidate relation now experiences:

```text
8 clear RGB probes
4 Gate-15 ambiguous / look-alike probes
1 clean probe
7 additional clean probes
```

The desired fast-clock decision is:

```text
pre-attack             YES
under ambiguity        NO
one clean later        NO
after sustained clear  YES
```

A permanent veto, cumulative mean, symmetric EMA and asymmetric eligibility
state are fit on 80 training trajectories.

The learned dynamic families are:

```text
symmetric EMA:
    tau                 1.5
    threshold           0.90
    training error      0.000

asymmetric eligibility:
    attack tau          1.5
    recovery tau        6.0
    threshold           0.75
    training error      0.000
```

Because both fit training perfectly, model choice is made on a separate
80-trajectory validation split:

```text
permanent veto         0.250000
cumulative mean        0.008333
symmetric EMA          0.002083
asymmetric state       0.000000
```

The asymmetric state therefore wins the frozen selection rule, but only by one
validation mistake made by the symmetric state.

Now evaluate on 240 untouched trajectories.

```text
                        pre      attack    +1 clean   recovered
symmetric genuine fast  1.000     0.000     0.000      1.000
asymmetric genuine fast 1.000     0.000     0.000      1.000
accidental fast         0.000     0.000     0.000      0.000
```

Both dynamic families have **zero held-out checkpoint error**.

The selected asymmetric state's median genuine authority is:

```text
pre-attack              0.7644
under ambiguity         0.6662
one clean later         0.7211
full recovery           0.9193
```

and downstream relation admission gives:

```text
pre-attack       genuine merge 0.825   accidental 0.000
ambiguity        genuine merge 0.000   accidental 0.000
one clean later  genuine merge 0.000   accidental 0.000
full recovery    genuine merge 0.825   accidental 0.000
```

Permanent veto never regains authority. Cumulative history does recover, but
prematurely re-authorizes about 10% of genuine relations after just one clean
sample on the final test.

The receipt therefore supports:

[
\boxed{
\text{relation authority is a recoverable local state}
}
]

It does **not** yet support the stronger claim that separate attack and recovery
timescales are necessary. Validation weakly prefers them; the final held-out test
cannot distinguish the two temporal families.

## Gate 17: false-clean recovery hits an observability boundary

The stronger hysteresis attacker exposed a causal limit rather than a better
timescale.

Construct paired worlds with:

```text
8 clear observations
2-5 ambiguous observations
1-4 clean-looking observations
--------------------------------
split point
--------------------------------
stable future:   clean continues
relapse future:  ambiguity returns
```

Everything above the split is generated once and copied verbatim into both
worlds.

CI reference, 40 training / 80 held-out pairs:

```text
max observed-prefix difference     0.000000
```

At the split:

| causal state | max paired state diff | best balanced accuracy |
|---|---:|---:|
| raw | 0.000000 | 0.500 |
| cumulative mean | 0.000000 | 0.500 |
| symmetric EMA | 0.000000 | 0.500 |
| asymmetric eligibility | 0.000000 | 0.500 |

The Gate-16 states do differ in how quickly they re-authorize a clean-looking
streak:

```text
clean streak       symmetric fast       asymmetric fast
1                  0.067                0.000
2                  1.000                0.769
3                  1.000                1.000
4                  1.000                1.000
```

But that cannot be interpreted as future prediction. The stable-recovery and
false-clean-relapse member of each pair have the exact same state.

One genuinely new RGB observation breaks the pair identity:

```text
post-split learned threshold       0.994541
training balanced accuracy         1.00000
held-out balanced accuracy         0.99375

median stable target               0.995967
median relapse target              0.674047
```

So the Gate-16 asymmetry question is now bounded more precisely:

[
\boxed{
\text{memory controls commitment latency}
\neq
\text{memory predicts an unobserved future}
}
]

Separate attack and recovery constants may still be useful engineering knobs,
but no prefix-only temporal mechanism can decide whether a currently identical
recovery streak will persist.

## Gate 18: preserve one more thing the RGB matcher already knew

Gate 17's chance result was a representation boundary, not a claim that the raw
RGB frames contained no other structure.

One tempting candidate was rejected before implementation: Gate 6's
"confidence" is already the normalized best-vs-second-best template score
margin, so renaming match margin would add no information.

Gate 18 instead preserves the **absolute shared translation vector**, which
Gates 14-17 discarded after deciding that the candidates moved together.

Every accepted prefix is conditioned so the old admission bit is identical:

```text
common-fate residual             0
confidence                       >= 0.95
ordinary relation admission      TRUE
```

All motions have equal speed.

Stable trajectories switch shared direction with probability 0.04 per
transition. Future-relapse trajectories begin identically, then switch with
probability 0.65 in the later prefix. This is deliberately a controlled
predictive precursor, not a natural-video assumption.

The learned measurement is

[
s = \frac{1}{T-1}\sum_t
||\hat v_t-\hat v_{t-1}||^2.
]

CI reference, 120 training / 240 held-out trials:

```text
old admission TRUE fraction            1.0000
minimum confidence                     0.993065
max common-fate residual               0.000000
estimated-motion accuracy              1.0000

learned surprise threshold             0.000000
median stable surprise                 0.000000
median relapse-prefix surprise         1.000000
```

Held-out classification:

```text
matched old admission                  0.5000
RGB motion surprise                    0.9042
oracle motion surprise                 0.9042
shuffled RGB surprise                  0.4542
```

The RGB cue exactly reaches the oracle ceiling in this controlled world.

Downstream relation admission:

| policy | stable merge | relapse false merge | balanced accuracy |
|---|---:|---:|---:|
| old matched admission | 0.8500 | 0.8000 | 0.5250 |
| **RGB surprise** | **0.7250** | **0.0167** | **0.8542** |
| shuffled surprise | 0.3417 | 0.3583 | 0.4917 |
| global cautious | 0.0000 | 0.0000 | 0.5000 |

So Gate 17 and Gate 18 now form a useful pair:

[
\boxed{
\text{same represented evidence} \Rightarrow \text{no memory can help}
}
]

but

[
\boxed{
\text{preserve an informative discarded observable}
\Rightarrow
\text{prediction can return}
}
]

## Next attacker

The obvious criticism is that Gate 18 **builds directional instability into the
future-relapse class**. That is acceptable for mechanism isolation, but it has
not earned general predictive status.

The next useful gate should therefore attack transfer:

- train the surprise threshold on one pair of switch probabilities,
- change the stable/relapse dynamics at test time,
- include a world where direction changes are benign rather than precursors,
- and compare raw surprise against a calibrated predictor that is allowed to
  learn whether surprise is informative in the current world.

If motion surprise remains useful only when the generator makes it predictive,
that is an important boundary: the observable carries information, but its
*meaning* is environment-dependent. The mechanism should then learn cue
reliability rather than hard-code "direction change = danger."

