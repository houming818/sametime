# TreeHeap Multiscale Geometry

Date: 2026-07-15
Status: claim not supported / Bag geometry positive, ordered geometry failed
Claim: `S3-TREEHEAP-GEOMETRY-C01`
Predict: `P-S3-TREEHEAP-GEOMETRY-01`

## Revised Question

The previous multiscale-mask proof graded every depth by exact token recovery.
Houming818 objects that this assumes a high node is a damaged copy of its
leaves. A more natural hypothesis is multiresolution geometry:

> Different TreeHeap depths represent different structural scales. Lower
> states carry fine token/order detail; upper states cover a wider language
> region and encode a coarser probability contour. Failure to reconstruct
> exact leaves from an upper state is not automatically an error.

This proof therefore does not ask the root to reproduce 64 token identities.
It asks whether every node becomes geometrically useful for the distributional
observables defined over its own receptive field.

## Scale-Indexed Spaces

For a balanced 64-leaf TreeHeap, a depth-`d` node covers `2^d` leaves. Let
`X_d` be the space of spans at that scale. Folding is treated as a coarse
projection rather than a general inverse:

```math
\pi_d:X_d\rightarrow X_{d+1}.
```

The learned state is:

```math
z_i^{(d+1)}=K_\theta(z_{2i}^{(d)},z_{2i+1}^{(d)},e_d),
```

where one shared recursive kernel receives a learned depth coordinate `e_d`.
No decoder is required to recover exact leaves from `z_i^(d)`.

## Exact Observable Geometry

Each real-text span has two label-free observable sketches.

### WordBag sketch

Every token `w` has a fixed random signed basis vector `b(w)`. For span `S`:

```math
B(S)=\sum_{w\in S}b(w).
```

It has an exact recursive compose law:

```math
B(L\Vert R)=B(L)+B(R).
```

### Directed adjacency sketch

Each ordered pair `(u,v)` has a fixed non-commutative tensor sketch `a(u,v)`.
For span `S=(w_1,...,w_n)`:

```math
A(S)=\sum_{j=1}^{n-1}a(w_j,w_{j+1}).
```

Its exact compose law is:

```math
A(L\Vert R)=A(L)+A(R)+a(last(L),first(R)).
```

These targets are not human semantic labels. They are deterministic
measurements of the real corpus. Bag captures content without order; adjacency
captures directional local geometry. Both expand their receptive range with
TreeHeap depth.

## Learning Protocol

A 64D TreeHeap state is trained from scratch on real Chinese token blocks.
Training alternates tasks rather than summing a large multi-loss:

```text
step 1: choose one depth, decode its Bag sketch
step 2: choose one depth, decode its adjacency sketch
step 3: next depth, Bag
step 4: next depth, adjacency
...
```

The encoder kernel is shared across all addresses and depths. Bag and adjacency
have separate read kernels, so their gradients do not need to collapse into one
output head. The state itself is shared and must support both views.

## Query Definition

For a held-out node state `q=z_i^(d)`, Query searches other states at the same
depth by cosine similarity. The returned nearest state is then evaluated in the
exact observable spaces:

```math
G_B(d)=cos(B_i,B_{NN(z_i)})-cos(B_i,B_{random}),
```

```math
G_A(d)=cos(A_i,A_{NN(z_i)})-cos(A_i,A_{random}).
```

Positive gains mean that a state-space neighbor is also a real corpus neighbor
at that scale. This is the first minimal version of Query TreeHeap over a World
TreeHeap geometry. It retrieves spans, not yet open-ended words such as
`food -> rice`.

## Real-Data Smoke

```text
data                real Chinese pretraining blocks
length              64 tokens
depths              1..6 (2..64-token receptive fields)
state dimension     64
sketch dimension    64
training blocks     200,000
validation blocks   512
seed                71501
host                io RTX 3090
```

## Predictions

```text
P1  Exact recursive Bag and adjacency targets match direct sequence scans
    with maximum absolute error <= 1e-6.

P2  On held-out data, learned Bag sketch cosine is >= 0.80 and adjacency
    sketch cosine is >= 0.60 at every depth 1..6.

P3  At depths 4..6, state-nearest-neighbor gains are >= 0.10 for Bag and
    >= 0.05 for adjacency relative to deterministic random neighbors.

P4  At root scale, shuffling token order preserves predicted Bag geometry
    substantially more than predicted adjacency geometry: Bag invariance
    minus adjacency invariance >= 0.10.

P5  Every depth has positive Bag and adjacency query gain; an upper depth is
    not required to recover exact token identity.

P6  WRITE, recursive FOLD, depth coordinates, Bag READ, and adjacency READ
    all receive finite non-zero gradients from their corresponding tasks.
```

## Decision

- Support only if P1-P6 pass and the result later reproduces across three
  seeds.
- Partial support if sketches decode but state-neighbor Query gains fail. That
  would prove readable statistics but not an intrinsic geometric index.
- Reject this implementation if exact algebra fails, upper-depth sketch quality
  collapses, permutation changes Bag as much as adjacency, or Query neighbors
  are no better than deterministic random controls.

Even a full pass proves only scale-indexed corpus geometry under Bag and
adjacency observables. It does not prove human semantic categories, world
knowledge, exact compression, consciousness, or superiority to Transformers.

## Next Step If Supported

Use a Query TreeHeap to search coarse upper-depth neighborhoods, then descend
through child addresses and detail states to refine a broad probability region
into concrete token/subheap candidates. That coarse-to-fine search is a later
claim, not part of this proof.

## Result

The preregistered single-seed run completed on `io` with 200,000 real Chinese
64-token blocks. It passed 3 of 6 gates:

```text
P1 exact recursive algebra             PASS   max error 4.77e-7
P2 held-out readout thresholds         FAIL
P3 upper-depth Query thresholds        FAIL
P4 Bag/order permutation separation    FAIL   gap 0.0096
P5 positive Query gain at every depth  PASS
P6 complete gradient path              PASS
```

### What appeared

The Bag observable produced a real and consistent signal. Held-out Bag readout
cosine rose with receptive field size:

```text
covered tokens       2      4      8      16     32     64
Bag read cosine    .267   .286   .306   .334   .377   .436
Bag Query gain     .184   .122   .095   .086   .071   .077
```

This is compatible with Houming818's contour hypothesis in one narrow sense:
upper nodes did not become meaningless as exact token detail disappeared.
Their coarse content sketch became easier to linearly read. At all six depths,
a nearest neighbour in learned state space was closer in exact Bag geometry
than a deterministic random neighbour.

### What did not appear

The directed-adjacency observable was nearly absent:

```text
covered tokens       2      4      8      16     32     64
Adj read cosine    .010   .012   .015   .017   .021   .027
Adj Query gain     .014   .000   .006   .004   .013   .002
```

Reversing all 64 tokens left both predicted views almost unchanged: Bag
invariance was `0.9759`, adjacency invariance was `0.9663`, and the required
gap was only `0.0096` rather than `0.10`. The state therefore learned mostly a
content contour, not a directional topology.

The random ordered-pair sketch is intentionally demanding: every adjacent
token pair has an unrelated fixed target direction. Failure means the current
64D shared FOLD plus alternating shallow readers cannot carry that ordered
information. It does not prove that all scale-indexed TreeHeap geometries are
impossible.

## Decision

`S3-TREEHEAP-GEOMETRY-C01` is **not supported by this implementation**. Keep
the Bag result as a supported sub-observation, but do not call the full state a
multiscale language geometry yet. The next proof should replace the arbitrary
pair hash with a learnable, low-rank non-commutative observable and compare it
against a matched flat recursive baseline. That would test whether order can
be transported through FOLD instead of requiring memorization of token-pair
hashes.

Evidence: `evidence/s3_treeheap_multiscale_geometry_smoke/summary.json`.
