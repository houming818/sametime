# S1 Multiresolution Internal-State Echo

## Question

Can a TreeHeap encoder preserve an observed token block in recursively folded
internal states, or does echo work only because the decoder can copy an ordered
leaf/string channel?

This experiment is deliberately narrower than masked-language modeling.  The
encoder observes the complete input.  There is no pre-encoding mask, training
noise, semantic label, syntax label, translation target, or teacher-forced
output token.  The decoder must reconstruct the observed block from non-leaf
TreeHeap states.

## Claim

`S1-INTERNAL-ECHO-C01`:

> Shared TreeHeap FOLD kernels can write recoverable information into a
> multiresolution internal state `H_state`.  When the decoder has no leaf or
> residual string bypass, held-out echo should remain possible, and removing
> causally used parent/root levels or destroying left/right address pairing
> should increase reconstruction loss.

This claim is about structural information transport and compression.  It does
not claim that the internal states are human-interpretable semantics, a world
model, or a better language model than a Transformer.

## Data Flow

For a length-`N=2^D` real-text block,

\[
H^{(0)}_i = WRITE(x_i),
\]

\[
H^{(d+1)}_j = FOLD(H^{(d)}_{2j}, H^{(d)}_{2j+1}),
\]

and the decoder-visible state is

\[
H_{state}=\{H^{(1)},H^{(2)},\ldots,H^{(D)}\}.
\]

`H^(0)` is never passed to the decoder.  To reconstruct position `i`, READ may
inspect only the ancestor chain

\[
H^{(1)}_{\lfloor i/2\rfloor},
H^{(2)}_{\lfloor i/4\rfloor},\ldots,H^{(D)}_0.
\]

This is an address-aware TreeHeap read, not attention over the original token
array.

## FOLD Variants

All variants allocate the same modules and parameter count.  They differ only
in the signals supplied to four output channels.

### Mean-only control

Every channel receives

\[
M(L,R)=\frac{L+R}{\sqrt{2}}.
\]

This uploads a coarse symmetric summary but cannot directly distinguish
`(L,R)` from `(R,L)`.

### Single learned control

Every channel receives the same learned nonlinear comparison

\[
J(L,R)=MLP([L,R]).
\]

This tests whether a generic learned binary merge is sufficient.

### Multichannel TreeHeap FOLD

The four channels receive distinct algebraic signals:

\[
M(L,R)=\frac{L+R}{\sqrt{2}},
\qquad
D(L,R)=\frac{L-R}{\sqrt{2}},
\]

\[
P(L,R)=\tanh(L)\odot\tanh(R),
\qquad
J(L,R)=MLP([L,R]).
\]

`M` carries a symmetric contour, `D` carries left/right orientation, `P`
carries multiplicative co-occurrence, and `J` leaves capacity for relations not
captured by the fixed bases.  These are channels of one FOLD operation, not
handwritten linguistic categories.

## Predict

`P-S1-INTERNAL-ECHO-01`:

1. multichannel FOLD beats mean-only held-out token NLL by at least `0.05`;
2. multichannel FOLD beats single learned FOLD NLL by at least `0.02`;
3. removing the closest parent level increases multichannel NLL by at least
   `0.10`;
4. at least one level intervention, including root zero/replacement, increases
   NLL by at least `0.02`;
5. destroying left/right pair addresses increases NLL by at least `0.10`;
6. training, gradients, reconstruction output, and all intervention metrics are
   finite and non-empty.

Predictions 3-5 are the anti-bypass gates.  Echo accuracy without those causal
effects is not accepted as TreeHeap evidence.

## Proof Protocol

Train the three variants from matched seeds on fixed-length blocks sampled from
the existing raw-Chinese corpus shards.  Optimize only surface-token
cross-entropy:

\[
L_{echo}=-\sum_i\log p(x_i\mid H_{state},i).
\]

After training, evaluate the same checkpoint under:

- `full`: all internal levels, no leaves;
- `drop_depth_d`: remove one internal resolution at a time;
- `root_zero`: remove the root;
- `root_shuffle`: replace each root with another sample's root;
- `destroy_addresses`: recursively mismatch right children before FOLD.

Primary metrics are held-out NLL, token top-1/top-5, block exact, and each
intervention's NLL increase.  The evidence must record parameter counts,
gradient norms, per-depth READ mass, examples, configuration, host, runtime,
and a machine-readable gate decision.

## Falsification and Boundary

The claim is rejected as written if the multichannel model cannot reconstruct
held-out blocks, if parent/address interventions are neutral, or if the decoder
receives leaves or an equivalent ordered-token cache.  A neutral root
intervention alone does not reject multiresolution storage: root may carry a
coarse contour while lower parents carry exact detail.  However, no level may
be called useful without a positive causal intervention.

Even a passing result proves only that shared recursive FOLD and addressed READ
formed a usable private codec.  Exact echo is not by itself evidence of
semantics, reasoning, or consciousness.

## Result

The registered run completed on `io` using 100,000 real-Chinese training
blocks, 8,192 held-out blocks, length 16, five epochs, and seed `71501`.  All
three variants had exactly `2,264,960` parameters.  The decoder received only
internal nodes.

| FOLD | Held-out NLL | Token top-1 | 16-token exact | Address-destroy NLL increase |
|---|---:|---:|---:|---:|
| mean-only | 0.616107 | 0.662140 | 0.003906 | +9.379443 |
| single learned | 0.004979 | 0.998772 | 0.980835 | +45.384604 |
| multichannel | 0.002010 | 0.999428 | 0.991211 | +54.368653 |

For the multichannel checkpoint:

- removing the closest parent level increased NLL by `74.425402`;
- removing levels 2, 3, and 4 changed NLL by approximately `2.2e-7`,
  `-7.8e-8`, and `-7.6e-8`;
- root zero and cross-sample root replacement changed NLL by `-7.5e-8` and
  `-2.6e-8`;
- READ mass over levels 1..4 was
  `0.999786/0.000084/0.000051/0.000079`.

Registered predictions P1, P3, P4, P5, P6 and the implementation gates passed.
P2 failed: multichannel beat the generic learned FOLD by only `0.002969` NLL,
below the registered `0.02` threshold.

The result is **partial support**.  It proves a narrow one-level internal codec:
a shared nonlinear FOLD writes each adjacent token pair into its parent, and an
addressed decoder reconstructs held-out 16-token blocks without reading leaves.
It does not prove a multiresolution protocol.  The model ignored every level
above the closest parents, including root.  In retrospect the registered P4
gate was too weak because P3 could satisfy it by itself; future multiresolution
claims must preregister a separate higher-level causal gate.

This is stronger than a leaf/string copy channel, but it may still be described
as eight parallel learned pair codes.  The next design problem is therefore not
more echo training.  It is to define a task whose answer cannot be recovered
from disjoint adjacent pairs, so that recursively larger receptive fields must
contribute.
