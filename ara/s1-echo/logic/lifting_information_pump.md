# TreeHeap Lifting Information Pump

## Motivation

`S1-INTERNAL-ECHO-C01` proved that a learned FOLD can encode adjacent token
pairs into the first parent level without exposing leaves to the decoder.  It
also found that `99.9786%` of READ mass stayed at that first level; every higher
parent and root was causally neutral.  Recursively calling FOLD did not create a
law that forced useful information upward.

This experiment replaces unrestricted multilevel READ with an explicit
information-pump protocol based on the lifting scheme used by invertible
wavelet transforms.

## Claim

`S1-LIFT-PUMP-C01`:

> A shared TreeHeap lifting kernel can provide an algebraically closed
> information pump.  Every local FOLD splits two child states into one upward
> parent and one addressed detail; only parents recurse toward root, and the
> same kernel reconstructs the children exactly by UNFOLD.  A real-corpus
> next-token objective should then send non-zero gradient into the learned
> predictor and make the root and recursive address pairings causally useful.

The claim has two independent parts:

1. **deductive closure**: FOLD/UNFOLD inversion follows from the equations and
   must hold before training;
2. **inductive learning**: corpus loss must improve a learned predictor over a
   frozen matched initialization and must depend on the produced root.

## Local Pump Algebra

For child states `L` and `R`, define a shared learned predictor `P_theta`:

\[
D = R - P_\theta(L),
\]

\[
U = L + \frac{1}{2}D.
\]

`D` is an addressed detail retained at the current depth.  `U` is the parent
sent upward.  Only `U` participates in the next FOLD.

UNFOLD uses the same predictor:

\[
L = U - \frac{1}{2}D,
\]

\[
R = D + P_\theta(L).
\]

Substitution gives exact recovery for any deterministic `P_theta`; the
predictor need not be linear or invertible.  If `P_theta(L)=L`, then

\[
U=\frac{L+R}{2},\qquad D=R-L,
\]

which is the familiar coarse-plus-detail split.

For a 16-leaf TreeHeap, the state is

\[
H_{state}=\left(root,D^{(4)},D^{(3)},D^{(2)},D^{(1)}\right).
\]

The decoder cannot jump directly to `D^(1)` or a first parent.  It starts from
root and applies UNFOLD in reverse depth order until leaves are reconstructed.

## Learning Pressure

Clean echo alone cannot train an exactly invertible predictor: FOLD and UNFOLD
cancel, so every predictor produces the same recovered leaves.  The learning
pressure therefore comes from a natural, unmasked corpus objective:

\[
L_{next}=-\log p(x_{17}\mid root(x_1,\ldots,x_{16})).
\]

There is no masked token, corrupted input, syntax label, semantic class, or
internal-state target.  The input is 16 observed tokens and the target is the
next naturally occurring token.  Echo is an algebraic audit, not an additional
training loss.

## Predict

`P-S1-LIFT-PUMP-01`:

1. nonlinear FOLD followed by UNFOLD has maximum absolute reconstruction error
   below `1e-5` for random continuous trees of depths 1 through 6;
2. native real-token state reconstruction MSE is below `1e-10` and token/block
   echo is at least `0.999/0.99`;
3. zeroing or cross-sample replacing root while retaining details damages echo
   by at least `0.10` token accuracy;
4. cross-sample shuffling any addressed detail level damages echo, with maximum
   token-accuracy loss at least `0.10`;
5. real-corpus next-token loss gives the predictor a finite non-zero gradient
   and changes its parameters by at least `1e-4`;
6. learned-predictor validation NLL beats a matched frozen-predictor model by at
   least `0.02`;
7. root shuffle raises learned-predictor next-token NLL by at least `0.10`, and
   recursively breaking at least one left/right pairing depth raises NLL by at
   least `0.02`.

## Controls and Interventions

- `learned_predictor`: all WRITE, predictor, root READ, and output parameters
  learn from `L_next`;
- `frozen_predictor`: identical initialization and declared parameter count,
  but predictor parameters remain frozen while the remaining modules learn;
- `root_zero` and `root_shuffle`: test root causality;
- `break_depth_d`: replace right subtrees at one FOLD depth with right subtrees
  from another sample;
- `detail_shuffle_d`: replace one stored detail level across samples during
  UNFOLD.

## Falsification and Boundary

Failure of predictions 1-4 rejects the algebraic implementation.  Passing those
predictions alone proves only a deterministic codec.  If predictor gradients or
the learned-over-frozen gate fail, the pump is mathematically valid but the
chosen corpus objective has not trained its internal coordinate system.  If
root/address interventions are neutral, no useful recursive learning claim is
allowed.

A passing pilot would establish a trainable information-flow primitive, not
semantic hierarchy, world knowledge, reasoning, compression optimality, or
superiority to Transformer/LSTM/MLP systems.

## Result

### Smoke correction

The first 2,048-block smoke used an unbounded predictor and failed P1: FP32
closure error grew to `2.12e-5` at depth 6 while next-token states became
numerically large.  This is retained under `evidence/.../smoke_v1/`.  Before the
registered main run, the implementation was corrected without changing the
lifting equations: `P_theta` received a bounded `2*tanh(z/2)` output, and the
next-token classifier was separated from the token embedding table.  Smoke v2
then passed algebraic closure.

### Main run

The main run completed on `io` with seed `71511`, 100,000 real-Chinese training
blocks, 8,192 held-out blocks, context 16, dimension 128, and three epochs.
Echo was not part of the training loss.

Deductive codec results:

- random nonlinear depth-1..6 closure maximum FP32 error: `3.70e-6`;
- native state reconstruction MSE/max error: `3.14e-14/2.86e-6`;
- native token/block echo: `1.0/1.0`;
- root zero token/block echo: `0.93042/0.0`;
- cross-sample root token/block echo: `0.93701/0.0078125`;
- detail-shuffle token drops at depths 1..4:
  `0.49878/0.25000/0.12354/0.06323`.

Inductive corpus results:

- learned/frozen predictor validation NLL: `8.03468/8.06377`;
- learned gain: `0.02910` NLL, passing the registered `0.02` gate;
- predictor maximum gradient norm / parameter delta: `0.50191/16.13119`;
- cross-sample root replacement increased next-token NLL by `0.21359`;
- breaking left/right pairing at FOLD depths 1..4 increased NLL by
  `0.03221/0.03838/0.04324/0.05634`.

P1, P2, P4, P5, P6, and P7 passed.  P3 failed because root intervention reduced
token accuracy by only `0.06958`, below the preregistered `0.10` threshold.
The machine decision therefore remains **not supported as fully written**.

The narrower result is nevertheless positive and should be retained:

1. the lifting pump is algebraically closed and numerically stable through six
   tested depths;
2. every addressed detail level is necessary for exact echo;
3. root is necessary for exact whole-block recovery even though most individual
   tokens remain nearest-neighbor readable after root removal;
4. natural next-token loss changes the predictor, improves over its frozen
   initialization, and makes root plus all four recursive pairing depths
   causally relevant.

This resolves the earlier `parent-1` shortcut at the mechanism level: decoding
now starts at root and recursively UNFOLDs.  It does not yet show that root or
details carry linguistic abstractions rather than a private invertible code.
