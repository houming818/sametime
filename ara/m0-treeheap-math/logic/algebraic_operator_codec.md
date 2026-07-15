# Algebraic Operator Codec

## Claim

`M0-OPCODEC-C01`:

> When observations are transformed by a finite family of native TreeHeap
> operators, a shared structural encoder should infer a compact probability
> bucket over `operator/address/depth/parameter`; a non-learned algebraic
> executor should then restore the target TreeHeap at unseen addresses and
> depths better than flat program prediction or direct array rewriting.

This is a controlled existence claim for two project hypotheses:

1. structural regularities can be extracted into an encoder/decoder protocol;
2. TreeHeap data transformations can be executed by TreeHeap's own address and
   subheap operators instead of an opaque network rewriting the full state.

## Canonical Number Law

One-indexed binary TreeHeap addresses obey:

\[
left(i)=2i,\qquad right(i)=2i+1
\]

Each synthetic legal state uses the same arithmetic law:

\[
A[2i]=2A[i]
\]

\[
A[2i+1]=2A[i]+1
\]

The root primitive is random, so samples are not one memorized array.

## Native Operators

The corruption program is one of:

```text
MIRROR(node, depth)
SUBTREE_PLUS(node, depth, delta)
```

`MIRROR` recursively exchanges left/right addresses inside one subheap.
`SUBTREE_PLUS` adds `delta` to every state in one bounded subheap.  Their
inverse programs use the same native operators: mirror is self-inverse and
plus is inverted by `-delta`.

## Learned Encoder and Fixed Executor

The structural encoder runs one shared convolutional kernel over every
candidate node.  Features are local arithmetic residuals plus depth-wise
subheap residual summaries.  It emits:

```text
P(operator)
P(address)
P(depth)
P(delta)
```

Depth is deliberately not a four-class learned head.  The number-law violation
forms a residual wave through relative subtree layers; the algebraic decoder
selects the layer with maximum normalized residual.  This is the recursive
`continue/stop` equivalent and can return depth 3 or 4 even though training only
contains operations of depth 1 or 2.

The network is not allowed to generate target array values.  After argmax
collapse, the fixed executor applies the predicted inverse operator.  Exact
restoration is checked against the original legal TreeHeap.

## Splits

```text
train:       shallow addresses, operation depth 1..2
IID:         same regime, new samples
OOD address: unseen deeper addresses, depth 1..2
OOD depth:   seen address regime, operation depth 3..4
OOD joint:   unseen addresses and operation depth 3..4
```

## Baselines

1. `flat_program`: one flattened MLP predicts the same operator tuple;
2. `flat_rewriter`: one flattened MLP directly predicts every target array
   value, bypassing TreeHeap operators;
3. oracle program + algebraic executor: exact upper bound.

## Predictions

If the claim is supported:

- structural program/execution restoration remains high on OOD address and
  OOD depth;
- flat program prediction loses unseen-address accuracy;
- flat direct rewriting may copy unchanged cells but fails exact restoration;
- oracle execution is exactly correct;
- predicted programs remain short while tested tree size grows.

## Falsification

Reject or narrow the claim if the structural encoder only works on trained
addresses/depths, if a flat direct rewriter matches OOD exact restoration, if
the executor secretly learns target arrays, or if native operator identities
fail exact rerun.

A pass does not prove language semantics, WMT, consciousness, or that this
operator family is sufficient for all reasoning.  It proves an algebraic
encoder/program/executor/decoder route exists under controlled structure laws.

## Result (2026-07-12)

The main claim is **partially supported and narrowed**.

| Split | Structural restore | Flat program restore | Flat rewrite restore |
|---|---:|---:|---:|
| IID | 0.9270 | 0.1535 | 0.7205 |
| OOD address | 0.8400 | 0.0000 | 0.0000 |
| OOD depth | 0.0440 | 0.0000 | 0.0000 |
| OOD address + depth | 0.0580 | 0.0000 | 0.0000 |

The structural model uses 41,671 parameters, versus 235,913 for flat program
prediction and 196,927 for direct flat rewriting.  The fixed executor reaches
1.0 exact restoration with the oracle program on every split.

The important diagnostic separates address inference from depth readout.  On
OOD depth, depth accuracy is 0.9365 **when read at the gold address**, but
end-to-end address accuracy is only 0.1075.  Therefore the failure is not simply
that categorical depth 3/4 was absent during training.  A deeper corruption
changes the residual field used by the address kernel, and the shallow-trained
address selector does not remain invariant to that change.

What is supported:

- a learned structural encoder can select a compact native-operator program;
- a fixed TreeHeap executor can exactly decode that program;
- shared structural scoring transfers strongly to unseen heap addresses;
- this route is smaller and much more OOD-address capable than both flat
  baselines in this controlled number-law world.

What is not supported:

- joint extrapolation over both address and recursive operator depth;
- semantic induction from natural language;
- a general claim that TreeHeap is better than MLP or Transformer.

The next claim must make address detection invariant to operator depth, for
example by recursively locating the residual boundary with one shared
`continue/stop` kernel instead of scoring the entire corrupted field once.
