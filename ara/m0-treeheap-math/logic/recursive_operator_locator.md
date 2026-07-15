# Recursive TreeHeap Operator Locator

## Claim

`M0-RECUR-C01`:

> If one content-aware `stop/left/right` kernel learns a reliable local routing
> rule on operator roots at address levels 0..2, repeated application of that
> same kernel should locate operator roots at unseen levels 3..5 without new
> depth classes or new parameters.

This tests the narrow form of the growth hypothesis: validate one local rule in
a bounded region, then let TreeHeap addresses grow by recursive composition.

## Address Recurrence

Starting from `i_0 = 1`, one action updates the address by

\[
i_{t+1}=
\begin{cases}
i_t & stop,\\
2i_t & left,\\
2i_t+1 & right.
\end{cases}
\]

The model never predicts an absolute address.  It predicts one of three local
actions and is called again on the resulting node.  Model parameters therefore
do not grow with tree depth.

## Input Boundary

At a current node `i`, the kernel reads algebraic residual features from:

```text
current node/subheap
left child/subheap
right child/subheap
```

The residuals are computed from the same number law and native `MIRROR` /
`SUBTREE_PLUS` corruptions used by `M0-OPCODEC-C01`.  Inputs do not contain the
gold address, target interval, path bits, or a boolean saying which child owns
the target.

Teacher forcing supplies the correct action only as a training label.  At test
time inference starts at root and follows its own previous action.

## Splits

```text
train: target address level 0..2, operator depth 1..2
IID:   target address level 0..2, operator depth 1..2
OOD-3: target address level 3
OOD-4: target address level 4
OOD-5: target address level 5
```

Operator depth remains 1..2 in every split.  This isolates recursive address
growth from the separate depth-extrapolation failure in `M0-OPCODEC-C01`.

## Baselines

1. `one_shot_structural`: score every absolute address once with a shared
   structural kernel;
2. `flat_address`: flatten the complete heap and classify an absolute address;
3. oracle path: exact upper bound.

## Metrics and Prediction

- teacher-forced local action accuracy;
- autonomous exact route/address accuracy by target level;
- end-to-end exact restoration after native executor;
- observed route accuracy versus the diagnostic approximation `p^steps`;
- parameter counts and mean executed steps.

Pre-registered support gate:

```text
OOD level 3..5 recursive route exact >= 0.80
OOD level 3..5 recursive route exact exceeds both learned baselines
no target/path leakage in route features
same kernel parameters used at every recursive step
```

Strong support additionally requires exact restoration `>= 0.75` at every OOD
level.  If local action accuracy is high but route accuracy decays with depth,
the claim is narrowed to local reliability and does not justify free growth.

## Falsification

Reject the growth claim if the recursive kernel needs absolute depth/address,
new per-level parameters, or oracle child flags; if autonomous routes do not
extrapolate beyond the trained levels; or if error multiplication makes deeper
routes unusable despite good shallow accuracy.

This experiment does not test natural language structure, unsupervised
discovery, WMT, consciousness, or unlimited recursion.
