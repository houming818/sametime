# TreeHeap Metric Self-Improvement

## Claim

`S1-SELF-METRIC-C01`:

> Starting from the same TreeHeap observer checkpoint, adding a TreeHeap-native
> contrastive distance objective should improve that model's own positive vs
> negative structural-distance margin while preserving echo, non-collapsed
> codes, and held-out context transfer.

This is a self-comparison.  It does not claim superiority over MLP,
Transformer, BoW, or another metric.

## Data Signal

The existing unlabeled observer corpus contains verb-object observations.
Object pairs sharing observed contexts are positives.  Pairs with disjoint
observed contexts are negatives.  Gold names such as `food` and `medicine` are
not used by the loss; they are audit labels only.

## TreeHeap Code and Distance

For object `o`, construct a three-node differentiable TreeHeap code:

```text
root:  mixture of composed prefix states selected by Theta_place[o]
left:  object leaf state
right: mixture of prefix-slot/address states selected by Theta_place[o]
```

After per-node normalization, use the M0 depth-weighted diff algebra:

\[
D_T(H_i,H_j)
=
\sqrt{\sum_{n\in\{root,left,right\}}w_n
\lVert H_i[n]-H_j[n]\rVert_2^2}
\]

with `w_root=1` and `w_left=w_right=0.72`.

The pull/push loss is multi-positive InfoNCE:

\[
L_{NCE}(i)
=
-\log
\frac{\sum_{j\in Pos(i)}\exp(-D_T(H_i,H_j)/\tau)}
{\sum_{k\ne i}\exp(-D_T(H_i,H_k)/\tau)}
\]

## Two Stages

```text
Stage A: L_context + 0.1 L_echo
Stage B: continue the same checkpoint with
         L_context + 0.1 L_echo + lambda L_NCE
```

Metrics are captured immediately before and after Stage B.  No external model
is used as the decision baseline.

## Registered Gates

Across seeds:

```text
negative_minus_positive_margin_after
  > margin_before + 0.05

echo_accuracy_after = 1.0
heldout_MRR_after >= heldout_MRR_before - 0.02
code_variance_after > 1e-4
effective_prefix_slots_after >= 2
```

The claim is rejected if margin does not improve, if echo/transfer collapses,
or if apparent improvement comes from mapping every object to one state.

## Scope

This tests whether an existing TreeHeap observer can improve its own learned
metric.  It does not yet prove natural-language semantics, a complete
encoder/decoder codec, TreeHeap-specific superiority, or operator edit
distance.

## Result (2026-07-13)

The registered self-improvement gates passed across 8 seeds on io.

| Metric | Before | After | Delta |
|---|---:|---:|---:|
| Positive structural distance | 1.3020 | 0.2454 | -1.0565 |
| Negative structural distance | 2.2377 | 2.3369 | +0.0992 |
| Negative-minus-positive margin | 0.9357 | 2.0915 | +1.1557 |
| Echo accuracy | 1.0000 | 1.0000 | 0.0000 |
| Held-out MRR | 0.4402 | 0.4606 | +0.0204 |
| Held-out Top-3 | 0.6319 | 0.6528 | +0.0208 |
| Cluster purity (audit only) | 0.8194 | 0.8819 | +0.0625 |
| Pairwise F1 (audit only) | 0.7246 | 0.8392 | +0.1146 |
| Full context-cell accuracy | 0.9032 | 0.9105 | +0.0073 |
| Code variance | 0.02935 | 0.02972 | +0.00037 |
| Effective prefix slots | 5.75 | 5.50 | -0.25 |

Every seed increased the structural-distance margin; mean gain was `1.1557`
with standard deviation `0.0490`.  Echo stayed exact in every seed.  Every
final code variance exceeded the collapse gate and every run retained 4..6
active prefix slots.

Interpretation: the same TreeHeap observer can learn to pull context-compatible
three-node codes closer and push incompatible codes farther apart without
destroying its stored input or held-out transfer.  The evidence supports
self-improvement of this metric on a controlled observation corpus.

It does not establish that the gain is uniquely TreeHeap-shaped.  There is no
matched external baseline in this claim, the corpus is a small controlled
world, and the metric does not yet include native operator-program distance.
