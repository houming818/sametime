# TreeHeap vs MLP Under a Shared Objective

## Claim

`S1-METRIC-ARCH-C01`:

> TreeHeap and MLP may optimize the same observational objective without being
> the same architecture or using the same representation algorithm.  A fair
> comparison must let each architecture build its native code, then compare
> task-level and scale-free outcomes rather than raw distances.

This experiment is diagnostic.  It does not pre-register that TreeHeap must
win.

## Shared Goal, Different Mechanisms

Both models receive the same unlabeled verb-object observations and optimize
the same functional requirements:

```text
preserve object identity (echo)
predict observed context
pull context-compatible objects together
push context-incompatible objects apart
transfer to held-out combinations
```

TreeHeap uses its native mechanism:

```text
object leaf
Theta_place -> latent prefix address
Theta_compose -> internal prefix state
three-node root/leaf/address code
depth-weighted TreeHeap diff
```

MLP uses its native mechanism:

```text
object embedding
dense nonlinear encoder
one continuous latent vector
Euclidean distance
```

The positive/negative observation relation and training budget are shared.  The
representation, distance implementation, and parameter topology are not.

## Why Raw Margin Is Not Compared

TreeHeap distance sums three weighted nodes; MLP distance measures one vector.
Their raw scales are not commensurate.  Cross-architecture comparison uses:

- pair-distance AUC: probability that a positive pair is closer than a
  negative pair;
- standardized margin `(mean_negative - mean_positive) / pooled_std`;
- nearest-neighbor category accuracy, with categories used only for audit;
- held-out context MRR/Top-3;
- echo and context accuracy;
- parameter count and seed variance.

Each model is also measured before and after its own contrastive stage.

## Registered Interpretation

```text
Both improve:
  the shared pull/push objective is architecture-general.

TreeHeap improves more on held-out transfer/pair AUC:
  evidence that explicit prefix/compose bias helps this corpus.

MLP improves more:
  current TreeHeap metric/placement is not yet competitive on this goal.

Mixed result:
  report the capability profile; do not collapse it into one winner.
```

No superiority statement is permitted from raw distance margin alone.

## Falsification and Scope

The comparison is invalid if either model receives gold category labels, if
their data/objective differs, if raw distance is treated as a common unit, or
if only one seed is reported.  This controlled corpus does not establish
natural-language semantics, WMT quality, or general architecture superiority.
