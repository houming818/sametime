# Annealed FOLD/UNFOLD Token Space

## Claim

`S1-ANNEAL-SPACE-C01`

TreeHeap can derive a token coordinate system from observed token-context
statistics without a randomly initialized trainable token embedding.  A
deterministic annealing protocol alternates:

```text
UNFOLD: parent prototype -> soft left/right responsibilities
FOLD:   responsibilities + observations -> updated child prototypes
```

The resulting root-to-leaf route probabilities are the token coordinates.
Semantic labels are never available to the algorithm and are used only for
post-hoc audit.

## Observable State

For token `i`, the observer first estimates a context distribution:

```text
x_i[c] = P(context=c | token=i)
```

The algorithm operates on the Hellinger coordinate `sqrt(x_i)`.  This is a
bounded, dimensionally consistent representation of a probability
distribution; it is not a learned embedding table.

At a node with child prototypes `mu_L` and `mu_R`:

```text
r_i(s) = softmax_s((log pi_s - ||sqrt(x_i)-mu_s||^2) / T)
```

UNFOLD computes `r_i(L), r_i(R)`.  FOLD updates each prototype by the
responsibility-weighted mean.  Temperature `T` decreases on a fixed schedule.
The split is initialized deterministically from the leading singular direction
of the node observations, not from a random token placement.

For a depth-`D` tree the route vector is:

```text
z_i = [P_i(right at depth 0), ..., P_i(right at depth D-1)]
```

## Experiment

Three arms use equal token and context cardinality:

1. `structured`: context observations retain a latent category law.
2. `shuffled`: each token keeps its observation count, but context IDs are
   independently permuted to destroy shared category structure.
3. `random_route`: the structured observations are kept, but routes are random.

The controlled corpus contains 64 tokens in 8 latent categories.  Each token
has category contexts, global contexts, and a private signature context.  The
algorithm sees counts only.  Independent held-out counts audit whether a leaf
prototype predicts the original context law.

Formal execution uses at least 12 data seeds, depth 4, and the same annealing
schedule for structured and shuffled arms.

## Metrics

```text
heldout_context_nll         lower is better
cluster_purity              labels used only here
pairwise_f1                 labels used only here
route_knn_top3              labels used only here
leaf_utilization            occupied leaves / available leaves
leaf_occupancy_entropy      normalized entropy of leaf mass
fold_conservation_max_abs   parent vs mass-weighted children
route_stability             pairwise co-cluster F1 across data seeds
```

Token-specific empirical prediction is reported as an upper-information
reference; a global unigram is a lower-information reference.  Neither is a
TreeHeap arm.

## Pre-registered Decision

Support this narrow pilot only if all conditions hold on the formal run:

```text
structured purity - shuffled purity >= 0.20
structured purity - random_route purity >= 0.20
structured pairwise_f1 - shuffled pairwise_f1 >= 0.20
structured heldout NLL < shuffled heldout NLL
structured heldout NLL < random_route heldout NLL
structured route stability > shuffled route stability
structured leaf utilization >= 0.75
structured occupancy entropy >= 0.80
fold conservation max abs <= 1e-6
```

If the algebraic conservation gate passes while semantic/control gaps fail,
the result supports only a numerically closed partition process.  It does not
support an induced semantic coordinate system.

## Boundaries

This experiment does not prove natural-language semantics, WMT translation,
sentence generation, or superiority over pretrained embeddings.  It tests the
smaller existence claim that an observed probability law plus a TreeHeap
FOLD/UNFOLD annealing protocol can generate non-random token coordinates.

## Formal Result

Executed on `io.grepcode.cn` as task `446` on 2026-09-16.  The formal run used
24 seeds (`19001..19024`), 64 tokens, 136 contexts, depth 4, 1,200 train and 600
held-out observations per token.  It emitted 72 complete arm records.

| Metric | Structured | Shuffled | Random route |
|---|---:|---:|---:|
| held-out context NLL | 2.7745 | 8.6505 | 3.6886 |
| cluster purity | 1.0000 | 0.3991 | 0.3906 |
| pairwise F1 | 0.7438 | 0.0740 | 0.0850 |
| route kNN top-3 | 0.8902 | 0.1148 | 0.1105 |
| leaf utilization | 0.9948 | 0.9974 | 0.9714 |
| occupancy entropy | 0.9441 | 0.9760 | 0.9471 |
| route stability | 0.7094 | 0.2296 | 0.2368 |

Structured FOLD conservation max-abs mean was `1.02e-16`.  All nine registered
gates passed, so the narrow controlled claim is supported.  Purity `1.0` and
pairwise-F1 `0.7438` are not contradictory: every leaf is category-pure, while
some categories are over-segmented into multiple leaves.

Evidence: `evidence/s1_annealed_fold_unfold_token_space/formal_seed19001/`.
