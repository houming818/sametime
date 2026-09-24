# S1 Probability-Residual Corpus Scale A14

Status: supported for the mechanical comparison claim; scale-quality result is mixed

## Question

Does increasing corpus exposure improve or degrade the frozen-vocabulary
TreeHeap probability-residual embedding when representation capacity and every
other experimental input remain fixed?

The experiment measures a context-field embedding. It does not measure
generation, translation, READ, Decoder compatibility, or downstream semantic
task quality.

## Controlled Contract

The following are fixed across every arm:

```text
SentencePiece model and hash
512 target token identities
1024 context token identities
context window = 4
maximum sentence length = 96 tokens
sealed test = valid English examples 1,000,000 through 1,099,999
fit/dev split seed = 20260924
F-search seed = 20260929
F-search budget = 1024 proposals
alpha = 0.05
temperature = 0.02
```

Only two independent variables are allowed:

```text
training examples: 50K, 100K, 200K, 500K, 1M
TreeHeap depth: 3 and 5
```

All count snapshots are produced by one forward scan. Each larger training
field strictly contains every smaller field. The sealed test follows the 1M
training prefix and is byte-identical for all ten arms.

## Competing Predictions

`H_more_data`: More observations reduce sampling noise. Sealed-test NLL should
decrease or plateau, context-neighbor sets should stabilize, and no large rise
in frequency leakage should appear.

`H_fixed_capacity`: At fixed vocabulary and TreeHeap capacity, more corpus can
expose incompatible senses or frequency domination. Sealed-test NLL may form a
U-shaped curve, adjacent partitions may remain unstable, or the fraction of
log token frequency explained by leaf identity may rise.

Neither prediction is a stopping rule. Every arm runs its full registered
budget even when intermediate quality is non-monotonic.

## Measurements

For every arm:

```text
dev and fixed sealed-test NLL
gain relative to deterministic initialization
random-routing control
leaf utilization and normalized occupancy entropy
frequency R-squared explained by leaf partition
fold conservation and residual closure
```

For adjacent corpus scales:

```text
sealed-test NLL delta
label-invariant partition co-cluster F1
top-10 context-neighbor Jaccard similarity
frequency-leakage delta
```

The context-neighbor metric is computed from the square-root conditional
probability coordinates under the same split seed. The partition metric is
label-invariant because left/right leaf numbers may flip without changing the
partition.

## Claim

`S1-F-SCALE-A14-C01`: The registered fixed-vocabulary corpus ladder can produce
ten mechanically valid, directly comparable TreeHeap embedding observations
against one byte-identical sealed test field.

This claim is supported only if all ten runs are finite, preserve algebraic
conservation and residual closure, retain the registered data identities, and
complete the full search budget. Whether more data improves embedding quality
is an experimental result, not a gate used to terminate the experiment.

## Artifact Policy

Every arm saves its summary, trace, and compact routing index. Full embedding
checkpoints are retained for the 1M endpoint at depths 3 and 5. Intermediate
large checkpoints are not required because counts, seeds, code, and routing
indices are sufficient to reproduce the scale curve.

## Result

The formal queue completed as taskd jobs `587` through `598`. All ten arms ran
the full 1024-proposal budget. Every value was finite, all leaves remained in
use, fold conservation and residual closure passed, and the two 1M endpoint
checkpoints reloaded exactly. The fixed sealed field contains 100,000 examples
and 3,670,609 context pairs; all five count manifests share SHA-256
`5b194125733e0b346de0886640231ef37745e9fd4cfbf5d5eca990af6b5f1512`.

Best sealed-test NLL by scale:

```text
train examples       50K       100K       200K       500K        1M
depth 3          5.419493   5.422999   5.413569   5.411712   5.409086
depth 5          5.349730   5.337926   5.328784   5.332392   5.331227
```

Neither curve is strictly monotonic. Depth 3 improves by `0.010406` NLL from
50K to 1M and reaches its best observation at 1M. Depth 5 improves by
`0.018502` from 50K to 1M, but its best observation is 200K; 500K and 1M are
respectively `0.003608` and `0.002443` worse than that point.

The square-root conditional-probability geometry becomes more stable with
more data. Adjacent-scale top-10 neighbor Jaccard rises from `0.659` to `0.703`,
`0.707`, and `0.815`. Tree partitions do not show the same convergence:
co-cluster F1 remains roughly `0.20-0.45`, with depth 5 especially unstable.
Frequency R-squared is non-monotonic and does not show a cumulative
high-frequency takeover.

Therefore A14 does not support the broad claim that larger corpora degrade the
embedding field. It supports a narrower diagnosis: the observed context field
stabilizes with scale, while the current fixed-budget Monte Carlo F search does
not yet recover a stable tree topology. The small NLL reversals cannot separate
finite search variance from a real fixed-capacity limit because A14 uses one
search seed per arm. Replicated search or warm-start continuation is required
before declaring the 200K depth-5 point a capacity optimum.

`S1-F-SCALE-A14-C01` is supported. This remains an embedding reconstruction
result and does not establish downstream semantic, READ, Decoder, translation,
or generation quality.
