# TreeHeap / Embedding Fixed-Point Loop

## Claim

`S1-TH-EMBED-FP-C01`

A single learnable TreeHeap and one token embedding table can form a closed
training loop: tokens repeatedly fall through the same TreeHeap, the resulting
vectors are scored only by fixed corpus positive/negative pairs, and one loss
delivers finite non-zero gradients to both the embedding table and TreeHeap.

This is a mechanism claim. It does not claim semantic superiority, generation,
or a solved background model.

## Minimal Model

```text
token id -> E[token] -> shared TH fall -> updated token vector
                                      -> repeat K times
updated positive/negative pair scores -> one contrastive loss
                                      -> update E and TH
```

The model contains only:

1. a trainable token table `E[V,d]`;
2. one shared TreeHeap containing node route weights, route biases, and node
   values.

No separate background-field model, teacher, or token-specific router is used.

## Smoke Contract

```text
corpus lines: 20,000 WMT massive English-side
vocabulary: 1,024
dimension: 32
tree depth: 6
repeated falls: 2
steps: 300
batch / negatives: 256 / 4
train/test pairs: capped at 200,000 / 30,000
seed: 19301
```

Smoke passes only if:

- embedding, route weight, route bias, and node value gradients are finite and
  greater than `1e-8`;
- held-out positive-vs-negative pair accuracy exceeds `0.52`;
- positive pairs share a longer path prefix than random negatives;
- leaf utilization exceeds `0.10` and normalized occupancy entropy exceeds
  `0.50`;
- all recorded losses are finite.

## Bounded Formal Queue

Only after smoke passes, prepare one shared 100K-line pair cache and run a
fixed-budget ladder with repeat counts `1,2,4,8`. Each arm uses the same cache,
seed, initialization contract, 4,096-token vocabulary, 64D table, depth 9,
1,000 steps, batch 512, and four negatives. Every arm also trains a matched
embedding-only baseline on exactly the same sampled batches.

The queue must run serially on io under the existing 270W limit. Quality
metrics are observations, not early-stop gates. OOM, CUDA faults, NaN/Inf,
damaged evidence, or loss of the power limit are the only automatic stop
conditions.

## Interpretation

The fixed corpus pair labels are the reference frame. If embeddings and TH
move together without improving held-out positive-vs-negative ordering, that
is parameter co-adaptation rather than useful landing. Repeat count is a
controlled variable, not an assumed improvement.

## First Formal Ladder Result

Tasks 460--463 completed the `1/2/4/8` repeat ladder on the shared 100K-line
cache. Tree held-out pair accuracy was `0.6675/0.6671/0.6649/0.6647` versus the
matched embedding-only `0.6687` in every arm. Repeat depth therefore did not
improve the primary task metric. It did expand route use: leaf utilization rose
from `0.1582` to `0.2832`, and normalized occupancy entropy from `0.3833` to
`0.5392`. Positive-minus-negative LCP remained positive in all arms. The
mechanism is differentiable and nontrivial, but no quality advantage is shown.

## Registered Follow-up

Before changing the architecture, run two r8 seed references (`19302/19303`)
at 1,000 steps and two longer 5,000-step arms (`r2/r8`, seed 19301). The causal
comparison keeps seed `19301` fixed; the additional seeds are robustness
references only and must not be pooled into the primary estimate. All use the
existing immutable pair cache and matched embedding-only arm. No metric-based
early stop is allowed.

## Follow-up Result

Tasks 464--467 completed the preregistered follow-up without runtime faults.
The primary comparison keeps seed `19301` fixed. At 1,000 steps, r1/r2/r4/r8
Tree-minus-baseline accuracy deltas were `-0.00120/-0.00155/-0.00380/-0.00400`.
At 5,000 steps, r2/r8 deltas were `-0.00365/-0.00100`. These fixed-seed results
do not establish a quality advantage.

Seeds `19302/19303` are reference runs, not members of the primary causal
estimate. Their r8/1,000-step deltas were `+0.00070/-0.00160`, compared with
`-0.00400` for seed `19301`; all three margin deltas were negative. The
references show seed sensitivity and do not reverse the fixed-seed decision.

Longer optimization also did not establish an advantage. At 5,000 steps,
r2 scored `0.67725` versus baseline `0.68090`, while r8 scored `0.67990`
versus `0.68090`. The r8 gap narrowed from `-0.00400` at 1,000 steps to
`-0.00100` at 5,000 steps, but r2 widened from `-0.00155` to `-0.00365`.
This is not a consistent fixed-seed training-time trend.

The structural signal remains real but optimization concentrates routes.
Every run kept positive LCP above negative LCP. However, r2 utilization and
entropy changed from `0.2168/0.4397` at 1,000 steps to `0.0352/0.2462` at
5,000, and r8 changed from `0.2832/0.5392` to `0.1582/0.4499`. Repeated falls
therefore expose nontrivial shared routing, but the current pair-NCE objective
does not preserve broad occupancy or deliver a semantic-quality gain over a
matched embedding table.

## Decision

`S1-TH-EMBED-FP-C01` is supported only as a mechanism claim. Under the tested
100K-line, 4,096-token pair-NCE contract, semantic advantage over the matched
embedding-only baseline is not supported. Do not scale this exact objective
unchanged. A successor must add an explicit anti-collapse or hierarchical
information objective and preregister a baseline comparison before using more
GPU time.
