# Root-Unfold Token Embedding

## Claim

`S1-ROOT-UNFOLD-EMBED-C01`

A unit root signal can be recursively split by shared TreeHeap node gates into
a token-dependent leaf probability vector. The resulting vector, rather than
a directly optimized token lookup or token-to-leaf matrix, is the token
embedding. A hard selector may choose one leaf for readout while gradients
update the shared unfold gates and leaf context decoders.

## Mathematical Boundary

For a deterministic map with no other state,

```text
U_theta(1) = U_theta(1).
```

Therefore identical scalar inputs cannot produce token-dependent embeddings
unless the gates also observe some token-dependent condition, mutable state,
or token-specific parameter. This experiment keeps root mass equal to `1`
for every token and supplies only a fixed real-corpus observation

```text
c_t = sqrt(P_train(context | token=t))
```

to shared node gates. There is no trainable token lookup and no direct
`token x leaf` parameter.

## Unfold Operator

At level `l`, token `t` has non-negative mass `h[l][t,n]` at node `n`.
The node computes

```text
g[t,n] = sigmoid((c_t W[n] + b[n]) / temperature)
left    = h[l][t,n] * (1 - g[t,n])
right   = h[l][t,n] * g[t,n].
```

Thus depth three produces

```text
[1] -> [2 values] -> [4 values] -> [8 values]
```

and every level must conserve unit mass. The final eight values are the
derived route embedding. The matched selectors are:

- local `argmax`, which exposes whether unfold alone crowds tokens;
- exact equal-capacity selection, which supplies mutual exclusion without
  becoming a trainable token-to-leaf table.

Both selectors use a straight-through gradient: hard one-hot routes are used
for context prediction, while derivatives follow the recursively generated
soft leaf vector.

## Preregistered Smoke

- host: `io.grepcode.cn`, RTX 3090 capped at 270 W;
- corpus: WMT massive English side, first 50,000 valid lines;
- 128 readable target pieces and 256 context pieces;
- depth `3`, hence exactly eight leaf values per token;
- every tenth line held out from gradients;
- deterministic corpus-derived split initialization;
- 160 full-batch Adam steps, learning rate `0.01`;
- temperature `0.50`, decoder blend `0.50`, balance weight `0.10`;
- seed `19501` is used only for the shuffled-input and random-route controls.

If this smoke passes, one registered successor may repeat the same algorithm
at 100,000 lines, 256 targets, 512 contexts, depth `4`, and 240 steps. No
larger run is authorized by this document.

## Gates

The capacity-selected arm supports the claim only if all are true:

- root mass is exactly one and maximum per-level conservation error is below
  `1e-6`;
- trainable token parameter count is zero;
- gate and decoder gradients are finite and gate parameters move;
- held-in context NLL decreases by at least `0.01`;
- hard occupancy is exactly equal across all leaves;
- held-out NLL is at least `0.05` below the global-context predictor and no
  more than `0.03` worse than the frozen initial unfold;
- shuffling token observations increases held-out NLL by at least `0.02`;
- held-out semantic-neighbor route-prefix length beats balanced random;
- token-to-token variance of the derived leaf vectors is nonzero.

The argmax arm is a required diagnostic but cannot by itself pass or fail the
capacity-selected claim. Its utilization and occupancy reveal how much of the
final separation comes from recursive gates versus the external exclusion
operator.

Passing does not prove that a raw token ID can route without observations,
that the route supports polysemy, or that a sentence decoder can use the
embedding for generation or translation.

## Result

Io tasks 497--498 completed both registered points and passed every gate.

| Lines / targets / leaves | Capacity NLL initial -> final | Global / shuffled NLL | Capacity LCP / random | Local argmax use | Conservation |
|---|---:|---:|---:|---:|---:|
| 50K / 128 / 8 | `4.38605 -> 4.36042` | `4.46868 / 4.59844` | `1.70833 / 0.84115` | all leaves, `10..24` | `1.19e-7` |
| 100K / 256 / 16 | `4.89755 -> 4.87234` | `5.02004 / 5.29377` | `2.06120 / 0.94010` | all leaves, `12..21` | `1.79e-7` |

The exact-capacity selector retained `16..16` tokens per leaf at both points.
More importantly, the local argmax diagnostic used every leaf without the
capacity projection. Therefore the recursive unfold itself produced a
nontrivial token-dependent field; the capacity operator regularized occupancy
but did not create all branching. Shuffling the fixed token observations
increased NLL by `0.23802` and `0.42143`, while hard-route agreement fell to
`0.11719` and `0.04297`, respectively.

The claim is supported through 256 targets as a token-context embedding
mechanism. The result does not remove the mathematical boundary: context
statistics provide the token-dependent condition. It does not show that
identical scalar inputs with no condition or mutable state can separate, nor
does it prove sentence-level decoding, polysemy, generation, or translation.
