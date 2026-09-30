# A19 Recursive Context Codec Dimension Ladder

Date: 2026-09-30

Status: completed; registered gate mechanically passed, interpretation remains
bounded by strong address-prior use

## Question

Can one shared recursive FOLD/READ operator compress a real 1,024-coordinate
token-context probability field into a small node state and recover useful
held-out context probabilities? How does that behavior change at state
dimensions `2/4/8/16`?

This experiment responds to a specific architecture concern: a global PCA or
random projection followed by a tree does not make dimension formation
recursive. A19 therefore permits no flat `1024 -> d -> 1024` bypass.

## Frozen Data

Reuse the exact A14 one-million-line count artifact:

```text
ara/s1-echo/evidence/s1_probability_residual_corpus_scale_a14/
  formal/counts/train_1000000/context_counts.pt
```

The artifact contains 512 target-token rows, 1,024 context coordinates, the
training counts, and the byte-identical sealed-test counts. The A14 split seed
and binomial fit/dev split are reused. Every dimension arm receives exactly
the same rows, counts, batches, seed family, and number of epochs.

## Recursive State Contract

For token row `t`, each context probability is initialized as a leaf state:

```text
h[t,c,0] = E_d(sqrt(p(c|t)), log(p(c|t)))
```

At every level, the same FOLD module is reused:

```text
h_parent = F_d(h_left, h_right)
```

After ten binary levels, one `d`-dimensional root remains. READ starts only
from that root and recursively reuses one shared module:

```text
(h_left_hat, h_right_hat) = G_d(h_parent_hat)
```

After ten READ levels, one shared scalar output head produces 1,024 context
logits. There is no direct root-to-vocabulary layer, leaf skip, original
probability skip, per-address embedding, or per-depth F/G parameter set.

The state dimension is the only architecture variable:

```text
d in {2, 4, 8, 16}
```

## Training Contract

- fixed split seed: `20260924`;
- fixed model seed: `20260930` plus a registered dimension offset;
- complete 1M A14 count field;
- 512 target rows and 1,024 context outputs;
- batch size 64 target rows;
- 200 complete epochs per arm;
- AdamW, fixed learning rate and weight decay;
- output heads start at the same global fit marginal;
- quality never terminates an arm early;
- OOM, CUDA failure, NaN/Inf, corrupt evidence, or reload failure may stop the
  queue.

## Measurements

For every dimension:

1. fit, search-dev, and sealed-test pair-weighted NLL/PPL;
2. fixed-epoch learning curve;
3. NLL after READ depth `0..10`, where unread descendants are broadcast;
4. root-shuffle damage across target-token rows;
5. zero-root and branchless-READ controls;
6. FOLD state RMS/variance by level;
7. root effective rank and pairwise cosine;
8. finite gradients, checkpoint hash, and exact reload equality;
9. parameter count and elapsed time.

The depth curve is the main recursion observation. If READ is genuinely using
successive resolution, later depths should add predictive information rather
than all useful information appearing in a direct root output.

## Registered Claim

`S1-RECURSIVE-CONTEXT-CODEC-A19-C01`: at one or more registered dimensions,
the shared ten-level recursive codec finishes the full budget, remains finite,
reloads exactly, beats the branchless and shuffled-root controls on sealed
context NLL, and exhibits a non-flat READ-depth information curve.

This is a usability gate, not an optimum search. No dimension is declared
best merely because it has the lowest observed NLL.

## Boundaries

A pass would establish only that a low-dimensional recursively produced state
can carry usable information about this fixed context field. It would not
establish translation, sentence generation, semantic grounding, universal
TreeHeap topology, exact reversibility, or superiority over Transformer/PCA.

The ordering of the 1,024 context coordinates remains a fixed experimental
topology. A later claim must separately test learned or permutation-robust
context topology.

## Formal Result

The registered `2/4/8/16` ladder completed 200 epochs per arm on `io` with
finite values and exact checkpoint reloads.

| state dimension | parameters | sealed-test NLL | PPL | full READ gain | shuffled-root damage | root effective rank |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 77 | 6.931472 | 1024.001 | 0.000000 | 0.000000 | 1.000 |
| 4 | 249 | 5.605970 | 272.046 | 1.325502 | 0.026981 | 1.030 |
| 8 | 881 | 5.597232 | 269.679 | 1.334240 | 0.040823 | 2.081 |
| 16 | 3297 | 5.572473 | 263.084 | 1.358999 | 0.068362 | 3.758 |

For every trainable arm, NLL decreased progressively as READ depth increased.
At `d=16`, the sealed-test curve from depth zero through ten was:

```text
6.9315, 6.7602, 6.5877, 6.4032, 6.2045, 6.0220,
5.8439, 5.6818, 5.6099, 5.5851, 5.5725
```

The result therefore supports the narrow registered statement that a shared
recursive operator can form and read a bounded context state, and that usable
capacity changes with state dimension. It rejects the stronger informal idea
that merely adding recursion makes very small state dimensions sufficient:
the `d=2` arm remained at the uniform distribution with a maximum gradient
norm of approximately `7.1e-7`.

## Interpretation

The formal gate passed, but it must not be read as proof of an approximately
invertible codec. At `d=16`, shuffling roots between target-token rows worsened
NLL by only `0.068362`, whereas removing recursive READ worsened it by
`1.358999`. Most measured gain therefore comes from the learned recursive
address structure and corpus-wide context prior; only a smaller, nonzero part
depends on which token-specific root is supplied.

This separates two facts:

1. READ depth is causally active and adds progressively finer address regions.
2. FOLD-to-root conditioning is still weak; roots remain highly aligned
   (`mean off-diagonal cosine = 0.9717` at `d=16`).

The next architecture question is not whether to replace recursion with a flat
array. It is how to increase token-conditional information retained by the
shared FOLD without allowing a direct leaf or address bypass. A follow-up
should compare against an explicit global-context prior and require meaningful
root-shuffle damage, while preserving the same recursive contract.

The post-hoc baseline audit makes this limitation quantitative:

| baseline or arm | sealed-test NLL | PPL |
|---|---:|---:|
| uniform over 1,024 contexts | 6.931472 | 1024.000 |
| global context prior, no target-token input | 5.555696 | 258.707 |
| `d=16` recursive codec | 5.572473 | 263.084 |
| direct per-token probability field | 5.116258 | 166.710 |

Thus `d=16` remains `0.016777` NLL worse than the explicit global-prior
baseline. Its shuffled-root damage is real but represents only about `5.03%`
of its total gain over uniform. The present codec has learned a useful address
decoder and a weak conditional channel; it has not yet learned a competitive
root-conditioned compression of the context field.

## Evidence

```text
ara/s1-echo/evidence/s1_recursive_context_codec_dimension_ladder_a19/
```

The post-hoc global-prior decomposition is diagnostic only and is not added to
the preregistered pass gate.
