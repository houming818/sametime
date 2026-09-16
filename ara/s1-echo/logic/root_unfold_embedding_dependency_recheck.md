# Root-Unfold Embedding Dependency Recheck

## Claim

`S1-ROOT-UNFOLD-DOWNSTREAM-C01`

The corpus-derived root-unfold token coordinate may repair earlier S1 probes
whose downstream state was built from a random trainable lookup or fixed
random token vectors. The new coordinate must improve the original task
metric under a matched data, seed, model, and optimization contract. Its own
context reconstruction score is an input audit, not downstream evidence.

## Dependency Inventory

### Direct dependencies

1. `S1-COMPACT-CONTENT-ROUTE-C01` constructs every compact subheap state by
   summing fixed random token vectors. It can directly consume the derived
   root-unfold vector without changing the route kernel.
2. `S1-TH-EMBED-FP-C01` starts both its embedding-only and recurrent TreeHeap
   arms from a random trainable embedding table. It can instead start both
   arms from the same derived root-unfold coordinate.

### Indirect dependencies, not changed in this experiment

1. `S1-WM-C01` uses frozen MiniLM embeddings as an external evaluation ruler.
   Replacing that ruler would change the Claim rather than repair its input.
2. `S1-WM-C02` explicitly tests a local SGNS coordinate. Root-unfold is a new
   coordinate-source comparison and requires a separate matched study.
3. S3 generation and translation models contain embedding tables, but those
   tables participate in a larger encoder/decoder protocol. The present token
   coordinate result does not authorize replacing them without a new Claim.

## Shared Embedding Contract

Only held-in token-context counts may train the coordinate. For token `t`,

```text
c_t = sqrt(P_train(context | token=t)).
```

A unit root mass is recursively split by shared node gates. The materialized
coordinate is

```text
e_t = sqrt(U_theta(1 | c_t)),
```

so `||e_t||_2 = 1` up to numerical error. There is no trainable token lookup
or direct token-to-leaf parameter in coordinate construction. Held-out route
labels and held-out corpus rows must not enter this stage.

## Experiment A: Compact Route

Run two arms on the original WMT compact-route contract:

- 20,000 rows, vocabulary 1,024, dimension 64;
- train lengths `3..24`, OOD lengths `25..32`;
- seed `42`, five route-kernel epochs;
- identical route model, batches, hidden size, and optimizer;
- arm A uses the original normalized random vectors;
- arm B uses root-unfold vectors trained only from the train-length rows.

The root-unfold arm supports a downstream repair only if:

- embedding mass conservation is below `1e-6`, gradients are finite, and the
  coordinate has no trainable token parameter;
- OOD step accuracy is at least `0.99`;
- OOD route exact is at least `0.99` and exceeds the matched random arm by at
  least `0.003`;
- compact materialization remains below 512 MiB.

## Experiment B: Fixed-Point Loop

Run two arms on one immutable 20K-line pair cache:

- vocabulary 1,024, dimension 32, depth 6, repeat count 2;
- 300 steps, batch 256, four negatives, seed `19301`;
- arm A initializes both the embedding-only and TreeHeap models from the same
  seeded random table;
- arm B initializes both models from the same root-unfold coordinate;
- after initialization, both arms retain the original trainable lookup and
  the original pair-NCE objective.

Two comparisons answer different questions:

1. Root TreeHeap minus root embedding-only measures whether TreeHeap becomes
   useful after the coordinate repair. Support requires at least `+0.002`
   held-out pair-accuracy delta.
2. Root TreeHeap minus random TreeHeap measures warm-start usefulness. Support
   requires at least `+0.005` held-out pair accuracy with no lower LCP margin.

The second comparison includes coordinate-construction compute and therefore
cannot by itself establish TreeHeap superiority. All quality points run to the
registered 300 steps; only OOM, CUDA faults, NaN/Inf, damaged evidence, or loss
of the 270 W power cap may stop execution.

## Interpretation Boundary

Passing either experiment supports only that a previously random coordinate
was a measurable bottleneck for that probe. Failure does not reject the new
embedding mechanism itself. Neither experiment proves sentence generation,
translation, polysemy, or a general world model.

## Result

Io tasks 501--504 completed every registered arm on 2026-09-16. Both arms in
each pair used the same rows, seed, model size, batches, and optimization
budget. Contract tasks 499--500 completed first and were not used as evidence.

### Compact route: direct substitution failed

| Coordinate | OOD step accuracy | OOD route exact | Compact memory |
|---|---:|---:|---:|
| fixed random | `0.99710` | `0.98260` | `324.84 MiB` |
| root-unfold | `0.89200` | `0.48199` | `324.84 MiB` |

The route-exact delta is `-0.50061`, opposite to the registered `+0.003`
gate. The new coordinate itself remained finite and nontrivial: conservation
error `1.19e-7`, utilization `0.84375`, normalized occupancy entropy
`0.87370`, and token-parameter count zero. The failure therefore occurs at
the old compact representation contract. Summing probability coordinates
does not retain the near-orthogonal token-presence evidence supplied by fixed
random vectors. `S1-COMPACT-CONTENT-ROUTE-C01` is not repaired by direct
replacement.

### Fixed point: pair ordering improved, hierarchical route collapsed

| Initialization | Embedding-only pair acc. | TreeHeap pair acc. | Tree - baseline | Leaf use | LCP margin |
|---|---:|---:|---:|---:|---:|
| random | `0.62090` | `0.62410` | `+0.00320` | `35/64` | `0.14980` |
| root-unfold | `0.62320` | `0.64350` | `+0.02030` | `1/64` | `0.00000` |

The root-initialized TreeHeap beats the root-initialized embedding-only arm by
`0.02030` and the random-initialized TreeHeap by `0.01940`. Those accuracy
gates pass. The registered warm-start condition nevertheless fails because
the LCP margin falls from `0.14980` to zero. All tokens choose one downstream
leaf, positive and negative pairs both have LCP depth `6`, and round flip rate
is zero.

This collapse was introduced downstream. Before fixed-point training, the
derived 32D coordinate used `31/32` leaves, had normalized occupancy entropy
`0.93336`, finite gradients, conservation error `1.19e-7`, and no trainable
token parameter. The old pair-NCE loop improves semantic pair ordering but
does not preserve or learn a hierarchical route from that coordinate.

## Decision

`S1-ROOT-UNFOLD-DOWNSTREAM-C01` is mixed:

- supported as evidence that the learned coordinate contains useful
  corpus-pair information under the fixed-point scoring task;
- rejected as a drop-in replacement for compact additive subheap states;
- not supported as a repair of fixed-point hierarchical routing.

The next architecture step must specify coordinate-compatible composition and
a route-preservation objective. More repetitions or a larger vocabulary would
not address either observed failure by themselves.
