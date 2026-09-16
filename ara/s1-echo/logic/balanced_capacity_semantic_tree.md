# Balanced-Capacity Semantic Tree

## Claim

`S1-BALANCED-CAPACITY-C01`

A TreeHeap can separate geometry from semantics: exact subtree capacities
guarantee uniform landing and mutual exclusion, while distances between
observed token-context probability laws learn the arrangement. NCE, negative
sampling, and a randomly initialized embedding are not required.

## State and Objective

For token `v`, the observer estimates `p(c|v)` from real corpus counts and uses
the Hellinger coordinate `x_v = sqrt(p(c|v))`. At every internal node, tokens
are assigned to two equal-capacity children by balanced two-means:

```text
cost(v, child) = ||x_v - prototype_child||^2
```

The lower-cost permutation is projected onto exact child capacities. Child
prototypes are then the FOLD means of their assigned observations. Recursing
this operation produces a balanced TreeHeap. Similar tokens may occupy
different leaves, but should share a longer prefix; finite child capacity
provides mutual exclusion.

## Fixed-Seed Smoke

- WMT massive English side, first 50,000 valid lines;
- 256 readable target pieces and 512 context pieces;
- symmetric context window 4; every tenth line held out;
- depth 6, hence 64 leaves and exactly 4 token types per leaf;
- all semantic construction is deterministic;
- seed `19401` is used only for the balanced-random reference and random-pair
  audit, never to search the semantic tree.

Controls preserve the same capacities:

1. balanced random permutation;
2. balanced frequency-rank ordering.

Held-out context counts are never used to construct routes. They define
held-out leaf NLL and held-out nearest-neighbor pairs for topology audit.

## Smoke Gates

- maximum child-capacity imbalance is exactly zero;
- leaf utilization and normalized occupancy entropy are each `1.0`;
- semantic-tree held-out context NLL beats balanced random;
- held-out nearest-neighbor LCP exceeds random-pair LCP by at least `0.25`;
- held-out nearest-neighbor LCP exceeds both balanced controls;
- all metrics and FOLD prototypes are finite.

Passing establishes only that exact balanced capacity and corpus-derived
semantic arrangement can coexist. It does not prove generation, translation,
polysemy resolution, differentiable end-to-end training, or superiority at
larger vocabulary scales.

## Smoke Result

Task 475 failed before computation because the CLI evidence path was absent
from the configuration dataclass; its taskd log is retained. Corrected task
476 passed all registered gates on 50,000 lines. The semantic tree used every
leaf with exactly four target pieces (`utilization=1`, normalized occupancy
entropy `=1`, maximum child imbalance `=0`). On held-out lines its context NLL
was `4.78872`, compared with `4.85255` for balanced random and `4.87903` for
balanced frequency ordering. Held-out nearest neighbors shared `2.02214`
route levels on average, versus `1.03255` for random pairs; the balanced-random
and balanced-frequency controls reached `0.94531` and `1.28255`. FOLD mean
conservation error was `5.55e-17`.

This supports the narrow claim that explicit capacity supplies exclusion while
corpus-derived probability distance supplies attraction, without NCE. It does
not yet show that the relationship survives larger vocabularies or can be
trained through a decoder loss.

## Scale Result

Tasks 477--478 extended the same four-tokens-per-leaf contract:

| Lines | Targets / contexts | Depth | Semantic / random / frequency NLL | Semantic / random / frequency neighbor LCP |
|---:|---:|---:|---:|---:|
| 200K | 512 / 1024 | 7 | 5.26116 / 5.33487 / 5.37730 | 2.45964 / 1.01888 / 1.25065 |
| 500K | 1024 / 2048 | 8 | 5.70540 / 5.78394 / 5.83967 | 2.79785 / 0.99219 / 1.27572 |

Both scale points retained exact capacity, full utilization, normalized
occupancy entropy `1.0`, and FOLD conservation error `5.55e-17`. Absolute NLL
is not comparable across rows because the context vocabulary changes; the
matched within-row gaps are the relevant measure. Semantic-minus-best-control
neighbor LCP margins grew from `0.73958` at 256 targets to `1.20898` at 512
and `1.52214` at 1024. The 512 margin is computed against its strongest
frequency control (`2.45964 - 1.25065`).

The raw scale summaries retained the script's P01 experiment label although
the tasks were preregistered as P02. Task IDs, commands, configurations, and
the report below provide the explicit mapping; raw summaries are preserved.
