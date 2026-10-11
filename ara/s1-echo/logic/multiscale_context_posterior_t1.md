# T1 Multiscale Context-Posterior Annealing Gate

Owner: Nio / Trinity

Created: 2026-10-11

## Question

T0 showed that corpus co-occurrence can produce a useful multiscale probability
coordinate. T1 asks the next narrower question: when held-out local context is
revealed one token at a time, does that fixed coordinate support a progressively
more informative posterior over the masked target token?

This is a lexical conditional-inference gate. It does not train FOLD/READ, claim
sentence compression, or establish generation, reasoning, multilingual alignment,
or an optimal TreeHeap topology.

## Construction

Use the T0 held-in/held-out split and vocabulary contract. The learned balanced
16-leaf context tree is built from held-in counts only. For target token `t`, scale
`K`, and context bin `b`, estimate

```math
p_K(b\mid t)=\frac{n_K(t,b)+\alpha}
{\sum_j n_K(t,j)+K\alpha}.
```

For a sealed held-out masked-token event with observed context bins
`b_1,\ldots,b_m`, use the registered conditional-independence reader

```math
\log q_K(t\mid b_{1:m})=
\log \pi(t)+\sum_{r=1}^{m}\log p_K(b_r\mid t)-\log Z.
```

The posterior entropy is

```math
H_K(T\mid b_{1:m})=-\sum_t q_K(t\mid b_{1:m})\log q_K(t\mid b_{1:m}).
```

T1 evaluates the same events at `m in {1,2,4}` and `K in {1,2,4,8,16}`.
Context tokens are selected by distance from the masked position with a fixed
left-before-right tie rule. Only events with four in-vocabulary context tokens are
used, so all context-count comparisons are paired.

## Controls

1. Learned T0 tree.
2. Three balanced random context-to-leaf assignments with identical counts,
   smoothing, dimensions, event order, and vocabulary.
3. Frequency-only prior (`K=1`).
4. Uncompressed context-token Naive Bayes reader over the full context vocabulary.
   This is a higher-capacity reference, not a matched-cost competitor.

## Registered Claims

`S1-MS-CONTEXT-T1-C01`: at `K=16, m=4`, learned-tree masked-token NLL is at least
`0.02` below the mean matched-random NLL, and the paired bootstrap 95% lower bound
of the random-minus-tree margin is positive.

`S1-MS-CONTEXT-T1-C02`: in the learned `K=16` field, revealing four contexts instead
of one reduces mean posterior entropy by at least `0.05` nat and masked-token NLL by
at least `0.05` nat.

`S1-MS-CONTEXT-T1-C03`: at `K=4, m=4`, the learned tree beats the mean matched-random
NLL by at least `0.01`. This tests whether useful evidence exists before the finest
registered scale.

Quality gates are evaluated only after the complete budget. They never terminate a
formal run early.

## Integrity Gates

- all 16 learned leaves are populated;
- probability rows conserve mass within `1e-6`;
- at least 2,000 paired held-out events exist for smoke;
- all metrics are finite;
- event extraction uses only held-out source lines;
- the full-context reference uses the same held-in counts and held-out events;
- summary and event contract are valid JSON.

## Interpretation Boundary

A pass would support probability annealing in a fixed lexical TreeHeap coordinate:
additional observed context narrows a token posterior and learned placement improves
over matched random placement. It would not prove that parent nodes contain sentence
meaning, that recursive FOLD is sufficient, or that an autoregressive Decoder can
use the field.

A failure would localize the break between static T0 retrieval and event-level
conditional composition. It would not erase T0's static partition evidence.

## Result

Io tasks `650` and `651` completed the registered smoke and formal budgets. The
formal run used 200,000 WMT lines, 8,192 target tokens, 4,096 context tokens,
50,000 paired masked-token events, and three balanced random-tree controls. All
integrity gates passed, archive hashes verified, and the 270 W GPU limit remained
active.

At the finest registered scale, learned-tree NLL for `m=1/2/4` contexts was
`6.938479/6.753027/6.745880`. Mean posterior entropy was
`7.025415/6.814301/6.470929`. The three random-tree NLL values at `m=4` were
`7.028900/7.031557/7.039332`; their mean exceeded the learned tree by `0.287383`
nat. The paired bootstrap 95% interval for that margin was
`0.276356..0.298175`. C01 passed.

From one to four contexts, learned-tree entropy fell by `0.554486` nat and NLL
fell by `0.192599` nat. C02 passed. At the coarser four-bin scale, mean random
NLL exceeded learned-tree NLL by `0.255499`, so C03 also passed.

The uncompressed 4,096-context reference reached NLL `6.250681/5.714359/6.044044`
for `m=1/2/4`. It was stronger than the tree but became worse after the second
context while continuing to lose entropy. This exposes correlated-evidence
overcounting in the registered conditional-independence reader. The TreeHeap field
loses lexical detail but regularizes that failure: its formal `m=4` NLL remained
slightly below `m=2`. Smoke did show a small `m=2 -> 4` rebound, so monotonicity is
not promoted to a universal claim.

Top-1 did not follow NLL monotonically (`0.05458` at `m=1`, `0.05010` at `m=4`).
Therefore the supported claim is posterior information and calibration under a
fixed lexical reader, not improved argmax decoding, sentence meaning, or generation.

Evidence:
`../evidence/s1_multiscale_context_posterior_t1/`.
