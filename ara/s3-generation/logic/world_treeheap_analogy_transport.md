# World TreeHeap Analogy Transport

Date: 2026-07-15
Status: full World claim not supported / finite analogy subclaim supported
Claim: `S3-WORLD-ANALOGY-C01`
Predict: `P-S3-WORLD-ANALOGY-01`

## Claim

In a finite, fully enumerable toy world, a learned parameter World TreeHeap and
TreeHeap-native `DIFF -> TRANSPORT -> APPLY` path can infer a relation operator
from `A:B`, transport it into a compatible but held-out context `C`, and
produce `D`. The result must cover non-copy relations and causally depend on
the World TreeHeap; surface target-token replacement is not sufficient.

## Relation Objects

```math
H_A=E(A),\quad H_B=E(B),\quad H_C=E(C),
```

```math
R_{AB}=DIFF_\psi(W,H_A,H_B),
```

```math
R_C=TRANSPORT_\omega(W,R_{AB},H_A,H_C),
```

```math
\hat H_D=APPLY_\eta(H_C,R_C),\qquad \hat D=G(\hat H_D).
```

`W` is a persistent 15-node parameter TreeHeap. A recursive probability
kernel returns `{stop,left,right}` at every visited node and reads a soft
background state. `R_AB` is a multiscale relation TreeHeap, not one vector:
each leaf and ancestor receives a relation state. `APPLY` gathers the relation
states along each output leaf's ancestor path.

## Finite World

All statements use eight fixed logical leaves. The vocabulary contains people,
foods, numbers, predicates, unit, and zero. Four relation families are used:

1. food replacement transported across `eat`, `buy`, and `price` templates;
2. numeric `+1/-1` transported between `price` and `buy` templates;
3. person replacement transported between subject and object roles;
4. subject/object role swap transferred to unseen person pairs.

Examples:

```text
A  P1 EAT SWEET_POTATO
B  P1 EAT POTATO
C  SWEET_POTATO COST N2 UNIT
D  POTATO COST N2 UNIT
```

```text
A  FOOD3 COST N2 UNIT
B  FOOD3 COST N3 UNIT
C  P4 BUY FOOD8 N5
D  P4 BUY FOOD8 N6
```

The validation split is not a random row sample: a stable hash reserves whole
person/food/number/template combinations. Every atomic symbol remains visible
in training, but selected combinations do not.

## Training

Training rotates three objectives instead of summing one large loss:

1. surface analogy token cross-entropy;
2. multiscale state alignment to the encoded `H_D`;
3. private encoder/decoder echo on the four statements.

A flat Transformer analogy model is trained on the same quadruples. A
deterministic lexical-replacement baseline copies the token changes visible in
`A:B` into matching tokens in `C`; it should solve direct replacement but fail
numeric transport and role swap.

## Predict

```text
P1  Full TreeHeap held-out sequence exact >= 0.90 and every non-copy relation
    family exact >= 0.80.

P2  Full TreeHeap exceeds deterministic lexical replacement by >= 0.20 exact
    on all examples and >= 0.30 on non-copy examples.

P3  Zeroing or heap-address-shuffling W increases token error by >= 0.10
    absolute. Otherwise W is decorative.

P4  Pairing C with the wrong B relation increases token error by >= 0.30.

P5  Non-target token preservation >= 0.95, so APPLY is selective rather than
    rewriting the entire statement.

P6  The flat Transformer baseline is reported. TreeHeap superiority is not
    claimed unless it wins at matched parameter/compute scale across seeds.
```

## Falsification and Boundaries

- If lexical replacement matches the model, the proof establishes no analogy.
- If W interventions do not hurt, the World TreeHeap claim is rejected even if
  outputs are accurate.
- If the model passes only seen combinations, it learned a table rather than
  transport.
- If the flat baseline wins, analogy may exist but there is no TreeHeap
  advantage.
- A toy pass would establish only finite-world operator induction and
  transport. It would not prove natural-language semantics, consciousness,
  multilingual reasoning, or WMT quality.

## Result

The preregistered single-seed smoke completed on `io` after 6,000 steps and
4,096 held-out hash-blocked combinations.

```text
TreeHeap token / sequence exact       1.0000 / 1.0000
Flat Transformer token / exact        1.0000 / 1.0000
Lexical replacement token / exact     0.9124 / 0.5466
Non-target preservation               1.0000
```

Every TreeHeap relation family reached sequence exact `1.0`, including the two
non-copy families `number_shift` and `role_swap`. This supports a narrow
result: `DIFF -> TRANSPORT -> APPLY` can learn finite-world analogy mappings
that lexical target substitution cannot solve.

The core World TreeHeap claim failed causal intervention:

```text
full W token accuracy       1.0000
zero W token accuracy       0.9984
address-shuffled W          0.9867
wrong B relation            0.8284
final route entropy         about 2.4e-5
```

The route collapsed to effectively one background node. Zeroing W changed
token accuracy by only `0.0016`, far below the registered `0.10` gate. The
network learned analogy through the direct encoded A/B/C and relation/apply
MLPs while treating W mostly as a constant bias. Wrong B caused a measurable
`0.1716` token drop, so the relation example mattered, but not by the required
`0.30` margin.

The flat Transformer used 73,574 parameters versus TreeHeap's 238,314 and also
reached exact `1.0`. There is no TreeHeap efficiency or quality advantage in
this toy.

Gate result:

```text
P1 full and non-copy exact   PASS
P2 lexical margin            PASS
P3 World causal dependence   FAIL
P4 wrong-relation margin     FAIL
P5 selective preservation    PASS
P6 flat baseline reported    PASS
```

## Decision

`S3-WORLD-ANALOGY-C01` is **not supported as a World TreeHeap claim**. Retain
only the narrower positive subclaim that the TreeHeap DIFF/TRANSPORT/APPLY path
can represent finite analogy operators. The World TreeHeap currently has no
meaningful causal role, and the smaller flat baseline matches the result.

The next revision must remove the direct bypass by making the transported
operator read a finite addressed code from W, then compare against a matched
flat memory. Merely regularizing route entropy would force traffic without
proving useful world knowledge.

Evidence: `evidence/s3_world_treeheap_analogy_smoke/summary.json`.
