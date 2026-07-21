# Bounded Rotation Search

Status: supported pilot
Claim: `M0-ROT-C01`
Predict: `P-ROT01`
Date: 2026-07-21

Architecture disposition: retained as mathematical boundary evidence; rejected
as a runtime growth strategy. The active direction is fixed-capacity
`M0-ROT-C02` in `logic/fixed_capacity_subheap_rotation.md`.

## Question

Can one ordered TreeHeap slice be expanded by a reliable rotation operator and
`CAT`, while preserving exact search with a fixed-size kernel?

The construction is:

```text
H_(n+1) = CAT(H_n, R_n(H_n))
```

`R_n` is not an arbitrary copy. It must be invertible, order preserving inside
the rotated slice, separated from the old slice by a known interval boundary,
and represented by a descriptor rather than a materialized copy.

The query kernel compares a target with the right-slice boundary. When it
chooses the rotated branch, it applies `R_n^-1` and repeats on `H_n`. After the
rotation bits are removed, the same binary-search kernel reads the base slice.

## Complexity Statement

After `n` recursive rotations, the logical candidate count is:

```text
M_n = |H_0| * 2^n
```

The expected retrieval path is:

```text
n rotation decisions + O(log |H_0|) local decisions
```

It is linear in construction depth `n`, and therefore logarithmic in logical
candidate count `M_n`. It is not linear in `M_n`, and it does not enumerate or
solve arbitrary candidates for free.

Explicit copies require `O(M_n)` payload storage. The lazy orbit stores the
base slice plus `n` rotation descriptors: `O(|H_0| + n)`.

## Learned Kernel

The inductive part uses one two-parameter logistic comparison kernel:

```text
x = (query - right_min) / current_range
p(right | x) = sigmoid(w*x + b)
```

It is trained only on shallow rotations and reused unchanged at unseen deeper
rotations. The interval is structural metadata; no precomputed left/right
answer flag is provided. This tests parameter sharing, not semantic
intelligence.

## Controls

1. `explicit_sorted`: materialize all rotated slices; exact but exponentially
   larger in construction depth.
2. `unordered_scan`: destroy local order and scan for equality; exact but
   linear in candidate count.
3. `broken_rotation`: permute payload positions inside rotated slices while
   keeping the old one-path comparator. It should fail because the transform
   no longer preserves searchable order.
4. `budget_exceeded`: request one rotation beyond the hard maximum. It must
   return `BUDGET_EXHAUSTED`, allocate no new orbit, and produce no answer.

## Predictions

For base size 31 and rotation depths through 16:

```text
P1 deterministic exact retrieval       = 1.0
P2 learned-kernel OOD exact retrieval   >= 0.999
P3 inverse/path equivariance            = 1.0
P4 max query steps                      <= n + ceil(log2(31)) + 1
P5 lazy/explicit storage ratio at n=16  >= 1000
P6 broken-rotation one-path accuracy    <= 0.10
P7 over-budget status                   = BUDGET_EXHAUSTED
```

## Falsification

Reject the claim if any exact algebraic identity fails, if the shared learned
kernel cannot transfer beyond trained depths, if lazy storage follows explicit
copy growth, or if destroying order does not damage the same one-path search.

## Boundaries

This experiment does not claim semantic learning, generation of independent
information, search over arbitrary unstructured spaces, cryptographic key
recovery, or superiority over a fully materialized sorted array. The narrow
claim is compact reuse of a regular, order-isomorphic orbit.

## Result

Executed on `io.grepcode.cn` with the registered seed and configuration.

```text
tested queries                       = 52,898
deterministic exact                  = 1.000000
learned OOD minimum exact            = 1.000000
inverse exact                        = 1.000000
depth-16 logical candidates          = 2,031,616
depth-16 mean TreeHeap comparisons   = 20.1682
depth-16 mean unordered scan         = 1,014,613.8684
depth-16 explicit/lazy storage       = 32,247.873x
depth-16 broken one-path exact       = 0.036000
over-budget request                  = BUDGET_EXHAUSTED
```

All registered gates passed. This supports `M0-ROT-C01` as a synthetic pilot.
The evidence is under `evidence/bounded_rotation_search_probe/`.

The result does not show that TreeHeap discovers the transform. The exact
rotation law and interval metadata were supplied. The learned component only
learned the shared left/right comparison and transferred it from depths 1--4
to depths 5--16. A fully materialized sorted array has the same asymptotic
comparison count; TreeHeap's measured benefit here is lazy structural reuse.

The route feature is `query - right_min`, not a precomputed left/right answer
flag. Even so, `right_min` is exact structural metadata maintained by the
construction. This is therefore not a content-aware or semantic route proof.
The next inductive gate is to learn the rotation descriptor and interval
summary from observed slice pairs, then rerun unseen-depth search without
handing those summaries to the model.
