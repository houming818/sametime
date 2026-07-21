# Bounded Rotation Search

Status: preregistered design
Claim: `M0-ROT-C01`
Predict: `P-ROT01`
Date: 2026-07-21

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
