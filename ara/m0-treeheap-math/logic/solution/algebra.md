# Minimal TreeHeap Algebra

This is the first pure-math toolbox draft.

## Object

```text
H = (name, v, head_v, slot, q, children)
```

The vector `v` is a structural signature. The `head_v` field preserves the
local root/head vector needed for exact synthetic recomposition. The `children`
field keeps exact synthetic structure so exact inverse tests are possible in M0.

## Operators

### compose

```text
compose(root, [left, right]) -> H
```

The pilot vector uses distinct left and right structural bases:

```text
v(H) = normalize(root.v + L @ left.v + R @ right.v)
```

Because `L != R`, composition is non-commutative.

The pilot stores `head_v` separately from `v(H)`. The first implementation did
not do this, and `transpose(transpose(H))` failed because the composed vector was
accidentally reused as the root vector. That failure is kept as a design lesson:
TreeHeap objects need a root reference, not only a collapsed whole-vector.

### decompose

```text
decompose(H) -> children(H)
```

M0 uses exact stored children. Later stages can replace this with learned or
approximate decompose.

### transpose

```text
transpose(H) -> H'
```

For binary heaps, transpose swaps left and right recursively.

### inverse_transpose

```text
inverse_transpose(transpose(H)) -> H
```

In the pilot, transpose is its own inverse.

### project / unproject

```text
z = P @ v
v_hat = pinv(P) @ z
```

This is not expected to perfectly reconstruct high-dimensional vectors when
the projection is lossy. The gate is whether relative energy ordering survives.

### match_subheap

```text
match_subheap(H, K) -> ProbabilityContainer[SubHeap]
```

Every subtree in `H` is scored against kernel `K`. Scores are converted into a
softmax probability container.

## Status

Design plus first synthetic probe.

## Open Extension: Primitive and Plus

The current M0 toolbox defines composition and matching on synthetic TreeHeap
objects. The next lower-level question is whether TreeHeap order can be
generated internally.

Integer order is generated from:

```text
primitive: 1
operator: plus / successor
order: n -> n + 1
```

TreeHeap should test the analogous structure:

```text
x_0 = origin
x_{n+1} = plus(x_n, p)
x_{n+base} ~= x_n
```

If this holds, convolution over TreeHeap addresses can be defined over the
generated cyclic orbit rather than over an externally assigned traversal index.

This is tracked as `P-MATH02`.
