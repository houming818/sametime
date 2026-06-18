# M0 Predict Registry

## P-MATH01: Minimal TreeHeap algebra

### Predict

A minimal TreeHeap algebra can satisfy closure, non-commutativity, approximate
inverse behavior, projection stability, local subheap matching, and probability
normalization on synthetic tree objects.

### Why This Comes Before Echo

Echo should test whether the toolbox preserves information. It should not be
the first test of whether the toolbox exists.

The intended route is:

```text
M0 pure math
-> M1 toolbox closure
-> M2 TreeHeap echo
-> M3 structure invariants
-> S2 translation
```

### Evidence Gates

E-MATH01 closure:

```text
compose(H1, H2) -> legal TreeHeap object
```

E-MATH02 non-commutativity:

```text
compose(root, left=A, right=B) != compose(root, left=B, right=A)
```

E-MATH03 inverse / reconstruction:

```text
inverse_transpose(transpose(H)) ~= H
compose(decompose(H)) ~= H
```

E-MATH04 projection stability:

```text
project(H) preserves nearest/energy ordering better than random projection baseline
```

E-MATH05 subheap kernel matching:

```text
match_subheap(H, K) ranks the gold subheap in top-k
```

E-MATH06 probability container:

```text
match probabilities are normalized and do not collapse to NaN/Inf
```

### Pass Criteria

Pilot pass:

```text
closure_ok = true
noncomm_margin > 0.05
transpose_inverse_error < 1e-9
compose_decompose_error < 1e-9
projection_top1_preserved = true
subheap_hit_at_1 >= 0.80
prob_mass_error < 1e-9
```

### Fail Criteria

P-MATH01 fails if:

```text
left/right order is not distinguishable
or inverse operations cannot recover structure in synthetic exact mode
or projection destroys all energy ordering
or subheap matching only returns root-level trivial matches
or probability containers are numerically unstable
```

