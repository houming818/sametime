# Fixed-Capacity Subheap Rotation

Status: design
Claim: `M0-ROT-C02`
Predict: `P-ROT02`
Date: 2026-07-21

## Architecture Decision

Rotation must not create a larger TreeHeap. The runtime receives a fixed node
pool of capacity `C` and must stay inside it:

```text
R_(S, phi): H_C -> H_C
```

`S` is an existing subheap and `phi` is one member of a finite, registered set
of orientations. `R` may permute payload addresses or rewrite edges inside
`S`, but it must satisfy:

```text
node_pool(R(H)) = node_pool(H)
capacity(R(H))  = capacity(H) = C
R^-1(R(H))      = H
R(H)[outside S] = H[outside S]
```

No recursive call may allocate another TreeHeap. A fixed scratch buffer, if an
implementation needs one, must be counted in the startup memory budget and
reused for every step.

## Kernel Use

Rotation is a change of structural viewpoint, not candidate generation. A
shared local kernel works in a canonical orientation:

```text
local_view = R_(S, phi)(H[S])
local_update = K_theta(query, local_view)
H'[S] = R_(S, phi)^-1(local_update)
```

The same `theta` is reused at different addresses and orientations. Mirror is
the two-orientation special case. A genuine left/right tree rotation may also
rewrite parent-child pointers while conserving the fixed node pool.

## Whole-Tree Enhancement

The local kernel expands its receptive field by repeated convolution over the
existing tree:

```text
H_(t+1) = ApplyAllSubheaps(H_t, K_theta, R, gates)
```

After `t` rounds, information may travel roughly `t` structural hops. This is
growth of accessible context, not growth of memory. Runtime is bounded by:

```text
space = O(C)
time  = O(T_max * C * local_kernel_cost)
```

Both `C` and `T_max` are hard limits. If the answer is unavailable at the
limit, return `UNRESOLVED`; do not allocate a larger state.

## Predict P-ROT02

On a fixed 127-node TreeHeap:

1. every registered rotation and inverse is exact;
2. node count, capacity, and resident state bytes remain constant;
3. a kernel trained on canonical subheaps transfers to unseen addresses and
   registered orientations after rotate-to-canonical/read/rotate-back;
4. removing rotation alignment damages that transfer;
5. repeated local rounds enlarge measured receptive field toward the root
   without allocating nodes;
6. requests beyond `T_max` return `BUDGET_EXHAUSTED` or `UNRESOLVED`.

## Planned Controls

```text
shared kernel + fixed rotation
shared kernel without rotation
address-specific kernel
matched flat MLP
random non-structural permutation
```

The proof must report exact node/edge conservation, peak allocated bytes,
orientation/address OOD accuracy, and accuracy by iteration count.

## Falsification

Reject the claim if rotation changes capacity, inverse recovery fails, the
shared kernel does not transfer across registered views, a random permutation
works equally well, or whole-tree propagation requires dynamically allocated
nodes.

## Boundary

This design does not say that rotation alone creates semantic knowledge. It is
an inductive-bias claim: a useful local rule may be reused under structural
symmetry while the system obeys a fixed physical memory budget.
