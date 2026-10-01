# A21 Frozen-Encoder Multiscale Flat READ

Date: 2026-10-01

Status: formal complete; both registered Claims supported

## Question

A20 showed that a recursive conditional-residual codec can carry target-specific
context information through a root bottleneck. It did not establish that decoding
only from the root is the best way to expose the already-computed multiresolution
states. This experiment asks whether retaining the identical A20 encoder and exposing
its level maps directly to READ is more useful than recursively unfolding only the
root.

This is a READ experiment. It does not replace, retrain, or reinterpret A20 FOLD.

## Frozen Material

- A20 `evidence_fold`, dimension 16 checkpoint;
- byte-identical A14 one-million-line count artifact;
- A20 fit/dev/test split seed `20260924`;
- all leaf, parent, ancestor, and root states are computed once and detached;
- every arm receives the same frozen state tensors and global output prior.

## Arms

All arms contain the same parameter allocation and start from the same initialized
weights. Parameters unused by an arm remain allocated but receive no gradient.

1. `root_unfold`: start from the frozen root and recursively apply one shared binary
   READ transform for ten levels;
2. `leaf_only`: read the frozen leaf map directly;
3. `flat_coarse`: align every non-leaf level to leaf coordinates, then learn a
   content-conditioned mixture over levels;
4. `flat_all`: the same flat mixture with the true frozen leaf level included.

For a node state at level `j > 0`, one shared READ transform predicts its immediate
left and right child views. Those views are repeated only over their own descendant
coordinates. Therefore the flat arms preserve dyadic address alignment; they do not
perform all-to-all attention and do not create cross-position information.

For output coordinate `c`, the flat state is

```text
z(c) = sum_j alpha_j(c) v_j(c)
alpha_j(c) = softmax_j(s(v_j(c)) + b_j)
```

The decoder emits a residual over the same fixed A20 global prior. Its output layer is
zero initialized, so all arms must have exactly the same step-zero logits and NLL.

## Training Contract

- smoke: 20 complete epochs;
- formal: 200 complete epochs only after smoke integrity gates pass;
- identical row split, batch order, optimizer, seed, updates, and output prior;
- frozen encoder parameters must not change;
- quality never stops an arm early;
- stop only for OOM, CUDA failure, NaN/Inf, corrupt evidence, or reload failure.

## Registered Claims

### Flat exposure claim

`S1-MULTISCALE-FLAT-READ-A21-C01`: under the frozen A20 encoder, `flat_all` obtains
sealed-test NLL at least `0.005` below `root_unfold` after the full formal budget.

### Coarse contribution claim

`S1-MULTISCALE-FLAT-READ-A21-C02`: disabling every coarse level from trained
`flat_all` worsens sealed-test NLL by at least `0.005`, and the mean READ probability
assigned to non-leaf levels is at least `0.10`.

### Coarse sufficiency observation

`flat_coarse` beating the no-input global prior is recorded as a separate observation.
It is not required for C01 or C02 because the experiment is about complementary
multiscale access, not root-only sufficiency.

## Integrity Gates

1. all arms start from exactly the global-prior logits;
2. all arms use identical frozen state hashes;
3. all arms complete the registered update budget with finite values;
4. every checkpoint reloads with exact output equality;
5. the A20 source checkpoint reloads and reproduces its frozen native output.

## Boundaries

A pass supports only multiscale READ access for this conditional context-field codec.
It does not establish long-range language composition, FPN superiority, Butterfly
redundancy, translation quality, semantic hierarchy, or consumer-hardware efficiency.
Those questions require a later real-sequence matched experiment.

## Formal Result

The smoke ran as io task `645`; all integrity gates passed. The preregistered
200-epoch formal therefore ran as io task `646`. All four arms completed 1,600
updates with finite values, exact step-zero prior equality, an unchanged frozen A20
encoder, and exact checkpoint reloads.

| READ arm | effective trainable parameters | Test NLL | Test PPL | row top-1 |
|---|---:|---:|---:|---:|
| `root_unfold` | 1,121 | 5.477452 | 239.236 | 0.6016 |
| `leaf_only` | 17 | 5.135705 | 169.984 | 0.7715 |
| `flat_coarse` | 1,148 | 5.119611 | 167.270 | 0.8340 |
| `flat_all` | 1,148 | **5.118257** | **167.044** | **0.8496** |

`flat_all` beat `root_unfold` by `0.359195` Test NLL, far above the registered
`0.005` gate. Disabling every coarse level from the trained `flat_all` checkpoint
worsened Test NLL by `0.115939`; non-leaf levels received `0.418750` mean READ
probability. C01 and C02 are therefore supported in this frozen conditional-field
codec.

The learned READ mass was concentrated at leaf level 0 (`0.581250`) and pair level 1
(`0.275864`). Levels 2--10 jointly received about `0.142886`. The result therefore
supports complementary near-leaf multiscale access, not equal usefulness of every
depth and not a root-semantic hierarchy.

The A20 native root codec obtained Test NLL `5.467703`; the newly trained
`root_unfold` reproduced it within `0.009749`. In contrast, `flat_all` nearly reached
the A19 direct conditional-field reference (`5.116258`). This locates the dominant
loss at the root-only recursive READ/FOLD bottleneck for this task. It does not show
that Flat created new information: leaf states were constructed from fit-split
conditional fields for the same 512 target rows, and the sealed test contains new
counts for those rows rather than unseen language sequences.

Allocated parameter count was exactly 1,148 in every arm. Effective trainable
parameter count differed because parameters inaccessible to `leaf_only` and
`root_unfold` received no gradient. The Flat versus root gap is much larger than the
27 effectively active parameter difference between those two arms, but a later
real-sequence experiment should use strict effective-capacity matching.

## Evidence

```text
ara/s1-echo/evidence/s1_multiscale_flat_read_a21/
  smoke_seed20261002/
  formal_seed20261002/
```
