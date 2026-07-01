# S1 Mirror / Chiral Kernel Flip

Status: supported pilot
Created: 2026-07-01
Updated: 2026-07-01
Owner: nio / Houming818
Review Engineer: Codex

## Terminology

Earlier drafts used the word `conjugate`. That wording is retired.

This claim is about **mirror** / **chiral flip**:

```text
left and right are swapped in the heap geometry,
so left and right must also be swapped in the local kernel slots.
```

It is not complex conjugation, and it is not a general group-theory conjugacy
claim. It is a concrete algebraic permutation proof.

## Problem

SPR-039 proved that a parameter TreeHeap `Theta` can learn a local subheap
convolution kernel. Houming818 then asked a sharper question:

```text
Can a geometric operation on TreeHeap, such as left/right mirror,
be implemented as an algebraic operation on heap addresses and kernel slots?
```

This matters because TreeHeap should not only store values. It should provide a
structured operator calculus: convolution, write, read, mirror, path move, and
eventually more complex geometry-aware operations.

After review, this SPR-040 claim is intentionally scoped tighter than a
rotation or 3D-fold claim:

```text
We do not claim that the kernel learns a rotation angle.
We do not claim that full 3D fold is solved.
We only claim that mirror turns root/left/right into structural directions,
and scalar loss can learn the mirrored slot assignment.
```

## Definitions

Use complete binary heap indices:

```text
        1
      /   \
     2     3
    / \   / \
   4   5 6   7
```

Physical mirror `M` swaps left and right recursively:

```text
M(1)=1
M(2)=3, M(3)=2
M(4)=7, M(5)=6, M(6)=5, M(7)=4
```

With zero-based array indices this is:

```text
P_m = (0, 2, 1, 6, 5, 4, 3)
```

So:

```text
H = [1,2,3,4,5,6,7]
P_m H = [1,3,2,7,6,5,4]
```

Local kernel:

```text
theta = [theta_root, theta_left, theta_right]
```

The mirror operation on kernel slots is:

```text
P_lr theta = [theta_root, theta_right, theta_left]
```

Equivalently:

```text
P_lr =
[[1,0,0],
 [0,0,1],
 [0,1,0]]
```

TreeHeap local convolution over internal nodes:

```text
K_theta(H)[i] =
  theta_root  * H[i]
+ theta_left  * H[left(i)]
+ theta_right * H[right(i)]
```

## Claim

`S1-KERNEL-MIRROR-C01`:

TreeHeap local convolution is equivariant under mirror / chiral flip:

```text
P_m K_theta(H) = K_{P_lr theta}(P_m H)
```

In words:

```text
first convolve, then mirror the tree
=
first mirror the tree, then convolve with the mirrored kernel
```

Additional narrow learning claim:

```text
When the target data is mirrored, gradient descent can recover the mirrored
kernel assignment:

root -> root
left -> original right
right -> original left
```

This is the current "structure-aware" result. The kernel slots are not anonymous
scalar positions; they are local structural directions.

## Predict

For random heaps and an asymmetric kernel:

```text
theta = [0.5, 1.25, -0.75]
P_lr theta = [0.5, -0.75, 1.25]
```

the following should happen:

1. `P_m K_theta(H)` and `K_{P_lr theta}(P_m H)` differ only by floating point
   noise.
2. If we mirror the heap but wrongly keep the original unflipped kernel, error
   should become large.
3. If we train a local kernel on mirrored data, gradient descent should recover
   `[0.5, -0.75, 1.25]`.
4. The learned assignment should specifically show:

```text
learned_root  ~= original_root
learned_left  ~= original_right
learned_right ~= original_left
```

## Proof Design

This proof has two parts.

### Part A: Deductive Operator Check

Directly compute both sides:

```text
left  = P_m K_theta(H)
right = K_{P_lr theta}(P_m H)
```

Measure:

```text
max_abs(left - right)
```

This checks the algebraic law itself.

### Part B: Inductive Learning Check

Create mirrored training pairs:

```text
input  = P_m H
target = internal_nodes(P_m K_theta(H))
```

Train a three-slot parameter TreeHeap:

```text
Theta = [theta_root, theta_left, theta_right]
```

with scalar MSE loss. If the data really implies the mirror rule, learned
`Theta` should converge to:

```text
P_lr theta = [0.5, -0.75, 1.25]
```

This checks that loss can recover the mirror assignment of the structural
slots. It does not require or imply a continuous rotation parameter.

## Executed Result

Script:

```text
ara/s1-echo/src/s1_mirror_kernel_symmetry_probe.py
```

Evidence:

```text
ara/s1-echo/evidence/s1_mirror_kernel_symmetry_probe/
```

Host:

```text
io.grepcode.cn
```

Summary:

```text
pilot_pass = true
test_max_flipped_error = 8.88e-16
ood_max_flipped_error = 3.55e-15
test_mean_unflipped_error = 6.4372
learned_theta = [0.5000000000000002, -0.7499999999999998, 1.2499999999999996]
theta_mirror_l2_error = 5.44e-16
left_slot_learns_original_right_error = 2.22e-16
right_slot_learns_original_left_error = 4.44e-16
learned_test_mse = 1.01e-30
learned_ood_mse = 9.76e-30
```

Decision:

```text
S1-KERNEL-MIRROR-C01 -> supported pilot
```

## Falsification

Reject or downgrade if:

1. Mirror equivariance error is not near zero under `P_lr theta`.
2. The unflipped kernel performs equally well on mirrored heaps.
3. A learned mirrored kernel does not recover `[root,right,left]`.
4. The proof only works for symmetric kernels where left and right weights are equal.
5. Deeper/vector kernels break the same law without a principled reason.

## Boundary

This proof does not prove language understanding, WMT translation, or learned
semantic mirror in real corpora. It only proves a local algebraic property of
TreeHeap convolution kernels:

```text
geometric mirror can be implemented as algebraic permutation.
scalar loss can learn the mirrored structural slot assignment.
```

It also does not prove:

```text
learned rotation angle
full 3D fold
latent plane projection weights
```
