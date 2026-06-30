# S1 Conjugate Kernel Symmetry

Status: supported pilot
Created: 2026-07-01
Updated: 2026-07-01
Owner: nio / Houming818
Review Engineer: Codex

## Problem

SPR-039 proved that a parameter TreeHeap `Theta` can learn a local convolution
kernel. The tested kernel was a simple local sum. Houming818's next question is
whether the TreeHeap kernel calculus also supports conjugate symmetric flipping:

```text
mirror tree + flipped kernel = mirrored result
```

This is the TreeHeap analogue of flipping a convolution kernel under a mirror
transform.

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

Local kernel:

```text
theta = [theta_root, theta_left, theta_right]
```

Conjugate / mirrored kernel:

```text
theta_mirror = [theta_root, theta_right, theta_left]
```

Convolution over internal nodes:

```text
K_theta(H)[i] =
  theta_root  * H[i]
+ theta_left  * H[left(i)]
+ theta_right * H[right(i)]
```

## Claim

`S1-KERNEL-CONJ-C01`:

TreeHeap local convolution is equivariant under mirror conjugation:

```text
M(K_theta(H)) = K_conj(theta)(M(H))
```

where:

```text
conj([root,left,right]) = [root,right,left]
```

Status after execution: `supported pilot`.

## Predict

For random heaps and asymmetric kernels such as:

```text
theta = [0.5, 1.25, -0.75]
```

the equality error should be near machine zero when the kernel is flipped, and
large when the original unflipped kernel is incorrectly reused on the mirrored
heap.

Additionally, if a kernel is learned from mirrored data, it should recover:

```text
theta_mirror ~= [0.5, -0.75, 1.25]
```

## Executed Result

Script:

```text
ara/s1-echo/src/s1_conjugate_kernel_symmetry_probe.py
```

Evidence:

```text
ara/s1-echo/evidence/s1_conjugate_kernel_symmetry_probe/
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
theta_conj_l2_error = 5.44e-16
learned_test_mse = 1.01e-30
learned_ood_mse = 9.76e-30
```

Decision:

```text
S1-KERNEL-CONJ-C01 -> supported pilot
```

## Falsification

Reject or downgrade if:

1. Mirror equivariance error is not near zero under the flipped kernel.
2. The unflipped kernel performs equally well on mirrored heaps.
3. A learned mirrored kernel does not recover `[root,right,left]`.
4. The proof only works for symmetric kernels where left and right weights are equal.

## Boundary

This proof does not prove language understanding, WMT translation, or learned
semantic conjugacy. It only proves a local algebraic property of TreeHeap
convolution kernels.
