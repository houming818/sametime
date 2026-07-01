# S1 Mirror / Chiral Kernel Flip Probe

Claim: `S1-KERNEL-MIRROR-C01`
Predict: `P-S1-KERNEL40`
Host: `io.grepcode.cn`

## Result

pilot_pass: `True`

```text
theta = [0.5, 1.25, -0.75]
theta_mirror = [0.5, -0.75, 1.25]
deductive_law = P_m K_theta(H) = K_{P_lr theta}(P_m H)
deductive_test_max_flipped_error = 8.88178e-16
deductive_test_mean_unflipped_error = 6.43722
learned_theta = [0.5000000000000002, -0.7499999999999998, 1.2499999999999996]
learned_theta_mirror_l2_error = 5.43896e-16
learned_test_mse = 1.00992e-30
learned_ood_mse = 9.76166e-30
```

## Meaning

This proof tests both a deductive mirror identity and an inductive learned
mirrored kernel. The retired word `conjugate` is intentionally avoided here:
the operation is a left/right address permutation plus a left/right kernel-slot
permutation.

## Boundary

It does not prove language understanding, WMT translation, or arbitrary group
equivariance.
