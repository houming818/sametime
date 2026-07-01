# S1 Mirror / Chiral Kernel Flip Probe

Claim: `S1-KERNEL-MIRROR-C01`
Predict: `P-S1-KERNEL40`
Host: `local`

## Result

pilot_pass: `True`

```text
theta = [0.5, 1.25, -0.75]
theta_mirror = [0.5, -0.75, 1.25]
deductive_law = P_m K_theta(H) = K_{P_lr theta}(P_m H)
deductive_test_max_flipped_error = 8.88178e-16
deductive_test_mean_unflipped_error = 6.27438
learned_theta = [0.4999999999999999, -0.7500000000000002, 1.2499999999999996]
learned_theta_mirror_l2_error = 5.08768e-16
left_slot_learns_original_right_error = 2.22045e-16
right_slot_learns_original_left_error = 4.44089e-16
learned_test_mse = 8.31162e-31
learned_ood_mse = 9.82215e-30
```

## Meaning

This proof tests both a deductive mirror identity and an inductive learned
mirrored kernel. The retired word `conjugate` is intentionally avoided here:
the operation is a left/right address permutation plus a left/right kernel-slot
permutation.

The learned part is specifically a slot-assignment proof:

```text
root stays root
left learns the original right coefficient
right learns the original left coefficient
```

It is not a rotation-angle proof or a full 3D fold proof.

## Boundary

It does not prove language understanding, WMT translation, or arbitrary group
equivariance.
