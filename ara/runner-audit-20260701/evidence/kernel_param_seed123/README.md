# S1 Kernel Parameter Learning Probe

Claim: `S1-KERNEL-LEARN-C01`
Predict: `P-S1-KERNEL39`
Host: `local`

## Result

pilot_pass: `True`

```text
learned_theta = [1.0000000000000004, 1.0000000000000004, 0.9999999999999998]
theta_l2_error = 6.66134e-16
theta_delta_l2 = 1.79468
treeheap_test_mse = 1.31769e-30
treeheap_ood_mse = 1.41524e-29
wrong_address_test_mse = 6.10314
flat_global_test_mse = 4.33342
```

## Meaning

This proof updates `Theta`, the parameter TreeHeap / shared local kernel. It
does not update only `H`, the per-sample heap state. That is the key distinction
from SPR-038.

## Boundary

This does not prove language understanding, WMT translation, or superiority
over every larger flat model.
