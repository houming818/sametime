# S1 Kernel Parameter Learning Probe

Claim: `S1-KERNEL-LEARN-C01`
Predict: `P-S1-KERNEL39`
Host: `local`

## Result

pilot_pass: `True`

```text
learned_theta = [1.0, 0.9999999999999998, 0.9999999999999998]
theta_l2_error = 3.14018e-16
theta_delta_l2 = 2.16488
treeheap_test_mse = 2.98963e-31
treeheap_ood_mse = 3.39411e-30
wrong_address_test_mse = 6.48336
flat_global_test_mse = 4.12069
```

## Meaning

This proof updates `Theta`, the parameter TreeHeap / shared local kernel. It
does not update only `H`, the per-sample heap state. That is the key distinction
from SPR-038.

## Boundary

This does not prove language understanding, WMT translation, or superiority
over every larger flat model.
