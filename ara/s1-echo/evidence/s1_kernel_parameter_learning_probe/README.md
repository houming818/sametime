# S1 Kernel Parameter Learning Probe

Claim: `S1-KERNEL-LEARN-C01`
Predict: `P-S1-KERNEL39`
Host: `io.grepcode.cn`

## Result

pilot_pass: `True`

```text
learned_theta = [0.9999999999999998, 0.9999999999999998, 1.0000000000000004]
theta_l2_error = 5.43896e-16
theta_delta_l2 = 1.15777
treeheap_test_mse = 8.78177e-31
treeheap_ood_mse = 8.92978e-30
wrong_address_test_mse = 5.92848
flat_global_test_mse = 3.5601
```

## Meaning

This proof updates `Theta`, the parameter TreeHeap / shared local kernel. It
does not update only `H`, the per-sample heap state. That is the key distinction
from SPR-038.

## Boundary

This does not prove language understanding, WMT translation, or superiority
over every larger flat model.
