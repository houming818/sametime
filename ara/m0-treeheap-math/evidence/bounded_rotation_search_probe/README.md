# Bounded Rotation Search Evidence

Claim: `M0-ROT-C01`
Predict: `P-ROT01`

## Result

```text
pilot_pass                         = True
tested_queries                     = 52898
deterministic_exact                = 1.000000
learned_ood_min_exact              = 1.000000
inverse_exact                      = 1.000000
deepest_logical_candidates         = 2031616
deepest_treeheap_mean_comparisons  = 20.168
deepest_unordered_mean_comparisons = 1014613.868
deepest_explicit/lazy_storage      = 32247.873x
deepest_broken_one_path_exact      = 0.036000
over_budget_status                 = BUDGET_EXHAUSTED
```

## Interpretation

The probe tests compact reuse of a regular order-isomorphic orbit. Search work
is linear in rotation depth and logarithmic in logical candidate count. The
explicit sorted baseline has the same asymptotic comparison count but stores
every payload. Destroying local order forces equality scanning or breaks the
one-path kernel.

This is not evidence for semantic reasoning, arbitrary exponential search, or
cryptographic key recovery.
