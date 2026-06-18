# TreeHeap Math Probe Evidence

This is synthetic M0 evidence for `P-MATH01`. It does not use language, WMT, or checkpoints.

## Verdict

`pilot_pass = True`

## Metrics

| Metric | Value |
|---|---:|
| `closure_ok` | `True` |
| `noncomm_distance` | `1.1930861484999091` |
| `noncomm_margin` | `0.7117272788711737` |
| `transpose_inverse_shape_ok` | `True` |
| `transpose_inverse_error` | `0.0` |
| `compose_decompose_error` | `0.0` |
| `projection_top1_preserved` | `True` |
| `projection_order_agreement` | `0.8333333333333334` |
| `subheap_hit_at_1` | `1.0` |
| `subheap_hit_at_3` | `1.0` |
| `role_swap_margin` | `0.9091461256178817` |
| `prob_mass_error` | `0.0` |

## Gates

| Gate | Pass |
|---|---:|
| `closure` | `True` |
| `noncomm` | `True` |
| `transpose_inverse` | `True` |
| `compose_decompose` | `True` |
| `projection` | `True` |
| `subheap` | `True` |
| `probability` | `True` |

## Interpretation

The pilot supports using TreeHeap as a synthetic algebra object before moving to echo.

A first implementation failed `transpose_inverse_error` because the object only stored
the collapsed whole-vector `v`. Adding `head_v` made the inverse exact in synthetic mode.
This is a useful design constraint: TreeHeap needs a root/head reference, not only a
single collapsed vector.

The role-swapped kernel is evaluated by score margin. In a heap that does not contain
the swapped structure, the best available candidate may still be returned as top-1, but
its score should be much lower than the gold kernel score.

The next experiment should replace exact synthetic inverse with approximate learned inverse,
then test whether TreeHeap-object echo preserves these algebraic invariants.
