# Primitive Plus Probe Evidence

This is synthetic M0 evidence for `P-MATH02`.

The experiment treats TreeHeap as an addressable array:

```text
arr[0] = root
next(i) = (i + 1) mod base
plus(H, primitive) writes primitive to next address
```

## Verdict

`pilot_pass = True`

## Metrics

| Metric | Value |
|---|---:|
| `closure_ok` | `True` |
| `root_ok` | `True` |
| `successor_ok` | `True` |
| `info_gain_pre_base` | `True` |
| `info_saturates_after_base` | `True` |
| `mod_fold_targets` | `[0, 1]` |
| `mod_fold_ok` | `True` |
| `overwritten_root_ok` | `True` |
| `kernel_hit_at_1` | `1.0` |
| `kernel_top_score` | `1.0` |
| `kernel_after_wrap_top_score` | `0.6731517435938855` |
| `wrap_breaks_old_kernel` | `True` |
| `address_laws_ok` | `True` |
| `summary_consistency_ok` | `True` |
| `summary_min_delta` | `0.4124184600459178` |
| `cycle_address_error` | `0` |

## Gates

| Gate | Pass |
|---|---:|
| `closure` | `True` |
| `root_reference` | `True` |
| `successor` | `True` |
| `info_gain_pre_base` | `True` |
| `info_saturation_after_base` | `True` |
| `mod_fold` | `True` |
| `overwritten_root` | `True` |
| `cyclic_kernel` | `True` |
| `wrap_breaks_old_kernel` | `True` |
| `address_laws` | `True` |
| `summary_consistency` | `True` |
| `summary_moves` | `True` |

## Interpretation

The toy supports the narrow claim that an addressable TreeHeap can use
`plus` as a successor operation with information gain before the base
is full, and as a modular overwrite/fold operation after the base is full.

This is not language evidence. It only validates the next mathematical
toolbox layer needed before TreeHeap-object echo.
