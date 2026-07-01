# Kernel Convolution Ops Probe Evidence

Verdict: `pilot_pass = True`

## What Was Tested

This deterministic toy tests the claim that TreeHeap operations can be defined
as kernel convolutions over the whole tree state:

```text
search:    kernel scans every local subheap and emits a score map
plus:      the same score map becomes a soft write/update field
conjugate: a mirrored kernel recovers the mirrored score map
```

## Key Metrics

| Check | Result |
|---|---:|
| search hit@1 | True |
| plus write hit@1 | True |
| plus target update error | 0.000000 |
| max non-target update norm | 0.750000 |
| max disjoint non-target update norm | 0.000000 |
| overlapping non-target patches | 3 |
| raw mirror hit@1 | False |
| conjugate mirror hit@1 | True |
| score-map equiv max error | 0.000000e+00 |

## Interpretation

The positive result supports `M0-SOFT-C08` as a toy operator-semantics pilot:
search, plus/write, and conjugate mirror can be expressed as TreeHeap kernel
convolutions that produce full-tree maps.

The non-target update norm includes overlapping TreeHeap patches. A parent patch
can observe a child update even when it was not selected as the write target.
For localization, use `max_disjoint_nontarget_update_norm`.

This is not a learned-kernel proof and not language evidence. It is a clean
operator-level proof target for the next learned C05/C06 experiments.
