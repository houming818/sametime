# Structural C05 Probe Evidence

Verdict: `pilot_pass = True`

## What Was Tested

A local 3-node subheap pattern was injected into synthetic TreeHeaps.
Models trained on shallow trees and were tested on deeper, unseen target addresses.

## Result Table

| Variant | Train acc | Test acc | Hit@3 | Mean rank | Feature dim |
|---|---:|---:|---:|---:|---:|
| flat_address | 0.397 | 0.000 | 0.000 | 17.16 | 7 |
| path_only | 0.397 | 0.000 | 0.000 | 35.13 | 9 |
| subheap_kernel | 1.000 | 1.000 | 1.000 | 1.00 | 13 |
| path_subheap_kernel | 1.000 | 1.000 | 1.000 | 1.00 | 22 |

## Interpretation

If `subheap_kernel` and `path_subheap_kernel` succeed while `flat_address`
and `path_only` fail, the relocation signal is carried by local subheap
structure rather than absolute memory slots.

This supports `M0-SOFT-C07` as a structural pilot. It does not by itself
upgrade `M0-SOFT-C05`, because C05 still needs the full write-mechanism
ablation against naive memory write and generic encoder soft plus.
