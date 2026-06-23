# GLM Audit Summary for Soft Plus Probe

Date: 2026-06-23
Source: `.squad/outbox/006-runner-reviewer-01-soft-treeheap-ara-audit.md`
Status: review evidence, not a new positive proof

## What GLM Re-ran

GLM / Runner independently re-ran `src/soft_plus_probe.py` and reported stable
passes across multiple seeds:

```text
seed=42   pilot_pass=True  loss=0.000774  collapse_acc=1.0
seed=7    pilot_pass=True  loss=0.000614  collapse_acc=1.0
seed=123  pilot_pass=True  loss=0.000311  collapse_acc=1.0
seed=999  pilot_pass=True  loss=0.000261  collapse_acc=1.0
```

GLM also removed the route CE auxiliary loss:

```text
route_ce_weight=0.0 still passes on seeds 42, 7, 123, 999.
```

This is useful: the current proof does not depend on route CE supervision.

## Key Ablation

GLM then ablated the kernel feature set:

| Feature set | Pilot pass | Collapse accuracy | Meaning |
|---|---:|---:|---|
| Current 13D features with alignment terms | true | 1.000 | Current proof passes |
| Remove alignment/sum features | false | 0.625 | Partially degraded |
| Raw/basic 8D features | false | 0.250 | Random-level routing |
| Random projection + side flags | false | 0.250 | Random-level routing |

## Interpretation

The audit does not falsify these scoped claims:

```text
M0-SOFT-C03: gradient reaches K_write and Plus_a in the toy.
M0-SOFT-C04: the toy collapses to the hard address with the current features.
```

It does show that a stronger claim is still open:

```text
M0-SOFT-C05 / P-SOFT02:
Can a clean kernel learn write routing from raw subheap geometry and beat
naive soft memory write / generic encoder soft plus?
```

## Practical Consequence

Future reports must not say:

```text
The Soft Plus kernel has learned routing from TreeHeap geometry.
```

They may say:

```text
The Soft Plus implementation has a working differentiable path and can collapse
in a synthetic toy. Clean kernel route learning is the next proof.
```
