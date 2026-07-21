# Rotation Selection Multi-Seed Drift Audit

Status: supported pilot
Parent: `P-ROT02-B`
Predict: `P-ROT02-C`
Date: 2026-07-21

## Reason

The single-seed structured world selected exact automorphisms strongly, but the
IID gate also locked onto one exact candidate despite tied validation fitness.
This follow-up distinguishes repeatable environmental selection from neutral
winner drift.

## Frozen Design

Repeat the complete `P-ROT02-B` run for eight seeds:

```text
1, 7, 19, 42, 73, 99, 314, 2026
```

Every seed rebuilds the fixed 24-candidate population with the same group
counts: 6 exact, 6 mild, 12 random. No order signal enters the loss.

## Predictions

```text
C1 structured exact winner count        >= 7/8
C2 structured exact mass mean           >= 0.90
C3 structured exact mass minimum        >= 0.75
C4 structured edge/loss Pearson mean    <= -0.90
C5 IID exact winner count                <= 4/8
C6 IID exact mass mean                   <= 0.50
C7 IID exact/random mean-loss gap        <= 0.02
C8 IID |edge/loss Pearson mean|          <= 0.40
C9 all exact-echo inverse errors         < 1e-12
```

## Interpretation

If structured winners are stable while IID winners follow population-level
drift, relational order is selected by the structured environment. If both
worlds repeatedly choose exact candidates, the architecture or optimization
contains an unaccounted structural bias. If neither does, the single-seed
positive was not robust.

## Result

All nine registered gates passed:

```text
structured exact winners            = 8/8
structured exact mass mean/min      = 0.999657 / 0.999215
structured edge/loss Pearson mean   = -0.997404

IID winner histogram                = 2 exact / 1 mild / 5 random
IID exact mass mean                 = 0.283272
IID exact/random loss gap mean      = 0.000992
IID edge/loss Pearson mean          = -0.018631

max exact-echo inverse error        = 0
```

The seed-42 IID exact winner was neutral lock-in, not a repeatable structural
effect. Across seeds, IID winners followed the population composition and its
exact-group probability stayed near the initial `6/24 = 0.25` share. In the
structured world, all seeds selected exact tree automorphisms with more than
`99.92%` exact-group mass.

Decision: in this controlled Gaussian tree world, parent-child predictive
structure supplies selection pressure that eliminates relation-destroying
rotations. Exact paired echo supplies no such pressure. Here "order" means
preservation of tree edges/prefix relations, not ascending scalar order.

Boundary: operators were selected from a fixed finite bank; the experiment did
not evolve a new rotation formula and did not use language data.
