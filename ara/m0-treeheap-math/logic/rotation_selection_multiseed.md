# Rotation Selection Multi-Seed Drift Audit

Status: preregistered follow-up
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
