# C06: Additive Canonical-View Dose

## Status

Preregistered. This experiment follows the negative fixed-budget C05 screen.
It does not reinterpret or overwrite C05.

## Problem

C05 increased the identity-view probability while keeping total target tokens
fixed. That changed two variables together:

1. the identity/Butterfly ratio;
2. the absolute number of Butterfly training tokens.

Native NLL worsened as identity exposure increased, while cross-view JS fell.
The result cannot distinguish a harmful view ratio from simple undertraining of
the native Butterfly route.

## Claim: S3-TREEHEAP-CANONICAL-DOSE-C06

With the native Butterfly dose held fixed, an additional limited identity-view
replay may reduce identity-versus-Butterfly JS without reproducing most of the
native-NLL penalty seen when identity examples replace Butterfly examples. Any
effect must exceed an equal-compute extra-Butterfly replay control.

This is a mechanism claim about continuation training from the same C03
checkpoint. It is not a claim of semantic correctness, production translation,
or general architecture superiority.

## Fixed data and doses

All arms start from the same checkpoint, corpus interval, tokenizer, batch
order, optimizer recipe and seed. C05 measured the following one-pass dose:

```text
base target tokens       6,308,579
20% selected tokens      1,272,632
base optimizer updates       6,056
```

The four arms are:

| Arm | Butterfly tokens | Identity tokens | Total tokens | Purpose |
|---|---:|---:|---:|---|
| `A_base` | 6,308,579 | 0 | 6,308,579 | native one-pass baseline |
| `S_substitute` | 5,035,947 | 1,272,632 | 6,308,579 | reproduce C05 replacement |
| `BB_add_butterfly` | 7,581,211 | 0 | 7,581,211 | equal-compute replay control |
| `BI_add_identity` | 6,308,579 | 1,272,632 | 7,581,211 | additive identity treatment |

For `BB` and `BI`, the same deterministic 20% examples are replayed at the same
points in the stream. Only the replay coordinate differs. `BB` uses Butterfly;
`BI` uses identity. This controls example identity, target text, order, update
count and added compute.

## Registered predictions

### P1: native-dose recovery

Let `N_X` be native Butterfly validation NLL for arm `X`. Define:

```text
recovery = (N_S - N_BI) / (N_S - N_A)
```

The screen passes this gate when `recovery >= 0.50`. A non-positive denominator
invalidates this gate rather than being silently reinterpreted.

### P2: view-specific JS effect

Let `J_X` be cross-view JS. The identity replay must outperform equal-compute
Butterfly replay:

```text
J_BB - J_BI >= 0.05
```

### P3: bounded native price at equal compute

```text
N_BI - N_BB <= 0.015
```

### P4: structural and source causality remain

The candidate must keep:

```text
source-shuffle NLL damage > 1.5
adjacent structural override damage > 0
communication delta RMS > 0
communication gradient norm > 0
```

## Interpretation matrix

| Observation | Interpretation |
|---|---|
| `BI` recovers native NLL and lowers JS versus `BB` | additive identity supplies a view-specific protocol signal |
| `BI` recovers NLL but does not beat `BB` on JS | generic extra optimization explains the result |
| `BI` remains close to `S` despite restored Butterfly dose | normalized view ratio dominates |
| `BB` is best on both native NLL and JS | spend added budget on native Butterfly only |
| directions vary across seeds | effect is too unstable to guide architecture |

## Execution budget

The scientific stop is the registered corpus/token dose, not wall-clock time.
Wall-clock is recorded as an engineering cost. Task timeout is a generous
four-hour fault guard per arm and is not an early-stopping criterion.

Queue:

```text
smoke
  -> A_base
  -> S_substitute
  -> BB_add_butterfly
  -> BI_add_identity
  -> summarize and notify
```

Smoke must write `SMOKE_PASS`. Every formal arm checks this marker before using
the GPU. The one-seed screen uses seed `9101`; seeds `9102` and `9103` require a
new confirmation registration and are not launched automatically.

## Falsification

C06 is not supported by the one-seed screen if any primary gate P1--P4 fails.
It must also be downgraded if dose accounting differs between matched arms,
the replay example sets differ, the starting checkpoint differs, or results are
selected from intermediate wakes instead of the registered final state.

## Files

```text
logic:    ara/s3-generation/logic/treeheap_canonical_view_dose.md
code:     ara/s3-generation/src/s3_treeheap_canonical_view_dose.py
evidence: ara/s3-generation/evidence/s3_treeheap_canonical_view_dose/
```
