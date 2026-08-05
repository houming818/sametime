# C06: Additive Canonical-View Dose

## Status

One-seed screen completed on `io`. The full C06 gate is not supported because
native-dose recovery was `0.3909`, below the registered `0.50` threshold. The
narrower view-specific protocol signal is supported: additive identity replay
reduced cross-view JS by `0.1375` against equal-compute Butterfly replay while
costing only `0.00281` native NLL.

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
count and added compute. Base and replay losses are summed with target-token
weighting before one shared backward pass, so a small replay subset does not
receive a second full-strength optimizer step.

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

## One-seed screening result

Tasks `105` and `110--115` completed on `io` on 2026-08-05. The corrected smoke
confirmed one token-weighted backward pass per parent batch. Every formal arm
used `6,054` optimizer updates, the same seed, checkpoint and line interval.

| Arm | Native NLL | Identity NLL | Cross-view JS | Source-shuffle damage |
|---|---:|---:|---:|---:|
| `A_base` | **3.272799** | 4.847785 | 0.242087 | +1.809397 |
| `S_substitute` | 3.282513 | 3.770985 | 0.102573 | +1.817281 |
| `BB_add_butterfly` | 3.275903 | 4.840925 | 0.242107 | +1.826621 |
| `BI_add_identity` | 3.278716 | 3.779678 | **0.104578** | +1.823747 |

Dose accounting passed exactly:

```text
A base Butterfly tokens     6,308,579
S identity tokens           1,272,632
BB extra Butterfly tokens   1,272,632
BI base Butterfly tokens    6,308,579
BI extra identity tokens    1,272,632
all optimizer updates           6,054
```

Derived registered quantities:

```text
native-dose recovery                 0.390900  FAIL (< 0.50)
JS specificity: JS_BB - JS_BI       0.137528  PASS (>= 0.05)
equal-compute native cost: N_BI-N_BB 0.002812  PASS (<= 0.015)
structural/source causality                    PASS
dose and replay matching                       PASS
```

The result rejects a pure absolute-dose explanation. Restoring all native
Butterfly tokens recovered only 39.1% of the replacement penalty, so normalized
view composition still matters. It also rejects a generic-extra-compute
explanation: extra Butterfly replay left JS unchanged (`0.242087 -> 0.242107`),
whereas the matched identity replay reduced it to `0.104578`.

The defensible observation is therefore:

> Identity exposure is a strong, view-specific control signal for the shared
> FOLD/Decoder protocol. Adding it instead of substituting it makes the native
> trade-off much cheaper, but this one seed did not meet the preregistered
> native-dose recovery gate.

Fixed Dreams remain mixed and do not show a uniform semantic or grammatical
improvement. They do not override the numerical screening decision. C06 must
remain unconfirmed until a separately registered multi-seed experiment.

## Files

```text
logic:    ara/s3-generation/logic/treeheap_canonical_view_dose.md
code:     ara/s3-generation/src/s3_treeheap_canonical_view_dose.py
evidence: ara/s3-generation/evidence/s3_treeheap_canonical_view_dose/
```
