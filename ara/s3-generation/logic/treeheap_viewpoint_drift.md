# TreeHeap Private-Protocol Viewpoint Drift

Status: preregistered / retrospective Dreams audit first / latent-state proof pending

## Claim S3-TREEHEAP-VIEW-DRIFT-C04

During end-to-end training, the Butterfly, lifting FOLD and Decoder may jointly
change their private coordinate protocol. Consequently, some non-monotonic
changes in fixed free-running Dreams may be caused by an internal coordinate
drift rather than semantic forgetting. This interpretation is supported only
when semantic roles and facts remain stable and a held-out alignment recovers
the relation between intermediate states from different checkpoints.

This claim does **not** say that every fluent or different Dream is correct.
Repetition, role reversal, missing negation, changed numbers and invented facts
remain generation errors unless an external reference set establishes another
valid interpretation.

## Motivation

For an encoder state `H` and decoder `D`, an invertible coordinate change `A`
can leave the externally observed function unchanged:

```text
H' = A(H)
D'(H') = D(A^-1(H')) = D(H)
```

The internal protocol is therefore not identifiable from raw coordinates
alone. TreeHeap makes this question observable because the fixed probe can be
tapped after each serial Butterfly stage and after FOLD without splitting the
main forward path into parallel views.

The current implementation computes one unified state:

```text
H0 -> B0 -> H1 -> B1 -> ... -> Hs -> FOLD -> root/details -> READ -> Decoder
```

The probe records `H0..Hs`; only `Hs` continues into FOLD. Recorded states are
detached observations and never enter loss or generation.

## Three phenomena that must remain separate

1. **Coordinate drift**: raw states change, but held-out alignment recovers
   them and output semantics remain stable.
2. **Surface-view drift**: wording changes while agent, patient, polarity,
   time, quantity and causal direction remain correct.
3. **Functional error**: a required relation changes, is omitted or is
   replaced by repetition/hallucination.

Only the conjunction of (1) and semantic stability supports this claim.

## Grammar probe contract

The expanded `ara/s3-generation/dreams.txt` contains fixed bilingual probes
chosen to expose distinctions that a bag of words cannot preserve:

| Family | Minimal distinction |
|---|---|
| agent/patient reversal | `A defeated B` versus `B defeated A` |
| active/passive | same event, different grammatical projection |
| negation | event versus explicitly negated event |
| temporal order | before versus after |
| quantifier | every versus not every |
| relative clause | local noun plus long modifier |
| attachment | instrument versus noun modifier |
| Chinese `把`/`被` | argument roles under different surface order |
| topic-comment | Chinese topicalization versus English canonical order |
| number/entity | exact facts that cannot be replaced by fluent prose |
| causal direction | cause and consequence must not be reversed |
| long composition | nested relations and coordination |

These probes are observation data, not training data. The same file, order,
tokenizer and greedy decoding settings must be used at every wake point.

## Human-visible annotation

For each Dream, record the following independent fields:

```text
agent/patient correct       0/1/NA
polarity correct            0/1/NA
time/order correct          0/1/NA
number/entity correct       0/1/NA
causal direction correct    0/1/NA
meaning acceptable          0/1
surface wording changed     0/1
severe repetition           0/1
```

The semantic-invariant score is the mean of applicable role/fact fields. It is
kept separate from fluency and lexical overlap. A second reviewer should audit
at least 20% of rows; disagreements stay in the evidence instead of being
silently reconciled.

## Latent trace capture

For checkpoint `t`, probe `i` and Butterfly stage `s`, record:

```text
H[t, i, s]
```

No full-corpus tensor dump is allowed. Only the fixed probe suite is captured.
At the current width/dimension this is tens of MiB per checkpoint, within the
RTX 3090 budget.

Two checkpoints are compared with an orthogonal Procrustes map fitted only on
a calibration subset:

```text
A* = argmin_A || X_cal A - Y_cal ||_F,  subject to A^T A = I
```

The map is then evaluated on held-out probes. Report raw normalized RMSE,
aligned normalized RMSE and alignment gain:

```text
gain = 1 - aligned_nrmse / raw_nrmse
```

Fitting and evaluating on the same probes is prohibited because a flexible
map could manufacture apparent agreement.

## Predictions

### P1: semantic/surface separation

Across two late checkpoints, at least 25% of grammar probes change surface
wording while the mean semantic-invariant score drops by no more than 0.05.

### P2: held-out coordinate recoverability

At two or more Butterfly stages, held-out Procrustes alignment reduces
normalized state error by at least 20% relative to the unaligned comparison.

### P3: structure-sensitive cases remain distinguishable

For agent/patient, negation, temporal and causal minimal pairs, the aligned
states must remain more separated between opposite meanings than between two
surface realizations of the same meaning. A coordinate map that collapses both
classes does not count as recovery.

### P4: output function is not merely frozen

Reference NLL or semantic-invariant score must improve somewhere along the
trajectory. If only raw coordinates change while task behavior is uniformly
bad, the experiment has found parameter drift, not useful viewpoint drift.

### P5: read-only observation

Enabling trace capture must reproduce native logits and greedy Dreams exactly
within FP32 deterministic tolerance (`max_abs_logit_diff <= 1e-6`, identical
token IDs).

## Falsification

Downgrade or reject the claim if any of the following occurs:

```text
alignment gives no held-out improvement
alignment works only on its fitting probes
opposite grammatical meanings collapse together
surface changes track role/fact errors rather than stable semantics
trace collection changes logits or decoded tokens
only one checkpoint exists, preventing a temporal comparison
```

If P1 passes but P2 fails, retain only a surface-generation observation. If P2
passes but P1/P3 fail, retain only coordinate reparameterization evidence; do
not call it semantic stability.

## Experiment stages

### Stage A: retrospective Dreams audit

Use all immutable snapshots from taskd 89. This stage can describe output
trajectory but cannot prove latent coordinate rotation because historical
intermediate tensors were not saved.

### Stage B: two-checkpoint trace smoke

Capture fixed probes from `checkpoint_best.pt` and `checkpoint_latest.pt` (or
two explicitly archived wake checkpoints), verify P5, then run held-out
alignment. CPU comparison is sufficient after GPU capture.

### Stage C: prospective trajectory

On the next training run, save detached probe traces and periodic checkpoints
at registered wake intervals. Do not save every training sample. Only after
Stage B passes should encoder/decoder cross-checkpoint swapping or learned
nonlinear alignment be considered.

## Evidence targets

```text
ara/s3-generation/evidence/s3_treeheap_viewpoint_drift/
  README.md
  command.sh
  grammar_annotations.tsv
  dreams_trajectory.json
  trace_<checkpoint>.pt
  alignment_summary.json
  summary.json
```

## Scope

This experiment concerns the identifiability and stability of a learned
TreeHeap private protocol. It does not establish consciousness, a unique
semantic geometry, human-like perception, or that an alternative translation
is correct merely because it is fluent.
