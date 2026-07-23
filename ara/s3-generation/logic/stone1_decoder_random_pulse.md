# STONE-1 C07: Random Pulse Decoder Training

Date: 2026-07-23
Status: preregistered single-seed mechanism probe
Milestone: `STONE-1` (still incomplete)
Claim: `S3-STONE1-DECODER-RANDOM-PULSE-C07`
Predict: `P-S3-STONE1-DECODER-RANDOM-PULSE-07`

## Correction To C06

C06 established only that a fixed two-percent route floor keeps every decoder
depth trainable and produced a useful mixed read on one seed. It did not
establish a stable private protocol, an optimal pressure, or persistence after
the pressure is removed.

C07 tests a different mechanism: only one decoder depth is opened at a time.
The selected depth is temporary, but all depths update the same persistent
decoder parameters. If the shared recursive decoder can accumulate compatible
information, competence at inactive depths should survive later pulses. If
updates conflict, performance should oscillate as one depth overwrites another.

## Training Schedules

The C04 encoder and complete `H_state` construction are frozen. All schedules
start from the same C04 decoder and use equal data, batches, optimizer settings,
seed, and update budget.

```text
random_1    sample a new depth every update
random_32   sample one depth and hold it for 32 updates
random_256  sample one depth and hold it for 256 updates
cyclic_32   visit depths 0..5 in order, 32 updates each
```

At update `t`, only the selected depth `d_t` is readable:

```text
loss_t = CE(decoder(H_state, force_depth=d_t), target)
```

There is no simultaneous depth floor in these arms.

## Primary Prediction

`random_32` is the registered primary arm. `random_1`, `random_256`, and
`cyclic_32` diagnose the effect of pulse duration and random ordering.

```text
G1 every depth receives at least 8% of primary-arm updates
G2 mean forced-depth validation NLL improves by at least 0.20
G3 at least five of six depths improve by at least 0.10
G4 final maximum forgetting on the fixed probe is at most 0.15 NLL
G5 branch gradients are finite and nonzero whenever depth > 0
G6 the frozen encoder checksum remains unchanged
```

Forgetting at depth `d` is:

```text
forget_d(t) = NLL_d(t) - min_{s <= t} NLL_d(s)
```

Passing supports persistent cross-depth decoder competence under temporary
random pressure. Failing G4 with sawtooth per-depth curves supports the
gradient-wash interpretation.

## Gradient Conflict Diagnostic

On a fixed probe batch, C07 measures the cosine matrix between shared GRU-cell
gradients produced by each forced depth. This diagnostic is not a pass gate:

```text
positive cosine    compatible or transferring updates
negative cosine    locally conflicting updates
near-zero cosine   largely independent updates
```

The experiment does not claim subjective private protocol, spontaneous route
selection, multi-seed stability, or STONE-1 completion.

Planned evidence:

```text
../evidence/s3_stone1_decoder_random_pulse_smoke/
../evidence/s3_stone1_decoder_random_pulse/
```
