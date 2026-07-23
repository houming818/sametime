# STONE-1 C05: Frozen-Encoder Decoder Pressure

Date: 2026-07-23
Status: preregistered single-seed mechanism probe
Milestone: `STONE-1` (still incomplete)
Claim: `S3-STONE1-FROZEN-PRESSURE-C05`
Predict: `P-S3-STONE1-FROZEN-PRESSURE-05`

## Question

C04 showed an ordered recursive encoder compressed into root, while the native
decoder stopped at root even after 62,500 updates. C05 asks the cheapest next
question: does the frozen C04 `H_state` contain information that the existing
recursive decoder can learn to read when its downward route is kept open?

This is a diagnostic, not the final joint encoder-decoder architecture.

## Frozen Experiment

Load the C04 checkpoint at update 62,500 and freeze every encoder parameter.
Create two arms from the same checkpoint:

1. `root_control`: force the decoder to stop at root.
2. `leaf_pressure`: forbid early stopping and force the decoder to traverse one
   learned left/right route through every visible TreeHeap level.

Train only decoder parameters for 15,625 updates per arm on the frozen
one-million-pair WMT split. Both arms receive the same batches, optimizer
settings, and update budget. The encoder checksum must remain unchanged.

The `leaf_pressure` arm does not claim that stopping depth was learned. It
opens the existing pipe by intervention so that the branch kernel can receive
gradient. The experiment measures whether useful information can flow through
that pipe.

## Prediction

```text
G1  leaf-pressure validation NLL improves by at least 0.30
G2  leaf-pressure branch kernel receives finite nonzero gradients
G3  shuffling at least one frozen detail level damages final NLL by >= 0.10
G4  the frozen encoder checksum is unchanged in both arms
G5  leaf-pressure final NLL is no worse than root control by more than 0.10
```

`G1+G2+G3+G4` supports the existence of a trainable recursive decoder channel
over frozen TreeHeap state. `G5` is a stronger usefulness comparison, not a
requirement for proving that the channel exists.

## Falsification

Reject the decoder-pressure claim under this recipe if the branch kernel gets
no gradient, forced recursive NLL does not materially improve, or detail
shuffling has no effect. Such a result would mean that merely opening the
existing route is insufficient; it would not prove that no other recursive
decoder can read C04 state.

Even a positive result does not establish spontaneous recursive emergence:
route depth is forced, one seed is used, and the source encoder remains frozen.

Planned evidence: `../evidence/s3_stone1_frozen_encoder_pressure_decoder/`.
