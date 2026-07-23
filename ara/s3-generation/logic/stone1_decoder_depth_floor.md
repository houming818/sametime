# STONE-1 C06: Decoder Depth Floor

Date: 2026-07-23
Status: preregistered single-seed mechanism probe
Milestone: `STONE-1` (still incomplete)
Claim: `S3-STONE1-DECODER-DEPTH-FLOOR-C06`
Predict: `P-S3-STONE1-DECODER-DEPTH-FLOOR-06`

## Question

C05 proved that a decoder can learn a useful recursive route over frozen C04
TreeHeap state when it is forced to read the deepest visible level. That
intervention keeps the gradient pipe open but removes stopping-depth choice.
C06 asks whether a decoder can retain nonzero pressure at every depth while
learning how much probability mass to assign to each level.

## Decoder

The C04 encoder is loaded and frozen. At each generated token, the new decoder
routes a probability distribution through every visible TreeHeap depth. It
collects one candidate context per depth, scores all depths, and forms:

```text
route_weight[d] = (1 - K * epsilon) * softmax(score)[d] + epsilon
```

where `K=6` visible levels and `epsilon=0.02`. Every level therefore receives
at least two percent route mass and gradient, while 88 percent remains
learnable. Scores are bounded with `tanh` before softmax to prevent an initial
root logit from numerically closing the pipe.

## Frozen Comparison

All arms start from the same C04 update-62,500 checkpoint and train only the
decoder for 15,625 updates:

1. `native_control`: original sequential learned stop.
2. `leaf_reference`: C05 mandatory deepest read.
3. `depth_floor`: learned depth distribution with a two-percent floor.

Data, batches, optimizer, seed, and update budget are equal.

## Prediction

```text
G1  depth-floor branch gradients are finite and nonzero
G2  every final depth has route mass >= 0.019
G3  depth-floor test NLL is within 0.10 of native control
G4  depth-floor test NLL is within 0.10 of deepest-read reference
G5  shuffling at least one detail depth damages NLL by >= 0.10
G6  all encoder checksums remain unchanged
```

Passing all gates supports a bounded-pressure learnable-depth mechanism. It
does not show that the floor can be removed, that the route is stable across
seeds, or that end-to-end encoder-decoder training avoids parameter freedom.

Planned evidence: `../evidence/s3_stone1_decoder_depth_floor/`.
