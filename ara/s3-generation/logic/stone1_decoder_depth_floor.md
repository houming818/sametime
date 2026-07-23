# STONE-1 C06: Decoder Depth Floor

Date: 2026-07-23
Status: supported bounded-pressure learnable-depth mechanism / single seed
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

## Smoke Result

The 200-update-per-arm smoke completed in `124.4` seconds with peak allocated
VRAM `2.251 GiB`. The depth-floor route was
`[0.3712, 0.3585, 0.0676, 0.0675, 0.0675, 0.0676]`: it neither collapsed to
root nor remained uniform. Five of six gates passed. The only failed smoke gate
was detail causality, whose maximum shuffle damage was `0.0800` rather than
`0.10`. The registered threshold is unchanged for the formal run.

Smoke evidence: `../evidence/s3_stone1_decoder_depth_floor_smoke/`.

## Formal Result

The formal three-arm run completed on `io` in `8163.6` seconds with peak
allocated VRAM `2.253 GiB`. All arms used the same frozen C04 encoder, one
million training pairs, 15,625 decoder-only updates, validation/test splits,
optimizer settings, seed, and batch order.

| Arm | Test NLL | PPL | BLEU-4 | Severe repetition | Route mass by depth |
|---|---:|---:|---:|---:|---|
| Native control | 3.5156 | 33.64 | 13.4823 | 3.35% | `[1, 0, 0, 0, 0, 0]` |
| Forced leaf reference | 3.4636 | 31.93 | 13.9564 | 1.95% | `[0, 0, 0, 0, 0, 1]` |
| Two-percent depth floor | **3.4117** | **30.32** | **14.4886** | **1.90%** | `[0.5446, 0.0911, 0.0911, 0.0911, 0.0911, 0.0911]` |

The bounded-pressure arm improved test NLL by `0.1039` over the native
root-collapse control and by `0.0519` over mandatory deepest reading. Its
maximum detail-shuffle damage was `0.1311` NLL, versus approximately zero for
the native root reader. Every registered gate passed:

```text
G1 branch gradients finite and nonzero                 PASS
G2 every depth route mass >= 0.019                     PASS
G3 within 0.10 NLL of native                           PASS (better by 0.1039)
G4 within 0.10 NLL of forced leaf                      PASS (better by 0.0519)
G5 at least one detail shuffle damage >= 0.10          PASS (0.1311)
G6 frozen encoder checksums unchanged                  PASS
```

The data contract also passed: tokenizer and split hashes matched, one million
training rows were unique, validation/test overlap was zero, and no new
training row matched the frozen evaluation sets.

## Interpretation Boundary

C06 supports the following narrow statement:

> A small fixed pressure supply can keep every recursive decoder level
> trainable while the remaining probability learns a useful, non-collapsed
> depth mixture. On this single seed, that mixture outperformed both root-only
> and forced-leaf extremes.

It does not show that the fixed floor can be removed, that the learned mixture
is stable across seeds, or that end-to-end encoder-decoder training preserves
the same result. STONE-1 therefore remains incomplete.

Formal evidence: `../evidence/s3_stone1_decoder_depth_floor/`.
