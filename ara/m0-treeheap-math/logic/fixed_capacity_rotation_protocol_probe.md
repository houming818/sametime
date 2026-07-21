# Fixed-Capacity Rotation Protocol Probe

Status: preregistered
Parent claim: `M0-ROT-C02`
Predict: `P-ROT02-A`
Date: 2026-07-21

## Narrow Question

Can nested subheap rotations act as a private encoder/decoder protocol without
copying or growing the TreeHeap?

This is a protocol-carrier proof, not an emergence proof. The encoder's private
rotation program is fixed for each protocol. The decoder receives only encoded
`H_state` and echo loss, and must learn the inverse program.

## Fixed State

Use one complete binary TreeHeap:

```text
capacity = 127 nodes
state_dim = 4
```

Every rotation is a permutation of addresses inside an existing subheap:

```text
R_(S)(H)[j] = H[pi_S(j)]
```

It conserves node count and state shape. Training may use one fixed-size tensor
scratch required by autograd, but no operation creates TreeHeap nodes.

## Private Programs

Register six overlapping subheap mirror operators. Their order matters because
the addressed subheaps overlap. Two encoder protocols use different hard bit
programs:

```text
P_A = [1, 1, 0, 1, 0, 1]
P_B = [0, 1, 1, 0, 1, 1]
```

Encoder execution is left to right. The exact inverse applies enabled mirrors
in reverse order.

The decoder has six trainable logits. A soft mirror is:

```text
SoftMirror(H, p) = (1-p)H + p R(H)
```

Decoder logits are optimized only by reconstruction MSE. At evaluation, each
probability collapses at `0.5` and the hard inverse program is executed.

## Predictions

```text
P1 exact hand inverse max error       < 1e-12
P2 learned decoder bit accuracy       = 1.0 for A and B
P3 paired hard echo max error          < 1e-6
P4 identity-decoder MSE                > 0.10
P5 cross-protocol decoder MSE          > 0.10
P6 one-bit-flip decoder MSE            > 0.05
P7 wrong inverse order MSE             > 0.05
P8 partial inverse remains lossy and
   complete inverse reaches exact echo
P9 TreeHeap node/state capacity         unchanged
```

## Controls

1. identity decoder;
2. decoder with one protocol bit flipped;
3. correct bits applied in encoder order instead of inverse order;
4. `Encoder_A + Decoder_B` and `Encoder_B + Decoder_A`;
5. partial inverse programs from zero through all six steps.

## Falsification

Reject the protocol-carrier predict if paired decoders cannot learn the exact
inverse, cross/wrong/order controls are harmless, or logical TreeHeap capacity
changes.

## Interpretation Boundary

Passing shows that fixed-capacity nested rotations can carry a seed/private
protocol and that recursion order is causally readable by a decoder. It does
not show that an unsupervised language model will discover this protocol, that
rotation improves WMT, or that this hand-registered operator bank is optimal.
