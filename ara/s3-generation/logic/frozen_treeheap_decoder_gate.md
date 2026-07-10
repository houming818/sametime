# S3 Frozen TreeHeap Decoder Gate

Date: 2026-07-10
Status: design / gated by S1 encoder proof
Related:

```text
S1-ENCODER-OBS-C01
S3-FROZEN-DECODER-C01
S3-SEM-HUFF-GEN-C01
```

## Why This Gate Exists

DeepSeek's review is accepted.

The S3 semantic Huffman plan is too large to prove in one step.  It mixes:

```text
encoder
tree placement
kernel route
compression pressure
probability buckets
surface decoder
```

If that system succeeds or fails, we will not know which part caused the
result.  Therefore S3 must start with a smaller gate:

```text
first prove a minimal S1 encoder can create useful internal TreeHeap subheaps;
then freeze that encoder;
then test whether a decoder can read the frozen subheaps.
```

## Mathematical Story

Let:

```text
x
  observed input

E_phi
  S1 encoder

H = E_phi(x)
  encoded TreeHeap state

S_i
  internal subheap or prefix state inside H

G_psi
  S3 decoder

y
  surface output
```

The minimal S3 gate is:

```text
H = freeze(E_phi(x))
y_hat = G_psi(S_i)
L_gen = CE(y_hat, y)
```

The important constraint is that `E_phi` is frozen.  This prevents the decoder
from secretly fixing a bad encoder during S3 training.

## What Counts As TreeHeap Evidence

The decoder may read:

```text
internal subheap state
path / prefix created by the encoder
kernel-read probability bucket
```

The decoder may not read:

```text
gold semantic label
oracle route answer
raw target text
target-in-left / target-in-right flags
```

In plain words:

```text
If the tree really stored useful structure, the decoder should read the tree.
If the decoder only works because we hand it the answer, the proof is invalid.
```

## Predict

If the claim is right:

```text
P1. The structured frozen encoder output beats shuffled encoder output.
P2. On structure-sensitive generation, TreeHeap decoder beats BoW.
P3. The effect remains after the encoder is frozen.
P4. The generated probability bucket is meaningful before final argmax.
```

## Falsification

Reject or keep blocked if:

```text
F1. S1 encoder does not beat shuffled controls.
F2. Frozen TreeHeap decoder does not beat shuffled TreeHeap decoder.
F3. BoW/flat decoder matches TreeHeap on the structure-sensitive task.
F4. The proof requires gold labels, object IDs, or precomputed answer flags.
F5. The result cannot be reproduced from saved S1 evidence.
```

## Experiment Order

1. Run `s1_encoder_minimal_observer_probe.py`.
2. Save the full S1 evidence directory.
3. If S1 is positive, select the best structured frozen encoder output.
4. Train only `G_psi`.
5. Compare to BoW, flat, shuffled, random/frequency TreeHeap.

## Boundary

This gate does not prove WMT.

It only proves the next necessary bridge:

```text
TreeHeap can encode useful internal structure,
and a decoder can read that frozen structure.
```

If this bridge fails, S3 should not proceed to a larger five-loss system.
