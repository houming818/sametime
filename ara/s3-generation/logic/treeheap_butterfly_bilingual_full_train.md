# TreeHeap Butterfly bilingual full training

Status: registered / implementation and smoke pending

## Claim S3-TREEHEAP-BUTTERFLY-BIDIR-C03

The supported Butterfly + adaptive lifting TreeHeap can be trained as one
shared Chinese-English sequence-to-sequence model over the complete local WMT
parallel corpus. Direction is explicit, but encoder, Butterfly communication,
FOLD/UNFOLD kernels, recursive READ and decoder parameters are shared.

This is a scale and product-readiness claim. It does not claim general
intelligence, production translation quality, or superiority to a large
Transformer.

## Data contract

Source file:

```text
/home/nio/datasets/wmt_massive/train.massive.zh-en.tsv
2.4 GiB
14,170,275 parallel rows
```

Every eligible pair is exposed in alternating directions across epochs:

```text
even orientation: English -> Chinese
odd orientation:  Chinese -> English
```

A stable row hash and epoch select the orientation, so two consecutive epochs
cover both directions without duplicating the disk corpus. Validation and test
rows use stable hash partitions and never enter training.

Raw source and target content may contain up to 253 SentencePiece pieces. A
direction token and EOS must also fit in the 256-leaf TreeHeap.

## Dynamic capacity

The model has a maximum capacity of 256 leaves, but each batch uses the nearest
power-of-two width required by that batch. All widths share the same learned
kernels:

```text
1..32 pieces    -> 32/64 leaves after direction and EOS
33..64 pieces   -> 128 leaves at the boundary
65..128 pieces  -> 256 leaves at the boundary
129..253 pieces -> 256 leaves
```

Length buckets vary batch size to control GPU memory. This is not a family of
independent models; it is one parameter set evaluated at different recursive
depths.

## Dream protocol

`dreams.txt` is a user-editable TSV probe list, not training data and not a
metric log:

```text
en2zh<TAB>source text<TAB>optional reference
zh2en<TAB>source text<TAB>optional reference
```

At every wake-up the current model translates every valid line and writes:

```text
dreams/latest.txt
dreams/step-XXXXXXXXXXXX.txt
```

The timestamped files are immutable observations of how generation changes
during training. They are never fed back into the optimizer.

## Wake-up and recovery

Training is divided into deterministic file blocks. Checkpoints are written
atomically at block boundaries and contain model, optimizer, epoch, next file
offset, counters and best validation state. Wake-up reports include throughput,
directional validation NLL, fixed free-running translations, repetition rate,
length buckets and ETA.

## Predictions

1. Both directional validation NLL values improve from the first wake-up.
2. Fixed dreams become source-dependent in both directions rather than sharing
   one dominant output.
3. The 33--64 and 65--128 buckets improve rather than remaining OOD.
4. Runtime replacement of Butterfly by identity continues to damage held-out
   NLL after scale training.
5. Saved and reloaded checkpoints reproduce fixed dream outputs exactly.

## Falsification and stop gates

Stop or downgrade the run if any of the following persists across two wake-ups:

```text
non-finite loss or gradients
source-independent dominant output
direction confusion
severe repeated n-grams
no improvement in either direction
Butterfly identity override causes no measurable damage
checkpoint reload changes deterministic output
```

The first 15-minute smoke must validate data direction, dynamic widths, CUDA
memory, atomic reload and dreams output before the 96-hour queue starts.
