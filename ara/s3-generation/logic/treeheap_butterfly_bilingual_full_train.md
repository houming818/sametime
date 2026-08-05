# TreeHeap Butterfly bilingual full training

Status: full scale run completed / bilingual path and Butterfly causality supported / product quality open

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

## Smoke result

Taskd 87 trained 29,944 examples (633,547 target tokens) from 30,000 raw
parallel rows in 226.1 seconds on `io`. Both directions were balanced. Mean
validation NLL fell from `10.5612` at initialization to `6.5007`; final test
NLL was `6.6654` for English-to-Chinese and `5.8854` for Chinese-to-English.
Replacing Butterfly with identity increased mean test NLL by `0.11908`.

The checkpoint strict-reload gate (taskd 88) reproduced directional NLL and
identity damage exactly. SMTP wake notifications succeeded. Immutable dream
snapshots show severe repetition at this early scale; they are retained as the
honest starting point for the full-run growth curve.

The 96-hour full run started as taskd 89. Its first 50,000-line block produced
49,897 eligible examples and 1,054,937 target tokens in 344.9 seconds. GPU
memory was about 4.76 GiB under the configured power limit.

## Full-run result

Taskd 89 completed after `345,626.9` seconds. It processed `50,009,218`
eligible bilingual examples and `1,057,491,121` target tokens. The final test
result was:

```text
English -> Chinese NLL   3.762406
Chinese -> English NLL   3.235606
mean Native NLL          3.499006
mean Identity NLL        4.952169
Identity damage         +1.453163
best valid mean NLL      3.422301
```

The run supports one shared dynamic-width bilingual model and strong causal use
of Butterfly communication at scale. Atomic checkpoints, resume and immutable
Dream snapshots worked. Dreams remained mixed and sometimes repetitive, so the
run does not establish product translation quality. It became the frozen
starting checkpoint for C05/C06 and the draft C07 product selector.
