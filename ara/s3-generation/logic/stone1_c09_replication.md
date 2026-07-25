# STONE-1 C09: Frozen-Platform Replication

Date: 2026-07-25
Status: supported / STONE-1 complete on frozen platform
Milestone: `STONE-1`
Claim: `S3-STONE1-FROZEN-PLATFORM-REPLICATION-C09`
Predict: `P-S3-STONE1-FROZEN-PLATFORM-REPLICATION-09`

## Question

C06 and C08 showed that a frozen C04 TreeHeap state can support useful
recursive decoding under a two-percent depth-pressure floor. Those product
results came from one seed. C09 asks whether the released fixed-root EOS recipe
is stable when initialization and batch order change.

This is a replication, not a scale experiment. The complete machine-readable
contract is `stone1_c09_platform_contract.json`.

## Frozen Platform

| Variable | Registered value |
|---|---:|
| Available WMT Massive corpus | 14,170,275 sentence pairs |
| Sampled training split | 1,000,000 unique pairs |
| Validation / test | 2,000 / 2,000 pairs |
| Direction | English to Chinese |
| Source length | 8 to 32 SentencePiece tokens |
| Vocabulary | 32,000 pieces |
| TreeHeap leaf width | 64 |
| State / decoder hidden width | 320 / 512 |
| Encoder | frozen C04 update-62,500 checkpoint |
| Tail convention | visible repeated EOS |
| Decoder route | learned depth route with 2% floor per level |
| Optimizer | AdamW, learning rate 0.002, gradient clip 1.0 |
| Batch / updates | 64 / 15,625 |
| Exposure | 1,000,000 pairs, exactly one nominal epoch |
| Seeds | 71901, 71902, 71903 |
| Host | io, one RTX 3090, serial execution |

The tokenizer, train, validation, test, and source checkpoint SHA-256 values
are frozen in the JSON contract. The program must stop before training if any
hash, file size, split size, model dimension, or optimization value differs.

Changing the training set to 3M or 14.17M, adding epochs, or enlarging the
model would answer a scaling question and therefore requires another claim.

## Registered Prediction

Product gates, computed across all three seeds:

```text
Q1 mean test NLL                     <= 3.90
Q2 mean token BLEU-4                 >= 13.5
Q3 test NLL population std           <= 0.05
Q4 non-empty generation              = 1.00 in every seed
Q5 severe repetition                 <= 0.10 in every seed
```

TreeHeap and training-integrity gates:

```text
S1 maximum detail-shuffle damage      >= 0.10 NLL in every seed
S2 branch kernel receives gradient    > 0 in every seed
S3 every visible depth keeps pressure >= 0.02 in every seed
S4 encoder checksum remains unchanged in every seed
S5 all gradients and metrics remain finite
```

Engineering gates:

```text
E1 peak allocated VRAM                <= 4 GiB
E2 each decoder checkpoint            <= 300 MiB
E3 all platform hashes and constants  match exactly
```

## Decision

```text
all Q + S + E pass
  -> C09 supported; STONE-1 can be signed against this exact platform

Q + E pass but S fails
  -> translation demo only; TreeHeap participation is not replicated

S + E pass but Q fails
  -> TreeHeap mechanism replicated; STONE-1 product gate remains open

any E failure
  -> invalid run; no scientific conclusion
```

The result does not establish state of the art, full-corpus scaling, general
conversation, world knowledge, or a removable pressure floor.

Planned evidence:

```text
../evidence/s3_stone1_c09_replication/
```

## Formal Result

The io run completed in `9,966.24` seconds (`2.77` hours). All three seeds used
the frozen one-million-pair split for exactly `15,625` decoder updates. Every
registered platform hash and constant matched before training.

| Seed | Valid NLL | Test NLL | BLEU-4 | Repetition | Maximum detail-shuffle damage |
|---:|---:|---:|---:|---:|---:|
| 71901 | 3.5340 | 3.4546 | 13.7066 | 0.0215 | +0.5648 |
| 71902 | 3.5370 | 3.4517 | 13.8713 | 0.0150 | +0.5634 |
| 71903 | 3.5306 | 3.4510 | 13.5945 | 0.0130 | +0.5747 |

Aggregate results:

```text
mean test NLL                  3.4524    gate <= 3.90
test NLL population std       0.00157   gate <= 0.05
mean token BLEU-4             13.7241   gate >= 13.5
minimum non-empty rate         1.0000
maximum severe repetition      0.0215   gate <= 0.10
minimum detail-shuffle damage +0.5634   gate >= 0.10
minimum branch-gradient rate   1.0000
peak allocated VRAM            2.27 GiB gate <= 4 GiB
```

All five product gates, five TreeHeap/integrity gates, and three engineering
gates passed. The encoder checksum was unchanged in every seed. Every branch
kernel observation received nonzero gradient, all six visible depths retained
their registered pressure, and disturbing frozen details caused substantial
held-out damage in every seed.

The measured model contains `50,267,778` parameters: `11,062,720` belong to
the frozen encoder and `39,205,058` to the trained decoder. Each decoder-only
checkpoint is about `149.6 MiB`.

Decision:

> `STONE-1` is complete against the exact `S3-STONE1-C09-PLATFORM-V1`
> contract. This is a reproducible TreeHeap translation PoC, not a claim of
> state-of-the-art translation or general intelligence.

The three large checkpoints are archived outside Git at:

```text
/mnt/nas/ara/s3_stone1_c09_replication/checkpoints/
```
