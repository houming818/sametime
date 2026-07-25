# STONE-2 D02: Chinese Dialogue Seq2Seq

Date: 2026-07-25
Status: registered / queued after product checkpoint
Claim: `S3-STONE2-DIALOGUE-C01`

## Question

Can the materialized TreeHeap translation checkpoint be continued on Chinese
instruction/response pairs and become a minimally usable dialogue Seq2Seq model
without losing recursive TreeHeap causality?

## Dataset

```text
source:
  /home/nio/datasets/pretrain/Chinese-Train-Datasets/
  belle_zh/Belle_open_source_1M.json

format:
  input  = instruction + optional input
  target = output

train / valid / test:
  100,000 / 1,000 / 1,000
```

Rows are assigned by a stable content hash before caps are applied. Source and
target are truncated to the existing 64-leaf TreeHeap contract. The existing
WMT SentencePiece model remains frozen.

BELLE is used for research proof only. Dataset and generated-weight licensing
must be audited before any commercial release.

## Training

```text
initial state       materialized legal D01 winner
objective           target-token cross entropy
encoder LR          0.00005
decoder LR          0.0002
batch               32
exposure            one 100K-pair epoch
route               depth floor
teacher forcing     yes during training
generation          greedy during held-out evaluation
```

## Predict

```text
P1 final validation NLL < initial validation NLL - 0.20
P2 test nonempty rate = 1.0
P3 severe repetition rate <= 0.10
P4 maximum detail-shuffle damage >= 0.10 NLL
P5 every visible depth route mass >= 0.019
P6 encoder receives nonzero gradient and changes
P7 checkpoint reload reproduces 32 frozen prompt outputs exactly
```

P1-P3 establish a minimal dialogue generator. P4-P7 establish that the artifact
still uses the TreeHeap protocol. No BLEU threshold is used as the primary
dialogue gate because many responses can be valid.

## Boundary

Passing does not establish factuality, safety, multi-turn memory, consciousness
or commercial readiness. It only establishes a single-turn Chinese
instruction/response PoC with measured TreeHeap causality.

## Evidence

```text
../evidence/s3_stone2_dialogue_100k/
```
