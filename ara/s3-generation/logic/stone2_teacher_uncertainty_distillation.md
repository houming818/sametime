# STONE-2 D01: Teacher Uncertainty Distillation

Date: 2026-07-25
Status: preregistered / smoke first
Claim: `S3-STONE2-TEACHER-UNCERTAINTY-D01`
Predict: `P-S3-STONE2-TEACHER-UNCERTAINTY-01`

## Question

STONE-1 learns from a deterministic corpus reference. A frozen translation
teacher can additionally assign probability to several candidate translations.
This experiment asks a narrow question:

> Does the teacher's ranking among several translations contain useful
> predictive information beyond its deterministic top-1 translation?

The teacher distribution is not called the real-world distribution. It is one
model's belief and may be wrong.

## Gradient Definition

For a gold reference `y`, teacher candidates `y_k`, teacher weights `q_k`, and
TreeHeap probability `p`, the four arms are:

```text
A gold:
  L_A = CE(y, p)

B top1:
  L_B = 0.5 CE(y, p) + 0.5 CE(y_1, p)

C topk:
  L_C = 0.5 CE(y, p) + 0.5 sum_k q_k CE(y_k, p)

D shuffled:
  L_D = 0.5 CE(y, p) + 0.5 sum_k shuffle(q)_k CE(y_k, p)
```

Each loss is a deterministic scalar once the teacher cache is frozen. The
teacher is evaluated under `torch.no_grad()` and never participates in student
backpropagation. Gradients are produced by the TreeHeap output loss and travel
through its decoder, depth route, `H_state`, recursive FOLD kernels, and
unfrozen encoder.

## Platform

Pilot/formal mechanism run:

```text
teacher              Helsinki-NLP/opus-mt-en-zh
teacher candidates   top 4 beam hypotheses
student start        C09 seed-71902 encoder plus decoder
train pairs          300,000
validation/test      frozen C09 2,000 / 2,000
student tokenizer    frozen TreeHeap 32K SentencePiece
heap                 fixed 64 leaves, repeated-EOS tail
route                six visible levels, 2% depth floor
student arms         gold / top1 / topk / shuffled-topk
student seed         71912
encoder              unfrozen in every arm
student batch        32 source rows
encoder / decoder LR 0.00005 / 0.0002
updates              one nominal 300K-pair epoch per arm
```

The teacher emits strings. Every string is retokenized with the TreeHeap
tokenizer, so teacher and student vocabularies need not share coordinates.

## Predictions

Primary uncertainty gates:

```text
U1 topk test NLL <= top1 test NLL - 0.02
U2 topk BLEU-4   >= top1 BLEU-4 + 0.20
U3 topk test NLL <= shuffled test NLL - 0.02
```

General distillation observations:

```text
D1 top1 improves over gold by at least 0.02 NLL or 0.20 BLEU-4
D2 every arm is nonempty and severe repetition <= 0.10
```

TreeHeap integrity:

```text
S1 encoder receives finite nonzero gradient in every arm
S2 encoder checksum changes in every arm
S3 topk maximum detail-shuffle damage >= 0.10 NLL
S4 all visible depth masses remain >= 0.019
```

## Decision

```text
U1 + U2 + U3 pass
  -> teacher uncertainty contains useful information beyond top-1 in this task

D1 passes but any U gate fails
  -> deterministic teacher output helps; soft uncertainty is not supported

topk equals shuffled
  -> teacher probability ordering carries no measured information

all teacher arms fail against gold
  -> reject this teacher/recipe; do not scale
```

Even a positive result proves behavioral transfer, not acquisition of the real
world, consciousness, or the teacher's internal representation.

Planned evidence:

```text
../evidence/s3_stone2_teacher_uncertainty_smoke/
../evidence/s3_stone2_teacher_uncertainty_300k/
```
