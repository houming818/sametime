# STONE-2 D01: Teacher Uncertainty Distillation

Date: 2026-07-26
Status: not supported under the registered recipe
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
teacher temperature  0.1
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

The first smoke at temperature `1.0` was rejected as an experimental
manipulation. Its mean entropy was `1.3853` nats versus the four-candidate
maximum `ln(4)=1.3863`, and its mean top-1 weight was only `0.2607`.
Temperature `0.1` is therefore preregistered before the formal run to create a
non-degenerate ranking. This calibration uses teacher diagnostics, not student
outcome metrics.

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
D3 mean teacher top-1 weight >= 0.30, mean unique candidates >= 2.0,
   and all-identical candidate fraction <= 0.05
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

## Result

The 300K run completed all four arms on the same frozen validation and test
splits:

| arm | test NLL | token BLEU-4 | severe repetition |
|---|---:|---:|---:|
| gold | 3.2375 | 15.5264 | 0.0150 |
| teacher top-1 | 3.3131 | 13.5238 | 0.0145 |
| teacher top-k | 3.3082 | 13.4217 | 0.0160 |
| shuffled top-k | 3.3084 | 13.3838 | 0.0145 |

`top-k` improved NLL over `top-1` by only `0.0049` and reduced BLEU-4 by
`0.1021`. More importantly, `top-k` and shuffled teacher weights differed by
only `0.0002` NLL and `0.0379` BLEU-4. U1, U2, U3 and D1 all failed. The
registered claim is therefore not supported: this experiment found no usable
signal in the teacher probability ordering, and every teacher arm lost to the
gold target arm.

The TreeHeap mechanism itself remained active. Every encoder changed and
received nonzero finite gradient; the largest detail-shuffle damage was between
`0.4511` and `0.4683` NLL, and all six route levels retained mass. This says the
negative result concerns the distillation target, not a bypassed encoder.
