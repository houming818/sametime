# Depth-curriculum lifting pretraining and WMT transfer

Claim ID: `S2-LIFT-PRETRAIN-TRANSFER-C01`

Status: **preregistered / formal single-seed run pending**

## Question

Can unlabeled text pretrain the same reversible TreeHeap used by translation,
and does that initialization improve a later English-to-Chinese checkpoint over
an architecture-identical model trained from scratch?

This is a transfer claim, not merely a lower pretraining loss claim. The
pretrained and scratch models must use identical WMT rows, batch order, depth
curriculum, optimizer, update count, model size, tokenizer, and decoder.

## Fixed architecture

Both stages use `AdaptiveRecursive(learned_update=True, alternate=False)`:

\[
r=b-P_\theta(a),\qquad p=a+U_\phi(r),
\]

with exact inverse

\[
a=p-U_\phi(r),\qquad b=r+P_\theta(a).
\]

The decoder begins at root. At each addressed node it places active probability
mass into STOP or passes the remainder only to that node's registered children.
All embedding, predictor, update, route, recurrent decoder, and output parameters
are transferred. The WMT optimizer state is reset.

## Why the previous pretraining checkpoint is insufficient

`S3-LIFT-SUBHEAP-PRETRAIN-C01` learned source/detail-dependent generation, but
root zero cost only `0.0042` NLL and adjacent sibling swapping cost only
`0.0022`. It remained detail-dominant. More optimizer updates alone would not
repair that missing pressure.

The new curriculum makes recursive depth an actual training intervention. Each
optimizer step computes one ordinary target cross-entropy at one deterministic
depth cap. The cycle is:

```text
6, 6, 6, 6, 5, 4, 3, 2, 1, 0
```

Thus 40% of updates retain unrestricted READ and 60% require the same decoder
to predict with a shallower addressed frontier. Losses are not summed into a
multi-objective mixture; one batch produces one CE and one gradient update.

## Stage A: unlabeled pretraining

- data: `/home/nio/datasets/pretrain` real Chinese documents;
- tokenizer: WMT Massive bilingual SentencePiece;
- input: 64-token block with one address-aligned subheap of width 1/2/4/8
  replaced by MASK;
- target: the removed token sequence plus EOS;
- updates: 50,000, batch 64, FP32, one seed;
- checkpoint selection: mean held-out NLL at caps 0/2/4/6.

This remains self-supervised. It does not use syntax, categories, summaries,
translations, or teacher-model embeddings.

## Stage B: matched WMT transfer

- data: deterministic 200k/5k/5k split from WMT Massive;
- direction: English to Chinese;
- model size: 256 state, 256 hidden, heap width 64;
- training: five epochs with the same depth-cap cycle;
- candidates: `scratch` and `pretrained`;
- checkpoint selection: unrestricted validation NLL;
- historical references: old TreeHeap token-BLEU4 `9.909`; matched flat sequence
  `10.572`.

## Predictions and gates

`P0 smoke`: 200 pretrain updates followed by 200 WMT updates remain finite,
use CUDA, save and reload the checkpoint, and preserve lifting closure below
`1e-10` MSE.

`P1 pretraining learns`: mean held-out cap-0/2/4/6 NLL falls by at least `0.20`
from initialization, and complete-source shuffle costs at least `0.10` NLL.

`P2 transfer`: pretrained WMT improves unrestricted test NLL over scratch by at
least `0.03` and token-BLEU4 by at least `0.20`.

`P3 historical progress`: pretrained WMT exceeds the old TreeHeap
token-BLEU4 `9.909`. The flat `10.572` score is an aspirational comparison, not
hidden if it remains ahead.

`P4 multiresolution use`: pretrained route mass places at least `0.05` on two
non-leaf depths and no more than `0.75` on leaves.

`P5 positive depth growth`: unrestricted test NLL beats forced-root by at least
`0.10`, and at least three successive depth openings improve NLL by `0.01`.

`P6 structural causality`: root shuffle costs at least `0.05`, at least three
detail-address shuffles cost `0.05`, and at least three pre-FOLD pair breaks
cost `0.05`.

`P7 algebra and fair stream`: closure remains below `1e-10`; both WMT models
record an identical SHA-256 digest for the first 1,024 training batches.

## Decisions

- P0 failure stops the formal queue.
- P1 pass and P2 fail means pretraining learned its own task but did not
  transfer; do not scale it again unchanged.
- P2 pass but P4/P6 fail supports ordinary parameter transfer, not a TreeHeap
  structural advantage.
- P1--P7 pass supports one-seed transfer evidence and justifies replication.
- Beating old TreeHeap but not flat is project progress, not architecture
  superiority.

## Boundaries

The run does not establish production translation, standard WMT SacreBLEU,
world knowledge, human-readable root summaries, compression efficiency,
Transformer superiority, or consciousness. WMT Massive contains noisy pairs;
all sample outputs and the matched flat reference remain visible.
