# Purified-corpus matched training A/B

Date: 2026-08-17

Status: preregistered, pending overnight evidence

## Claim

At a fixed TreeHeap architecture, initialization, optimizer, training-row
count, step count, and external evaluation set, training on locally purified
parallel pairs should improve held-out translation metrics relative to a
deterministic raw sample.

## Data

The source is the already scored 100,000-row pilot. No source row is rewritten.

- Reserve 1,024 validation and 1,024 test pairs with reranker score at least
  0.9999 as a shared external evaluation pool. Their TSV whitespace is
  deterministically padded so the existing content-hash splitter assigns the
  requested split; parsing strips this padding, so sentence content is
  unchanged.
- `raw`: deterministically sample 40,000 remaining rows without score filtering.
- `purified`: deterministically sample 40,000 remaining rows with score at
  least 0.98.
- Both training files contain the same number of rows. Evaluation rows are
  disjoint from both training files.

## Predict

The purified arm should have lower external test NLL. BLEU/chrF, source-shuffle
sensitivity, repetition, and generated examples are secondary diagnostics.
The claim is rejected if the purified arm is equal or worse within ordinary
run noise.

## Controls

- Same C13 `ref_zero` TreeHeap architecture and parent checkpoint.
- Same seed, 3,000 optimizer steps, batch size 16, and learning rate.
- Same external validation/test file.
- One serial GPU queue on `io`; the 270 W protection remains enabled.
- Outputs are evidence only. They do not replace or delete the original corpus.

## Limitation

This is a one-seed screening experiment. A positive result must be repeated
with multiple seeds and a larger shadow corpus before changing product data.
