# Purified-corpus matched training A/B

Date: 2026-08-17

Status: one-seed screening complete; NLL supported, generation mixed

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

## Result

Both serial tasks completed on `io` with the same initial model hash, seed,
architecture, optimizer budget, 38,000 training rows, and shared 1,024-row
validation and 1,024-row test sets.

| Metric | Raw | Purified | Direction |
|---|---:|---:|---|
| best validation NLL | 7.5629 | **7.1604** | purified better |
| test NLL | 7.5784 | **7.1441** | purified better |
| test perplexity | 1955.5 | **1266.7** | purified better |
| source-shuffle delta | 1.2496 | **1.3268** | purified more source-sensitive |
| pair-break delta | 0.1154 | **0.1793** | purified more structure-sensitive |
| runtime-identity delta | -0.0940 | **0.0294** | purified favors native FOLD |
| adjacent repetition | **3.77%** | 5.77% | raw better |
| EN->ZH SacreBLEU | **1.426** | 0.939 | raw better |
| ZH->EN SacreBLEU | 0.651 | **0.923** | purified better |
| EN->ZH chrF2 | **3.064** | 2.821 | raw better |
| ZH->EN chrF2 | 14.636 | **14.711** | purified slightly better |

Purification reduced test NLL by 0.4343, or about 5.73% of the raw NLL. In
perplexity terms the reduction is about 35.2%. Source shuffling and pair
breaking also caused larger damage after purified training, while replacing
native FOLD with runtime identity changed from a misleading improvement in the
raw arm to a small degradation in the purified arm. This suggests that cleaner
pairs improved conditional and structural learning rather than merely making
the output distribution easier.

The generated sentences remain poor in both arms. EN->ZH overlap declined,
ZH->EN overlap improved, and repetition increased. Therefore this experiment
supports the narrow claim that local purification improves the learned
probability/structure signal. It does **not** yet support a claim of overall
translation-quality improvement or product readiness.

## Decision

Run a matched multi-seed confirmation before promoting the filter. Keep the
original corpus immutable. If the NLL and structural deltas repeat but
generation remains mixed, investigate language-direction balance and decoding
before scaling the purified corpus.
