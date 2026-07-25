# STONE-2 Productization Route

Date: 2026-07-25
Status: registered / D01 running
Claim: `S3-STONE2-PRODUCT-C01`

## Goal

Turn a completed causal experiment into an artifact a person can actually run,
without selecting an invalid control arm or calling translation a dialogue
model.

## Frozen Route

1. Finish D01 `gold/top1/topk/shuffled` under the registered 300K platform.
2. Exclude `shuffled`, because it is a falsification control rather than a
   deployable learning target.
3. Select the lowest test-NLL arm among `gold/top1/topk`; use token BLEU-4 only
   as a tie breaker.
4. Retrain exactly that arm from the same STONE-1 checkpoint and save the full
   TreeHeap encoder/decoder state.
5. Run a translation CLI and preserve its input/output examples.
6. Add SacreBLEU, chrF++, TER, repetition, length-ratio and language-ID audits.
7. Open a separate Chinese prompt/response claim for dialogue. Translation
   evidence must not be relabeled as dialogue evidence.

## Predict

The materialized checkpoint should reproduce the selected arm within:

```text
absolute test NLL drift <= 0.02
absolute internal token BLEU-4 drift <= 0.20
nonempty rate = 1.0
severe repetition rate <= 0.10
checkpoint reload gives byte-identical greedy output on 32 frozen prompts
```

TreeHeap existence remains gated by:

```text
maximum detail-shuffle damage >= 0.10 NLL
every visible depth route mass >= 0.019
```

## Boundaries

- Selection and materialization share the same test split, so this produces a
  product artifact, not independent confirmation.
- Internal token BLEU-4 is not comparable with published SacreBLEU.
- A translation checkpoint may be queried interactively, but it is not a
  conversation model.
- Dialogue requires prompt/response data, multi-turn context and a separate
  held-out evaluation.

## Evidence

```text
../evidence/s3_stone2_teacher_uncertainty_300k/
../evidence/s3_stone2_product_checkpoint/
```
