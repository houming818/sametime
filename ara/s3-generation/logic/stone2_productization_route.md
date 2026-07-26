# STONE-2 Productization Route

Date: 2026-07-26
Status: supported artifact / gold arm materialized
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

## Result

The selection rule chose `gold`, because it beat both teacher arms. A complete
`201,083,096` byte checkpoint was saved with SHA-256
`0d19a4b0814c82a7c8a86281d54a03afcfd0bff4ac42399099227837608e6e49`.
The deterministic rerun reproduced test NLL `3.2375`, internal token BLEU-4
`15.5264`, nonempty rate `1.0`, and maximum detail-shuffle damage `0.4511` NLL.

The stricter product audit found a mean hypothesis/reference length ratio of
`0.9023`; `32.35%` of sentences repeated at least one bigram, `16.85%` repeated
a trigram, and `8.90%` repeated a four-gram. These are occurrence rates, not
the earlier severe-repetition classifier, so they expose a quality weakness
without contradicting its `1.5%` result. Standard SacreBLEU/chrF/TER remain
unreported because `sacrebleu` was unavailable on io.
