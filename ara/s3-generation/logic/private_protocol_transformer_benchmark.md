# Small Transformer Benchmark for the TreeHeap Private Protocol

Date: 2026-07-20
Result date: 2026-07-20
Status: supported for the registered small-Transformer comparison / top-model claim open
Claim: `S3-PRIVATE-PROTOCOL-TF-C02`
Predict: `P-S3-PRIVATE-PROTOCOL-TF-02`

## Question

The first private-protocol battle established a structurally causal TreeHeap
protocol but compared it only with a GRU flat-sequence baseline.  Similarity to
that baseline is a viability result, not a competitive result.  This addendum
asks:

> Under the exact frozen WMT split and approximately 27M parameter budget, how
> far is the best registered TreeHeap result from a small Transformer trained
> with both the identical recipe and a conventional Transformer recipe?

This is a stronger architecture baseline, not an industry-top result.  No claim
about state of the art is permitted from this run.

## Frozen Inputs

The benchmark must reuse the original Stage A values without resampling:

```text
data       /home/nio/datasets/wmt_massive/train.massive.zh-en.tsv
tokenizer  /home/nio/datasets/wmt_massive/sp_bpe_massive.model
split      30,000 / 2,000 / 2,000
scan       300,000 rows
length     8..32 pieces
seeds      71901, 71902, 71903
direction  English -> Chinese
```

The script reads the previous `summary.json` and rejects incompatible data,
split, seed, vocabulary, or model-budget settings.

## Model

The small Transformer uses:

```text
d_model              256
attention heads      4
encoder layers       2
decoder layers       2
feed-forward width   512
learned positions    128
```

It has separate source and target embeddings, causal decoder self-attention,
encoder-decoder cross-attention, and one vocabulary projection.  The parameter
count must remain within 5% of the registered TreeHeap h1 count (`27,619,714`).

## Two Training Controls

`same_recipe` isolates architecture under the old optimizer budget:

```text
4 epochs, AdamW, constant lr=0.002, no label smoothing, dropout=0
```

`standard_recipe` is the stronger practical reference:

```text
8 epochs, AdamW(beta2=0.98), peak lr=0.0005,
10% warmup plus cosine decay, label smoothing=0.1, dropout=0.1
```

The second recipe has more compute and is therefore not a compute-matched
causal comparison.  It is included deliberately as a harder quality ceiling.
Both recipes select the best validation-NLL checkpoint and report test NLL,
perplexity, token BLEU-4, exact generation, parameters, and elapsed time.

## Predictions and Decisions

Primary comparison uses the previous best TreeHeap NLL (`h1 = 6.1231`).

```text
competitive: TreeHeap h1 is no more than 0.02 NLL worse than standard_recipe
weak parity:  the gap is in (0.02, 0.10]
behind:       the gap is greater than 0.10
```

The same-recipe result separates architecture from recipe sensitivity.  If the
same-recipe Transformer fails but the standard recipe wins, optimization is a
major confound.  If both win, the current TreeHeap mechanism is simply behind.
If neither wins, Stage A still does not establish a top-model result; it only
shows that this small Transformer configuration is not a stronger reference.

## Falsification Boundary

Close NLL values do not prove that either model is a global optimum, that their
function classes are equal, or that the dataset has reached its irreducible
entropy.  This run provides trained upper bounds only.  Industry-level claims
still require a standard public benchmark, a strong published checkpoint, and
scaling curves.

## Result

The formal `io` run completed successfully in about eight minutes.  The
Transformer had `27,278,337` parameters, a `1.236%` gap from TreeHeap h1.  Mean
three-seed test results were:

| Model | NLL | token BLEU-4 | Mean train time |
|---|---:|---:|---:|
| flat GRU | 6.0401 | 5.3530 | 111.6 s |
| TreeHeap h1 | 6.1231 | 4.9719 | 405.4 s |
| Transformer same recipe | 6.4423 +/- 0.0062 | 2.9422 +/- 0.1529 | 52.2 s |
| Transformer standard recipe | 6.5330 +/- 0.0043 | 2.8941 +/- 0.1916 | 100.8 s |

TreeHeap h1 beat the same-recipe and standard-recipe Transformers by `0.3192`
and `0.4099` NLL respectively, so the registered small-Transformer competitive
gate passed.  It also beat both by about two BLEU-4 points.  This rejects the
narrow hypothesis that the current TreeHeap is simply worse than any small
Transformer at this data scale.

The result does not establish a top-model standard.  Flat GRU remained the best
model, TreeHeap h1 was `3.63x` slower than flat and `7.77x` slower than the
same-recipe Transformer, and the nominal standard Transformer recipe was worse
than the identical old recipe.  Its validation NLL was still decreasing at
epoch 8, so the run is not evidence that the Transformer class reached its own
optimum.  A public strong checkpoint or a tuned scaling curve remains required.

Evidence is stored in
`ara/s3-generation/evidence/s3_private_protocol_transformer_benchmark_full/`.
All six checkpoints are archived on `io` at
`/mnt/nas/ara/s3-generation/evidence/s3_private_protocol_transformer_benchmark_full/`.
