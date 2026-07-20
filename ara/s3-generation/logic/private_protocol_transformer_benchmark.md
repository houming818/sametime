# Small Transformer Benchmark for the TreeHeap Private Protocol

Date: 2026-07-20
Status: registered
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

