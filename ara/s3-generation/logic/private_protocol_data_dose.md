# Controlled Data-Dose Test For The Private Protocol

Date: 2026-07-20  
Claim: `S3-PRIVATE-PROTOCOL-DATA-DOSE-C03`  
Predict: `P-S3-PRIVATE-PROTOCOL-DATA-DOSE-03`

## Question

SPR-064 used only 30,000 training pairs sampled from a WMT-derived file that
contains more than fourteen million rows. Its similar Flat and TreeHeap NLL may
therefore be a small-data result rather than an architecture ceiling.

The causal question is:

> With the number of optimizer updates held fixed, does increasing the number
> of distinct training pairs lower held-out NLL?

## Why Fixed Steps Matter

If every dataset is trained for four epochs, a one-million-row dataset receives
about 33 times more parameter updates than a 30,000-row dataset. An improvement
could then be caused by either more distinct evidence or simply more compute.

This proof gives every arm exactly 15,625 optimizer updates with batch size 64,
approximately one million sample exposures:

| Unique train pairs | Approximate times each row is reused |
|---:|---:|
| 30,000 | 33.3 |
| 100,000 | 10.0 |
| 300,000 | 3.3 |
| 1,000,000 | 1.0 |

Validation and test rows, tokenizer, initialization seed, model dimensions,
optimizer, learning rate, batch size, update count, and evaluation cadence are
frozen. The training sets are nested, so every smaller set is a prefix of every
larger set.

## Claim

`S3-PRIVATE-PROTOCOL-DATA-DOSE-C03`:

Under an equal optimizer-update budget and frozen evaluation split, increasing
the number of unique WMT sentence pairs from 30K to 1M should reduce the h1
TreeHeap held-out NLL if the SPR-064 result was materially data-limited.

This is a data-sufficiency claim. It is not a TreeHeap superiority claim.

## Predict

Before execution:

1. Primary: TreeHeap h1 test NLL at 1M is at least `0.10` lower than at 30K.
2. Shape: Spearman correlation between `log10(unique_rows)` and h1 test NLL is
   at most `-0.80`.
3. Supporting: at least two of the three adjacent dose increases lower h1 NLL.
4. Diagnostic: Flat GRU and a parameter-matched small Transformer run under the
   same data-dose contract. Their curves distinguish a corpus-wide data effect
   from a TreeHeap-specific effect.

## Falsification And Interpretation

- If h1 improves by less than `0.03`, the hypothesis that 30K diversity was the
  main bottleneck is not supported under this training recipe.
- If all architectures improve similarly, more data helps, but the result does
  not establish a TreeHeap advantage.
- If h1 closes its NLL gap to the best control as dose grows, that is a reason
  to preregister a multi-seed scale test; it is not yet proof of better scaling.
- If NLL worsens with dose, inspect optimization schedule and sample quality
  before blaming capacity. Equal steps mean the 1M arm sees each row only once.

## Evidence Contract

The implementation must save split hashes, source/tokenizer identity, all
controlled variables, per-evaluation traces, elapsed time, peak GPU memory,
examples, and decision gates using `ara/EXPERIMENT_REPORT_TEMPLATE.md`.
