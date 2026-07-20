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

## Result

Status: `supported pilot / single seed`.

The formal run completed on `io` in 5.32 hours. Every arm used 15,625 AdamW
updates and the same 2K validation / 2K test rows.

| Model | 30K NLL | 100K NLL | 300K NLL | 1M NLL | 30K to 1M gain |
|---|---:|---:|---:|---:|---:|
| TreeHeap h1 | 6.2671 | 5.1454 | 4.2558 | 4.0198 | 2.2473 |
| Flat GRU | 6.0319 | 5.0532 | 4.2025 | 3.9365 | 2.0954 |
| Small Transformer | 6.5373 | 5.3375 | 4.1761 | 3.9201 | 2.6172 |

All four registered gates passed for h1: the 30K-to-1M gain was `2.2473`
against the preregistered `0.10` threshold, dose/NLL Spearman was `-1.0`, all
three adjacent doses improved, and every gradient remained finite.

The strongest explanation is a general data-diversity effect rather than a
TreeHeap-only effect, because all three architectures improved monotonically.
At 1M, h1 remained behind Flat by `0.0833` NLL and Transformer by `0.0997` NLL.
Its gap to the best control nevertheless narrowed from `0.2352` at 30K to
about `0.08-0.10` at 300K-1M.

The best-step trace rejects the idea that 30K represented the corpus optimum.
All 30K models peaked at step 1,000 and then overfit badly; h1 final validation
NLL rose from `6.2998` to `10.7979`. The 1M h1 and Transformer arms were still
best at the final step, so the one-pass 1M result is not a convergence claim.

The examples also expose noisy and sometimes mojibake/misaligned web pairs.
This proof is valid as a controlled NLL data-dose test, but the corpus quality
limits product-level translation conclusions.

Evidence: `../evidence/s3_private_protocol_data_dose_full/`.
