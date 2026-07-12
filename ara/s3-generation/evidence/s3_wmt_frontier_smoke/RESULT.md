# S3 WMT Fixed-Bandwidth Frontier Smoke

## Setup

- host: `io`, RTX 3090, existing 270W limit retained;
- data: WMT17 English-Chinese;
- train / valid / test: 5,000 / 500 / 500;
- source BPE length before EOS: 9--24;
- state dimension: 192;
- epochs: 5;
- matched compressed-memory bandwidth: `K=4` states.

## Independent Training Results

| Encoder | Parameters | Test NLL | PPL | token-BLEU4 |
|---|---:|---:|---:|---:|
| learned TreeHeap frontier | 12,896,898 | 6.5988 | 734.2 | 0.226 |
| fixed TreeHeap frontier | 12,896,898 | 6.6012 | 736.0 | 0.159 |
| random TreeHeap frontier | 12,896,898 | 6.7020 | 814.0 | 0.185 |
| flat four-vector compression | 12,748,673 | **6.5080** | **670.5** | **0.431** |
| unrestricted leaf GRU oracle | 12,896,897 | 6.3541 | 574.8 | 0.792 |

The learned frontier beats random and is numerically tied with fixed, but it
does not beat the matched-bandwidth flat compressor.  The primary claim gate
therefore fails.

## Same-Checkpoint Route Intervention

The trained learned-frontier checkpoint was evaluated three times while only
the route policy was replaced:

| Route used at test time | NLL | Delta from learned | token-BLEU4 |
|---|---:|---:|---:|
| learned | 6.5988 | - | 0.226 |
| fixed | 6.6085 | +0.0097 | 0.215 |
| random | 6.6272 | +0.0285 | 0.183 |

Across 7,469 merge decisions, learned route choices agreed with fixed only
`9.22%` and with random only `10.16%`.  The route has a small causal effect, but
both NLL deltas are below the preregistered `0.05` gate.

## Decision

```text
S3-WMT-FRONTIER-C01
  -> main claim not supported at smoke
  -> weak causal route signal retained
```

This is stronger than the prior all-leaf experiment because the internal
frontier is now unavoidable.  It is still not evidence of a TreeHeap advantage:
four ordered flat pooled vectors preserve translation information better than
the current learned compose/merge system.

Likely next question: does TreeHeap need a residual/value-preservation channel
inside each composed node?  The current nonlinear compose must simultaneously
compress lexical identity and learn structure, and may destroy information that
flat pooling keeps cheaply.
