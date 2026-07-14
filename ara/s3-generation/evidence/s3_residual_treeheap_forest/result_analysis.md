# Result Analysis

Date: 2026-07-14
Host: `io`
Evidence: `full/summary.json`, `full/trace.jsonl`, `queue.log`

## Decision

```text
S3-RESIDUAL-FOREST-C01          -> rejected as written
S3-TREEHEAP-ROOT-COMPRESS-C01  -> supported full-corpus / single-seed
```

## Why The Main Claim Failed

The four-head residual and no-residual models have exactly `6,310,341`
parameters. The residual model first logged `NaN` at step `62,400`; the
no-residual model completed all data at validation NLL `6.236525`.

All parameter tensors in the final residual checkpoint are non-finite. This
means the run proves numerical divergence but cannot identify the first tensor
that diverged. The unconstrained scalar residual gain is a leading repair
target, not a proven sole cause.

## What Survived

The no-residual model compressed each 64-token block recursively to one root.
The decoder had no direct access to leaves. Its structure interventions were
large:

```text
destroy left/right pairing: +2.625219 NLL
ablate head 0:              +1.185258 NLL
ablate head 1:              +2.466763 NLL
ablate head 2:              +1.157449 NLL
ablate head 3:              +3.485271 NLL
```

Therefore the current evidence supports use of addresses and multiple heads,
but not human-readable specialization or superiority over other architectures.
