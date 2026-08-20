# Local purification scale extension

Date: 2026-08-20

Status: preregistered, queued after smoke-proven 40K--200K ladder

Partial execution update (taskd `240`, 2026-08-20): the 300K arm completed with
the registered initialization, seed, 18,750 steps, and shared 1,024/1,024
validation/test split. Test NLL was `5.7716216` (PPL `321.06`), an improvement
of `0.3662023` over 200K. EN->ZH / ZH->EN BLEU was `5.8134 / 2.4467`, adjacent
repetition was `0.07476`, source-shuffle delta was `+2.7220`, depth-0 pair-break
delta was `+0.5841`, and runtime-identity delta was `+0.3394`. The NLL gain is
well above the `0.05` stopping threshold. The positive identity delta is a new
structural diagnostic, not yet a replicated structural claim. The 400K arm is
running as taskd `241`.

## Question

Does the held-out NLL improvement observed from 40K through 200K purified
parallel rows continue at 300K and 400K, or has the useful-data boundary become
small enough to stop before exhausting the local candidate pool?

## Immutable controls

- Source scored pool: `NioScore-ZHEN-1M-v1`.
- Acceptance rule: local reranker score `>= 0.98`.
- Available accepted candidates after evaluation exclusion: `514,188`.
- Dataset seed: `14106`; model seed: `14108`.
- Same C13 `ref_zero` architecture, tokenizer, initial checkpoint, optimizer,
  learning rate, batch size, and shared 1,024/1,024 validation/test rows as the
  completed 40K--200K ladder.
- One pass per arm: 300K uses 18,750 steps and 400K uses 25,000 steps at batch
  size 16.
- Runs are serial on the protected `io` RTX 3090. Power and frequency limits
  must remain unchanged.

The dataset builder is rerun with the complete nested size list
`40000,80000,120000,160000,200000,300000,400000,500000`. Existing dataset
hashes must remain unchanged. A hash mismatch invalidates the extension before
training starts.

Amendment after taskd `236--238`: the first extension preparation kept the
validation and test rows fixed but selected its loader-only train sentinel from
the new maximum-size boundary. That changed the byte hash of `shared_eval.tsv`,
so the registered identity gate correctly stopped both training arms before GPU
work began. The rerun fixes `eval_filler_index=200000`, exactly matching the
original ladder construction. Training may restart only if all six existing
40K--200K and evaluation hashes then match.

## Predict

If additional high-confidence rows still carry useful learning signal, test
NLL should continue to decrease. The primary quantities are:

```text
gain_300K = NLL_200K - NLL_300K
gain_400K = NLL_300K - NLL_400K
```

BLEU, repetition, source shuffle, pair break, and runtime identity are retained
as secondary diagnostics. They do not replace the primary NLL stopping rule.

## Decision gate

- If both `gain_300K` and `gain_400K` are below `0.05`, stop. Do not run 500K.
- If either gain is at least `0.05`, 500K may be registered as the final
  candidate-pool boundary arm.
- Stop immediately on OOM, non-finite values, evidence corruption, changed
  initialization/evaluation hashes, GPU instability, or clear structural
  diagnostic regression.

This remains a one-seed resource-planning screen. It cannot establish a final
scaling law or a structural superiority claim.
