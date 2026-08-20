# Local purification scale ladder

Date: 2026-08-18

Status: preregistered, pending evidence

## Question

Does increasing the absolute amount of locally purified parallel data continue
to improve TreeHeap learning, or does the benefit reach a visible boundary?

## Data and ladder

### Dataset names

To keep the source corpus, scored pool, accepted candidates, and training
subsets distinct, this experiment uses the following stable names:

| Name | Meaning |
|---|---|
| `WMT-Massive-ZHEN-14M` | immutable 14,170,275-row source corpus |
| `NioScore-ZHEN-1M-v1` | deterministic one-million-row shadow sample scored by the local reranker |
| `NioClean-ZHEN-S098-v1` | accepted candidate pool with local score `>= 0.98` |
| `NioClean-ZHEN-S098-40K-v1` | nested 40K training subset |
| `NioClean-ZHEN-S098-80K-v1` | nested 80K training subset |
| `NioClean-ZHEN-S098-120K-v1` | nested 120K training subset |
| `NioClean-ZHEN-S098-160K-v1` | nested 160K training subset |
| `NioClean-ZHEN-S098-200K-v1` | nested 200K training subset |

`NioClean` identifies the SameTime/Nio local purification pipeline, `ZHEN`
identifies Chinese-English parallel data, and `S098` records the acceptance
threshold. The suffix is part of the dataset identity: changing the scorer,
threshold, ordering, deduplication, or source snapshot requires a new version.

Score a deterministic one-million-row shadow sample with the local
`bge-reranker-v2-m3`. No DeepSeek API is used. Construct nested purified
training sets at an arithmetic progression:

```text
40K, 80K, 120K, 160K, 200K
```

All sets use the same score threshold (`score >= 0.98`) and deterministic order.
The 40K set is a strict subset of 80K, and so on. A shared high-confidence
evaluation set is excluded from every training set.

## Training control

- Same C13 `ref_zero` TreeHeap architecture and initial checkpoint.
- Same seed, optimizer, learning rate, batch size, tokenizer, validation, and
  test rows.
- Train each arm for one pass: `steps = ceil(rows / batch_size)`.
- Each physical training TSV includes 1,000 extra candidates to absorb
  tokenizer eligibility rejection, plus one valid/test loader sentinel. These
  sentinels are discarded when the shared evaluation set replaces them.
- Run serially on the protected `io` RTX 3090.

## Predict

If purified-data quantity carries useful learning energy, test NLL should tend
to fall as the ladder grows. Generation and structural interventions are
secondary diagnostics and need not be strictly monotonic at this scale.

## Boundary rule

For each additional 40K rows, calculate:

```text
marginal_gain = previous_test_nll - current_test_nll
```

Plot the absolute test-NLL curve and a marginal-gain histogram. An operational
plateau is flagged if two consecutive increments improve test NLL by less than
0.05. This threshold is a resource-planning rule, not a mathematical constant.

## Decision

- Positive and not plateaued: expand local scoring before buying API review.
- Positive but plateaued: stop scaling this filter and inspect model/decoder.
- Flat or negative: do not spend DeepSeek budget on full-corpus cleaning.

This is a one-seed scaling screen. It cannot by itself establish a final
learning law.
