# Local purification scale ladder

Date: 2026-08-18

Status: completed, one-seed scaling evidence recorded

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

## Result

All five arms completed from the same initial checkpoint and seed. Each arm
used the same 1,024-row validation set and 1,024-row test set.

| Purified rows | Test NLL | PPL | Repetition | EN->ZH BLEU | ZH->EN BLEU | Shuffle delta | Pair-break delta | Runtime identity delta | Gain from previous arm |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 40K | 7.2390 | 1392.73 | 0.1223 | 1.7427 | 0.7631 | +1.1336 | +0.1979 | -0.0328 | - |
| 80K | 6.7582 | 861.13 | 0.0909 | 3.9104 | 1.2082 | +1.5239 | +0.3380 | -0.0208 | 0.4808 |
| 120K | 6.5127 | 673.67 | 0.1004 | 4.6215 | 1.3190 | +1.8557 | +0.3657 | -0.0426 | 0.2455 |
| 160K | 6.2796 | 533.55 | 0.0797 | 4.5340 | 1.6655 | +1.9451 | +0.4640 | -0.0326 | 0.2332 |
| 200K | 6.1378 | 463.04 | 0.0672 | 4.6925 | 2.0782 | +2.1308 | +0.4825 | -0.0240 | 0.1417 |

The four marginal NLL gains are `0.4808`, `0.2455`, `0.2332`, and `0.1417`.
The last two are both above the preregistered `0.05` plateau threshold, so the
screen has **not** reached its operational boundary by 200K rows.

## Evidence-limited conclusion

Increasing this high-confidence parallel subset from 40K to 200K consistently
improved held-out NLL and PPL. Repetition also fell overall, although neither
repetition nor BLEU was strictly monotonic at every intermediate arm. Source
shuffle and pair-break interventions became more damaging as scale increased,
which is compatible with stronger use of source and pair alignment.

The runtime identity intervention remains slightly better than the native
Butterfly route at every arm. This experiment therefore supports a purified
data quantity effect, but it does **not** support a claim that the native
TreeHeap route is already superior to the identity route. Generation BLEU also
remains low, so the result is a training-signal milestone rather than a product
quality milestone.

The preregistered rule permits another scale test. Automatic expansion is held,
however, until the next dataset release and teacher-calibration protocol are
registered. This avoids mixing a scale change with a scorer or corpus-family
change.

Machine-readable aggregation and the plotted curves are stored beside the raw
summaries in `formal_seed14108/comparison.json` and
`formal_seed14108/scale_ladder.png`.
