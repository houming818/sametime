# Non-parallel local-judge calibration and data ladder

Date: 2026-08-22

Status: executed; QA and medical ladders authorized, mono ladder rejected

Claims: `NIO-NONPAR-CAL-C01`, `NIO-NONPAR-LADDER-C01`

## Purpose

Test whether the local BGE relation score plus deterministic integrity flags
can rank useful examples inside mono, general-QA and medical-QA corpora. The
test separates data-quality evidence from later TreeHeap training evidence.

No source row is rewritten or deleted. A selected ladder is a reproducible
view over immutable rows, not a corrected corpus.

## Stage A: 100K shadow pools

For seed `15103`, independently sample and score exactly 100,000 rows per
family:

| Family | Sources | Relation |
|---|---|---|
| `mono` | news2016zh, wiki_zh | adjacent span -> next span |
| `qa` | baike2018qa, BELLE, webtext2019zh | prompt -> answer |
| `medical` | Chinese-medical CSV files | question -> answer |

The scorer is the already-smoked local `bge-reranker-v2-m3`. Each pool also
contains a length-bucketed shuffled-right control, source identity, row id,
integrity flags and a content hash. Passing requires exactly 100,000 finite
observed/control values and the Stage-1 observed-vs-shuffled AUC gate.

## Stage B: Qwen calibration labels

From each 100K pool, sort records separately inside every source and divide
them into ten equal BGE rank bins. Deterministically sample 500 records per
family, balanced across source and rank bin. This produces 1,500 Qwen reviews
without treating the review set as a natural corpus-frequency sample.

The frozen local judge is `Qwen/Qwen3-8B`, BF16, greedy decoding, thinking
disabled. It emits the schema registered in `local_qwen_judge_smoke.md`.

Derived labels:

```text
strict_usable = relation == matched and text_quality == usable
acceptable    = relation in {matched, partial} and text_quality != corrupt
mismatch      = relation == mismatch
```

These are Qwen calibration labels, not ground truth. Every medical judgment
must remain `medical_unverified`.

Calibration gates:

```text
C0 1500/1500 source rows and BGE scores have stable hashes
C1 at least 1470/1500 outputs satisfy the exact JSON schema
C2 every medical output remains medical_unverified
C3 source text is unchanged
C4 per-family acceptable and mismatch rates are reported for all ten bins
```

Prediction, not a hard infrastructure gate: higher BGE rank bins should show
increasing acceptable rate and decreasing mismatch rate. A family that lacks
this ordering does not receive an automatic score-based ladder.

## Stage C: nested data ladders

For a family that passes the ordering audit, create a deterministic quality
order while preserving source and length coverage:

1. Partition the 100K pool by source and `floor(log2(combined_length))`.
2. Inside each stratum, rank rows by integrity class first, BGE score second,
   and stable row id last.
3. Convert each within-stratum rank to a percentile.
4. Merge strata by percentile, yielding one coverage-preserving global order.
5. Ladder views are prefixes of that order, so nesting is exact.

Integrity class places mojibake, extreme repetition and too-short records
behind intact rows. URL/contact and low-CJK flags are reported but are not
universal deletion rules.

Registered sizes:

```text
mono:     10K -> 20K -> 40K -> 80K
qa:       10K -> 20K -> 40K -> 80K
medical:   5K -> 10K -> 20K -> 40K
```

For each family, construct a matched raw-control order from the same strata
using seed `15104`. Each level records row count, SHA-256, exact subset checks,
source proportions, length distribution, score quantiles, integrity flags,
character diversity and Qwen-label estimates.

## Data-ladder predictions

```text
L0 all row counts, hashes and nested-subset checks are exact
L1 selected source-distribution JS divergence from the 100K pool <= 0.02
L2 selected length-distribution JS divergence from the 100K pool <= 0.02
L3 selected mismatch rate is lower than its matched raw control
L4 selected strict-usable rate is higher than its matched raw control
L5 the quality advantage may shrink as the selected prefix approaches 80K/40K
```

`L3` and `L4` must hold for at least two adjacent sizes before that family is
allowed into a TreeHeap training ladder. Failure is a result: it means the
current local-cleaner ranking is not useful for that corpus schema.

## Later training gate

Passing this data ladder authorizes a separate, matched-compute experiment:

```text
selected view vs raw control
-> TreeHeap natural-text pretrain or response task train
-> immutable held-out evaluation
-> NLL/PPL, repetition, free generations and structural interventions
```

The training experiment must not skip the registered pretrain -> task-train ->
proof -> report pipeline. Data-ladder success alone is not a TreeHeap model
claim.

## Executed result

The run used taskd jobs `274` through `278` on `io` under the registered
seeds. The three shadow scorers each processed exactly 100,000 rows and passed
their finite-value and shuffled-control gates:

| Family | Observed median | Shuffled median | AUC |
|---|---:|---:|---:|
| mono | 0.968856 | 0.010860 | 0.870648 |
| QA | 0.920361 | 0.000138 | 0.953783 |
| medical | 0.962953 | 0.000020 | 0.987869 |

These values establish that the local relation scorer detects corpus
structure. They do not establish factual correctness or justify one shared
threshold across the three schemas.

Qwen3-8B then reviewed 500 source/score-bin-balanced examples per family.
All 1,500 outputs satisfied the registered schema, all medical rows remained
`medical_unverified`, and all source hashes were unchanged. The run took
2,264.14 seconds and peaked at 17,434,524,672 GPU bytes.

| Family | Strict usable | Acceptable | Mismatch |
|---|---:|---:|---:|
| mono | 0.270 | 0.932 | 0.068 |
| QA | 0.324 | 0.568 | 0.432 |
| medical | 0.034 | 0.436 | 0.564 |

The low medical `strict_usable` rate is expected under this label definition:
the judge is forbidden to certify medical truth. Medical authorization below
therefore concerns question-answer relation and text integrity only.

## Ladder result

All selected and raw-control orders contain exactly 100,000 unique rows.
All six gzip streams passed integrity checks, and SHA-256 computed over their
decompressed JSONL matched the registered summaries. Every ladder is an exact
prefix ladder. Source and length JS divergence remained far below the `0.02`
gate.

### QA

| Rows | Selected acceptable | Raw acceptable | Selected mismatch | Raw mismatch |
|---:|---:|---:|---:|---:|
| 10K | 0.7800 | 0.5636 | 0.2200 | 0.4364 |
| 20K | 0.6465 | 0.5743 | 0.3535 | 0.4257 |
| 40K | 0.6602 | 0.5302 | 0.3398 | 0.4698 |
| 80K | 0.6065 | 0.5572 | 0.3935 | 0.4428 |

QA passes `L3/L4`: the selected order has lower estimated mismatch and higher
strict-usable rate than the matched raw control at all four sizes.

### Medical

| Rows | Selected acceptable | Raw acceptable | Selected mismatch | Raw mismatch |
|---:|---:|---:|---:|---:|
| 5K | 0.6538 | 0.5263 | 0.3462 | 0.4737 |
| 10K | 0.6122 | 0.3810 | 0.3878 | 0.6190 |
| 20K | 0.6765 | 0.4409 | 0.3235 | 0.5591 |
| 40K | 0.6176 | 0.3889 | 0.3824 | 0.6111 |

Medical passes `L3/L4` for relation quality. This is not medical validation.

### Mono

| Rows | Selected acceptable | Raw acceptable | Selected mismatch | Raw mismatch |
|---:|---:|---:|---:|---:|
| 10K | 0.9615 | 0.9545 | 0.0385 | 0.0455 |
| 20K | 0.9352 | 0.9432 | 0.0648 | 0.0568 |
| 40K | 0.9444 | 0.9457 | 0.0556 | 0.0543 |
| 80K | 0.9352 | 0.9369 | 0.0648 | 0.0631 |

Mono fails `L3/L4`. Adjacent-span BGE score is not a useful general-purpose
quality ordering for continuous monolingual text: most low-scoring adjacent
spans remain legitimate prose, so relation similarity and corpus usability are
different variables here.

## Decision

`NIO-NONPAR-CAL-C01` is supported as a local calibration result. The scorer
has useful schema-dependent signal, and Qwen3-8B is operational as a frozen
third-layer local judge on the 3090.

`NIO-NONPAR-LADDER-C01` is partially supported:

```text
QA ladder:      authorized for a later matched-compute training experiment
medical ladder: authorized for relation experiments, not factual claims
mono ladder:    rejected; keep raw or design a mono-specific quality signal
```

No TreeHeap training claim follows from this result. The next training run
must compare selected versus coverage-matched raw data under identical model,
initialization, token budget and evaluation sets.
