# Full non-parallel corpus cleaning

Date: 2026-08-22

Status: executed; all registered full-corpus gates passed

Claim: `NIO-NONPAR-FULL-C01`

## Motivation

The 100K calibration ladder established a schema boundary:

- relation scoring is useful for general QA and medical QA;
- adjacent-span relation scoring is not a valid general quality ranker for
  monolingual prose.

The full run therefore must not apply one universal deletion threshold. It
creates immutable, content-addressed shadow metadata for every usable source
row. Source files are never rewritten or deleted.

## Registered corpus families

```text
QA:
  baike2018qa
  BELLE 1M and 2M
  webtext2019zh prompt/content pairs

medical QA:
  all Chinese-medical-dialogue-data CSV members

monolingual:
  news2016zh
  wiki_zh
```

QA and medical rows receive a frozen `bge-reranker-v2-m3` relation score plus
deterministic integrity flags. Monolingual rows receive integrity flags only;
their relation score is recorded as null.

## Storage protocol

Rows are streamed in source order into independent 100K gzip JSONL shards.
Every completed shard records:

- family, mode, shard number and global row interval;
- first and last stable source keys;
- per-source row counts;
- integrity-flag counts;
- finite score statistics and a 1000-bin histogram when scoring is enabled;
- compressed-file SHA-256 and decompressed-content SHA-256.

A `.done.json` is written only after the gzip member is closed and renamed.
On restart, an existing shard is skipped only after its row interval, boundary
keys and compressed SHA-256 have been verified.

## Predictions and gates

```text
F0 20K smoke for mono, QA and medical completes without OOM or non-finite data
F1 every valid yielded source row occurs exactly once in the full shard sequence
F2 all QA and medical scores are finite
F3 all shard indexes and global row intervals are contiguous
F4 every recorded compressed SHA-256 verifies after the run
F5 source text remains byte-equivalent after JSON decode; no correction is made
```

Failure of any gate stops release materialization. A low relation score is not
called an error row. It is only a ranking signal whose interpretation remains
schema-specific.

## Execution order

```text
20K smoke: mono -> QA -> medical
full scoring: medical -> QA
full integrity scan: mono
finalize and audit all three families
```

Medical runs before QA because it is the smaller relation corpus and provides
an early complete full-family result. The estimated relation-scoring wall time
on the 3090 is roughly 12-15 hours; actual throughput is recorded per shard.

## What this claim does not establish

This is a data-system claim, not a TreeHeap model-quality claim. After the full
metadata is complete, selected and coverage-matched raw views must be compared
under the same TreeHeap initialization, token budget, pretrain/task-train
pipeline and immutable evaluation set.

## Smoke result and formal queue

The registered 20K smoke completed on `io`:

| Family | Mode | Rows | Shards | Seconds | Mean score | Hash audit |
|---|---|---:|---:|---:|---:|---|
| mono | integrity only | 20,000 | 2 | 4.87 | n/a | pass |
| QA | relation score | 20,000 | 2 | 76.62 | 0.699765 | pass |
| medical | relation score | 20,000 | 2 | 74.54 | 0.909579 | pass |

All smoke gates passed. No source row was changed, QA and medical produced
exactly 20,000 finite scores each, shard intervals were contiguous, and all
six compressed-file hashes verified independently.

The full queue was submitted as taskd jobs `285` through `290`:

```text
285 medical full score
286 medical finalize and hash audit
287 QA full score
288 QA finalize and hash audit
289 mono full integrity scan
290 mono finalize and hash audit
```

Full artifacts are written under
`ara/data-quality/evidence/full_nonparallel_cleaning/formal_seed15106/` on
`io`. Large shards remain remote; compact summaries are pulled into the local
ARA after each family completes.

## Formal progress: 2026-08-23

The medical family and its independent finalizer have completed:

```text
valid QA records: 792099
shards:           8
finite scores:    792099
mean score:       0.768880
median bin:       approximately 0.959
hash audit:       pass
all F1-F5 gates:  pass
```

Approximate score-view sizes derived from the registered 1000-bin histogram:

| Score floor | Records | Fraction |
|---:|---:|---:|
| 0.90 | 476,619 | 60.17% |
| 0.95 | 414,549 | 52.34% |
| 0.98 | 331,730 | 41.88% |
| 0.99 | 270,894 | 34.20% |

These are candidate view sizes, not correctness rates. In particular, records
below `0.98` are not declared wrong. The full medical distribution is broad
and apparently bimodal: the approximate median is high while the mean is much
lower. Later training should therefore compare nested score views rather than
adopt one destructive threshold.

Integrity flags over the 792,099 parsed QA records were: 498 too-short, 201
URL/contact, 12 extreme-repetition, 5 low-CJK and zero mojibake. Physical CSV
line counts are not a record denominator because quoted answers may contain
embedded newlines; `792099` is the count of parsed, non-empty question-answer
records yielded by the registered reader.

The remainder of the queue subsequently completed without manual restart.

### Full QA result

```text
valid QA records: 8451252
shards:           85
finite scores:    8451252
mean score:       0.714814
median bin:       approximately 0.947
hash audit:       pass
all F1-F5 gates:  pass
```

Source composition was 1,422,653 `baike2018qa`, 2,917,412 BELLE and
4,111,187 `webtext2019zh` records.

| Score floor | Records | Fraction |
|---:|---:|---:|
| 0.90 | 4,774,996 | 56.50% |
| 0.95 | 4,187,496 | 49.55% |
| 0.98 | 3,381,543 | 40.01% |
| 0.99 | 2,761,430 | 32.67% |

Integrity flags were 137,480 too-short, 89,784 low-CJK, 50,611 URL/contact,
11,660 extreme-repetition and 388 mojibake. Flags may overlap and remain
metadata rather than automatic deletion commands.

### Full monolingual result

```text
valid prose records: 2984702
shards:              30
news2016zh:           2186399
wiki_zh:               798303
hash audit:           pass
all F1-F5 gates:      pass
```

The monolingual run intentionally produced no relation score. It recorded
126,020 URL/contact, 11,232 low-CJK, 8,476 extreme-repetition and 5,713
mojibake flags. The input reader already required at least 80 characters, so
the registered `too_short` flag count is zero.

## Final decision

Taskd jobs `285-290` all reached `done` in their registered order. An empty
taskd queue after `290` is normal queue exhaustion, not loss of the queued
tasks.

Across the three non-parallel families, 12,228,053 records now have immutable
shadow metadata:

```text
QA relation-scored:       8451252
medical relation-scored:   792099
monolingual integrity:     2984702
```

`NIO-NONPAR-FULL-C01` is supported as a data-pipeline claim. It authorizes
materializing nested, source/length-preserving training views. It does not yet
show that any view improves a TreeHeap model; that requires the separately
matched pretrain -> task-train -> proof comparison.
