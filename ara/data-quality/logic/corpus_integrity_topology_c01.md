# Full corpus integrity and duplicate-topology audit

Date: 2026-09-01

Status: supported operational / formal run complete

Claim: `NIO-CORPUS-INTEGRITY-C01`

## Question

Before another product-scale TreeHeap run, do the released mono and parallel
training views have a reproducible physical identity and a measured duplicate,
length, and language-orientation profile?

This is an operational data claim, not a model-quality claim.  The audit does
not rewrite, filter, rank, or delete a source row.

## Immutable inputs

| Release | File | Registered rows | Registered bytes |
|---|---|---:|---:|
| `NioText-ZH-Integrity-2985K-v1` | `data.jsonl` | 2,972,976 | 5,147,059,454 |
| `NioClean-ZHEN-S098-7M-v2` | `pairs.tsv` | 7,304,358 | 1,535,169,936 |

The formal run computes file SHA-256 while parsing.  The resulting hashes,
sizes, row counts, host, command, and timestamps are part of the evidence.

## Measurements

For the mono JSONL release:

- JSON and schema failures;
- blank text and duplicate IDs;
- 128-bit BLAKE2b text-digest duplicates;
- source-prefix counts derived from the registered `id` field;
- UTF-8 replacement characters, text-length histogram and quantiles;
- aggregate Han and ASCII-letter ratios.

For the bilingual TSV release:

- malformed rows and blank source/target fields;
- pair-, source-, and target-digest duplicates;
- source and target length distributions;
- aggregate source-Han and target-ASCII-letter ratios;
- descriptive low-orientation counts.  These counts are diagnostics, not
  automatic declarations that a row is wrong.

Digest equality is treated as practical exact equality for this audit.  The
128-bit collision probability is negligible at this scale, but the report
must call the measurement `digest duplicates`, not a mathematical proof of
string identity.

## Gates

```text
G0 both inputs exist and their byte sizes match the preregistration
G1 parsed row counts equal 2,972,976 and 7,304,358
G2 JSON/schema/TSV malformed counts are zero
G3 every numeric output is finite and both SHA-256 values are recorded
G4 progress JSONL, per-release summaries and combined summary are complete
G5 the source files retain their original byte size and modification time
```

A gate failure stops interpretation and preserves the partial evidence.  No
threshold for filtering is selected from this run.

## Compute assignment

The formal scan runs on CPU node `n14.grepcode.cn` under `nice`, with at most
8 GiB memory per scanner and no GPU dependency.  Mono and parallel scans may
run concurrently because they read independent files.  `io` remains dedicated
to D10 GPU training; `ni` remains dedicated to the evidence observer.

## Expected value

The result provides the physical data baseline for later pretrain and task
training reports.  It can reveal accidental duplication or schema damage, but
it cannot by itself establish that cleaner data improves NLL, BLEU, generation,
or TreeHeap structure use.

## Formal result

The formal `n14` run completed both scans on the preregistered releases.  All
five gates passed.

| Measurement | Mono release | Parallel release |
|---|---:|---:|
| Rows | 2,972,976 | 7,304,358 |
| Bytes | 5,147,059,454 | 1,535,169,936 |
| Parse/schema failures | 0 | 0 |
| Blank records | 0 | 0 source / 0 target |
| Full-record digest duplicates | 100,024 (3.3644%) | 267 (0.003655%) |
| Source-side digest duplicates | - | 590,214 (8.0803%) |
| Target-side digest duplicates | - | 361,523 (4.9494%) |
| Scan time | 461.46 s | 409.41 s |

Physical identities:

```text
data.jsonl  04a90d88b51755561645d0fec962fc8bd5e642d099423348417de9318e22c94e
pairs.tsv   299134867398720cc6d407eadd6de4fb237812319d113fbe12071758e79d92c8
```

The mono release contains a measurable repeated-document component.  The
parallel release has almost no repeated complete pairs, while repeated source
or target sides are much more common.  That asymmetry is compatible with
templates and one-to-many or many-to-one translation relations; it is not a
license to remove those rows without a matched downstream experiment.

The source files retained their registered size and modification timestamp
after the scan.  Evidence:

```text
ara/data-quality/evidence/corpus_integrity_topology_c01_n14/
```
