# Nio dataset registry

This directory records immutable identities for datasets used by SameTime.
Large data stays on `io` or NAS; Git stores release manifests, checksums, and
provenance rather than copying the corpus.

## Naming

```text
<family>-<language/task>-<selection>-<size>-<version>
```

Examples:

- `NioClean-ZHEN-S098-200K-v1`
- `NioQA-ZH-Baike-v1`
- `NioText-ZH-Wiki-v1`

Only a dataset with a directory under `releases/` and a root SHA-256 is a
released dataset. A filename, row count, or informal label is not sufficient.

## Version rule

Any change to source files, cleaning model, threshold, normalization,
deduplication, ordering, split, or seed creates a new version. Existing release
directories are append-only and must not be regenerated in place.

## Current releases

| Release | Status | Purpose |
|---|---|---|
| `NioClean-ZHEN-S098-v1` | released | WMT Massive shadow-score pool and nested purified training sets |

## Completed evidence pools

These pools have completed full-corpus computation and hash audit. They are
usable as evidence-backed source pools, but are not immutable releases until a
directory under `releases/` records their root hash and members.

| Dataset ID | Records | Status | Evidence |
|---|---:|---|---|
| `NioScore-ZHEN-14M-v1` | 14,170,275 | full bilingual relation scoring complete | `full_parallel_cleaning/formal_14m_v1` |
| `NioScore-ZH-QA-8451K-v1` | 8,451,252 | full QA relation scoring complete | `full_nonparallel_cleaning/formal_seed15106/qa` |
| `NioScore-ZH-MedQA-792K-v1` | 792,099 | full medical-QA relation scoring complete | `full_nonparallel_cleaning/formal_seed15106/medical` |
| `NioAudit-ZH-Text-2985K-v1` | 2,984,702 | full monolingual integrity scan complete | `full_nonparallel_cleaning/formal_seed15106/mono` |

The medical pool contains relation metadata, not medical truth labels. The
monolingual pool intentionally has no adjacent-span relation score because the
calibration experiment rejected that signal as a general prose-quality ranker.

## Planned releases

The following names are reserved for the next materialization stage. `S090`,
`S095` and `S098` are score-view identifiers, not claims that excluded records
are wrong.

| Planned release | Parent pool | Expected role |
|---|---|---|
| `NioClean-ZHEN-S098-7M-v2` | `NioScore-ZHEN-14M-v1` | full high-relation bilingual training view |
| `NioQA-ZH-S090-v1` | `NioScore-ZH-QA-8451K-v1` | broad QA task-training view, about 4.775M rows |
| `NioQA-ZH-S095-v1` | `NioScore-ZH-QA-8451K-v1` | middle QA view, about 4.187M rows |
| `NioQA-ZH-S098-v1` | `NioScore-ZH-QA-8451K-v1` | narrow QA view, about 3.382M rows |
| `NioMedQA-ZH-S090-v1` | `NioScore-ZH-MedQA-792K-v1` | broad medical-relation view, about 477K rows |
| `NioMedQA-ZH-S095-v1` | `NioScore-ZH-MedQA-792K-v1` | middle medical-relation view, about 415K rows |
| `NioMedQA-ZH-S098-v1` | `NioScore-ZH-MedQA-792K-v1` | narrow medical-relation view, about 332K rows |
| `NioText-ZH-Integrity-2985K-v1` | `NioAudit-ZH-Text-2985K-v1` | monolingual pretraining view with deterministic integrity policy |

After these task-specific releases exist, two pipeline-level mixtures may be
registered separately:

```text
NioPretrain-ZH-Core-v1  = monolingual pretraining release plus registered auxiliaries
NioTask-ZH-QA-v1       = selected general-QA task-training release
```

Their composition, tokenizer snapshot, ordering and mixture weights are not
yet registered, so these two names remain plans rather than datasets.
