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

Question-answer, instruction, monolingual text, and medical corpora are present
on `io` but are not yet Nio dataset releases. They retain their upstream names
until task-specific cleaning and immutable manifests are completed.
