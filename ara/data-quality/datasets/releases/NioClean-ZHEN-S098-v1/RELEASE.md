# NioClean-ZHEN-S098-v1

This directory is an immutable dataset release record. The data files remain
on `io`; this release stores their content identities and provenance.

## Release identity

- root SHA-256: `04a17619a78dbfba964cabffe7029aa8e8193533fc0c1bb8e88a54c95093f1c0`
- schema: `nio.dataset-release.v1`

## Provenance

- `acceptance_rule`: `score>=0.98`
- `candidate_rows`: `514188`
- `dataset_seed`: `14106`
- `parent`: `WMT-Massive-ZHEN-14M`
- `policy`: `non-destructive-shadow-filter`
- `sample_seed`: `14105`
- `scored_pool`: `NioScore-ZHEN-1M-v1`
- `scorer`: `BAAI/bge-reranker-v2-m3`

## Members

| Logical file | Lines | Bytes | SHA-256 |
|---|---:|---:|---|
| `evaluation/shared_eval.tsv` | 2,049 | 429,644 | `094965f1252eab5682595f991f135c5949bc0d9283424e2f38a1251176a968e7` |
| `metadata/dataset_summary.json` | 34 | 956 | `a444deabac400872edd19889791d18f17ac05a252a14328aae7e62afafb7f6ba` |
| `scored_pool/manifest.jsonl` | 1,000,000 | 447,449,086 | `e909c4af4bffe52a93c89fbaec8ec92f8d0bc568792c075b2dd9113eb80bf806` |
| `training/purified_120000.tsv` | 121,002 | 27,371,699 | `3f2ba077db60f0645e195f5fa696572aab8d0b94aa65d5eceb944c638044520a` |
| `training/purified_160000.tsv` | 161,002 | 36,447,630 | `f3c729d1802113077e5445cc04ecd15ebdc0465572867f066053f4449615fb3f` |
| `training/purified_200000.tsv` | 201,002 | 45,480,409 | `72c4c5a76aecb5b3fc52f9b0157940ca781e0a6c24b17dfa2d531eecdcfde2ac` |
| `training/purified_40000.tsv` | 41,002 | 9,271,459 | `826a6823ccd7d3b739e2a378682d9cf72c05eb866d997148467c3e507395cdae` |
| `training/purified_80000.tsv` | 81,002 | 18,314,711 | `de897be34c8cabb4e3beee53619efddf2083d85c30848fcc28b8354eb2d5b95b` |

Changing any member, scorer, threshold, ordering, seed, source snapshot, or
normalization rule requires a new release version. Never overwrite this record.
