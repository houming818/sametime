# Full local parallel-corpus shadow cleaning

Date: 2026-08-21

Status: preregistered for serial execution after the 100K calibration, 1M
shadow pass, and 40K--500K downstream scale screen.

## Scope

Score all `14,170,275` valid rows of `WMT-Massive-ZHEN-14M` with the already
calibrated `BAAI/bge-reranker-v2-m3`. This is a non-destructive shadow filter:
the source corpus remains immutable and every observed score is retained.

The previous `NioScore-ZHEN-1M-v1` was a deterministic one-million-row sample,
not a completed full-corpus cleaning run. The 500K training ladder established
that accepted-row quantity still improved held-out NLL inside that sample. It
therefore permits this full scoring pass, but does not predetermine its quality.

## Execution design

- Process the source once in 250,000-row shards.
- Write compressed score manifests and accepted-candidate TSVs atomically.
- Write one `done.json` per shard with row counts and SHA-256 identities.
- On restart, skip completed shards rather than recomputing them.
- Keep the registered acceptance rule `score >= 0.98` unchanged.
- Do not delete low-score rows and do not mix monolingual or QA corpora into
  this bilingual scorer.
- Run serially on the protected `io` RTX 3090 without changing its power or
  frequency limits.

## Predict and gates

The run is operationally successful only if all 14,170,275 valid rows are
accounted for, shard indexes are contiguous, every shard has both score and
accepted-candidate hashes, and all values are finite. Any OOM, GPU fault,
missing shard, row-count mismatch, or damaged evidence stops finalization.

This run produces `NioScore-ZHEN-14M-v1`. A later `NioClean` release must be a
separate immutable manifest. No downstream quality claim is made until a
matched training probe compares the full-pool release with the existing 1M
sample release.
