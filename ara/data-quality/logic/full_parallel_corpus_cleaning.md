# Full local parallel-corpus shadow cleaning

Date: 2026-08-21

Status: completed and structurally validated.

## Scope

Score all `14,170,275` valid rows of `WMT-Massive-ZHEN-14M` with the already
calibrated local snapshot `/home/nio/models/bge-reranker-v2-m3`. This is a
non-destructive shadow filter:
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

## Result

The formal run completed on 2026-08-22 in 50,703.6 seconds (14.08 hours).
All 14,170,275 source rows were scored in 57 contiguous shards. The final
shard contains 170,275 rows; every preceding shard contains 250,000 rows.

- Rows accepted at `score >= 0.98`: 7,304,358 (51.5470%).
- Completed shard indexes: 0 through 56, without gaps.
- Every member records both manifest and accepted-candidate SHA-256 values.
- Aggregate flags: 224,540 mojibake, 1,858,573 direction-suspect,
  163 length-ratio-suspect, and 2,378,766 number-mismatch observations.
- Task 252 completed without OOM, non-finite values, GPU failure, or row-count
  mismatch; task 253 independently finalized the aggregate summary.

The evidence supports successful, reproducible full-corpus scoring. It does
not by itself prove that all 7,304,358 accepted pairs are correct translations:
the reranker score is a compatibility signal, not a calibrated probability or
human correction. The next valid claim requires a matched downstream training
comparison against the earlier sample-derived dataset.
