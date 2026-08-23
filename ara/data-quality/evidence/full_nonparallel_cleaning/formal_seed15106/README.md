# Full non-parallel cleaning evidence

Taskd jobs `285-290` completed on `io` on 2026-08-23.

Compact summaries are retained here for:

- `medical/summary.json`: 792,099 relation-scored records;
- `qa/summary.json`: 8,451,252 relation-scored records;
- `mono/summary.json`: 2,984,702 integrity-scanned records.

All summaries report exact row intervals, finite-score counts where
applicable, source counts, integrity flags, compressed-file hashes and the
registered `F1-F5` gates.

The large gzip JSONL shards remain on `io` at:

```text
/home/nio/log/holds/SameTime/ara/data-quality/evidence/
  full_nonparallel_cleaning/formal_seed15106/
```

Every shard hash was recomputed by the finalizer. The source corpora were not
rewritten or deleted. Scores below any displayed threshold remain available;
thresholds define candidate training views, not ground-truth error labels.
