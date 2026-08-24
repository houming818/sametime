# STONE-2 C03-D02 task-gradient Gram audit

- Host: `io`
- Valid task: `307`
- Checkpoint: C03 PT `checkpoint_best.pt`
- Checkpoint SHA-256: `24d2b03c7a5f7441e3169ed40741544f62b1ce7db3e37ade7021558c226cb202`
- Rows: 24 unique WMT test rows in three batches
- Width contract: source 65..128 pieces, 128-leaf TreeHeap, READ depths 0..7
- Result: exact untie/reconstruction passed; grouped READ conflict supported;
  grouped branch conflict not supported.

Key measurements:

```text
READ coarse:middle median cosine = -0.159591
READ coarse:middle negative fraction = 0.666667
READ cancellation ratio median = 0.700306
branch cancellation ratio median = 0.820232
max READ reconstruction relative error = 2.24e-7
max branch reconstruction relative error = 3.96e-7
```

Decision: `register_grouped_read_matched_smoke`. This audit does not authorize
formal training and does not retroactively pass C03 S7.

Files:

- `pt_summary.json`: valid preregistered result;
- `impl_smoke.json`: two-row implementation smoke only.
