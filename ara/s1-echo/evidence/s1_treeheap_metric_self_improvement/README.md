# TreeHeap Metric Self-Improvement

Executed on io on 2026-07-13 with 8 seeds.  The same TreeHeap observer was
measured before and after adding the three-node TreeHeap diff/InfoNCE loss.

```text
positive distance: 1.3020 -> 0.2454
negative distance: 2.2377 -> 2.3369
margin:            0.9357 -> 2.0915
echo accuracy:     1.0000 -> 1.0000
held-out MRR:      0.4402 -> 0.4606
```

All registered self-comparison gates passed. See `summary.json` for per-seed
results and `trace.jsonl` for the complete before/after records.
