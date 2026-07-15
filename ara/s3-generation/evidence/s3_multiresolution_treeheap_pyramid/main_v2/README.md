# Multiresolution TreeHeap Pyramid

Claim `S3-TREEHEAP-PYRAMID-C01` is supported as a bounded mechanism claim by a
three-seed, one-million-real-text-block-per-seed run.

```text
k=0/8/16/32/64 MSE  1.9334 / 0.8803 / 0.7622 / 0.6516 / 0.5526
k=64 address shift  3.3147 MSE
k=64 token top-1    0.9964
k=64 sequence exact 0.8268
root NLL delta       0.0
```

See `result_analysis.md` for interpretation and boundaries, `summary.json` for
the aggregate, and the per-seed metrics/traces/checkpoints for raw evidence.

The run does not establish entropy-coded compression, learned scale
specialization, or universal superiority over flat architectures.
