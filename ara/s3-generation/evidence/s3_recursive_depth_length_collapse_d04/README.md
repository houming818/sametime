# S3 Recursive Depth Length Collapse D04

Claim: `S3-RECURSIVE-DEPTH-LENGTH-COLLAPSE-D04`

Taskd: `318` (`done`, 145.8 seconds)

Frozen checkpoints: seeds `10101`, `10102`, `10103`.

Formal matrix:

```text
READ depth: 5, 6, 7
target block size: 4, 2, 1
nominal output length: 1/4, 1/2, 1
rows: 256 identical WMT test rows per seed
```

Gates:

```text
P0 implementation/evidence contract: PASS
P1 shallow shorter-output benefit: FAIL
P2 exact 4/2/1 scale alignment: FAIL
P3 cross-seed replication: PASS
```

All three seeds preferred `block=1` at every READ depth. The shorter outputs
had the intended address counts but worse NLL, lower block probability mass,
and higher repetition. This rejects zero-training adjacent-token bag collapse
for the current checkpoint. It does not reject a learned low-resolution target
protocol.

Primary evidence:

- `formal/summary.json`
- `formal/seed_10101/summary.json`
- `formal/seed_10102/summary.json`
- `formal/seed_10103/summary.json`
- `formal.log`
