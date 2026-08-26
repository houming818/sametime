# S3 Recursive Depth Probability Exposure D03

Claim: `S3-RECURSIVE-DEPTH-PROBABILITY-EXPOSURE-D03`

Taskd:

- `316`: failed during smoke because variable-depth examples were indexed as fixed-depth;
- `317`: completed after explicitly treating shorter trees as saturated at leaf.

Evidence contract:

- frozen checkpoints only;
- no optimizer, backward pass, or parameter update;
- 3 seeds: `10101`, `10102`, `10103`;
- 256 identical WMT test rows per seed;
- native, runtime identity, pair-break-depth-0, and source-shuffle conditions;
- full-depth capped READ checked against the original READ on identical batches.

Result:

```text
P0 evidence/equivalence: PASS
P1 recursive exposure:  PASS
P2 seed replication:    PASS
P3 structural cause:    PASS
decision: supported_frozen_recursive_exposure
```

The result supports a reproducible, structure-dependent cumulative-depth effect
on the vocabulary probability field. It does not establish fixed linguistic
semantics for individual depths or a usable root-only representation.

Primary files:

- `self_test/summary.json`
- `smoke/summary.json`
- `formal/summary.json`
- `formal/seed_10101/summary.json`
- `formal/seed_10102/summary.json`
- `formal/seed_10103/summary.json`
- `formal.log`
