# S3 Recursive Depth Free Generation Length D05

Claim: `S3-RECURSIVE-DEPTH-FREE-LENGTH-D05`

## Contract

- Frozen READ checkpoints: seeds `10101`, `10102`, `10103`
- Same 128 WMT test rows per seed
- READ depths: `5`, `6`, `7`
- One greedy run and eight temperature `0.8`, top-p `0.9` samples per row/depth
- Common `max_output=256`; EOS alone determines natural stopping
- No target read, teacher forcing, target grouping, or depth-specific length cap

## Tasks

- `319`: implementation self-test, done
- `320`: first smoke with common cap 64, done
- `321`: censoring smoke with common cap 256, done
- `322`: three-seed formal run, done in 151.8 seconds

## Result

```text
seed 10101 sample median length d5/d6/d7 = 256/13/12
seed 10102 sample median length d5/d6/d7 = 256/14/13
seed 10103 sample median length d5/d6/d7 = 256/11/11

seed 10101 cap rate d5/d6/d7 = 80.47%/6.15%/0.20%
seed 10102 cap rate d5/d6/d7 = 87.50%/6.93%/0.68%
seed 10103 cap rate d5/d6/d7 = 87.50%/6.25%/0.00%
```

The exact shallow/full length ratio is right-censored, but the proposed direction
is rejected: the depth-5 median has a lower bound of 256 while the depth-7 median
is 11-13. Shallower READ does not naturally produce a shorter sentence in these
checkpoints. It destabilizes EOS and often enters repetitive high-frequency loops.

## Gates

```text
P0 implementation/evidence contract: PASS
P1 natural monotonic length response: FAIL
P2 near-binary 1/4, 1/2, 1 scale: FAIL
P3 <=10% cap rate at every depth: FAIL
P4 cross-seed support: FAIL
```

The machine summary conservatively reports `inconclusive_length_censored` for
exact ratios. The one-sided bound is sufficient to reject the shorter-at-shallow-
depth direction.

## Integrity

All seeds use test-row SHA-256:

```text
f8fc7061ac388886bbf1ee0b7770275ba694371a4d5fece7e56f42c9f4fa304e
```

Checkpoint state SHA-256:

```text
10101 5547eeced555989cacc2e663441729fbb1b971969852eae86165ff3d193d3a1e
10102 21f9d123d1135b73dbb3f2803da014aac64ff0a9c789c2d4396c6e52766f701d
10103 e78643f646e8e9abc9ade6a9bef54f45fbd27a658109be9067aa9631c6c99137
```

Primary files:

- `formal/summary.json`
- `formal/seed_10101/summary.json` and `records.json`
- `formal/seed_10102/summary.json` and `records.json`
- `formal/seed_10103/summary.json` and `records.json`
- `taskd-322-formal.log`
