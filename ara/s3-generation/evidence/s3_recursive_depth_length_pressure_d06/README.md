# S3 Recursive Depth Length Pressure D06

Claim: `S3-RECURSIVE-DEPTH-LENGTH-PRESSURE-D06`

## Contract

- Frozen READ checkpoints: seeds `10101`, `10102`, `10103`
- Same 128 WMT test rows and checkpoint hashes as D05
- Depth budgets derived only from encoded source length:
  `Bd=max(2, ceil((Lx-2)*2^(d-7)))`
- EOS mixture: `P'=(1-g)P+g*delta_EOS`
- `g=sigmoid(20*(progress-0.85))`
- One greedy generation and eight temperature `0.8`, top-p `0.9` samples
- Common global `max_output=128`; per-row hard capacity is a safety wall
- No target read and no teacher forcing during generation

## Tasks

- `323`: pressure algebra self-test, done
- `324`: one-seed smoke, done in 14.8 seconds
- `325`: three-seed formal, done in 55.1 seconds
- `326`: deterministic rerun after correcting only the decision label, done in 55.4 seconds

## Formal Result

```text
seed 10101 median budget d5/d6/d7 = 4/8/15
seed 10101 median length d5/d6/d7 = 3/6/9

seed 10102 median budget d5/d6/d7 = 4/8/15
seed 10102 median length d5/d6/d7 = 3/6/10

seed 10103 median budget d5/d6/d7 = 4/8/15
seed 10103 median length d5/d6/d7 = 3/6/9
```

Paired generation:

```text
mean monotonic fraction       = 0.9590
mean strict monotonic fraction = 0.8031
mean median L5/L7             = 0.2831
mean median L6/L7             = 0.5608
```

All seed/depth cells had `pre_wall_eos_rate=1.0` and `hard_wall_rate=0.0`.
The differentiable EOS pressure, not hard truncation, produced the length response.

## Gates

```text
P0 implementation/evidence contract: PASS (3/3)
P1 depth-length coupling:             PASS (3/3)
P2 near-binary capacity response:     PASS (3/3)
P3 soft pressure effectiveness:       PASS (3/3)
P4 full-depth quality safety:         FAIL (0/3)
P5 cross-seed replication:            PASS
```

Decision:

```text
supported_soft_pressure_coupling_with_quality_cost
```

Depth-7 BLEU4 fell from a D05 baseline mean of `5.817` to `2.741`. Repetition
fell rather than rose, so the main cost is premature removal of translation detail.
The pressure mechanism works, but its fixed calibration is not product-safe.

## Integrity

Test-row SHA-256:

```text
f8fc7061ac388886bbf1ee0b7770275ba694371a4d5fece7e56f42c9f4fa304e
```

Primary files:

- `formal/summary.json`
- `formal/seed_10101/summary.json` and `records.json`
- `formal/seed_10102/summary.json` and `records.json`
- `formal/seed_10103/summary.json` and `records.json`
- `taskd-326-formal-r1.log`
