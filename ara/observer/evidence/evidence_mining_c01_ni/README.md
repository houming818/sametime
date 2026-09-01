# TreeHeap Observer C01 on `ni`

Status: running / first formal cycle complete

## Runtime

- Host: `ni.grepcode.cn`
- CPU: Intel i5-4690, 4 cores
- Service: `treeheap-observer-c01.service`
- Start: 2026-08-31 22:29 Asia/Shanghai
- Input: `/home/nio/treeheap-observer/input/ara`
- Output: `/home/nio/treeheap-observer/output/c01`
- Schedule: one incremental scan every 900 seconds, 192 cycles (48 hours)
- Limits: `CPUQuota=50%`, `MemoryMax=2G`, `Nice=10`

On cycle 29 the first service instance stopped because SQLite attempted to use
the nearly full root-backed `/tmp` filesystem.  No evidence was corrupted.  The
runner now sets `TMPDIR=/home/nio/treeheap-observer/tmp` and SQLite uses memory
for temporary sorting; the remaining cycles resume under the original CPU and
memory limits.  Root filesystem cleanup remains a separate maintenance task.

The input snapshot contains ARA JSON, JSONL, and Dreams text only.  It excludes
checkpoints, tensors, datasets, archives, and other binary artifacts.

## Smoke

The local format smoke used the historical bilingual full-training evidence:

- 26 files;
- 390 parsed records;
- 246 metrics;
- 144 generations;
- 0 parse errors;
- 6 high-adjacent-repetition diagnostics.

## First formal cycle

- 969 source files / 82,065,971 bytes;
- 123,521 parsed records;
- 70,653 normalized metric values;
- 5,461 source/generation records;
- 0 parse errors;
- 685 findings requiring classification or review.

Observed findings include exact output reuse across distinct inputs, high
adjacent repetition, and non-finite historical fields.  A non-finite field can
mean either numerical failure or an intentionally undefined ablation metric;
C01 records it but does not infer which explanation is correct.

## Operational result

The full first scan took 4.68 seconds with a maximum resident set of 379,340
KiB.  This supports the engineering prediction that evidence mining can run on
the existing `ni` host without competing for the 3090 or threatening the
control plane.  Scientific interpretations remain open.

The remote SQLite database is intentionally not committed.  The compact
`summary.json`, `report.md`, and `run.jsonl` are the portable evidence.
