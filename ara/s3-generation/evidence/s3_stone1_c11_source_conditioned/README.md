# C11 Evidence

- Host/task: `io`, taskd task `53`
- Code commit: `74a65f4`
- Runtime: 27,041 seconds (7.51 hours)
- Updates: 10,000
- Parameters: 50,679,947
- Remote checkpoint: `/home/nio/log/holds/SameTime/ara/s3-generation/evidence/s3_stone1_c11_source_conditioned/checkpoint_latest.pt`
- Checkpoint bytes: 608,222,139

`summary.json` and `trace.jsonl` are the machine evidence. The large checkpoint
remains on io and is intentionally not copied into Git.

## Audit conclusion

Matched sources beat shuffled and empty sources for every tested context length,
so source conditioning is supported. Free generation remains highly repetitive.
The automatic `unique_output_fraction=1.0` only says the eight complete strings
were not identical; it does not mean each string was healthy.

Post-hoc character distinct-n over the eight outputs:

| metric | mean |
|---|---:|
| distinct-1 | 0.2774 |
| distinct-2 | 0.3568 |
| distinct-4 | 0.4514 |

The lowest per-output distinct-4 was `0.1111`. This post-hoc metric is a
diagnostic, not a preregistered success gate. See the logic document for the
claim-level interpretation.
