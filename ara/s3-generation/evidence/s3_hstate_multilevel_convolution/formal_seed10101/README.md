# C11 Formal Training Evidence

Claim: `S3-HSTATE-MULTILEVEL-CONV-C11`

Taskd job `172` ran on `io` for 25,000 optimizer steps and completed normally.
Task `173` copied the full directory, including checkpoints, to:

```text
/mnt/nas/ara-evidence/s3_hstate_multilevel_convolution/formal_seed10101
```

Git retains `summary.json` and `trace.jsonl`. The approximately 540 MB progress
checkpoint and final checkpoint remain on `io` and NAS.

Important boundary: `summary.json` reports the deterministic validation split
used for checkpoint selection. Independent test results are stored separately
under `../test_audit_seed10101/`.
