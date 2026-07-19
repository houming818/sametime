# Decoder depth-growth pilot

Completed on `io.grepcode.cn` on 2026-07-19 with exit code 0.

- Claim: `S3-DECODER-DEPTH-GROWTH-C01`
- Result: rejected for the current FOLD/READ mechanism
- Passed: long-context coverage and finite execution
- Failed: depth growth, structural advantage, link causality, unseen-arity transfer
- Primary artifact: `summary.json`
- Training trace: `stdout.log`
- Queue and GPU trace: `launcher.log`, `gpu_before.csv`, `gpu_after.csv`

The three large checkpoints remain on `io` in this directory and are omitted
from Git. They are not needed to audit the recorded metrics.
