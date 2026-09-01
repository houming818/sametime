# D10 recovery R1 evidence

Claim: `S3-STRUCTURAL-PROTOCOL-FULL-PIPELINE-D10-RECOVERY-R1`

- `canary_seed11001/`: isolated copy of the step-275000 D10 Stage B checkpoint,
  checkpoint audits, 2,000-step recovery trace, wakes and GPU records.
- The original D10 failure evidence and checkpoints remain unchanged.
- taskd `#344`: canary passed, advancing step 275000 to 277000 and cursor
  4400986 to 4432986.  Valid mean NLL changed from 4.15140285 to 4.14703905.
- taskd `#345`: formal 25,000-step segmented continuation is running.  Each
  segment restarts the Python/CUDA process while preserving model, optimizer,
  step and cursor state.
- Large checkpoint files remain on `io` and are intentionally excluded here.
