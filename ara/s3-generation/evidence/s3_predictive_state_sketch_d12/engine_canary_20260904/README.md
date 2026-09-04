# D12-E1 GPU throughput evidence

- Claim: `S3-PREDICTIVE-STATE-ENGINE-CANARY-D12-E1`
- Host: `io.grepcode.cn`, RTX 3090, 270W limit
- taskd: `360`
- Command: `bash ara/s3-generation/scripts/run_predictive_state_engine_canary_d12.sh`
- Decision: use batch 64 for the current D12/TreeHeap-63M graph

Stable target-token throughput after the 50-step warmup was 814.03/s at batch 16,
1,342.57/s at batch 32, and 2,189.63/s at batch 64. Batch 64 was 2.69x the baseline,
with 5.08 GiB peak memory, 74 C peak temperature, and 240.49W peak draw. Every arm
passed finite-value, frozen-source, update, and reload checks.

`throughput_summary.json` is the selection record. Each batch directory contains the paired
summaries, traces, GPU samples, exit code, and comparison output. Large checkpoints were
excluded from the local Git copy.
