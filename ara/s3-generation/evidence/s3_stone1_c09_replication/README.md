# STONE-1 C09 Evidence

Claim: `S3-STONE1-FROZEN-PLATFORM-REPLICATION-C09`

Host: `io`, NVIDIA GeForce RTX 3090

Result: `stone1_supported_on_frozen_platform`

The formal run used the frozen `S3-STONE1-C09-PLATFORM-V1` contract:
one million WMT Massive sentence pairs, 2K validation, 2K test, batch 64,
15,625 updates, and seeds `71901/71902/71903`.

All registered Q/S/E gates passed. See `summary.json` for complete metrics,
per-depth interventions, traces, examples, platform hashes, and parameter
counts.

The decoder checkpoints are not committed to Git. They are archived at:

```text
/mnt/nas/ara/s3_stone1_c09_replication/checkpoints/
```

| File | Bytes | SHA-256 |
|---|---:|---|
| `decoder_eos_seed71901.pt` | 156,825,809 | `8eacfd273a7b085de69be2ff2b83a1ec76c12ed07b45a5ceb98aeab9b46fd7e7` |
| `decoder_eos_seed71902.pt` | 156,825,809 | `96b7389ec276612955cc5e2c8c45ac3cf235d40f14fc50878c3b0b230362b23b` |
| `decoder_eos_seed71903.pt` | 156,825,809 | `406d3b8837a68dc7c30037210ae91f7e371bb9036b1334cb5598b5de38adf633` |

The taskd run was task `19`, exited `0`, and occupied the execution slot for
`9,969.3` seconds. The scientific script reported `9,966.24` seconds.

The submitted task did not set `CODE_COMMIT`, so generated `summary.json`
records `git_commit: unknown`. The preregistered files used by the run were
synced from local commit `20736ad`; their platform content is independently
captured by `platform_contract.json` and the hashes checked in `summary.json`.
