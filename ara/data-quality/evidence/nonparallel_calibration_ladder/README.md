# Non-parallel calibration ladder evidence

This directory contains compact local evidence for the 2026-08-22 execution
of `NIO-NONPAR-CAL-C01` and `NIO-NONPAR-LADDER-C01`.

Local files retained in git-sized form:

- three 100K shadow-pool `summary.json` files;
- Qwen3-8B `summary.json` and all 1,500 `judgments.jsonl` rows;
- overall and per-family ladder summaries.

Large immutable artifacts remain on `io` under:

```text
/home/nio/log/holds/SameTime/ara/data-quality/evidence/nonparallel_calibration_ladder/
```

The remote directory contains the three 100K manifests and the full selected
and raw-control gzip orders. Independent post-run audit confirmed:

```text
each of six order files: 100000 decompressed JSONL rows
gzip integrity:          pass
summary/content SHA-256: pass
```

Decompressed order hashes:

| Family | Selected | Raw control |
|---|---|---|
| mono | `e8da37125c41c3e43e7892b1981d24a08eaa8ff71bc22eded46503d05a618166` | `ec5d6a1d109f0134f4f80c8902eafe79ae9b96dff10dd674d8e2b9bc52f418bf` |
| QA | `d859c7dcf1bc972709c5f5ea15ecee3317d2fbce3d267938ec2f133cbcba69d0` | `07deb9dc3f7d63e6e1fb2822236fb5591dcbc356763b2061b85ee7064622ef6e` |
| medical | `cabec8a479a8bab230255106ae2df2b7bebab02c0cf3cef60b97f6b738098b27` | `566395bf765d2b1fdf1ee534d6e0ee5acc997af2d55fcb6d780101ad2a0e151e` |

The Qwen labels are calibration-model estimates, not human ground truth. The
medical judgments are explicitly unverified and must not be interpreted as
medical correctness labels.
