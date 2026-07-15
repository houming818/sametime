# Multiresolution TreeHeap Pyramid: Result Analysis

Date: 2026-07-14
Host: `io`
Claim: `S3-TREEHEAP-PYRAMID-C01`

## Decision

```text
mechanism claim                 supported
implemented equal-rate controls positive, bounded
scale specialization           open
entropy-coded compression      not tested
```

Each seed saw one million real-text blocks. The frozen full-corpus
four-head/no-residual TreeHeap checkpoint was identical before and after the
run. The audit validation NLL stayed exactly `6.2997973263`.

## Main Metrics

| k | Rate | TreeHeap MSE | Flat MSE | Haar MSE | Address-shift MSE | Token top-1 | Sequence exact |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1.56% | 1.9334 | 1.9104 | 1.9110 | 1.9334 | 0.0650 | 0.0000 |
| 8 | 5.66% | 0.8803 | 1.8416 | 1.8232 | 3.0678 | 0.5105 | 0.0000 |
| 16 | 9.77% | 0.7622 | 1.7459 | 1.7471 | 3.1684 | 0.7684 | 0.0000 |
| 32 | 17.97% | 0.6516 | 1.7047 | 1.5979 | 3.2665 | 0.9501 | 0.0599 |
| 64 | 34.38% | 0.5526 | 1.6042 | 1.2881 | 3.3147 | 0.9964 | 0.8268 |

The TreeHeap rate-distortion curve was strictly monotonic in all three seeds.
At `k=64`, MSE was 28.6% of root-only MSE. Moving detail codes to the wrong
addresses made reconstruction substantially worse than root-only.

## Important Boundary

The flat attention baseline was intentionally given matched or slightly more
stored floats, but its k=64 results were seed-sensitive:

```text
1.5245 / 1.3767 / 1.9113
```

The TreeHeap results were stable:

```text
0.5526 / 0.5499 / 0.5554
```

This supports the TreeHeap codec against these implemented controls. It does
not prove superiority over every possible flat autoencoder or Transformer.
Random-pairing training and per-level detail ablations remain required before
claiming learned scale specialization. Rates count float activations, not
quantized or entropy-coded bits.

## Reproducibility

- Script: `ara/s3-generation/src/s3_multiresolution_treeheap_pyramid.py`
- Raw summary: `summary.json`
- Per-seed metrics: `metrics_seed_71401.json` through `metrics_seed_71403.json`
- Traces: `trace_seed_*.jsonl`
- Checkpoints: `checkpoint_seed_*.pt`
- NAS mirror: `/mnt/nas/sametime/ara/s3-generation/evidence/s3_multiresolution_treeheap_pyramid/main_v2/`
