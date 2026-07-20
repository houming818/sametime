# Controlled Data-Dose Experiment Report

## Experiment Card

| Field | Value |
|---|---|
| Experiment ID | `s3_private_protocol_data_dose_full` |
| Claim / Predict | `S3-PRIVATE-PROTOCOL-DATA-DOSE-C03` / `P-S3-PRIVATE-PROTOCOL-DATA-DOSE-03` |
| Status | `supported_pilot_single_seed` |
| Code commit | `7bbb89c` |
| Host / device | `io` / `NVIDIA GeForce RTX 3090` |
| Runtime | `5.32 h` |

## Dataset Card

| Field | Value |
|---|---|
| Source | `/home/nio/datasets/wmt_massive/train.massive.zh-en.tsv` |
| Source bytes / declared rows | 2,520,995,022 / 14,170,275 |
| Train doses | 30,000, 100,000, 300,000, 1,000,000 |
| Validation | 2,000 rows, `86ac3b0fe60302b397416a6fabb8921f6890565ebb373cd04768fa72bddf6f76` |
| Test | 2,000 rows, `b893d33aa433214f87a6c16ee357178beffb12adcde53576a5d178e2813660c0` |
| Tokenizer | `/home/nio/datasets/wmt_massive/sp_bpe_massive.model`, SHA-256 `9956eff597852f8c684c4ad23243d15889da6a9b138f8fd025570147324cc731` |
| Leakage removal | train/eval overlaps removed: 0; duplicate train rows removed: 0 |

## Variable Contract

| Type | Variables |
|---|---|
| Independent | Number of unique training pairs: 30K, 100K, 300K, 1M |
| Controlled | split hashes, tokenizer, model seed `71901`, `15,625` updates, batch `64`, AdamW, LR `0.002`, validation cadence `500` |
| Dependent | held-out NLL and PPL (lower is better); BLEU4 (higher is better) |
| Known nuisance | single seed; equal steps imply fewer repetitions at larger doses |

## Results

| Model | Unique rows | Reuse | Best step | Valid NLL | Test NLL lower-is-better | PPL | BLEU4 | Time | Peak VRAM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| h1 | 30,000 | 33.33x | 1,000 | 6.2998 | 6.2671 | 526.9 | 5.401 | 56.9 min | 1.84 GiB |
| h1 | 100,000 | 10.00x | 3,000 | 5.1984 | 5.1454 | 171.6 | 7.778 | 57.0 min | 1.85 GiB |
| h1 | 300,000 | 3.33x | 14,000 | 4.3186 | 4.2558 | 70.5 | 11.079 | 56.7 min | 1.84 GiB |
| h1 | 1,000,000 | 1.00x | 15,625 | 4.0536 | 4.0198 | 55.7 | 12.085 | 56.7 min | 1.84 GiB |
| flat | 30,000 | 33.33x | 1,000 | 6.0520 | 6.0319 | 416.5 | 5.596 | 15.7 min | 1.44 GiB |
| flat | 100,000 | 10.00x | 4,500 | 5.0937 | 5.0532 | 156.5 | 8.950 | 15.5 min | 1.44 GiB |
| flat | 300,000 | 3.33x | 14,000 | 4.2703 | 4.2025 | 66.9 | 11.918 | 15.4 min | 1.44 GiB |
| flat | 1,000,000 | 1.00x | 15,625 | 3.9659 | 3.9365 | 51.2 | 13.006 | 15.3 min | 1.44 GiB |
| transformer | 30,000 | 33.33x | 1,000 | 6.5818 | 6.5373 | 690.4 | 3.145 | 6.9 min | 1.48 GiB |
| transformer | 100,000 | 10.00x | 6,000 | 5.3757 | 5.3375 | 208.0 | 6.320 | 6.9 min | 1.48 GiB |
| transformer | 300,000 | 3.33x | 15,625 | 4.2246 | 4.1761 | 65.1 | 11.597 | 6.8 min | 1.48 GiB |
| transformer | 1,000,000 | 1.00x | 15,625 | 3.9609 | 3.9201 | 50.4 | 12.302 | 6.7 min | 1.48 GiB |

## Curves And Decision Gates

```json
{
  "curves": {
    "h1": {
      "doses": [
        30000,
        100000,
        300000,
        1000000
      ],
      "test_nll": [
        6.26710072318349,
        5.14543889724367,
        4.255821958224378,
        4.019758249489361
      ],
      "spearman_log_dose_vs_nll": -1.0,
      "nll_improvement_first_to_last": 2.247342473694129,
      "adjacent_improvements": 3
    },
    "flat": {
      "doses": [
        30000,
        100000,
        300000,
        1000000
      ],
      "test_nll": [
        6.031940800187375,
        5.053230043158837,
        4.202527606580746,
        3.9365029901563995
      ],
      "spearman_log_dose_vs_nll": -1.0,
      "nll_improvement_first_to_last": 2.095437810030975,
      "adjacent_improvements": 3
    },
    "transformer": {
      "doses": [
        30000,
        100000,
        300000,
        1000000
      ],
      "test_nll": [
        6.537277812595862,
        5.337504335699014,
        4.176113218852175,
        3.9200728865052823
      ],
      "spearman_log_dose_vs_nll": -1.0,
      "nll_improvement_first_to_last": 2.61720492609058,
      "adjacent_improvements": 3
    }
  },
  "gates": {
    "P1_h1_improves_at_least_0_10": true,
    "P2_h1_spearman_at_most_minus_0_80": true,
    "P3_two_of_three_adjacent_doses_improve": true,
    "P4_all_gradients_finite": true
  }
}
```

## Interpretation Boundary

This fixed-update, single-seed curve tests data diversity under one training recipe. It does not establish a universal scaling law, dataset optimum, semantic heads, or TreeHeap superiority. A common gain across models is a corpus effect.

Per-evaluation values are preserved in `trace.jsonl`; exact split identities are in `dataset_manifest.json`.

## Reviewer Analysis

The preregistered data-sufficiency claim passed by a wide margin. TreeHeap h1
improved by `2.2473` NLL from 30K to 1M, and every adjacent dose improved.
This is primarily a general corpus-diversity effect: Flat and Transformer also
had perfect monotonic dose curves. At 1M, h1 remained behind Flat by `0.0833`
NLL and Transformer by `0.0997` NLL.

The best-step trace is as important as the final table. All 30K models peaked
at step 1,000. Continued reuse drove final validation NLL to `10.7979` for h1,
`10.8155` for Flat, and `14.7291` for Transformer. At 1M, h1 and Transformer
were still best at step 15,625. Thus 30K exhausted its evidence and overfit;
the one-pass 1M arm may still be undertrained.

Runtime remains a material negative: h1 took about `56.8` minutes per dose,
versus `15.5` for Flat and `6.8` for Transformer under the same update count.
Examples also reveal noisy, occasionally mojibake or misaligned web pairs, so
the controlled NLL result should not be presented as clean WMT product quality.
