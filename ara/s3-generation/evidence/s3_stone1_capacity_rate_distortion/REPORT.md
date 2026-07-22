# STONE-1 C03 Capacity and Rate-Distortion Report

## Experiment Card

| Field | Value |
|---|---|
| Status | `capacity_not_supported_stone1_incomplete` |
| Stage B authorized | `False` |
| Host / device | `io` / `NVIDIA GeForce RTX 3090` |
| Runtime | 7.44 h |

## Per-run Results

| Arm | Seed | Steps | Params | Test NLL | Bits/token | BLEU-4 | Train NLL | Time | VRAM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| base_28m_long | 71901 | 31,250 | 27,620,482 | 3.9023 | 5.6299 | 11.074 | 3.8899 | 98.0 min | 1.63 GiB |
| balanced_50m_equal | 71901 | 15,625 | 50,267,778 | 4.1532 | 5.9918 | 10.234 | 4.2391 | 49.9 min | 2.10 GiB |
| base_28m_long | 71902 | 31,250 | 27,620,482 | 3.6824 | 5.3125 | 13.452 | 3.6829 | 97.9 min | 1.63 GiB |
| balanced_50m_equal | 71902 | 15,625 | 50,267,778 | 4.1532 | 5.9918 | 10.009 | 4.1953 | 49.8 min | 2.10 GiB |
| base_28m_long | 71903 | 31,250 | 27,620,482 | 3.6639 | 5.2859 | 13.707 | 3.6669 | 98.1 min | 1.63 GiB |
| balanced_50m_equal | 71903 | 15,625 | 50,267,778 | 4.1343 | 5.9645 | 10.125 | 4.1951 | 49.8 min | 2.10 GiB |

## Aggregate

```json
{
  "base_28m_long": {
    "nll_mean": 3.7495310920326714,
    "nll_std": 0.10829771901868383,
    "bits_per_token_mean": 5.409429912134515,
    "bleu4_mean": 12.744374330170666,
    "nonempty_mean": 1.0,
    "severe_repetition_mean": 0.034,
    "final_train_nll_mean": 3.74656929175059,
    "seconds_mean": 5878.945271333058,
    "parameters": 27620482,
    "trainable_parameters": 27620482,
    "peak_vram_max": 1751621120
  },
  "balanced_50m_equal": {
    "nll_mean": 4.146898676345821,
    "nll_std": 0.008942560485286107,
    "bits_per_token_mean": 5.982710155433123,
    "bleu4_mean": 10.122499573616897,
    "nonempty_mean": 1.0,
    "severe_repetition_mean": 0.07066666666666667,
    "final_train_nll_mean": 4.209857027689616,
    "seconds_mean": 2990.2111554145813,
    "parameters": 50267778,
    "trainable_parameters": 50267778,
    "peak_vram_max": 2257053696
  }
}
```

## Gates

```json
{
  "C1_50m_nll_gain_at_least_0_08": false,
  "C2_50m_bleu_gain_at_least_0_75": false,
  "C3_50m_nll_std_at_most_0_08": true,
  "C4_50m_within_0_02_of_28m_long": false,
  "C5_force_algebraic_damage_at_least_0_10": true,
  "C6_address_damage_at_least_0_10": false,
  "C7_depth_growth": false,
  "C8_closure_below_1e_5": true,
  "C9_generation_nondegenerate": true,
  "Q1_nll_at_most_3_90": false,
  "Q2_bleu4_at_least_13_5": false,
  "Q3_nll_std_at_most_0_05": true,
  "E1_peak_vram_below_10gib": true,
  "E2_all_gradients_finite": true,
  "E3_parameter_counts_exact": true
}
```

## Boundary

C03 tests whole-system capacity under one frozen WMT and optimizer contract. It does not isolate codec-only capacity or establish general scaling laws, dialogue, or world knowledge.
