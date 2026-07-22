# STONE-1 C03 Capacity and Rate-Distortion Report

## Experiment Card

| Field | Value |
|---|---|
| Status | `smoke_only` |
| Stage B authorized | `False` |
| Host / device | `io` / `NVIDIA GeForce RTX 3090` |
| Runtime | 0.06 h |

## Per-run Results

| Arm | Seed | Steps | Params | Test NLL | Bits/token | BLEU-4 | Train NLL | Time | VRAM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| base_28m_long | 71901 | 500 | 27,620,482 | 7.6087 | 10.9771 | 1.309 | 1.7439 | 1.6 min | 1.63 GiB |
| balanced_50m_equal | 71901 | 500 | 50,267,778 | 7.6270 | 11.0034 | 1.874 | 0.3778 | 1.6 min | 2.10 GiB |

## Aggregate

```json
{
  "base_28m_long": {
    "nll_mean": 7.608746682553357,
    "nll_std": 0.0,
    "bits_per_token_mean": 10.977101106300081,
    "bleu4_mean": 1.3086216317567076,
    "nonempty_mean": 1.0,
    "severe_repetition_mean": 0.0859375,
    "final_train_nll_mean": 1.743890527486801,
    "seconds_mean": 94.68836212158203,
    "parameters": 27620482,
    "trainable_parameters": 27620482,
    "peak_vram_max": 1750702080
  },
  "balanced_50m_equal": {
    "nll_mean": 7.626962677856826,
    "nll_std": 0.0,
    "bits_per_token_mean": 11.003381232389252,
    "bleu4_mean": 1.8743997520387323,
    "nonempty_mean": 1.0,
    "severe_repetition_mean": 0.2890625,
    "final_train_nll_mean": 0.3778022558987141,
    "seconds_mean": 95.88670492172241,
    "parameters": 50267778,
    "trainable_parameters": 50267778,
    "peak_vram_max": 2257968640
  }
}
```

## Gates

```json
{
  "C1_50m_nll_gain_at_least_0_08": false,
  "C2_50m_bleu_gain_at_least_0_75": false,
  "C3_50m_nll_std_at_most_0_08": true,
  "C4_50m_within_0_02_of_28m_long": true,
  "C5_force_algebraic_damage_at_least_0_10": true,
  "C6_address_damage_at_least_0_10": false,
  "C7_depth_growth": false,
  "C8_closure_below_1e_5": true,
  "C9_generation_nondegenerate": false,
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
