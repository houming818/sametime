# STONE-1 Private Protocol Translation Report

## Experiment Card

| Field | Value |
|---|---|
| Status | `smoke_only` |
| Host / device | `io` / `NVIDIA GeForce RTX 3090` |
| Runtime | 0.02 h |
| Git commit | `4c3d275` |

## Results

| Variant | Seed | Best step | Test NLL | PPL | BLEU-4 | Nonempty | Repetition | Time | VRAM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| identity | 71901 | 100 | 7.7940 | 2425.9 | 1.771 | 1.000 | 0.516 | 0.3 min | 1.66 GiB |
| learned_structural | 71901 | 100 | 7.7372 | 2292.0 | 2.107 | 1.000 | 0.406 | 0.3 min | 1.69 GiB |
| frozen_random | 71901 | 100 | 7.7739 | 2377.8 | 2.200 | 1.000 | 0.430 | 0.3 min | 1.66 GiB |

## Aggregate

```json
{
  "identity": {
    "nll_mean": 7.793958773343708,
    "nll_std": 0.0,
    "bleu4_mean": 1.7706230759831822,
    "nonempty_mean": 1.0,
    "severe_repetition_mean": 0.515625,
    "seconds_mean": 20.275712728500366,
    "parameters": 27769097,
    "peak_vram_max": 1777566720
  },
  "learned_structural": {
    "nll_mean": 7.737167907959093,
    "nll_std": 0.0,
    "bleu4_mean": 2.106564681529249,
    "nonempty_mean": 1.0,
    "severe_repetition_mean": 0.40625,
    "seconds_mean": 20.9535710811615,
    "parameters": 27769097,
    "peak_vram_max": 1819386880
  },
  "frozen_random": {
    "nll_mean": 7.773934682775678,
    "nll_std": 0.0,
    "bleu4_mean": 2.2001812117819566,
    "nonempty_mean": 1.0,
    "severe_repetition_mean": 0.4296875,
    "seconds_mean": 20.15516185760498,
    "parameters": 27769097,
    "peak_vram_max": 1777763840
  }
}
```

## Structural Intervention

```json
{
  "seed": 71901,
  "normal": {
    "nll": 7.737167907959093,
    "ppl": 2291.972106814998,
    "tokens": 2249
  },
  "force_identity": {
    "nll": 7.7654018105269005,
    "ppl": 2357.6056089808844,
    "tokens": 2249
  },
  "force_random": {
    "nll": 7.746693843096932,
    "ppl": 2313.909606150034,
    "tokens": 2249
  },
  "address_swap": {
    "nll": 7.754503303551578,
    "ppl": 2332.050735733251,
    "tokens": 2249
  },
  "damage_nll": {
    "force_identity": 0.028233902567807334,
    "force_random": 0.00952593513783917,
    "address_swap": 0.017335395592485092
  },
  "latency": {
    "samples": 5,
    "p50_ms": 19.494237145408988,
    "min_ms": 19.44804796949029,
    "max_ms": 19.54899006523192
  }
}
```

## Decision Gates

```json
{
  "Q1_nll_at_most_3_90": false,
  "Q2_bleu4_at_least_13_5": false,
  "Q3_nll_std_at_most_0_05": true,
  "Q4_nonempty_is_one": true,
  "Q5_repetition_at_most_0_10": false,
  "S1_learned_beats_identity_0_05": true,
  "S2_learned_beats_random_0_10": false,
  "S3_force_identity_damage_0_10": false,
  "S4_force_random_damage_0_10": false,
  "S5_address_damage_0_10": false,
  "S6_closure_below_1e_5": true,
  "E1_latency_p50_at_most_1000ms": true,
  "E2_vram_at_most_4gib": true,
  "E3_checkpoint_at_most_300mib": true,
  "E4_finite": true,
  "E5_checkpoint_created": true
}
```

## Boundary

This test can support a fixed-capacity TreeHeap translation PoC. It cannot establish dialogue, world knowledge, semantic rotation, or superiority over industry-scale Transformers.
