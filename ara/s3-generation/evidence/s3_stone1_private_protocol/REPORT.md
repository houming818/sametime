# STONE-1 Private Protocol Translation Report

## Experiment Card

| Field | Value |
|---|---|
| Status | `not_supported_under_recipe` |
| Host / device | `io` / `NVIDIA GeForce RTX 3090` |
| Runtime | 7.57 h |
| Git commit | `4c3d275` |

## Results

| Variant | Seed | Best step | Test NLL | PPL | BLEU-4 | Nonempty | Repetition | Time | VRAM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| identity | 71901 | 15,625 | 4.0799 | 59.1 | 11.151 | 1.000 | 0.042 | 49.5 min | 1.66 GiB |
| learned_structural | 71901 | 15,625 | 4.3858 | 80.3 | 8.995 | 1.000 | 0.078 | 51.2 min | 1.69 GiB |
| frozen_random | 71901 | 15,500 | 4.1359 | 62.5 | 10.731 | 1.000 | 0.033 | 49.9 min | 1.66 GiB |
| identity | 71902 | 15,500 | 4.0691 | 58.5 | 11.361 | 1.000 | 0.038 | 49.8 min | 1.66 GiB |
| learned_structural | 71902 | 15,500 | 4.1538 | 63.7 | 9.961 | 1.000 | 0.043 | 51.0 min | 1.69 GiB |
| frozen_random | 71902 | 15,625 | 4.1121 | 61.1 | 10.587 | 1.000 | 0.048 | 49.9 min | 1.66 GiB |
| identity | 71903 | 15,625 | 4.0666 | 58.4 | 11.009 | 1.000 | 0.039 | 49.7 min | 1.66 GiB |
| learned_structural | 71903 | 15,625 | 4.1411 | 62.9 | 10.407 | 1.000 | 0.046 | 50.9 min | 1.69 GiB |
| frozen_random | 71903 | 15,625 | 4.1149 | 61.2 | 10.841 | 1.000 | 0.049 | 49.9 min | 1.66 GiB |

## Aggregate

```json
{
  "identity": {
    "nll_mean": 4.071877074325834,
    "nll_std": 0.005801094242744635,
    "bleu4_mean": 11.173549989485323,
    "nonempty_mean": 1.0,
    "severe_repetition_mean": 0.03966666666666667,
    "seconds_mean": 2979.6748099327087,
    "parameters": 27769097,
    "peak_vram_max": 1778750976
  },
  "learned_structural": {
    "nll_mean": 4.2269050024429164,
    "nll_std": 0.11247114016312774,
    "bleu4_mean": 9.787451935577256,
    "nonempty_mean": 1.0,
    "severe_repetition_mean": 0.05583333333333333,
    "seconds_mean": 3061.3424132665,
    "parameters": 27769097,
    "peak_vram_max": 1819453440
  },
  "frozen_random": {
    "nll_mean": 4.1209664445537895,
    "nll_std": 0.010646156396396569,
    "bleu4_mean": 10.719838203021604,
    "nonempty_mean": 1.0,
    "severe_repetition_mean": 0.043000000000000003,
    "seconds_mean": 2992.3747523625693,
    "parameters": 27769097,
    "peak_vram_max": 1778750976
  }
}
```

## Structural Intervention

```json
{
  "seed": 71903,
  "normal": {
    "nll": 4.141139266948594,
    "ppl": 62.8744114002465,
    "tokens": 33414
  },
  "force_identity": {
    "nll": 4.129340210801189,
    "ppl": 62.136912143530225,
    "tokens": 33414
  },
  "force_random": {
    "nll": 4.1445807737073155,
    "ppl": 63.091166880819735,
    "tokens": 33414
  },
  "address_swap": {
    "nll": 5.635948209921833,
    "ppl": 280.32459789506896,
    "tokens": 33414
  },
  "damage_nll": {
    "force_identity": -0.01179905614740484,
    "force_random": 0.0034415067587216797,
    "address_swap": 1.494808942973239
  },
  "latency": {
    "samples": 20,
    "p50_ms": 27.419887948781252,
    "min_ms": 27.350300922989845,
    "max_ms": 27.591509046033025
  }
}
```

## Decision Gates

```json
{
  "Q1_nll_at_most_3_90": false,
  "Q2_bleu4_at_least_13_5": false,
  "Q3_nll_std_at_most_0_05": false,
  "Q4_nonempty_is_one": true,
  "Q5_repetition_at_most_0_10": true,
  "S1_learned_beats_identity_0_05": false,
  "S2_learned_beats_random_0_10": false,
  "S3_force_identity_damage_0_10": false,
  "S4_force_random_damage_0_10": false,
  "S5_address_damage_0_10": true,
  "S6_closure_below_1e_5": false,
  "E1_latency_p50_at_most_1000ms": true,
  "E2_vram_at_most_4gib": true,
  "E3_checkpoint_at_most_300mib": true,
  "E4_finite": true,
  "E5_checkpoint_created": true
}
```

## Boundary

This test can support a fixed-capacity TreeHeap translation PoC. It cannot establish dialogue, world knowledge, semantic rotation, or superiority over industry-scale Transformers.
