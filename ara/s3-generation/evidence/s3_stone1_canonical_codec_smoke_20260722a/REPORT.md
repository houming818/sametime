# STONE-1 C02 Canonical Codec Report

## Experiment Card

| Field | Value |
|---|---|
| Status | `smoke_only` |
| Host / device | `io` / `NVIDIA GeForce RTX 3090` |
| Runtime | 0.02 h |
| Git commit | `22b5ef7` |

## Results

| Variant | Seed | Best step | Test NLL | PPL | BLEU-4 | Nonempty | Repetition | Time | VRAM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| canonical_algebraic | 71901 | 100 | 7.7796 | 2391.3 | 0.666 | 1.000 | 0.406 | 0.3 min | 1.56 GiB |
| canonical_learned | 71901 | 100 | 7.7469 | 2314.3 | 1.586 | 1.000 | 0.414 | 0.3 min | 1.63 GiB |
| canonical_frozen | 71901 | 100 | 7.7815 | 2395.9 | 0.661 | 1.000 | 0.305 | 0.3 min | 1.60 GiB |

## Aggregate

```json
{
  "canonical_algebraic": {
    "nll_mean": 7.77960907972988,
    "nll_std": 0.0,
    "bleu4_mean": 0.6655291601684714,
    "nonempty_mean": 1.0,
    "severe_repetition_mean": 0.40625,
    "seconds_mean": 19.149812698364258,
    "parameters": 27620482,
    "trainable_parameters": 27323650,
    "peak_vram_max": 1678806528
  },
  "canonical_learned": {
    "nll_mean": 7.746874913155847,
    "nll_std": 0.0,
    "bleu4_mean": 1.5856689419541738,
    "nonempty_mean": 1.0,
    "severe_repetition_mean": 0.4140625,
    "seconds_mean": 20.20519995689392,
    "parameters": 27620482,
    "trainable_parameters": 27620482,
    "peak_vram_max": 1751405568
  },
  "canonical_frozen": {
    "nll_mean": 7.781513572004224,
    "nll_std": 0.0,
    "bleu4_mean": 0.661399176442175,
    "nonempty_mean": 1.0,
    "severe_repetition_mean": 0.3046875,
    "seconds_mean": 20.075624704360962,
    "parameters": 27620482,
    "trainable_parameters": 27323650,
    "peak_vram_max": 1720199680
  }
}
```

## Codec Intervention and Depth Growth

```json
{
  "seed": 71901,
  "normal": {
    "nll": 7.746874913155847,
    "ppl": 2314.3286238333762,
    "tokens": 2249
  },
  "force_algebraic": {
    "nll": 7.8853765736160515,
    "ppl": 2658.1258176610486,
    "tokens": 2249
  },
  "address_swap": {
    "nll": 7.746874913155847,
    "ppl": 2314.3286238333762,
    "tokens": 2249
  },
  "damage_nll": {
    "force_algebraic": 0.1385016604602045,
    "address_swap": 0.0
  },
  "depth_growth": [
    {
      "visible_levels": 1,
      "nll": 7.746874478935082,
      "ppl": 2314.327618904049,
      "tokens": 2249
    },
    {
      "visible_levels": 2,
      "nll": 7.746875347376612,
      "ppl": 2314.3296287631406,
      "tokens": 2249
    },
    {
      "visible_levels": 3,
      "nll": 7.746875347376612,
      "ppl": 2314.3296287631406,
      "tokens": 2249
    },
    {
      "visible_levels": 4,
      "nll": 7.746875347376612,
      "ppl": 2314.3296287631406,
      "tokens": 2249
    },
    {
      "visible_levels": 5,
      "nll": 7.746875347376612,
      "ppl": 2314.3296287631406,
      "tokens": 2249
    },
    {
      "visible_levels": 6,
      "nll": 7.746874913155847,
      "ppl": 2314.3286238333762,
      "tokens": 2249
    }
  ],
  "root_to_full_gain_nll": -4.3422076512200647e-07,
  "improving_depth_transitions": 1,
  "learned_output_parameter_norm": 7.879814624786377,
  "latency": {
    "samples": 5,
    "p50_ms": 19.482083152979612,
    "min_ms": 19.472103798761964,
    "max_ms": 19.489557947963476
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
  "S1_learned_beats_algebraic_0_05": false,
  "S2_learned_beats_frozen_0_10": false,
  "S3_force_algebraic_damage_0_10": true,
  "S4_address_damage_0_10": false,
  "S5_root_to_full_gain_0_50": false,
  "S6_four_of_five_depths_improve": false,
  "S7_closure_below_1e_5": true,
  "E1_latency_p50_at_most_1000ms": true,
  "E2_vram_at_most_4gib": true,
  "E3_checkpoint_at_most_300mib": true,
  "E4_finite": true,
  "E5_checkpoint_created": true
}
```

## Boundary

C02 can support or reject one canonical codec recipe inside the unfinished STONE-1 milestone. It cannot establish dialogue, world knowledge, human-readable depth semantics, or industry superiority.
