# STONE-1 C02 Canonical Codec Report

## Experiment Card

| Field | Value |
|---|---|
| Status | `c02_not_supported_stone1_incomplete` |
| Host / device | `io` / `NVIDIA GeForce RTX 3090` |
| Runtime | 7.26 h |
| Git commit | `decc78e` |

## Results

| Variant | Seed | Best step | Test NLL | PPL | BLEU-4 | Nonempty | Repetition | Time | VRAM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| canonical_algebraic | 71901 | 15,625 | 4.1304 | 62.2 | 10.563 | 1.000 | 0.066 | 46.9 min | 1.56 GiB |
| canonical_learned | 71901 | 15,625 | 4.1831 | 65.6 | 9.878 | 1.000 | 0.079 | 49.0 min | 1.63 GiB |
| canonical_frozen | 71901 | 15,500 | 4.0977 | 60.2 | 10.497 | 1.000 | 0.047 | 48.3 min | 1.60 GiB |
| canonical_algebraic | 71902 | 15,625 | 4.1032 | 60.5 | 10.854 | 1.000 | 0.054 | 47.2 min | 1.56 GiB |
| canonical_learned | 71902 | 15,625 | 3.9909 | 54.1 | 11.889 | 1.000 | 0.029 | 48.9 min | 1.63 GiB |
| canonical_frozen | 71902 | 15,625 | 4.0778 | 59.0 | 10.866 | 1.000 | 0.054 | 48.3 min | 1.60 GiB |
| canonical_algebraic | 71903 | 15,625 | 4.1078 | 60.8 | 10.965 | 1.000 | 0.056 | 47.0 min | 1.56 GiB |
| canonical_learned | 71903 | 15,625 | 3.9876 | 53.9 | 12.093 | 1.000 | 0.035 | 49.0 min | 1.63 GiB |
| canonical_frozen | 71903 | 15,625 | 4.0975 | 60.2 | 10.697 | 1.000 | 0.044 | 48.2 min | 1.60 GiB |

## Aggregate

```json
{
  "canonical_algebraic": {
    "nll_mean": 4.113798709905232,
    "nll_std": 0.011868211445805582,
    "bleu4_mean": 10.79372942769775,
    "nonempty_mean": 1.0,
    "severe_repetition_mean": 0.058333333333333334,
    "seconds_mean": 2820.354402065277,
    "parameters": 27620482,
    "trainable_parameters": 27323650,
    "peak_vram_max": 1680035328
  },
  "canonical_learned": {
    "nll_mean": 4.053849851815637,
    "nll_std": 0.09139879528634272,
    "bleu4_mean": 11.286529740065545,
    "nonempty_mean": 1.0,
    "severe_repetition_mean": 0.04783333333333333,
    "seconds_mean": 2938.833192666372,
    "parameters": 27620482,
    "trainable_parameters": 27620482,
    "peak_vram_max": 1751481344
  },
  "canonical_frozen": {
    "nll_mean": 4.091022128792074,
    "nll_std": 0.009332642066072021,
    "bleu4_mean": 10.686495574906301,
    "nonempty_mean": 1.0,
    "severe_repetition_mean": 0.0485,
    "seconds_mean": 2895.6324274539948,
    "parameters": 27620482,
    "trainable_parameters": 27323650,
    "peak_vram_max": 1720848896
  }
}
```

## Codec Intervention and Depth Growth

```json
{
  "seed": 71903,
  "normal": {
    "nll": 3.9875581585726603,
    "ppl": 53.9230569175979,
    "tokens": 33414,
    "route_mass_by_level": [
      0.03185531124472618,
      0.18498142063617706,
      0.11936819553375244,
      0.12913523614406586,
      0.23689445853233337,
      0.2977653741836548
    ]
  },
  "force_algebraic": {
    "nll": 5.085975377087449,
    "ppl": 161.73761749350192,
    "tokens": 33414,
    "route_mass_by_level": [
      0.03395398333668709,
      0.22330324351787567,
      0.15166987478733063,
      0.11072472482919693,
      0.21691422164440155,
      0.2634339928627014
    ]
  },
  "address_swap": {
    "nll": 5.342027410853235,
    "ppl": 208.93588004550273,
    "tokens": 33414,
    "route_mass_by_level": [
      0.031756218522787094,
      0.13456927239894867,
      0.004083706997334957,
      2.570299693616107e-05,
      5.123939990880899e-05,
      7.402102346532047e-05
    ]
  },
  "damage_nll": {
    "force_algebraic": 1.0984172185147885,
    "address_swap": 1.3544692522805746
  },
  "depth_growth": [
    {
      "visible_levels": 1,
      "nll": 4.636612096480424,
      "ppl": 103.19414288849799,
      "tokens": 33414,
      "route_mass_by_level": [
        1.0
      ]
    },
    {
      "visible_levels": 2,
      "nll": 4.6249504543730735,
      "ppl": 101.99771941655362,
      "tokens": 33414,
      "route_mass_by_level": [
        0.031824786216020584,
        0.9681751728057861
      ]
    },
    {
      "visible_levels": 3,
      "nll": 4.388611891985068,
      "ppl": 80.52855902337562,
      "tokens": 33414,
      "route_mass_by_level": [
        0.03173719346523285,
        0.15500891208648682,
        0.8132538795471191
      ]
    },
    {
      "visible_levels": 4,
      "nll": 4.186017861677085,
      "ppl": 65.76040185592944,
      "tokens": 33414,
      "route_mass_by_level": [
        0.031830769032239914,
        0.18615439534187317,
        0.12011154741048813,
        0.6619033217430115
      ]
    },
    {
      "visible_levels": 5,
      "nll": 4.040125447686131,
      "ppl": 56.83347198584547,
      "tokens": 33414,
      "route_mass_by_level": [
        0.03186345472931862,
        0.18914039433002472,
        0.11670368164777756,
        0.13213932514190674,
        0.5301531553268433
      ]
    },
    {
      "visible_levels": 6,
      "nll": 3.9875581585726603,
      "ppl": 53.9230569175979,
      "tokens": 33414,
      "route_mass_by_level": [
        0.03185531124472618,
        0.18498142063617706,
        0.11936819553375244,
        0.12913523614406586,
        0.23689445853233337,
        0.2977653741836548
      ]
    }
  ],
  "root_to_full_gain_nll": 0.6490539379077638,
  "improving_depth_transitions": 5,
  "learned_output_parameter_norm": 69.8724136352539,
  "learned_output_parameter_rms": 0.181722953915596,
  "latency": {
    "samples": 20,
    "p50_ms": 23.241838556714356,
    "min_ms": 22.785732988268137,
    "max_ms": 24.031918961554766
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
  "S1_learned_beats_algebraic_0_05": true,
  "S2_learned_beats_frozen_0_10": false,
  "S3_force_algebraic_damage_0_10": true,
  "S4_address_damage_0_10": true,
  "S5_root_to_full_gain_0_50": true,
  "S6_four_of_five_depths_improve": true,
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
