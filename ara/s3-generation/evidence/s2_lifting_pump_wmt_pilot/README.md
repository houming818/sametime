# S2 Lifting Pump WMT

```json
{
  "derived": {
    "recursive_nll": 6.547321340330487,
    "recursive_gain_over_root": 0.12868484024154014,
    "recursive_gap_to_full": 0.04537244648363359,
    "recursive_gap_to_flat": 0.19396146500434774,
    "source_shuffle_damage": 0.5807709531645324,
    "root_shuffle_damage": 0.6981946565168409,
    "detail_shuffle_damage": [
      0.009587762146189505,
      0.02569338408246402,
      0.06145277415401651,
      0.16300153062737266,
      0.131728737992173
    ],
    "pair_break_damage": [
      0.17068061032376658,
      0.15412846275102865,
      0.16171893530597803,
      0.20387088744267867,
      0.07124693627450984
    ],
    "force_root_damage": 0.6203263670639627,
    "force_leaf_damage": 0.1424921739108953
  },
  "gates": {
    "P1_historical_root_exclusive_gain": true,
    "P2_recursive_over_root": true,
    "P3_near_full_expand": true,
    "P4_source_causal": true,
    "P5_root_and_detail_causal": true,
    "P6_recursive_pairs_causal": true,
    "P7_multiresolution_route": true,
    "P8_closure_finite_nonempty": true
  },
  "decision": "supported_pilot"
}
```
