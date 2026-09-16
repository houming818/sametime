# TH Embedding Fixed Point

```json
{
  "evaluation": {
    "pairs": 20000,
    "baseline_pair_accuracy": 0.680899977684021,
    "tree_pair_accuracy": 0.6798999905586243,
    "baseline_margin": 0.12051625549793243,
    "tree_margin": 0.11954225599765778,
    "leaf_utilization": 0.158203125,
    "occupancy_entropy": 0.4499260460120475,
    "positive_lcp": 3.3461499214172363,
    "negative_lcp": 2.5794498920440674,
    "lcp_margin": 0.766700029373169,
    "round_flip_rate": [
      0.0185546875,
      0.015380859375,
      0.01953125,
      0.0185546875,
      0.0205078125,
      0.015380859375,
      0.01953125
    ]
  },
  "gradients": {
    "embedding": 1.0732585191726685,
    "route_weight": 0.00024823201238177717,
    "route_bias": 0.0017076722579076886,
    "node_value": 0.13405843079090118
  },
  "gates": {
    "all_gradients_finite_positive": true,
    "tree_pair_accuracy_gt_0_52": true,
    "positive_lcp_gt_negative_lcp": true,
    "leaf_utilization_gt_0_10": true,
    "occupancy_entropy_gt_0_50": false,
    "finite_losses": true
  }
}
```
