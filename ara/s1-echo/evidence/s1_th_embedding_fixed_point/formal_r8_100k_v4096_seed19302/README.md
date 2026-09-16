# TH Embedding Fixed Point

```json
{
  "evaluation": {
    "pairs": 20000,
    "baseline_pair_accuracy": 0.6683499813079834,
    "tree_pair_accuracy": 0.6690499782562256,
    "baseline_margin": 0.11445480585098267,
    "tree_margin": 0.11326541751623154,
    "leaf_utilization": 0.15234375,
    "occupancy_entropy": 0.4715754344361022,
    "positive_lcp": 2.2900500297546387,
    "negative_lcp": 1.695199966430664,
    "lcp_margin": 0.5948500037193298,
    "round_flip_rate": [
      0.021484375,
      0.02294921875,
      0.025634765625,
      0.024658203125,
      0.02734375,
      0.0234375,
      0.023193359375
    ]
  },
  "gradients": {
    "embedding": 1.080620527267456,
    "route_weight": 0.0002691914269234985,
    "route_bias": 0.0013305250322446227,
    "node_value": 0.14542554318904877
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
