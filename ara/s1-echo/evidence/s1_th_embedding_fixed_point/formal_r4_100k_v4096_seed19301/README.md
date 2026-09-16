# TH Embedding Fixed Point

```json
{
  "evaluation": {
    "pairs": 20000,
    "baseline_pair_accuracy": 0.6686999797821045,
    "tree_pair_accuracy": 0.664900004863739,
    "baseline_margin": 0.1109059676527977,
    "tree_margin": 0.10919755697250366,
    "leaf_utilization": 0.21875,
    "occupancy_entropy": 0.4618634195906911,
    "positive_lcp": 2.6112000942230225,
    "negative_lcp": 1.8988499641418457,
    "lcp_margin": 0.7123500108718872,
    "round_flip_rate": [
      0.037841796875,
      0.03271484375,
      0.04248046875
    ]
  },
  "gradients": {
    "embedding": 1.1088377237319946,
    "route_weight": 0.00012748163135256618,
    "route_bias": 0.0006100151222199202,
    "node_value": 0.0669352188706398
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
