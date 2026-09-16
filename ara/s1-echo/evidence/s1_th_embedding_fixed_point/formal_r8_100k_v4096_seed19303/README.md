# TH Embedding Fixed Point

```json
{
  "evaluation": {
    "pairs": 20000,
    "baseline_pair_accuracy": 0.6659500002861023,
    "tree_pair_accuracy": 0.6643499732017517,
    "baseline_margin": 0.11040060967206955,
    "tree_margin": 0.10965117812156677,
    "leaf_utilization": 0.162109375,
    "occupancy_entropy": 0.4497819057380362,
    "positive_lcp": 2.4969000816345215,
    "negative_lcp": 1.9089499711990356,
    "lcp_margin": 0.5879499912261963,
    "round_flip_rate": [
      0.026611328125,
      0.024658203125,
      0.031005859375,
      0.03271484375,
      0.02587890625,
      0.025634765625,
      0.02587890625
    ]
  },
  "gradients": {
    "embedding": 1.1017719507217407,
    "route_weight": 0.0002329755952814594,
    "route_bias": 0.0013324441388249397,
    "node_value": 0.13668565452098846
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
