# S1 Compact Content TreeHeap Route Probe

Claim: `S1-COMPACT-CONTENT-ROUTE-C01`

Compact 8D subheap states replace dense vocab-count route features.

```json
{
  "complexity": {
    "train_route_steps": 1280,
    "ood_route_steps": 672,
    "compact_memory_mb": 0.2568359375,
    "dense_prior_memory_mb": 6191.25,
    "memory_reduction_x_vs_dense_prior": 24105.855513307986,
    "batches_per_epoch": 10
  },
  "metrics": {
    "train": {
      "steps": 1280,
      "routes": 320,
      "step_acc": 0.5406250357627869,
      "route_exact": 0.0
    },
    "ood": {
      "steps": 672,
      "routes": 168,
      "step_acc": 0.4330357313156128,
      "route_exact": 0.0
    },
    "flat_length_matrix": {
      "train": {
        "token_acc": 1.0,
        "exact": 1.0
      },
      "ood": {
        "token_acc": 0.061500612646341324,
        "exact": 0.0
      }
    },
    "trace": [
      {
        "epoch": 1,
        "loss": 0.9903792440891266,
        "step_acc": 0.540625,
        "epoch_sec": 0.03528189659118652,
        "eta_sec": 0.0
      }
    ]
  },
  "pass_checks": {
    "compact_memory_under_512mb": true,
    "ood_route_exact_ge_0_99": false,
    "ood_step_acc_ge_0_99": false,
    "flat_length_matrix_fails_unseen_lengths": true
  },
  "pilot_pass": false,
  "limits": [
    "root-unfold vectors are trained from held-in context counts only",
    "query token is supervised",
    "unique-token query positions only",
    "not translation",
    "not unsupervised span discovery"
  ]
}
```
