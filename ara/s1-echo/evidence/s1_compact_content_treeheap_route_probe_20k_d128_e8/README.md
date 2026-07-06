# S1 Compact Content TreeHeap Route Probe

Claim: `S1-COMPACT-CONTENT-ROUTE-C01`

Compact 128D subheap states replace dense vocab-count route features.

```json
{
  "complexity": {
    "train_route_steps": 352110,
    "ood_route_steps": 44130,
    "compact_memory_mb": 637.59228515625,
    "dense_prior_memory_mb": 6191.25,
    "memory_reduction_x_vs_dense_prior": 9.710359024314036,
    "batches_per_epoch": 344
  },
  "metrics": {
    "train": {
      "steps": 352110,
      "routes": 58685,
      "step_acc": 0.9997017979621887,
      "route_exact": 0.9982107864019767
    },
    "ood": {
      "steps": 44130,
      "routes": 7355,
      "step_acc": 0.9972581267356873,
      "route_exact": 0.9836845683208701
    },
    "flat_length_matrix": {
      "train": {
        "token_acc": 1.0,
        "exact": 1.0
      },
      "ood": {
        "token_acc": 0.002882178872823715,
        "exact": 0.0
      }
    },
    "trace": [
      {
        "epoch": 1,
        "loss": 0.07767227579180161,
        "step_acc": 0.9662605435801312,
        "epoch_sec": 1.3742430210113525,
        "eta_sec": 9.619714498519897
      },
      {
        "epoch": 2,
        "loss": 0.001447364105303267,
        "step_acc": 0.9996222771293062,
        "epoch_sec": 1.2618637084960938,
        "eta_sec": 7.908437490463257
      },
      {
        "epoch": 3,
        "loss": 0.0009136501176003904,
        "step_acc": 0.9997443980574252,
        "epoch_sec": 1.2301125526428223,
        "eta_sec": 6.443823575973511
      },
      {
        "epoch": 4,
        "loss": 0.00029781527924797337,
        "step_acc": 0.9999091193093067,
        "epoch_sec": 1.2600858211517334,
        "eta_sec": 5.126413822174072
      },
      {
        "epoch": 5,
        "loss": 0.0005656174127882484,
        "step_acc": 0.9998040385106927,
        "epoch_sec": 1.2412748336791992,
        "eta_sec": 3.8206347942352292
      },
      {
        "epoch": 6,
        "loss": 0.0012982421311439308,
        "step_acc": 0.9995200363522763,
        "epoch_sec": 1.2419779300689697,
        "eta_sec": 2.536578893661499
      },
      {
        "epoch": 7,
        "loss": 0.0001007261565947558,
        "step_acc": 0.9999715997841584,
        "epoch_sec": 1.2420740127563477,
        "eta_sec": 1.264549732208252
      },
      {
        "epoch": 8,
        "loss": 0.00021579683626496941,
        "step_acc": 0.9999375195251484,
        "epoch_sec": 1.2470676898956299,
        "eta_sec": 0.0
      }
    ]
  },
  "pass_checks": {
    "compact_memory_under_512mb": false,
    "ood_route_exact_ge_0_99": false,
    "ood_step_acc_ge_0_99": true,
    "flat_length_matrix_fails_unseen_lengths": true
  },
  "pilot_pass": false,
  "limits": [
    "fixed random token vectors, not learned semantic embeddings",
    "query token is supervised",
    "unique-token query positions only",
    "not translation",
    "not unsupervised span discovery"
  ]
}
```
