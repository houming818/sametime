# S1 Compact Content TreeHeap Route Probe

Claim: `S1-COMPACT-CONTENT-ROUTE-C01`

Compact 64D subheap states replace dense vocab-count route features.

```json
{
  "complexity": {
    "train_route_steps": 352110,
    "ood_route_steps": 44130,
    "compact_memory_mb": 324.84228515625,
    "dense_prior_memory_mb": 6191.25,
    "memory_reduction_x_vs_dense_prior": 19.059249004549983,
    "batches_per_epoch": 344
  },
  "metrics": {
    "train": {
      "steps": 352110,
      "routes": 58685,
      "step_acc": 0.999789834022522,
      "route_exact": 0.9987390304166311
    },
    "ood": {
      "steps": 44130,
      "routes": 7355,
      "step_acc": 0.9970995187759399,
      "route_exact": 0.9825968728755948
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
        "loss": 0.08086497386677037,
        "step_acc": 0.9655420181193377,
        "epoch_sec": 0.5103673934936523,
        "eta_sec": 2.0414772033691406
      },
      {
        "epoch": 2,
        "loss": 0.0024349678173803636,
        "step_acc": 0.9992133140211865,
        "epoch_sec": 0.32348203659057617,
        "eta_sec": 1.250837802886963
      },
      {
        "epoch": 3,
        "loss": 0.0019357437815925359,
        "step_acc": 0.9993524750788106,
        "epoch_sec": 0.32317209243774414,
        "eta_sec": 0.7713929812113444
      },
      {
        "epoch": 4,
        "loss": 0.0021773293867591087,
        "step_acc": 0.9992275141291074,
        "epoch_sec": 0.32091283798217773,
        "eta_sec": 0.3695067763328552
      },
      {
        "epoch": 5,
        "loss": 0.001402975015634626,
        "step_acc": 0.9995086762659396,
        "epoch_sec": 0.3210580348968506,
        "eta_sec": 0.0
      }
    ]
  },
  "pass_checks": {
    "compact_memory_under_512mb": true,
    "ood_route_exact_ge_0_99": false,
    "ood_step_acc_ge_0_99": true,
    "flat_length_matrix_fails_unseen_lengths": true
  },
  "pilot_pass": false,
  "limits": [
    "fixed random token vectors",
    "query token is supervised",
    "unique-token query positions only",
    "not translation",
    "not unsupervised span discovery"
  ]
}
```
