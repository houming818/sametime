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
      "step_acc": 0.9998239278793335,
      "route_exact": 0.998943511970691
    },
    "ood": {
      "steps": 44130,
      "routes": 7355,
      "step_acc": 0.9973034262657166,
      "route_exact": 0.9838205302515296
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
        "loss": 0.07815983035919576,
        "step_acc": 0.9665246655874584,
        "epoch_sec": 1.0486786365509033,
        "eta_sec": 4.194723129272461
      },
      {
        "epoch": 2,
        "loss": 0.002881954640878708,
        "step_acc": 0.9991394734599983,
        "epoch_sec": 0.8832199573516846,
        "eta_sec": 2.897902250289917
      },
      {
        "epoch": 3,
        "loss": 0.0017672245752708793,
        "step_acc": 0.9993581551219789,
        "epoch_sec": 0.8827369213104248,
        "eta_sec": 1.8764694531758626
      },
      {
        "epoch": 4,
        "loss": 0.0015594915161249174,
        "step_acc": 0.9994802760500979,
        "epoch_sec": 0.8789801597595215,
        "eta_sec": 0.9234293699264526
      },
      {
        "epoch": 5,
        "loss": 0.0019398560821281712,
        "step_acc": 0.9993070347334639,
        "epoch_sec": 0.8660986423492432,
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
    "fixed random token vectors, not learned semantic embeddings",
    "query token is supervised",
    "unique-token query positions only",
    "not translation",
    "not unsupervised span discovery"
  ]
}
```
