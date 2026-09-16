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
      "step_acc": 0.9273493885993958,
      "route_exact": 0.6139729061940871
    },
    "ood": {
      "steps": 44130,
      "routes": 7355,
      "step_acc": 0.8920009136199951,
      "route_exact": 0.48198504418762744
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
        "loss": 0.30104009504059037,
        "step_acc": 0.8459742694044474,
        "epoch_sec": 0.3380014896392822,
        "eta_sec": 1.3520097732543945
      },
      {
        "epoch": 2,
        "loss": 0.18699601975945063,
        "step_acc": 0.9153162364033967,
        "epoch_sec": 0.4084603786468506,
        "eta_sec": 1.1197503805160522
      },
      {
        "epoch": 3,
        "loss": 0.16476216512713138,
        "step_acc": 0.9255232739768823,
        "epoch_sec": 0.6024625301361084,
        "eta_sec": 0.8993452390034994
      },
      {
        "epoch": 4,
        "loss": 0.15140500755842068,
        "step_acc": 0.9315980801454091,
        "epoch_sec": 0.6160764694213867,
        "eta_sec": 0.49128592014312744
      },
      {
        "epoch": 5,
        "loss": 0.14677296854037827,
        "step_acc": 0.9333674135923433,
        "epoch_sec": 0.6021625995635986,
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
