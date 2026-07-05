# S1 Content TreeHeap Route Probe

Claim: `S1-CONTENT-ROUTE-C01`

The kernel receives query plus `arr[i]`, `arr[2i]`, `arr[2i+1]` content summaries.
It does not receive target interval flags or precomputed left/right answer bits.

```json
{
  "metrics": {
    "train": {
      "steps": 352110,
      "routes": 58685,
      "step_acc": 0.9998466372489929,
      "route_exact": 0.9991139132657408
    },
    "ood": {
      "steps": 44130,
      "routes": 7355,
      "step_acc": 0.9983457922935486,
      "route_exact": 0.990210740992522
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
        "loss": 0.06171910825063492,
        "step_acc": 0.9748061685268808,
        "epoch_sec": 12.85201644897461,
        "eta_sec": 51.408071517944336
      },
      {
        "epoch": 2,
        "loss": 0.0032445134881176644,
        "step_acc": 0.9989832722728693,
        "epoch_sec": 12.946133375167847,
        "eta_sec": 38.69732987880707
      },
      {
        "epoch": 3,
        "loss": 0.001996012564008805,
        "step_acc": 0.9993808752946522,
        "epoch_sec": 12.77202320098877,
        "eta_sec": 25.71354087193807
      },
      {
        "epoch": 4,
        "loss": 0.0016220624815867248,
        "step_acc": 0.9995115162875238,
        "epoch_sec": 12.675229549407959,
        "eta_sec": 12.811405539512634
      },
      {
        "epoch": 5,
        "loss": 0.0011585159576237194,
        "step_acc": 0.9996648774530686,
        "epoch_sec": 12.898380756378174,
        "eta_sec": 0.0
      }
    ]
  },
  "pass_checks": {
    "kernel_reads_heap_content": true,
    "no_geometry_answer_features": true,
    "ood_route_exact_ge_0_99": true,
    "ood_step_acc_ge_0_99": true,
    "flat_length_matrix_fails_unseen_lengths": true
  },
  "pilot_pass": true,
  "limits": [
    "query token is supervised",
    "uses bag/count summaries, not full semantic vectors",
    "unique-token query positions only",
    "not translation",
    "not unsupervised span discovery",
    "needs pointer/shared-flat baselines next"
  ]
}
```
