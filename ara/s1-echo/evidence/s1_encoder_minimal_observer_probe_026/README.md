# S1 Encoder Minimal Observer Probe

This run tests the DS/Houming minimal gate:

```text
L = L_echo + L_context
Theta_place learns object -> prefix placement.
Theta_compose learns prefix/internal-node state from assigned leaves.
Gold categories are hidden during training and used only for audit.
```

## Config

```json
{
  "seeds": 8,
  "epochs": 1200,
  "dim": 32,
  "k_values": [
    4,
    6,
    8
  ],
  "lr": 0.03,
  "echo_weight": 0.1,
  "device": "cuda",
  "report_every": 0
}
```

## Summary

```json
{
  "claim": "S1-ENCODER-OBS-C01",
  "experiment": "P-S1-ENCODER-OBS01-minimal",
  "config": {
    "seeds": 8,
    "epochs": 1200,
    "dim": 32,
    "k_values": [
      4,
      6,
      8
    ],
    "lr": 0.03,
    "echo_weight": 0.1,
    "device": "cuda",
    "report_every": 0
  },
  "vocab": {
    "verbs": [
      "eat",
      "cook",
      "order",
      "take",
      "prescribe",
      "buy",
      "drink",
      "pour",
      "serve",
      "wear",
      "wash",
      "fold",
      "drive",
      "park",
      "repair",
      "visit",
      "leave",
      "enter"
    ],
    "objects": [
      "rice",
      "noodle",
      "apple",
      "amoxicillin",
      "ibuprofen",
      "aspirin",
      "water",
      "milk",
      "tea",
      "shirt",
      "hoodie",
      "coat",
      "car",
      "truck",
      "bike",
      "paris",
      "museum",
      "school"
    ],
    "categories": [
      "food",
      "medicine",
      "beverage",
      "clothing",
      "vehicle",
      "place"
    ],
    "object_category": [
      0,
      0,
      0,
      1,
      1,
      1,
      2,
      2,
      2,
      3,
      3,
      3,
      4,
      4,
      4,
      5,
      5,
      5
    ]
  },
  "summary": {
    "by_mode": {
      "shuffled": {
        "cluster_purity": {
          "mean": 0.4444444444444445,
          "std": 0.07507376941024183,
          "n": 24
        },
        "pairwise_f1": {
          "mean": 0.16293176595173714,
          "std": 0.06568440236585513,
          "n": 24
        },
        "heldout_top1": {
          "mean": 0.05555555555555557,
          "std": 0.032764879145532694,
          "n": 24
        },
        "heldout_top3": {
          "mean": 0.13657407407407404,
          "std": 0.06340483857875287,
          "n": 24
        },
        "heldout_mrr": {
          "mean": 0.1854053314378069,
          "std": 0.033888511517879985,
          "n": 24
        },
        "heldout_beats_other_category": {
          "mean": 0.06481481481481484,
          "std": 0.03539011429281145,
          "n": 24
        },
        "full_context_cell_acc": {
          "mean": 0.6206275721391042,
          "std": 0.024923706916541933,
          "n": 24
        }
      },
      "structured": {
        "cluster_purity": {
          "mean": 0.7060185185185186,
          "std": 0.10029367327101385,
          "n": 24
        },
        "pairwise_f1": {
          "mean": 0.5348986325194723,
          "std": 0.12149632783828351,
          "n": 24
        },
        "heldout_top1": {
          "mean": 0.14814814814814817,
          "std": 0.04536092116265143,
          "n": 24
        },
        "heldout_top3": {
          "mean": 0.44444444444444436,
          "std": 0.1236847064202136,
          "n": 24
        },
        "heldout_mrr": {
          "mean": 0.36814098324514993,
          "std": 0.06159792280150572,
          "n": 24
        },
        "heldout_beats_other_category": {
          "mean": 0.4212962962962962,
          "std": 0.13401262815599962,
          "n": 24
        },
        "full_context_cell_acc": {
          "mean": 0.8652263308564822,
          "std": 0.03347353261301221,
          "n": 24
        }
      }
    },
    "structured_minus_shuffled": {
      "cluster_purity": 0.2615740740740741,
      "pairwise_f1": 0.3719668665677352,
      "heldout_top1": 0.09259259259259259,
      "heldout_top3": 0.30787037037037035,
      "heldout_mrr": 0.18273565180734302,
      "heldout_beats_other_category": 0.3564814814814814,
      "full_context_cell_acc": 0.2445987587173779
    }
  },
  "decision_hint": {
    "support_if": "structured beats shuffled on cluster_purity, pairwise_f1, and heldout transfer",
    "reject_or_redesign_if": "shuffled matches structured, or heldout transfer does not improve"
  }
}
```

## Example Structured Assignment

- `rice` gold=`food` learned_prefix=`7`
- `noodle` gold=`food` learned_prefix=`7`
- `apple` gold=`food` learned_prefix=`7`
- `amoxicillin` gold=`medicine` learned_prefix=`5`
- `ibuprofen` gold=`medicine` learned_prefix=`6`
- `aspirin` gold=`medicine` learned_prefix=`5`
- `water` gold=`beverage` learned_prefix=`0`
- `milk` gold=`beverage` learned_prefix=`0`
- `tea` gold=`beverage` learned_prefix=`0`
- `shirt` gold=`clothing` learned_prefix=`2`
- `hoodie` gold=`clothing` learned_prefix=`2`
- `coat` gold=`clothing` learned_prefix=`2`
- `car` gold=`vehicle` learned_prefix=`1`
- `truck` gold=`vehicle` learned_prefix=`4`
- `bike` gold=`vehicle` learned_prefix=`1`
- `paris` gold=`place` learned_prefix=`2`
- `museum` gold=`place` learned_prefix=`2`
- `school` gold=`place` learned_prefix=`2`

## Interpretation Gate

Support requires structured corpus to beat shuffled control on cluster
purity/pairwise-F1 and held-out transfer. If shuffled matches it, the
encoder is fitting frequency noise rather than observation structure.

This proof does not claim natural-language semantics or WMT translation.
It only checks whether a learnable TreeHeap placement/compose kernel can
induce reusable internal nodes from observation statistics.
