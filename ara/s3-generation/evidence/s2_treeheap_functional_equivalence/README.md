# WMT TreeHeap Functional Equivalence

```json
{
  "claim": "S2-TREEHEAP-FUNCTIONAL-EQUIV-C01",
  "terminology": "TreeHeap/subheap only; functional equivalence is a relation, not a new object type",
  "host": "io",
  "seconds": 8.807716846466064,
  "config": {
    "checkpoint_dir": "ara/s3-generation/evidence/s3_wmt_frontier_smoke",
    "output": "ara/s3-generation/evidence/s2_treeheap_functional_equivalence/summary.json",
    "groups": 8,
    "batch_size": 16,
    "seed": 71501,
    "device": "cuda"
  },
  "source_checkpoint_config": {
    "data": "/mnt/nas/datasets/wmt17/train.zh-en",
    "spm_model": "/mnt/nas/datasets/wmt17/sp_bpe.model",
    "evidence_dir": "ara/s3-generation/evidence/s3_wmt_frontier_smoke",
    "model": "all",
    "seed": 17,
    "train_samples": 5000,
    "valid_samples": 500,
    "test_samples": 500,
    "max_scan": 100000,
    "min_len": 9,
    "max_len": 24,
    "dim": 192,
    "hidden": 192,
    "batch_size": 32,
    "epochs": 5,
    "lr": 0.002,
    "device": "cuda",
    "num_workers": 0
  },
  "models": {
    "learned_frontier": {
      "checkpoint": "ara/s3-generation/evidence/s3_wmt_frontier_smoke/checkpoint_best_learned_frontier.pt",
      "test_examples": 500,
      "distance_matched_examples": 500,
      "distance_match_coverage": 1.0,
      "groups_per_address": 8,
      "baseline_nll": 6.567046642303467,
      "interventions": {
        "same": {
          "nll": 6.585699081420898,
          "nll_delta": 0.018652290105819702,
          "mean_kl": 0.022079771384596825,
          "argmax_flip_rate": 0.10180994123220444,
          "cosine_distance": 0.6201698184013367
        },
        "different": {
          "nll": 6.587571144104004,
          "nll_delta": 0.020525086671113968,
          "mean_kl": 0.021363073959946632,
          "argmax_flip_rate": 0.10600896179676056,
          "cosine_distance": 0.6202221512794495
        },
        "random": {
          "nll": 6.592098236083984,
          "nll_delta": 0.02505127713084221,
          "mean_kl": 0.033323463052511215,
          "argmax_flip_rate": 0.1425551027059555,
          "cosine_distance": 0.8632275462150574
        }
      },
      "different_minus_same_nll_delta": 0.0018727988936007023,
      "different_minus_same_bootstrap_95": [
        -0.003949011754989624,
        0.007450367069244385
      ],
      "distance_match_error": 5.233287811279297e-05
    },
    "fixed_frontier": {
      "checkpoint": "ara/s3-generation/evidence/s3_wmt_frontier_smoke/checkpoint_best_fixed_frontier.pt",
      "test_examples": 500,
      "distance_matched_examples": 465,
      "distance_match_coverage": 0.9300000071525574,
      "groups_per_address": 8,
      "baseline_nll": 6.588379859924316,
      "interventions": {
        "same": {
          "nll": 6.597074508666992,
          "nll_delta": 0.008695456199347973,
          "mean_kl": 0.010821543633937836,
          "argmax_flip_rate": 0.09140835702419281,
          "cosine_distance": 0.3837623596191406
        },
        "different": {
          "nll": 6.594222545623779,
          "nll_delta": 0.005843945778906345,
          "mean_kl": 0.010200649499893188,
          "argmax_flip_rate": 0.07996297627687454,
          "cosine_distance": 0.38404878973960876
        },
        "random": {
          "nll": 6.606139659881592,
          "nll_delta": 0.017761122435331345,
          "mean_kl": 0.029654357582330704,
          "argmax_flip_rate": 0.13400331139564514,
          "cosine_distance": 0.7898979187011719
        }
      },
      "different_minus_same_nll_delta": -0.0028515097219496965,
      "different_minus_same_bootstrap_95": [
        -0.007925566934770152,
        0.0019853275052962734
      ],
      "distance_match_error": 0.00028643012046813965
    },
    "random_frontier": {
      "checkpoint": "ara/s3-generation/evidence/s3_wmt_frontier_smoke/checkpoint_best_random_frontier.pt",
      "test_examples": 500,
      "distance_matched_examples": 455,
      "distance_match_coverage": 0.9100000262260437,
      "groups_per_address": 8,
      "baseline_nll": 6.6703972816467285,
      "interventions": {
        "same": {
          "nll": 6.682459831237793,
          "nll_delta": 0.01206196192651987,
          "mean_kl": 0.032090868800878525,
          "argmax_flip_rate": 0.11770322173833847,
          "cosine_distance": 0.6733991503715515
        },
        "different": {
          "nll": 6.676748752593994,
          "nll_delta": 0.00635165860876441,
          "mean_kl": 0.025053314864635468,
          "argmax_flip_rate": 0.12007533013820648,
          "cosine_distance": 0.6731358170509338
        },
        "random": {
          "nll": 6.686590194702148,
          "nll_delta": 0.016192492097616196,
          "mean_kl": 0.03884636238217354,
          "argmax_flip_rate": 0.14985211193561554,
          "cosine_distance": 0.8792923092842102
        }
      },
      "different_minus_same_nll_delta": -0.005710303317755461,
      "different_minus_same_bootstrap_95": [
        -0.011965583051953997,
        0.0007451235069023361
      ],
      "distance_match_error": 0.0002633333206176758
    }
  },
  "gates": {
    "P1_distance_matched": true,
    "P2_learned_functional_gap": false,
    "P3_stronger_than_tree_controls": false
  },
  "decision": "not_supported"
}
```
