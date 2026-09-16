# S1 Annealed FOLD/UNFOLD Token Space

Claim: `S1-ANNEAL-SPACE-C01`

Elapsed seconds: `1.409`

No trainable token embedding is used. Labels are audit-only.

## Means

| arm | heldout NLL | purity | pair F1 | route top3 | utilization | occupancy H | stability |
|---|---:|---:|---:|---:|---:|---:|---:|
| structured | 2.791609 | 1.000000 | 0.754941 | 0.768229 | 1.000000 | 0.941123 | 0.638921 |
| shuffled | 7.631868 | 0.398438 | 0.100211 | 0.143229 | 0.968750 | 0.958499 | 0.207036 |
| random_route | 3.732527 | 0.375000 | 0.070343 | 0.109375 | 0.968750 | 0.947733 | 0.226105 |

## Gates

- `purity_vs_shuffled_ge_0_20`: `true`
- `purity_vs_random_ge_0_20`: `true`
- `pairwise_f1_vs_shuffled_ge_0_20`: `true`
- `nll_better_than_shuffled`: `true`
- `nll_better_than_random`: `true`
- `stability_better_than_shuffled`: `true`
- `leaf_utilization_ge_0_75`: `true`
- `occupancy_entropy_ge_0_80`: `true`
- `conservation_le_1e_6`: `true`

Decision: `supported pilot`

The result is limited to a controlled probability-law corpus. It does not prove WMT semantics or generation.

## Config

```json
{
  "seeds": 2,
  "seed_start": 19001,
  "categories": 8,
  "tokens_per_category": 8,
  "contexts_per_category": 8,
  "global_contexts": 8,
  "train_observations": 300,
  "test_observations": 150,
  "depth": 4,
  "temperatures": [
    4.0,
    2.0,
    1.0,
    0.5,
    0.25,
    0.125,
    0.0625
  ],
  "em_steps": 12,
  "alpha": 0.05,
  "device": "cuda"
}
```
