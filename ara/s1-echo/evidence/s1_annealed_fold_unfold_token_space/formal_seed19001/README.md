# S1 Annealed FOLD/UNFOLD Token Space

Claim: `S1-ANNEAL-SPACE-C01`

Elapsed seconds: `27.206`

No trainable token embedding is used. Labels are audit-only.

## Means

| arm | heldout NLL | purity | pair F1 | route top3 | utilization | occupancy H | stability |
|---|---:|---:|---:|---:|---:|---:|---:|
| structured | 2.774492 | 1.000000 | 0.743772 | 0.890191 | 0.994792 | 0.944072 | 0.709364 |
| shuffled | 8.650482 | 0.399089 | 0.073965 | 0.114800 | 0.997396 | 0.976023 | 0.229626 |
| random_route | 3.688609 | 0.390625 | 0.084977 | 0.110460 | 0.971354 | 0.947120 | 0.236842 |

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
  "seeds": 24,
  "seed_start": 19001,
  "categories": 8,
  "tokens_per_category": 8,
  "contexts_per_category": 8,
  "global_contexts": 8,
  "train_observations": 1200,
  "test_observations": 600,
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
  "em_steps": 30,
  "alpha": 0.05,
  "device": "cuda"
}
```
