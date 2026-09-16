# Real-Corpus Annealed Token Space

Claim: `S1-ANNEAL-SPACE-REAL-C01`

## Mean held-out context NLL

- `tree`: `5.358955 +/- 0.019523`
- `frequency_shuffle`: `5.820611 +/- 0.028594`
- `random_route`: `5.461013 +/- 0.002996`
- `flat_kmeans`: `5.362574 +/- 0.003063`
- `expected_sgns`: `5.203250 +/- 0.000802`
- `token_reference`: `5.208899 +/- 0.000502`
- `global_reference`: `5.569782 +/- 0.000036`

## Gates

- `tree_nll_better_than_global`: `true`
- `tree_nll_better_than_frequency_shuffle`: `true`
- `tree_nll_better_than_random_route`: `true`
- `tree_stability_better_than_frequency_shuffle`: `true`
- `tree_stability_better_than_random_route`: `true`
- `tree_sgns_overlap_gap_ge_0_02`: `true`
- `tree_flat_nll_gap_le_0_15`: `true`
- `tree_leaf_utilization_ge_0_75`: `true`
- `tree_occupancy_entropy_ge_0_75`: `true`
- `fold_conservation_le_1e_6`: `true`

Decision: `supported controlled real-corpus pilot`

This is a corpus-structure audit, not a generation or translation result.
