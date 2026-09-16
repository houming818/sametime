# Real-Corpus Annealed Token Space

Claim: `S1-ANNEAL-SPACE-REAL-C01`

## Mean held-out context NLL

- `tree`: `4.363298 +/- 0.003738`
- `frequency_shuffle`: `4.764650 +/- 0.048135`
- `random_route`: `4.449122 +/- 0.010933`
- `flat_kmeans`: `4.374225 +/- 0.000308`
- `expected_sgns`: `4.320400 +/- 0.003769`
- `token_reference`: `4.375818 +/- 0.000634`
- `global_reference`: `4.504460 +/- 0.000160`

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
